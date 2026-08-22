"""StandardExecutor — JSON tool-calling, no sandbox.

Validates Pydantic args, invokes tool.invoke(args, ctx), records a tool.call
OTEL span. Phase 5 enforces capability mask + Trust-vs-Capability gate inline.
Phase 6 A4 adds inline authz (Axis 2) + CredentialBroker authn (Axis 1).
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from pydantic import BaseModel

from voussoir.agent.policy import PolicyViolation, PolicyViolationError
from voussoir.auth.decision import AuthzDecision
from voussoir.auth.errors import AuthenticationFailedError, MissingCredentialError
from voussoir.auth.protocol import Authorizer, CredentialBroker
from voussoir.container import Container
from voussoir.guardrails.trust import Trust
from voussoir.observability import span as otel_span
from voussoir.observability.logging_setup import get_logger
from voussoir.observability.metrics import AUTHZ_DECISIONS, CAPABILITY_DENIALS, TAINT_EXFIL_BLOCKS
from voussoir.tools.protocol import Capability, Tool, ToolContext

_log = get_logger(__name__)

_fallback_warned = False
# Separate native-structlog logger for the warn-once path so structlog.testing.capture_logs()
# can intercept it (stdlib-backed _log is not intercepted by capture_logs).
_fallback_log = structlog.get_logger(__name__)


def _warn_fallback_once() -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    _fallback_log.warning(
        "authz_fallback_used",
        authorizer="_FallbackAllowAllAuthorizer",
        hint=(
            "StandardExecutor fell back to _FallbackAllowAllAuthorizer because no "
            "Authorizer is bound on the container. Bind a concrete Authorizer "
            "(or use default_container() which binds AllowAllAuthorizer with its "
            "own one-time warning) for production."
        ),
    )


_CAP_TO_TRUST_ON_OUTPUT: dict[Capability, Trust] = {
    Capability.READ_PUBLIC: Trust.UNTRUSTED,
    Capability.READ_PRIVATE: Trust.INTERNAL,
    Capability.WRITE_PRIVATE: Trust.INTERNAL,
    Capability.EXFILTRATION: Trust.INTERNAL,
}


class _FallbackAllowAllAuthorizer:
    """Used by StandardExecutor when no Authorizer is bound on the container.

    `default_container()` binds a fail-closed `DenyByDefaultAuthorizer`, so this
    fallback only fires for hand-built containers that omit an Authorizer. It
    exists so the executor never has a None Authorizer; it warns once per
    process that authz is unenforced and should not be relied on in production.
    """

    name = "allow_all_fallback"

    async def authorize(
        self,
        principal: Any,
        tool: Any,
        args: Any,
        ctx: Any,
    ) -> AuthzDecision:
        del principal, tool, args, ctx
        return AuthzDecision(decision="ALLOW", authorizer_name=self.name)


def _resolve_authorizer(container: Container | None) -> Any:
    """Resolve Authorizer from container; fall back to _FallbackAllowAllAuthorizer."""
    if container is None:
        _warn_fallback_once()
        return _FallbackAllowAllAuthorizer()
    try:
        # Cast to suppress mypy type-abstract complaint: Authorizer is a
        # runtime_checkable Protocol so container.resolve accepts it at runtime.
        return container.resolve(cast("type[Any]", Authorizer))
    except LookupError:
        _warn_fallback_once()
        return _FallbackAllowAllAuthorizer()


def _resolve_broker(container: Container | None) -> CredentialBroker:
    """Resolve CredentialBroker from container; raise MissingCredentialError if absent."""
    if container is None:
        raise MissingCredentialError(
            "Tool declares auth_requirement but ToolContext has no container"
        )
    try:
        # Cast to suppress mypy type-abstract complaint: CredentialBroker is a
        # runtime_checkable Protocol so container.resolve accepts it at runtime.
        result: CredentialBroker = container.resolve(cast("type[Any]", CredentialBroker))
        return result
    except LookupError as exc:
        raise MissingCredentialError(
            "Tool declares auth_requirement but no CredentialBroker is bound"
        ) from exc


def _redact_masked_fields(result: Any, masked_fields: list[str]) -> Any:
    """Apply masked_fields redaction to a tool result, recursively.

    Dicts: keys in masked_fields replaced with '[REDACTED]'; non-masked values
    are themselves recursed into so nested matches are caught.
    Lists / tuples: each element is recursed into (preserving the container
    type for tuples).
    Pydantic BaseModels: dump -> recurse -> validate back to the same class
    when possible; falls back to the dumped dict on validation failure.
    Primitives / other: returned unchanged (no field structure to redact).

    A key matches if it appears anywhere in the tree — `masked_fields=["ssn"]`
    redacts `{"user": {"ssn": "..."}}` and `[{"ssn": "..."}, {"ssn": "..."}]`
    alike, which is the depth users expect when they declare a field to be
    sensitive.
    """
    masked = frozenset(masked_fields)
    return _redact(result, masked)


def _redact(result: Any, masked: frozenset[str]) -> Any:
    if isinstance(result, dict):
        return {k: "[REDACTED]" if k in masked else _redact(v, masked) for k, v in result.items()}
    if isinstance(result, list):
        return [_redact(item, masked) for item in result]
    if isinstance(result, tuple):
        return tuple(_redact(item, masked) for item in result)
    if isinstance(result, BaseModel):
        try:
            dumped = result.model_dump()
        except Exception:
            return result
        redacted = _redact(dumped, masked)
        try:
            return type(result).model_validate(redacted)
        except Exception:
            return redacted
    return result


def _trust_for_output(tool: Tool) -> Trust:
    """Return the Trust tag a tool's output should carry.

    Tool-declared `output_trust` overrides the default mapping. Otherwise the
    Cap-to-Trust mapping is iterated in declaration order so the lowest-trust
    bit wins: a `READ_PUBLIC | READ_PRIVATE` tool returns `UNTRUSTED` because
    `READ_PUBLIC` is listed first in `_CAP_TO_TRUST_ON_OUTPUT`. A
    `Capability.NONE` tool falls through to `Trust.SYSTEM`.
    """
    override = cast("Trust | None", getattr(tool, "_output_trust", None))
    if override is not None:
        return override
    for cap, trust in _CAP_TO_TRUST_ON_OUTPUT.items():
        if tool.capability & cap:
            return trust
    return Trust.SYSTEM


class StandardExecutor:
    """v1 tool executor — direct invoke, no sandbox."""

    name = "standard"

    async def invoke(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> Any:
        with otel_span(
            "tool.call",
            tool_name=tool.name,
            run_id=ctx.run_id,
            capability=int(tool.capability),
            trust_in=str(sorted(str(t) for t in ctx.taint)),
        ) as span:
            # PHASE 6 A4+A6: AUTHZ (Axis 2) — §12.1 ordering step 1.
            authorizer = _resolve_authorizer(ctx.container)
            with otel_span(
                "authz.check",
                authorizer_name=authorizer.name,
            ) as authz_span:
                decision = await authorizer.authorize(ctx.principal, tool, args, ctx)
                authz_span.set_attribute("decision", decision.decision)
                authz_span.set_attribute("reason", decision.reason)

            record = AuthzDecision(
                decision=decision.decision,
                reason=decision.reason,
                masked_fields=list(decision.masked_fields),
                authorizer_name=authorizer.name,
            )
            ctx.authz_decisions.append(record)
            AUTHZ_DECISIONS.add(
                1,
                {
                    "decision": decision.decision,
                    "authorizer_name": authorizer.name,
                    "tool_name": tool.name,
                },
            )
            if decision.decision == "DENY":
                # Prepend authorizer name so chained-authorizer denials are debuggable
                # without inspecting AgentResult.authz_decisions.
                reason = (
                    f"[{authorizer.name}] {decision.reason}"
                    if decision.reason
                    else f"[{authorizer.name}]"
                )
                raise PolicyViolationError(PolicyViolation.AUTHZ_DENIED, reason)

            # Rule 1 — Capability mask: tool's caps must be subset of agent's allowed.
            with otel_span(
                "capability.check",
                required=int(tool.capability),
                allowed=int(ctx.allowed_capabilities),
            ) as cap_span:
                cap_passed = (tool.capability & ctx.allowed_capabilities) == tool.capability
                cap_span.set_attribute("passed", cap_passed)
                if not cap_passed:
                    span.set_attribute("policy.violation", PolicyViolation.CAPABILITY_DENIED.value)
                    CAPABILITY_DENIALS.add(
                        1,
                        {
                            "tool_name": tool.name,
                            "required_capability": str(tool.capability),
                        },
                    )
                    raise PolicyViolationError(
                        PolicyViolation.CAPABILITY_DENIED,
                        f"Tool {tool.name!r} requires {tool.capability!r}; "
                        f"agent allows {ctx.allowed_capabilities!r}",
                    )
            # Rule 2 — Trust gate: EXFILTRATION cannot run with UNTRUSTED in taint.
            with otel_span(
                "taint.check",
                tool_caps=int(tool.capability),
                run_taint=str(sorted(str(t) for t in ctx.taint)),
            ) as taint_span:
                taint_passed = not (
                    (tool.capability & Capability.EXFILTRATION) and Trust.UNTRUSTED in ctx.taint
                )
                taint_span.set_attribute("passed", taint_passed)
                if not taint_passed:
                    span.set_attribute("policy.violation", PolicyViolation.TAINT_EXFILTRATION.value)
                    TAINT_EXFIL_BLOCKS.add(1, {"tool_name": tool.name})
                    raise PolicyViolationError(
                        PolicyViolation.TAINT_EXFILTRATION,
                        f"Tool {tool.name!r} (EXFILTRATION) cannot run with "
                        f"UNTRUSTED content in run taint",
                    )

            # PHASE 6 A4: AUTHN (Axis 1) — §12.1 ordering step 5.
            auth_req = getattr(tool, "auth_requirement", None)
            if auth_req is not None:
                broker = _resolve_broker(ctx.container)
                ctx.credentials = await broker.resolve(auth_req, ctx.principal, ctx)

            # Argument logging is opt-IN, not opt-out. This line runs at INFO
            # on every tool call and routinely carries user data; requiring
            # each tool to remember `sensitive_args` made leaking the default
            # (audit, minor). Now: names always, values only for tools that
            # declare `log_args=True`, and declared `sensitive_args` are
            # redacted even then.
            _sensitive_args = set(getattr(tool, "sensitive_args", []) or [])
            _dumped_args = args.model_dump()
            if getattr(tool, "log_args", False):
                _logged_args: dict[str, object] = {
                    k: "[REDACTED]" if k in _sensitive_args else v for k, v in _dumped_args.items()
                }
            else:
                _logged_args = dict.fromkeys(_dumped_args, "[NOT_LOGGED]")
            _log.info(
                "tool.invoke",
                tool=tool.name,
                run_id=ctx.run_id,
                args=_logged_args,
            )

            # §12.1 step 6: tool.invoke with one-shot 401/403 retry (step 7).
            try:
                result = await tool.invoke(args, ctx)
            except AuthenticationFailedError:
                if ctx.credentials is None or ctx.container is None:
                    raise
                broker = _resolve_broker(ctx.container)
                try:
                    ctx.credentials = await broker.refresh(ctx.credentials)
                except MissingCredentialError as exc:
                    # v1.0.4 E6 / I2: broker.refresh() raising MissingCredentialError
                    # signals a permanent break (revoked refresh token, broker offline)
                    # rather than a transient tool failure. Wrap as
                    # AuthenticationFailedError so the outer except-block treats it
                    # as a clear auth signal instead of an opaque TOOL_ERROR string.
                    raise AuthenticationFailedError(
                        f"Tool {tool.name!r} auth refresh failed permanently: {exc}. "
                        "Check the broker's refresh-token state."
                    ) from exc

                try:
                    result = await tool.invoke(args, ctx)
                except AuthenticationFailedError as exc:
                    # v1.0.4 E6 / I2: second auth failure after a refresh attempt
                    # is a strong "credentials are revoked" signal. Re-raise with
                    # a clear permanent-failure framing rather than the raw 401.
                    raise AuthenticationFailedError(
                        f"Tool {tool.name!r} authentication still failed after refresh. "
                        "Credentials may have been revoked; bind a fresh broker."
                    ) from exc

            # §12.1 step 8: MASK redaction.
            if decision.decision == "MASK" and decision.masked_fields:
                result = _redact_masked_fields(result, decision.masked_fields)

            # Tag output trust into the run's taint set.
            trust_out = _trust_for_output(tool)
            ctx.taint.add(trust_out)
            span.set_attribute("trust_out", str(trust_out))
            return result
