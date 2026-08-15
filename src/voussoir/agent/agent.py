"""Agent — voussoir's user-facing agent class.

Phase 1 surface:
- single-pass run when no tools are bound
- multi-step tool-calling loop (Task 1.13) when tools are present, with budget
  checks via AgentPolicy at every step boundary
"""

from __future__ import annotations

import contextvars
import copy
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, NamedTuple, cast

from ctxforge.protocols.llm import ChatMessage, ILLMProvider

from voussoir.agent import delegation
from voussoir.agent.cascade import Decision, RequestCascade
from voussoir.agent.context import AgentContext
from voussoir.agent.delegate import IDelegate, NamedDelegate
from voussoir.agent.dispatch import (
    apply_guardrail_verdict,
    last_sub_result_var,
    make_delegate_invoker,
    parent_ctx_var,
)
from voussoir.agent.policy import AgentPolicy
from voussoir.agent.result import AgentEvent, AgentResult, CascadeOutcome, FinishReason, Step
from voussoir.agent.stream_events import (
    cascade_failed_event,
    cascade_passed_event,
    done_event,
    post_dispatch_events,
    pre_dispatch_events,
    token_event,
)
from voussoir.agent.turn import tool_turn_dispatch, tool_turn_prepare
from voussoir.auth.principal import Principal, default_principal
from voussoir.container import Container
from voussoir.executors import IToolExecutor
from voussoir.executors.standard import StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain, GuardrailPayload, IGuardrailChain
from voussoir.observability import span as otel_span
from voussoir.observability.logging_setup import get_logger
from voussoir.observability.metrics import (
    CASCADE_ESCALATIONS,
    COST_USD,
    DURATION_MS,
    TOKENS_IN,
    TOKENS_OUT,
)
from voussoir.observability.sink import (
    BufferedTelemetrySink,
    ITelemetrySink,
    NullTelemetrySink,
)
from voussoir.tools import Capability
from voussoir.tools.registry import ToolRegistry

# Module-level constants and helpers. Sub-module imports above are flat per
# the project's "constants after all imports" rule.

_log = get_logger(__name__)

# `FinishReason` is the canonical Literal alias for AgentResult.finish_reason;
# canonical home is voussoir.agent.result, re-imported above.


class _DelegationLineage(NamedTuple):
    """One-shot lineage stash passed from `Agent.delegate` to a sub-agent's
    next `Agent.run` invocation. Named fields prevent positional-tuple
    swap bugs.
    """

    chain: list[str]
    depth: int
    parent_run_id: str
    max_depth: int


# Counts cascade re-entries (NOT SAS attempts). _run_with_cascade increments
# on entry and resets on exit; checks `current >= cascade.max_cascade_depth`
# before doing any work. Defends against escalation chains that loop back
# to a cascade-aware Agent (e.g. A's escalation is B; B's escalation is A).
_cascade_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "voussoir_cascade_depth", default=0
)


def _cost_from_tokens(tokens_in: int, tokens_out: int, *, model: str | None = None) -> float:
    """Estimated USD cost from running token totals, priced per model.

    Uses the per-model table in voussoir.llm.pricing (with a conservative
    fallback for unknown models) instead of the previous hardcoded
    ``$10/$30 per 1M`` formula, which was wrong for every current provider.
    """
    from voussoir.llm.pricing import compute_cost

    return compute_cost(model, tokens_in, tokens_out)


def _estimate_stream_tokens(
    llm: ILLMProvider, messages: list[ChatMessage], full_output: str
) -> tuple[int, int]:
    """Use this to estimate tokens_in/tokens_out after an llm.stream() call.

    ILLMProvider.stream() yields raw `str` chunks without usage totals, so
    Agent.stream's simple path can't pull tokens from the LLM response the
    way the tool-using path does. v1.0.3 fills the gap with client-side
    count_tokens() — a best-effort approximation (provider's tokenizer may
    differ slightly) that beats reporting 0. Returns (0, 0) silently if
    count_tokens raises or returns a non-int (e.g. an un-configured
    MagicMock in tests, or a provider that doesn't implement it).
    """
    try:
        _in = sum(llm.count_tokens(m.content) for m in messages if m.content)
        _out = llm.count_tokens(full_output)
        if isinstance(_in, int) and isinstance(_out, int):
            return _in, _out
    except Exception:
        pass
    return 0, 0


async def _safe_hook_fanout(
    middlewares: list[Any],
    hook_name: Literal["after_step", "after_run", "on_error"],
    *args: Any,
) -> None:
    """Call *hook_name* on every middleware; isolate each from sibling failures."""
    warning_key = f"middleware_{hook_name}_failed"
    for mw in middlewares:
        try:
            await getattr(mw, hook_name)(*args)
        except Exception:
            _log.warning(warning_key, middleware=type(mw).__name__, exc_info=True)


class _RunSetup(NamedTuple):
    """Shared scaffolding: registry + messages + synthetic_tools per run."""

    registry: ToolRegistry
    messages: list[ChatMessage]
    synthetic_tools: list[Any]


class Agent:
    """A voussoir agent: an LLM driving a tool-call loop, optionally with
    declared sub-agent delegates and an SAS-first cascade gate.
    """

    def __init__(
        self,
        name: str,
        *,
        instructions: str | list[str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[Any] | None = None,
        skills: list[str] | None = None,
        policy: AgentPolicy | None = None,
        container: Container | None = None,
        description: str = "",
        delegates: list[IDelegate | str] | None = None,
        cascade: RequestCascade | None = None,
        max_delegation_depth: int = 3,
        allowed_capabilities: Capability = Capability.READ_PUBLIC | Capability.READ_PRIVATE,
        # v1.1.0 F4: per-Agent overrides (precedence: run > init > container > fallback)
        executor: IToolExecutor | None = None,
        guardrail_chain: IGuardrailChain | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.allowed_capabilities = allowed_capabilities
        self.tools = tools or []
        # Reserve the synthetic delegate-tool prefix unconditionally — even
        # if no delegates are declared today, a user tool whose name starts
        # with `delegate_to_` will silently shadow a future synthetic tool
        # or get clobbered when delegates are added later.
        for t in self.tools:
            if getattr(t, "name", "").startswith(delegation.DELEGATE_TOOL_PREFIX):
                raise ValueError(
                    f"tool name {t.name!r} reserved: prefix "
                    f"{delegation.DELEGATE_TOOL_PREFIX!r} is for voussoir's "
                    f"synthetic delegate tools."
                )
        self.skills = list(skills) if skills else []
        self.policy = policy or AgentPolicy()
        self.middleware: list[Any] = []
        # Phase 4.5a P1 #21: container= is now required. Pre-4.5a the property
        # below lazily constructed default_container() — convenient for toy
        # examples but it failed late, deep in `.run()`, when (e.g.) the LLM
        # binding couldn't be resolved.
        if container is None:
            raise TypeError(
                f"Agent({name!r}) requires container=. Pass default_container() "
                "explicitly, or wire your own container. The lazy fallback was "
                "removed in Phase 4.5a (P1 #21)."
            )
        self._container = container
        raw_delegates = list(delegates) if delegates else []
        # Phase 4.5a P1 #23: fail-loud on unresolvable string delegates. Pre-4.5a
        # a typo or forgotten bind_agent_registry call surfaced only at first
        # delegate-tool invocation, six layers deep in a traceback.
        string_delegates = [d for d in raw_delegates if isinstance(d, str)]
        if string_delegates:
            from voussoir.agent.registry import AgentRegistry

            if not container.has(AgentRegistry):
                raise ValueError(
                    f"Agent {name!r} declares string delegates {string_delegates!r} "
                    "but no AgentRegistry is bound on the container. Call "
                    "bind_agent_registry(container) before constructing this Agent, "
                    "or pass the Agent instance directly instead of a name string."
                )
        self.delegates: list[IDelegate] = []
        for entry in raw_delegates:
            if isinstance(entry, str):
                self.delegates.append(NamedDelegate(entry))
            else:
                # Agent satisfies IDelegate; any other IDelegate impl
                # (future RemoteDelegate, PluginDelegate, user-defined)
                # passes through unchanged.
                self.delegates.append(entry)
        if self.delegates:
            delegation.check_delegate_collisions([d.name for d in self.delegates])
        self.cascade = cascade
        self.max_delegation_depth = max_delegation_depth
        # Lineage stash; populated by Agent.delegate, consumed one-shot by
        # Agent.run on the next invocation.
        self._delegation_lineage: _DelegationLineage | None = None
        # B5: resolve guardrail chain from the container; default to an empty chain
        # (no-op screen() that produces no audit records). Screening sites gate on
        # `if self.guardrail_chain.count():` so an empty chain skips the call.
        # v1.1.0 F2: resolve via IGuardrailChain Protocol key so custom chain
        # implementations bound under IGuardrailChain are picked up correctly.
        self.guardrail_chain: IGuardrailChain = container.resolve(
            IGuardrailChain,  # type: ignore[type-abstract]
            default=DefaultGuardrailChain([]),
        )
        # v1.1.0 F4: init-override slots for executor + guardrail_chain.
        # Consulted by _resolve_executor / _resolve_guardrail_chain;
        # run-kwarg beats these, these beat the container-resolved values.
        self._executor_override: IToolExecutor | None = executor
        self._guardrail_chain_init_override: IGuardrailChain | None = guardrail_chain

    def _with_container(
        self,
        container: Container,
        *,
        allowed_capabilities: Capability | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Shallow-copy with `_container` replaced; optional `allowed_capabilities`
        and `tools` overrides bake clamping into the clone (see Agent.delegate)."""
        clone = copy.copy(self)
        clone._container = container
        if allowed_capabilities is not None:
            clone.allowed_capabilities = allowed_capabilities
        if tools is not None:
            clone.tools = tools
        return clone

    @property
    def container(self) -> Container:
        # Phase 4.5a P1 #21: __init__ raises if container is None, so this
        # property always has a non-None value.
        assert self._container is not None
        return self._container

    def _resolve_executor(self, run_kwarg: IToolExecutor | None) -> IToolExecutor:
        """v1.1.0 F4: precedence ladder.

        run-kwarg > init-kwarg > container resolve > fallback StandardExecutor.
        """
        if run_kwarg is not None:
            return run_kwarg
        if self._executor_override is not None:
            return self._executor_override
        try:
            return self._container.resolve(IToolExecutor)  # type: ignore[type-abstract]
        except LookupError:
            return StandardExecutor()

    def _resolve_guardrail_chain(self, run_kwarg: IGuardrailChain | None) -> IGuardrailChain:
        """v1.1.0 F4: precedence ladder.

        run-kwarg > init-kwarg > self.guardrail_chain (container-resolved at __init__).
        """
        if run_kwarg is not None:
            return run_kwarg
        if self._guardrail_chain_init_override is not None:
            return self._guardrail_chain_init_override
        return self.guardrail_chain

    def _build_delegate_tools(self, ctx: AgentContext) -> list[Any]:
        """Synthesize the delegate-tool list for this Agent's delegates,
        gated by ctx.delegation_depth < ctx.max_delegation_depth.

        Returns [] when delegation is unavailable (depth cap reached, or
        no delegates declared). Extracted from Agent.run's inline loop so
        Agent.stream can use the same synthesis path (Phase 4c §3.5).
        """
        if not self.delegates or ctx.delegation_depth >= ctx.max_delegation_depth:
            return []
        synthetic: list[Any] = []
        for d in self.delegates:
            # Phase 5 A6: fail-loud at synthesis time for local Agent delegates.
            # AgentRef/NamedDelegate have no local tools; skip clamping for them.
            if isinstance(d, Agent):
                # Return value intentionally ignored — this call is the raise-only
                # synthesis-time check. Agent.delegate re-runs it at invoke time
                # to obtain the filtered tool list for the clone.
                delegation.clamp_tools(d, parent_mask=self.allowed_capabilities)
            synthetic.append(
                delegation.make_delegate_tool(
                    target_name=d.name,
                    target_description=d.description,
                    invoke=make_delegate_invoker(d),
                )
            )
        return synthetic

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        """IDelegate.delegate — execute self as a sub-agent for the caller.

        Owns local-Agent-specific setup that previously lived in the
        module-level helper that used to live here: child container scoping,
        lineage stash, and forwarding to `self.run(...)`.

        Depth-cap checking and post-run cost-aggregation stash via
        `last_sub_result_var` remain caller-side (the synthetic-tool
        invoker in `make_delegate_invoker`).
        """
        base_container = self._container if self._container is not None else parent_ctx.container
        child_container = base_container.child()
        # Phase 5 A6: enforce capability clamping at run time (defensive re-check).
        clamped_mask = parent_ctx.allowed_capabilities & self.allowed_capabilities
        clamped_tools = delegation.clamp_tools(self, parent_mask=parent_ctx.allowed_capabilities)
        sub_target = self._with_container(
            child_container,
            allowed_capabilities=clamped_mask,
            tools=clamped_tools,
        )

        sub_target._delegation_lineage = _DelegationLineage(
            chain=list(parent_ctx.delegation_chain) + [parent_ctx.agent_name],
            depth=parent_ctx.delegation_depth + 1,
            parent_run_id=parent_ctx.run_id,
            max_depth=parent_ctx.max_delegation_depth,
        )

        return await sub_target.run(
            task, user_id=parent_ctx.user_id, principal=parent_ctx.principal
        )

    async def run(
        self,
        input: str,
        *,
        session_id: str = "default",
        user_id: str = "local",
        principal: Principal | None = None,
        # v1.1.0 F4: per-call overrides (win over init-kwarg and container)
        executor: IToolExecutor | None = None,
        guardrail_chain: IGuardrailChain | None = None,
    ) -> AgentResult[str]:
        resolved_principal = principal or default_principal()
        resolved_executor = self._resolve_executor(executor)
        resolved_chain = self._resolve_guardrail_chain(guardrail_chain)
        if self.cascade is None:
            return await self._run_normal(
                input,
                session_id=session_id,
                user_id=user_id,
                principal=resolved_principal,
                executor=resolved_executor,
                guardrail_chain=resolved_chain,
            )
        return await self._run_with_cascade(
            input,
            session_id=session_id,
            user_id=user_id,
            principal=resolved_principal,
            executor=resolved_executor,
            guardrail_chain=resolved_chain,
        )

    async def _run_with_cascade(
        self,
        input: str,
        *,
        session_id: str = "default",
        user_id: str = "local",
        principal: Principal | None = None,
        executor: IToolExecutor,
        guardrail_chain: IGuardrailChain,
    ) -> AgentResult[str]:
        """Single-agent-first, escalate-to-multi-agent on validator failure.

        Attempt 0 (when sas_attempt_first): run self with delegates
        suppressed via _force_sas. Subsequent attempts: run
        cascade.escalation. Validator gates PASS → return; FAIL/AMBIGUOUS
        → next attempt. Exhausted attempts return the last result with
        finish_reason='error'.
        """
        # Real guard, not assert — asserts vanish under `python -O`, and
        # the caller contract (run() dispatches here only when cascade is
        # set) is enforced by mypy but not at runtime.
        cascade = self.cascade
        if cascade is None:
            raise RuntimeError("_run_with_cascade called with self.cascade=None")

        # Refuse cascade re-entry past the depth cap. This fires when an
        # escalation chain loops back to a cascade-aware Agent (directly
        # or via siblings); without it, such a config would recurse until
        # Python's own recursion limit.
        current_depth = _cascade_depth_var.get()
        if current_depth >= cascade.max_cascade_depth:
            raise RuntimeError(
                f"max_cascade_depth ({cascade.max_cascade_depth}) exceeded; "
                f"cascade.escalation likely loops back into a cascade-aware "
                f"Agent. Check the escalation chain for cycles."
            )
        depth_token = _cascade_depth_var.set(current_depth + 1)

        last_result: AgentResult[str] | None = None
        history: list[CascadeOutcome] = []

        try:
            for attempt in range(cascade.max_attempts):
                if attempt == 0 and cascade.sas_attempt_first:
                    attempt_name = self.name
                    result = await self._run_normal(
                        input,
                        session_id=session_id,
                        user_id=user_id,
                        principal=principal,
                        _force_sas=True,
                        executor=executor,
                        guardrail_chain=guardrail_chain,
                    )
                else:
                    if cascade.escalation is None:
                        break
                    attempt_name = cascade.escalation.name
                    result = await cascade.escalation.run(
                        input, session_id=session_id, user_id=user_id, principal=principal
                    )

                # Scope a buffered sink for the validate() call so any LLM
                # judge or other cost-emitting validator's records roll into
                # this attempt's result. Per-attempt buffer prevents
                # double-counting across cascade iterations. Fall back to a
                # fresh NullTelemetrySink if the container hasn't bound one
                # (some legacy tests construct ad-hoc containers).
                sink = self.container.resolve(
                    ITelemetrySink,  # type: ignore[type-abstract]
                    default=NullTelemetrySink(),
                )
                buffer = BufferedTelemetrySink()
                with (
                    sink.scoped(buffer),
                    otel_span("cascade.validate", validator_name=cascade.verifier.name) as cv_span,
                ):
                    decision = await cascade.verifier.validate(result, task=input)
                    cv_span.set_attribute("verdict", decision.value)
                from voussoir.agent.telemetry import merge_buffered_telemetry_into_result

                result = merge_buffered_telemetry_into_result(result, buffer.records)
                will_escalate = decision != Decision.PASS
                history.append(
                    CascadeOutcome(
                        attempted=attempt_name,
                        escalated=will_escalate,
                        reason=decision.value,
                    )
                )

                if decision == Decision.PASS:
                    return result.model_copy(update={"cascade_history": history})

                CASCADE_ESCALATIONS.add(1)
                last_result = result
                if cascade.escalation is None:
                    break

            if last_result is None:
                raise RuntimeError(
                    "cascade configured with no executable path; check "
                    "RequestCascade.sas_attempt_first / escalation"
                )
            return last_result.model_copy(
                update={"finish_reason": "error", "cascade_history": history}
            )
        finally:
            _cascade_depth_var.reset(depth_token)

    async def _run_normal(
        self,
        input: str,
        *,
        session_id: str = "default",
        user_id: str = "local",
        principal: Principal | None = None,
        _force_sas: bool = False,
        executor: IToolExecutor,
        guardrail_chain: IGuardrailChain,
    ) -> AgentResult[str]:
        t0 = time.monotonic()

        effective_middleware = self._effective_middleware()

        with otel_span(
            "agent.run",
            agent_name=self.name,
            model=self.model or "",
            cascade_attempt=0,
        ) as agent_span:
            async with await AgentContext.open(
                container=self.container,
                run_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                principal=principal,
                guardrail_chain=guardrail_chain,
            ) as ctx:
                await ctx.record_user_message(input)

                # Apply delegation lineage from one-shot stash (set by Agent.delegate).
                ctx.agent_name = self.name
                ctx.max_delegation_depth = self.max_delegation_depth
                ctx.allowed_capabilities = self.allowed_capabilities
                lineage = self._delegation_lineage
                if lineage is not None:
                    ctx.delegation_chain = lineage.chain
                    ctx.delegation_depth = lineage.depth
                    ctx.parent_run_id = lineage.parent_run_id
                    ctx.max_delegation_depth = lineage.max_depth
                    self._delegation_lineage = None

                # B5/B6: input-stage guardrail screening. Decisions accumulate on
                # ctx.guardrail_decisions (shared with _dispatch_one's tool_call +
                # tool_output stage records). Skipped when no chain was explicitly
                # bound on the container (default empty chain is a no-op).
                if guardrail_chain.count():
                    with otel_span("guardrail.input") as gi_span:
                        input_verdict = await guardrail_chain.screen(
                            GuardrailPayload(stage="input", content=input),
                            ctx,
                        )
                        gi_span.set_attribute("verdict", input_verdict.verdict)
                        gi_span.set_attribute("reason", input_verdict.reason or "")
                        gi_span.set_attribute(
                            "n_guardrails",
                            guardrail_chain.count(),
                        )
                    input, is_blocked = apply_guardrail_verdict(
                        input_verdict,
                        input,
                        stage="input",
                        ctx=ctx,
                        blocked_content=f"[blocked by input guardrail: {input_verdict.reason}]",
                    )
                    if is_blocked:
                        duration_ms = (time.monotonic() - t0) * 1000
                        agent_span.set_attribute("finish_reason", "blocked")
                        return AgentResult[str](
                            output=input,
                            trace_id=ctx.trace_id,
                            steps=[],
                            tokens_in=0,
                            tokens_out=0,
                            cost_usd=0.0,
                            duration_ms=duration_ms,
                            delegation_chain=[],
                            cascade_history=[],
                            guardrail_decisions=list(ctx.guardrail_decisions),
                            authz_decisions=list(ctx.authz_decisions),
                            taint=set(ctx.taint),
                            finish_reason="blocked",
                        )

                # Prepare before_run side-effects (e.g. SkillActivationMiddleware
                # populates ctx.skill_content) then build the registry / messages.
                ctx.skills_active = []
                ctx.skill_content = []
                for mw in effective_middleware:
                    await mw.before_run(ctx, input)
                setup = self._run_setup(
                    ctx,
                    input,
                    skill_content=getattr(ctx, "skill_content", []),
                    force_sas=_force_sas,
                )
                registry, messages, synthetic_tools = setup
                effective_tools = list(self.tools) + synthetic_tools
                llm: ILLMProvider = self.container.resolve(ILLMProvider)
                steps: list[Step] = []
                tokens_in = 0
                tokens_out = 0
                finish_reason: FinishReason = "completed"
                final_content = ""

                # Make ctx available to synthetic delegate tools via ContextVar.
                # last_sub_result_var is reset to None at run-loop entry so a
                # leftover from a previous (concurrent or earlier) run on the
                # same Task can't be misread as a phantom delegation; both
                # tokens are reset on exit, success or exception.
                _parent_ctx_token = parent_ctx_var.set(ctx)
                _sub_result_token = last_sub_result_var.set(None)
                try:
                    for step_idx in range(self.policy.max_steps + 1):
                        # Budget check before each LLM call.
                        duration_s = time.monotonic() - t0
                        violation = self.policy.check(
                            steps=step_idx,
                            duration_s=duration_s,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            cost_usd=_cost_from_tokens(tokens_in, tokens_out, model=self.model),
                        )
                        if violation is not None:
                            # AgentPolicy.check() only returns the 5 budget variants
                            # (MAX_STEPS/DURATION/INPUT_TOKENS/OUTPUT_TOKENS/COST);
                            # those values match the FinishReason literal one-for-one.
                            # Non-budget PolicyViolation variants (STREAMING_NOT_SUPPORTED,
                            # DELEGATE_NOT_FOUND, Phase 4.5a) are always raised, never
                            # returned, so this assignment is sound at runtime.
                            finish_reason = cast("FinishReason", violation.value)
                            final_content = ""
                            break

                        # Phase 4.5b B4b: per-turn body split into a
                        # two-phase API (tool_turn_prepare + tool_turn_dispatch)
                        # so Agent.stream can emit pre-dispatch events
                        # between the two calls. The run path doesn't emit
                        # events; the two-phase shape just mirrors stream's
                        # so both paths share one body.
                        # _cost_from_tokens(tokens_in, tokens_out) at the tail
                        # already accounts for aggregated sub-agent tokens, so
                        # dispatched.cost_delta is discarded here.
                        with otel_span("reason." + str(step_idx)):
                            prepared = await tool_turn_prepare(
                                llm=llm,
                                messages=messages,
                                registry=registry,
                                ctx=ctx,
                                model=self.model,
                                temperature=self.temperature,
                                has_tools=bool(effective_tools),
                                steps=steps,
                            )
                            tokens_in += prepared.tokens_in_delta
                            tokens_out += prepared.tokens_out_delta
                            if prepared.finish_reason == "stop":
                                final_content = prepared.response.content
                                break
                            # Append the assistant turn (with tool calls)
                            # BEFORE dispatching them so tool-result messages
                            # line up with their owning assistant message.
                            messages.append(prepared.assistant_message)
                            _steps_before = len(steps)
                            dispatched = await tool_turn_dispatch(
                                prepared=prepared,
                                registry=registry,
                                executor=executor,
                                ctx=ctx,
                                steps=steps,
                            )
                            tokens_in += dispatched.tokens_in_delta
                            tokens_out += dispatched.tokens_out_delta
                            messages.extend(dispatched.tool_result_messages)
                            for _step in steps[_steps_before:]:
                                await _safe_hook_fanout(
                                    effective_middleware, "after_step", ctx, _step
                                )
                except BaseException as _loop_exc:
                    await _safe_hook_fanout(effective_middleware, "on_error", ctx, _loop_exc)
                    raise
                finally:
                    last_sub_result_var.reset(_sub_result_token)
                    parent_ctx_var.reset(_parent_ctx_token)

                # B5/B6: output-stage guardrail screening. Records are appended to
                # ctx.guardrail_decisions alongside any tool_call + tool_output records
                # accumulated by _dispatch_one during the turn loop.
                # Skipped when no chain was explicitly bound on the container.
                try:
                    if guardrail_chain.count():
                        with otel_span("guardrail.output") as go_span:
                            output_verdict = await guardrail_chain.screen(
                                GuardrailPayload(stage="output", content=final_content),
                                ctx,
                            )
                            go_span.set_attribute("verdict", output_verdict.verdict)
                            go_span.set_attribute("reason", output_verdict.reason or "")
                        final_content, _ = apply_guardrail_verdict(
                            output_verdict,
                            final_content,
                            stage="output",
                            ctx=ctx,
                            blocked_content=f"[blocked by output guardrail: {output_verdict.reason}]",
                        )

                    await ctx.record_assistant_message(final_content)
                    duration_ms = (time.monotonic() - t0) * 1000
                    agent_span.set_attribute("finish_reason", finish_reason)
                    _run_attrs = {"model": self.model or "", "agent_name": self.name}
                    TOKENS_IN.add(tokens_in, _run_attrs)
                    TOKENS_OUT.add(tokens_out, _run_attrs)
                    COST_USD.add(
                        _cost_from_tokens(tokens_in, tokens_out, model=self.model),
                        {**_run_attrs, "delegation_depth": ctx.delegation_depth},
                    )
                    DURATION_MS.record(duration_ms, {"agent_name": self.name})
                    _result = AgentResult[str](
                        output=final_content,
                        trace_id=ctx.trace_id,
                        steps=steps,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=_cost_from_tokens(tokens_in, tokens_out, model=self.model),
                        duration_ms=duration_ms,
                        delegation_chain=delegation.build_chain(
                            self.name, ctx.delegation_chain, steps
                        ),
                        cascade_history=[],
                        guardrail_decisions=list(ctx.guardrail_decisions),
                        authz_decisions=list(ctx.authz_decisions),
                        taint=set(ctx.taint),
                        finish_reason=finish_reason,
                    )
                    await _safe_hook_fanout(effective_middleware, "after_run", ctx, _result)
                    return _result
                except BaseException as _tail_exc:
                    await _safe_hook_fanout(effective_middleware, "on_error", ctx, _tail_exc)
                    raise

    async def stream(
        self,
        input: str,
        *,
        session_id: str = "default",
        user_id: str = "local",
        principal: Principal | None = None,
        # v1.1.0 F4: per-call overrides (win over init-kwarg and container)
        executor: IToolExecutor | None = None,
        guardrail_chain: IGuardrailChain | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Token-stream variant of run().

        Phase 4c lifts the Tranche 3.5a limitation: tool-using agents now
        emit tool_started / tool_finished / delegation_started /
        delegation_finished events around concurrent tool dispatch.

        LLM-mode tradeoff:
          - Simple agents (no tools, no delegates) use llm.stream() and
            yield incremental `token` events. Preserves today's UX
            byte-for-byte.
          - Tool-using agents use llm.chat() and yield ONE synthetic
            `token` event per assistant turn carrying the full content.
            Phase 5 may wire structured token streaming for the tool path.

        Cascade is run-only. Phase 4.5a fails loud when self.cascade is set
        (previously: silent bypass of the SAS gate — a policy violation that
        looked like a working stream). Use .run() for cascade-gated execution.

        B6: DefaultGuardrailChain is wired into stream at input + output stages.
        tool_call + tool_output stages are handled by _dispatch_one (shared
        with the run path), so streaming agents automatically see per-tool
        guardrail screening when they go through tool_turn_dispatch.
        """
        if self.cascade is not None and self.cascade.max_cascade_depth > 1:
            from voussoir.agent.policy import PolicyViolation, PolicyViolationError

            raise PolicyViolationError(
                PolicyViolation.STREAMING_NOT_SUPPORTED,
                f"Agent {self.name!r} has cascade.max_cascade_depth={self.cascade.max_cascade_depth}; "
                "streaming with cascade retries (depth > 1) is not yet supported. "
                "Use .run() for retry-capable cascade-gated execution.",
            )
        # v1.1.0 F4: resolve overrides before entering the context.
        executor = self._resolve_executor(executor)
        guardrail_chain = self._resolve_guardrail_chain(guardrail_chain)
        stream_middleware = self._effective_middleware()
        async with await AgentContext.open(
            container=self.container,
            run_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            principal=principal,
            guardrail_chain=guardrail_chain,
        ) as ctx:
            ctx.agent_name = self.name
            ctx.max_delegation_depth = self.max_delegation_depth
            ctx.allowed_capabilities = self.allowed_capabilities
            _parent_ctx_token = parent_ctx_var.set(ctx)
            _sub_result_token = last_sub_result_var.set(None)
            try:
                await ctx.record_user_message(input)

                # B6: input-stage guardrail screening for stream. Mirrors
                # _run_normal's input screening. BLOCK yields an error event
                # and returns immediately; REWRITE replaces the user input.
                if guardrail_chain.count():
                    input_verdict = await guardrail_chain.screen(
                        GuardrailPayload(stage="input", content=input),
                        ctx,
                    )
                    input, is_blocked = apply_guardrail_verdict(
                        input_verdict,
                        input,
                        stage="input",
                        ctx=ctx,
                        blocked_content=f"[blocked by input guardrail: {input_verdict.reason}]",
                    )
                    if is_blocked:
                        yield done_event(input, ctx.trace_id)
                        return

                # Prepare before_run side-effects (e.g. SkillActivationMiddleware
                # populates ctx.skill_content). Mirrors _run_normal so streaming
                # agents declaring `skills=[...]` get skill content in their
                # prompt instead of silently empty skill content (v1.0.2 D7 #2).
                ctx.skills_active = []
                ctx.skill_content = []
                for mw in stream_middleware:
                    await mw.before_run(ctx, input)

                # Per-run state shared by both branches. Tracked here (and not
                # only inside the tool-using branch) so the single pseudo_result
                # built below can carry real finish_reason / token totals
                # regardless of which branch ran (v1.0.2 D7 #1+#4).
                t0 = time.monotonic()
                tokens_in = 0
                tokens_out = 0
                finish_reason: FinishReason = "completed"

                # Simple-agent fast path: incremental token streaming.
                # C7: `return` removed — both branches fall through to the
                # shared cascade gate below, which fires after `done`.
                if not self.tools and not self.delegates:
                    llm: ILLMProvider = self.container.resolve(ILLMProvider)
                    messages = self._build_messages(
                        input,
                        skill_content=getattr(ctx, "skill_content", []),
                    )
                    buffer: list[str] = []
                    async for chunk in llm.stream(messages=messages, model=self.model):
                        buffer.append(chunk)
                        yield token_event(chunk, ctx.trace_id)
                    full = "".join(buffer)
                    # v1.0.3 (c): estimate tokens client-side via count_tokens
                    # since ILLMProvider.stream() doesn't surface usage totals
                    # (unlike .chat). Best-effort; silently keeps 0 on failure.
                    tokens_in, tokens_out = _estimate_stream_tokens(llm, messages, full)
                    # B6: output-stage guardrail screening (simple path).
                    if guardrail_chain.count():
                        out_verdict = await guardrail_chain.screen(
                            GuardrailPayload(stage="output", content=full),
                            ctx,
                        )
                        full, is_blocked = apply_guardrail_verdict(
                            out_verdict,
                            full,
                            stage="output",
                            ctx=ctx,
                            blocked_content=f"[blocked by output guardrail: {out_verdict.reason}]",
                        )
                        if is_blocked:
                            finish_reason = "blocked"
                    await ctx.record_assistant_message(full)
                    yield done_event(full, ctx.trace_id)
                    final_content = full

                else:
                    # Tool-using path: multi-turn chat + structured events.
                    llm = self.container.resolve(ILLMProvider)
                    registry, messages, synthetic_tools = self._run_setup(
                        ctx,
                        input,
                        skill_content=getattr(ctx, "skill_content", []),
                    )
                    effective_tools = list(self.tools) + synthetic_tools
                    steps: list[Step] = []

                    # stream emits tool_started / delegation_started events
                    # BETWEEN the llm.chat and the dispatch_tool_calls, so
                    # consumers observe tool starts in real time.
                    # tool_turn_prepare returns after llm.chat; pre-dispatch
                    # events fire next; tool_turn_dispatch then executes the
                    # tools while the consumer awaits the next event. Do
                    # NOT fold the two calls back into one — the split is
                    # what restores real-time pre-dispatch event timing.
                    final_content = ""
                    # v1.0.2 D7 #1: budget-gated for-loop (was `while True:`).
                    # Mirrors _run_normal's per-turn policy.check so a
                    # misbehaving LLM that never emits finish_reason=stop is
                    # capped at policy.max_steps and the other budget caps
                    # (duration, tokens, cost) — not an infinite loop.
                    for step_idx in range(self.policy.max_steps + 1):
                        duration_s = time.monotonic() - t0
                        violation = self.policy.check(
                            steps=step_idx,
                            duration_s=duration_s,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            cost_usd=_cost_from_tokens(tokens_in, tokens_out, model=self.model),
                        )
                        if violation is not None:
                            # AgentPolicy.check() only returns the 5 budget
                            # variants; matches FinishReason literals 1:1.
                            # See _run_normal L568-574 for the parity note.
                            finish_reason = cast("FinishReason", violation.value)
                            break

                        prepared = await tool_turn_prepare(
                            llm=llm,
                            messages=messages,
                            registry=registry,
                            ctx=ctx,
                            model=self.model,
                            temperature=self.temperature,
                            has_tools=bool(effective_tools),
                            steps=steps,
                        )
                        tokens_in += prepared.tokens_in_delta
                        tokens_out += prepared.tokens_out_delta
                        if prepared.response.content:
                            yield token_event(prepared.response.content, ctx.trace_id)
                        if prepared.finish_reason == "stop":
                            final_content = prepared.response.content
                            break
                        # Append the assistant turn (with tool calls)
                        # BEFORE dispatch so tool-result messages line up
                        # with their owning assistant message.
                        messages.append(prepared.assistant_message)
                        # Emit pre-dispatch events from the declarations —
                        # BEFORE the tool bodies execute.
                        for ev in pre_dispatch_events(
                            prepared.tool_call_declarations, ctx.trace_id
                        ):
                            yield ev
                        steps_before = len(steps)
                        dispatched = await tool_turn_dispatch(
                            prepared=prepared,
                            registry=registry,
                            executor=executor,
                            ctx=ctx,
                            steps=steps,
                        )
                        tokens_in += dispatched.tokens_in_delta
                        tokens_out += dispatched.tokens_out_delta
                        messages.extend(dispatched.tool_result_messages)
                        for ev in post_dispatch_events(dispatched.tool_call_outcomes, ctx.trace_id):
                            yield ev
                        for _step in steps[steps_before:]:
                            await _safe_hook_fanout(stream_middleware, "after_step", ctx, _step)

                    # B6: output-stage guardrail screening (tool-using path).
                    if guardrail_chain.count():
                        out_verdict = await guardrail_chain.screen(
                            GuardrailPayload(stage="output", content=final_content),
                            ctx,
                        )
                        final_content, is_blocked = apply_guardrail_verdict(
                            out_verdict,
                            final_content,
                            stage="output",
                            ctx=ctx,
                            blocked_content=f"[blocked by output guardrail: {out_verdict.reason}]",
                        )
                        if is_blocked:
                            finish_reason = "blocked"
                    await ctx.record_assistant_message(final_content)
                    yield done_event(final_content, ctx.trace_id)

                # v1.0.2 D7 #4: build the post-done AgentResult ONCE and reuse
                # it for both the cascade gate AND the after_run hook fanout.
                # Pre-fix this was constructed twice with identical data; now
                # carries real tokens/cost/duration_ms. Tool-using path tracks
                # tokens through dispatch deltas; simple-path estimates via
                # _estimate_stream_tokens (v1.0.3 c).
                duration_ms = (time.monotonic() - t0) * 1000
                pseudo_result: AgentResult[str] = AgentResult(
                    output=final_content,
                    trace_id=ctx.trace_id,
                    steps=[],
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=_cost_from_tokens(tokens_in, tokens_out, model=self.model),
                    duration_ms=duration_ms,
                    delegation_chain=[],
                    cascade_history=[],
                    guardrail_decisions=list(ctx.guardrail_decisions),
                    authz_decisions=list(ctx.authz_decisions),
                    taint=set(ctx.taint),
                    finish_reason=finish_reason,
                )

                # C7: Single-pass cascade gate — fires AFTER `done`, only
                # when max_cascade_depth == 1. Emits cascade_passed or
                # cascade_failed; no retry (deferred to a future task).
                # v1.0.2 D7: skip cascade on budget violation -- verifying an
                # empty, budget-cut output would produce a spurious
                # cascade_failed event misrepresenting the outcome.
                if (
                    self.cascade is not None
                    and self.cascade.max_cascade_depth == 1
                    and finish_reason == "completed"
                ):
                    decision = await self.cascade.verifier.validate(pseudo_result, task=input)
                    cascade_span_id = str(uuid.uuid4())
                    if decision == Decision.PASS:
                        yield cascade_passed_event(
                            self.cascade.verifier.name, span_id=cascade_span_id
                        )
                    else:
                        yield cascade_failed_event(
                            self.cascade.verifier.name,
                            decision.value,
                            getattr(decision, "reason", ""),
                            span_id=cascade_span_id,
                        )

                await _safe_hook_fanout(stream_middleware, "after_run", ctx, pseudo_result)
            except BaseException as _stream_exc:
                await _safe_hook_fanout(stream_middleware, "on_error", ctx, _stream_exc)
                raise
            finally:
                last_sub_result_var.reset(_sub_result_token)
                parent_ctx_var.reset(_parent_ctx_token)

    def _run_setup(
        self,
        ctx: AgentContext,
        input: str,
        *,
        skill_content: list[str] | None = None,
        force_sas: bool = False,
    ) -> _RunSetup:
        """Synthesize tools, populate registry, and build the initial message list."""
        synthetic_tools = [] if force_sas else self._build_delegate_tools(ctx)
        registry = ToolRegistry()
        registry.register_many(list(self.tools) + synthetic_tools)
        messages = self._build_messages(input, skill_content=skill_content or [])
        if synthetic_tools:
            messages.insert(
                -1, ChatMessage(role="system", content=delegation.DELEGATE_SYSTEM_PROMPT)
            )
        return _RunSetup(registry=registry, messages=messages, synthetic_tools=synthetic_tools)

    def _effective_middleware(self) -> list[Any]:
        """Compose the middleware list for this run.

        User-supplied middlewares (`self.middleware`) come first; then, if
        `self.skills` is non-empty AND the user didn't already include a
        `SkillActivationMiddleware`, auto-construct one against the bound
        ISkillStore. If no ISkillStore is bound, log and skip — skills are
        a "best-effort enrichment", never a hard requirement.
        """
        out = list(self.middleware)
        if not self.skills:
            return out

        from voussoir.skills.adapter import SkillActivationMiddleware

        if any(isinstance(mw, SkillActivationMiddleware) for mw in out):
            return out

        from voussoir.protocols import ISkillStore

        try:
            skill_store = self.container.resolve(ISkillStore)
        except LookupError:
            from voussoir.observability.logging_setup import get_logger

            get_logger(__name__).warning(
                "agent.skills set but ISkillStore not bound; skipping skill activation. "
                "Call bind_skill_store(container) to enable."
            )
            return out

        from ctxforge.engine.services.skill_matcher import SkillMatcher

        out.append(
            SkillActivationMiddleware(
                matcher=SkillMatcher(),
                skill_store=skill_store,
                agent_skills_hint=self.skills,
            )
        )
        return out

    def _build_messages(
        self,
        user_input: str,
        *,
        skill_content: list[str] | None = None,
    ) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        if self.instructions:
            sys_text = (
                self.instructions
                if isinstance(self.instructions, str)
                else "\n\n".join(self.instructions)
            )
            msgs.append(ChatMessage(role="system", content=sys_text))
        for content in skill_content or []:
            msgs.append(ChatMessage(role="system", content=content))
        msgs.append(ChatMessage(role="user", content=user_input))
        return msgs
