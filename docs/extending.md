# Extending voussoir

voussoir's extension points are Protocol interfaces. To add a new capability,
implement the Protocol and bind the implementation on a `Container` — no framework
internals to touch.

Seven recipes cover the most common extensions:

1. [Custom `@tool`](#custom-tool) — wrap a Python function the agent can call.
2. [Custom `Guardrail`](#custom-guardrail) — add input/output policy checks.
3. [Custom `Authorizer`](#custom-authorizer) — write your own authz policy.
4. [Custom `CredentialBroker`](#custom-credentialbroker) — source credentials from a new place.
5. [Custom `IDelegate`](#custom-idelegate) — wrap a remote service as a sub-agent.
6. [Custom `ITelemetrySink`](#custom-itelemetrysink) — capture run telemetry to a new backend.
7. [Plugin packaging](#plugin-packaging) — ship a plugin via entry points.

---

## Custom `@tool` {#custom-tool}

The `@tool` decorator turns any `async` Python function into a `Tool` that the agent
can discover and call. It auto-derives a strict Pydantic v2 input schema from the
function's type-annotated parameters, so callers never write JSON Schema by hand.
Three security kwargs control the authorization model: `capability` declares what
access the tool needs (checked by the executor before invocation), `output_trust`
marks how much the downstream pipeline should trust the tool's output (defaults to
the capability's natural trust level), and `roles` restricts which principals may
invoke the tool at all.

```python
from voussoir.tools import Capability, ToolContext, tool
from voussoir.guardrails.trust import Trust

@tool(
    capability=Capability.READ_PUBLIC,
    name="search_docs",
    description="Full-text search internal documentation.",
    output_trust=Trust.UNTRUSTED,   # outputs from external search are taint-marked
    roles=["engineer"],             # only principals whose roles include 'engineer'
)
async def search_docs(
    query: str,
    *,
    limit: int = 5,
    ctx: ToolContext,
) -> list[str]:
    """Pydantic v2 derives the input schema from the type hints above.

    `ctx` is injected by the executor at call time — it carries the caller's
    Principal, resolved Credentials, allowed Capability set, and taint state.
    Do not include `ctx` in the schema; the decorator skips it automatically.
    """
    # Use ctx.credentials.headers for auth when a CredentialBroker is bound.
    headers = ctx.credentials.headers if ctx.credentials else {}
    _ = headers  # ...make HTTP call to your search backend...
    return [f"result-{i}: {query}" for i in range(limit)]
```

`output_trust=Trust.UNTRUSTED` ensures that any content returned by this tool enters
the run's taint set as untrusted: the executor will then block a subsequent call to
any `Capability.EXFILTRATION` tool (email, webhooks, image rendering with URLs) until
the taint is cleared. `roles` is consumed by the built-in `RoleAuthorizer`; pair it
with any `Authorizer` bound on the container. The `ctx: ToolContext` parameter is
your handle to per-invocation state — check `ctx.principal.classification` for data
classification, read `ctx.credentials` if your tool needs upstream auth, and append
authz decisions to `ctx.authz_decisions` if you implement secondary checks inside
the tool body.

**Redacting sensitive arguments from logs (v1.0.1+).** By default, the executor
emits an INFO `tool.invoke` log line that includes every argument name and value.
For tools that accept secrets (passwords, API keys, tokens), pass
`sensitive_args=["arg_name", ...]` to redact those values from the log — they are
replaced with `"[REDACTED]"`. The tool function always receives the real values;
redaction is logging-only.

```python
@tool(
    capability=Capability.READ_PRIVATE,
    sensitive_args=["password", "api_key"],
)
async def authenticate(username: str, password: str, api_key: str) -> str:
    """Log line will show password=[REDACTED] api_key=[REDACTED]."""
    ...
```

**See also:** `examples/02_research_agent/` for a runnable multi-tool agent;
`tests/tools/test_decorator_authz_kwargs.py` for the locked invariants on roles,
domains, and auth_requirement.

---

## Custom `Guardrail` {#custom-guardrail}

Guardrails are soft policies that the executor runs on every message that passes
through the pipeline. The `DefaultGuardrailChain` groups your implementations by
`stage` (`"input"`, `"tool_call"`, `"tool_output"`, `"output"`) and calls each
matching guardrail in order; the first non-`ALLOW` verdict short-circuits the
chain. A `BLOCK` verdict halts execution and raises a policy error; `REWRITE`
replaces the content in-place before forwarding; `AMBIGUOUS` defers to the
configured `LLMGuardrailJudge` for a second opinion.

```python
import re

from voussoir.guardrails import (
    DefaultGuardrailChain,
    Guardrail,
    GuardrailPayload,
    GuardrailVerdict,
)


class NoSocialSecurityNumbers:
    """Block any tool output containing an SSN-like pattern."""

    name = "no-ssn"
    stage = "tool_output"  # one of: "input", "tool_call", "tool_output", "output"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", payload.content):
            return GuardrailVerdict(
                verdict="BLOCK",
                reason="SSN pattern detected in tool output",
            )
        return GuardrailVerdict(verdict="ALLOW")


# Build a chain and bind it on the container.
from voussoir import Container

container = Container()
chain = DefaultGuardrailChain([NoSocialSecurityNumbers()])
container.bind(DefaultGuardrailChain, chain)
```

The `Guardrail` Protocol requires exactly two class-level attributes (`name: str`,
`stage: Literal[...]`) and one async method (`screen(payload, ctx) -> GuardrailVerdict`).
`GuardrailPayload.content` always carries the text to screen; for `tool_call` and
`tool_output` stages, `payload.tool_name` and `payload.tool_args` are also populated.
Verdicts are ephemeral — the audit-log record written to `AgentResult.guardrail_decisions`
is a separate `GuardrailDecision` object produced by the chain, not the raw verdict.

**See also:** `src/voussoir/guardrails/builtin/` for richer PII and prompt-injection screens;
`tests/security/lethal_trifecta/` for the 30-attack corpus that guardrails are validated against.

---

## Custom `Authorizer` {#custom-authorizer}

The `Authorizer` Protocol is Axis 2 (authorization) of voussoir's two-axis auth
model. The `StandardExecutor` calls `authorize()` inline before capability checks,
taint checks, and guardrail-chain evaluation. Returning `DENY` raises a
`PolicyViolationError` immediately; returning `MASK` allows the call but strips
`masked_fields` from the tool's output before handing it back to the LLM; returning
`ALLOW` proceeds normally. Every decision is appended to `AgentResult.authz_decisions`
for the run's audit trail.

```python
from datetime import UTC, datetime

from voussoir.auth import AuthzDecision
from voussoir.tools.protocol import Tool, ToolContext
from voussoir.auth.principal import Principal
from pydantic import BaseModel


class BusinessHoursAuthorizer:
    """Only allow tool calls during business hours (09:00–17:00 UTC)."""

    name = "business_hours"

    async def authorize(
        self,
        principal: Principal,
        tool: Tool,
        args: BaseModel,
        ctx: ToolContext,
    ) -> AuthzDecision:
        hour = datetime.now(UTC).hour
        if 9 <= hour < 17:
            return AuthzDecision(
                decision="ALLOW",
                authorizer_name=self.name,
                reason=f"hour {hour} UTC within business window",
            )
        return AuthzDecision(
            decision="DENY",
            authorizer_name=self.name,
            reason=f"hour {hour} UTC outside business window (09-17 UTC)",
        )


# Bind on the container (replaces the default DenyByDefaultAuthorizer).
from voussoir.auth import Authorizer
from voussoir import Container

container = Container()
container.bind(Authorizer, BusinessHoursAuthorizer())  # type: ignore[type-abstract]
```

!!! tip "Chaining authorizers"
    To combine policies (e.g. business hours **and** role checks), wrap multiple
    authorizers with `voussoir.auth.authorizers.ChainedAuthorizer`, which evaluates
    each in order and returns the first non-`ALLOW` decision.

The built-in `RoleAuthorizer` and `DomainAuthorizer` (from `voussoir.auth.authorizers`)
follow the exact same shape if you want to inspect a production reference. A `MASK`
decision requires populating `masked_fields` with the JSON path keys to strip from
the tool output dict before it reaches the LLM.

**See also:** `src/voussoir/auth/authorizers/role.py` for the RBAC pattern;
`tests/auth/test_authorizers.py` for the invariants.

---

## Custom `CredentialBroker` {#custom-credentialbroker}

A `CredentialBroker` resolves an `AuthRequirement` declared on a `@tool` into a
ready-to-use `Credentials` object. The executor calls `resolve()` once before the
first tool invocation; if the tool raises `AuthenticationFailedError` (signalling a
401 or 403 from its upstream service), the executor calls `refresh()` on the broker
and retries once. Raise `MissingCredentialError` with a human-readable message when
your backend can't supply credentials — the error surfaces as an actionable agent
error rather than an opaque crash.

```python
import httpx

from voussoir.auth import (
    AuthRequirement,
    AuthType,
    Credentials,
    MissingCredentialError,
)
from voussoir.auth.principal import Principal
from voussoir.tools.protocol import ToolContext


class VaultCredentialBroker:
    """Resolve API credentials from HashiCorp Vault KV v2."""

    name = "vault"

    def __init__(self, *, vault_url: str, vault_token: str) -> None:
        self._url = vault_url.rstrip("/")
        self._token = vault_token

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        if requirement.auth_type not in (AuthType.BEARER, AuthType.API_KEY):
            raise MissingCredentialError(
                f"VaultCredentialBroker does not support auth_type={requirement.auth_type!r}"
            )
        service = requirement.service or ""
        path = f"{self._url}/v1/secret/data/{service}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, headers={"X-Vault-Token": self._token})
        if resp.status_code == 404:
            raise MissingCredentialError(
                f"Vault secret not found for service {service!r}; check path {path}"
            )
        resp.raise_for_status()
        secret = resp.json()["data"]["data"]
        token = secret.get("token") or secret.get("api_key")
        if not token:
            raise MissingCredentialError(
                f"Vault secret for {service!r} has no 'token' or 'api_key' field"
            )
        header_key = "Authorization" if requirement.auth_type == AuthType.BEARER else "X-API-Key"
        header_val = f"Bearer {token}" if requirement.auth_type == AuthType.BEARER else token
        return Credentials(auth_type=requirement.auth_type, headers={header_key: header_val})

    async def refresh(self, creds: Credentials) -> Credentials:
        # Vault dynamic secrets are short-lived; re-resolve from the same path.
        # In a real implementation, use the stored service name from creds.metadata.
        return creds


# Bind on the container.
from voussoir.auth import CredentialBroker
from voussoir import Container

container = Container()
container.bind(
    CredentialBroker,  # type: ignore[type-abstract]
    VaultCredentialBroker(vault_url="https://vault.example.com", vault_token="..."),
)
```

The `refresh()` path is intentionally separate from `resolve()` so brokers can
implement a cheap in-memory token refresh (e.g. OAuth2 refresh grant) without
re-hitting every discovery step. If your backend has no refresh concept — like
`EnvCredentialBroker`, which reads env vars once — just return `creds` unchanged
from `refresh()`. Tools signal expired credentials by raising
`AuthenticationFailedError`, **not** `MissingCredentialError`; the executor only
catches the former for its retry loop.

**See also:** `src/voussoir/auth/brokers/oauth2.py` for a full refresh-token flow with
in-memory caching; `src/voussoir/auth/brokers/env.py` for a minimal reference impl;
`tests/auth/test_brokers.py`.

---

## Custom `IDelegate` {#custom-idelegate}

`IDelegate` is the uniform abstraction over "where a sub-agent runs." The parent
agent uses `IDelegate.delegate(task, parent_ctx=ctx)` regardless of whether the
target is a local `Agent`, a remote peer via `AgentRef`, or a third-party LLM
service you've wrapped yourself. Implementing this Protocol is how you integrate a
non-voussoir system — a LangChain agent, a hosted AI API, a legacy RPC service —
into a voussoir delegation tree.

```python
import httpx

from voussoir.agent.context import AgentContext
from voussoir.agent.result import AgentResult


class MyLegacyServiceDelegate:
    """Wrap a legacy JSON-over-HTTP reasoning service as an IDelegate."""

    name = "legacy-reasoner"
    description = "Delegates to the on-prem legacy reasoning service."

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        import time

        async with httpx.AsyncClient(timeout=60.0) as client:
            t0 = time.monotonic()
            resp = await client.post(
                f"{self._base_url}/reason",
                json={"prompt": task, "trace_id": parent_ctx.trace_id},
                headers={"X-API-Key": self._api_key},
            )
            duration_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        body = resp.json()
        return AgentResult[str](
            output=body["answer"],
            trace_id=parent_ctx.trace_id,
            steps=[],
            tokens_in=body.get("tokens_in", 0),
            tokens_out=body.get("tokens_out", 0),
            cost_usd=0.0,
            duration_ms=duration_ms,
            delegation_chain=[self.name],
            cascade_history=[],
            guardrail_decisions=[],
            finish_reason="completed",
        )


# Register so the parent agent can find it by name.
from voussoir.agent import Agent, AgentBuilder, register_agent
from voussoir import Container

container = Container()
delegate = MyLegacyServiceDelegate(base_url="https://legacy.internal", api_key="...")

# Option A: pass directly as a delegate to AgentBuilder.
# agent = AgentBuilder(...).delegates([delegate]).build()

# Option B: register by name so other agents can find it via NamedDelegate.
from voussoir.agent.registry import AgentRegistry
registry = AgentRegistry()
registry.add(delegate)
container.bind(AgentRegistry, registry)
```

The `IDelegate` Protocol requires only three attributes: `name: str`,
`description: str`, and `async def delegate(task, *, parent_ctx) -> AgentResult[str]`.
The parent agent owns depth-cap checks, `DELEGATION_REFUSED` wrapping, and cost
aggregation; your implementation only needs to execute the task and return a well-formed
`AgentResult`. Raise `PolicyViolationError` if the delegate can't accept the task —
the parent converts it to a `DELEGATION_REFUSED` tool output automatically.

**See also:** `src/voussoir/a2a/agent_ref.py` for the A2A reference implementation
(JWT auth, JSON-RPC transport, typed error hierarchy);
`examples/03_multi_agent_research/` for a working multi-delegate setup;
`tests/agent/test_plugins_e2e.py` for plugin-loaded delegate tests.

---

## Custom `ITelemetrySink` {#custom-itelemetrysink}

`ITelemetrySink` receives structured usage records — LLM token counts, tool-call
durations, delegation events, guardrail timings — from every step of an agent run.
It is separate from distributed tracing (`get_tracer()`): the sink accumulates the
usage that ends up in `AgentResult.tokens_in`, `tokens_out`, `cost_usd`, and `steps`.
Implement this Protocol to forward run economics to your data lake, billing system,
or observability backend without touching agent internals.

```python
import json
from typing import Any

import httpx

from voussoir.observability import ITelemetrySink, StepKind
from collections.abc import Iterator
from contextlib import contextmanager


class DatadogSink:
    """Forward voussoir run events to a Datadog custom metrics endpoint."""

    def __init__(self, *, api_key: str, metric_prefix: str = "voussoir") -> None:
        self._api_key = api_key
        self._prefix = metric_prefix

    def record_llm_call(
        self,
        *,
        name: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._post(
            series=[
                {"metric": f"{self._prefix}.llm.tokens_in", "points": [[0, tokens_in]]},
                {"metric": f"{self._prefix}.llm.tokens_out", "points": [[0, tokens_out]]},
                {"metric": f"{self._prefix}.llm.cost_usd", "points": [[0, cost_usd]]},
                {"metric": f"{self._prefix}.llm.duration_ms", "points": [[0, duration_ms]]},
            ],
            tags=[f"model:{name}"],
        )

    def record_step(
        self,
        *,
        kind: StepKind,
        name: str,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._post(
            series=[
                {"metric": f"{self._prefix}.step.duration_ms", "points": [[0, duration_ms]]},
            ],
            tags=[f"kind:{kind}", f"name:{name}"],
        )

    @contextmanager
    def scoped(self, child: ITelemetrySink) -> Iterator[None]:
        # For a custom sink that doesn't need scoping, simply yield.
        # The built-in NullTelemetrySink / InMemoryTelemetrySink handle
        # scoping via _BaseSink; re-use it by subclassing if you need it.
        yield

    def _post(self, series: list[dict[str, Any]], tags: list[str]) -> None:
        # Fire-and-forget; swap for an async client in production.
        try:
            import httpx as _httpx
            _httpx.post(
                "https://api.datadoghq.com/api/v1/series",
                headers={"DD-API-KEY": self._api_key, "Content-Type": "application/json"},
                content=json.dumps({"series": series}),
                timeout=2.0,
            )
        except Exception:
            pass  # telemetry must never crash the agent run


# Bind on the container.
from voussoir import Container

container = Container()
container.bind(ITelemetrySink, DatadogSink(api_key="..."))  # type: ignore[type-abstract]
```

!!! note "Async vs sync"
    `ITelemetrySink` methods are intentionally synchronous so they can be called
    from any context without `await`. If your backend requires async I/O, buffer
    records and flush them at run boundaries, or use `asyncio.create_task` inside
    `record_*` — but never let a failed flush propagate into the agent run.

The three methods in the Protocol are `record_llm_call()`, `record_step()`, and the
`scoped(child)` context manager (used by the cascade validator to redirect sub-run
emissions into a `BufferedTelemetrySink`). If you don't need scoping in your custom
sink, a `yield`-only `scoped()` is sufficient.

**See also:** `src/voussoir/observability/sink.py` for the `NullTelemetrySink`,
`BufferedTelemetrySink`, and `InMemoryTelemetrySink` reference implementations;
`examples/05_observability/` for a wired-up observability example.

---

## Cooperative interrupts {#cooperative-interrupts}

A tool can pause the agent for external input — operator approval, a
human-in-the-loop step, an out-of-band lookup — by calling
`ctx.interrupt(kind, payload)`:

```python
from voussoir.tools.protocol import ToolContext
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability


@tool(capability=Capability.WRITE_PRIVATE)
async def approve_purchase(amount: float, ctx: ToolContext) -> dict[str, str | bool]:
    if amount > 10_000:
        outcome = await ctx.interrupt(
            kind="manager_approval_required",
            payload={"amount": amount, "vendor": "ACME"},
        )
        # On resume, `outcome` is the operator-supplied payload.
        if not outcome["approved"]:
            return {"approved": False, "reason": str(outcome["reason"])}
    return {"approved": True, "reason": "under threshold"}
```

`ctx.interrupt(...)` raises `InterruptRequest` — a `VoussoirError`
carrying `kind` and `payload`. In a vanilla voussoir process the
exception bubbles out of `Agent.run` to the caller. Runtime layers
(such as impost) catch it in a `Middleware.on_error` hook, persist
the interrupt, and resume the agent later — on resume, the cached
`outcome` payload short-circuits the second `ctx.interrupt(...)`
call so the tool body proceeds as if nothing happened.

The `IInterruptable` Protocol is the structural type behind the
method: anything with `async def interrupt(kind, payload) -> Mapping`
satisfies it. Runtime layers may install custom resolvers via
container DI when they need to override the default
"raise InterruptRequest" behavior — see
`voussoir.testing.make_runtime_container` for the starter pattern
used by impost.

---

## Plugin packaging {#plugin-packaging}

voussoir discovers third-party delegates via the `voussoir.delegates` entry-point
group. Ship a plugin as a regular Python package; the loader calls each registered
factory with the host application's `Container` and adds the returned `IDelegate`
to the agent registry automatically.

```toml
# my-voussoir-plugin/pyproject.toml
[project]
name = "my-voussoir-plugin"
version = "0.1.0"
dependencies = ["voussoir>=1.0"]

[project.entry-points."voussoir.delegates"]
my_remote_agent = "my_voussoir_plugin:make_delegate"
```

```python
# my_voussoir_plugin/__init__.py
from voussoir.agent.context import AgentContext
from voussoir.agent.result import AgentResult
from voussoir import Container


class _MyRemoteAgentDelegate:
    name = "my_remote_agent"
    description = "Delegates tasks to My Remote Agent service."

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        # ...call your remote service, return AgentResult[str]...
        raise NotImplementedError


def make_delegate(container: Container) -> _MyRemoteAgentDelegate:
    """Entry-point factory. Receives the host Container; returns an IDelegate.

    Pull any config your delegate needs from the container here — e.g.:
        cfg = container.resolve(MyPluginConfig)
    """
    return _MyRemoteAgentDelegate()
```

After `pip install my-voussoir-plugin`, enable plugin loading when bootstrapping
the agent registry:

```python
from voussoir.agent import bind_agent_registry
from voussoir import Container

container = Container()
# load_plugins=True scans the voussoir.delegates entry-point group.
# allowed_plugin_names restricts which delegate names are accepted;
# omit it (or pass None) to accept all successfully-loaded plugins.
bind_agent_registry(
    container,
    load_plugins=True,
    allowed_plugin_names={"my_remote_agent"},
)
```

!!! warning "Security: use `allowed_plugin_names`"
    `allowed_plugin_names=None` (the default) accepts every plugin found in the
    environment. In production, always pass an explicit allowlist so that a
    compromised dependency cannot register a rogue delegate.

The loader is fault-tolerant: a broken import, a factory that raises, a factory that
returns a non-`IDelegate` object, or a name collision with an already-registered
delegate all log a structured warning and skip the plugin without crashing startup.
Name collisions always favour the application's own registrations over plugins.

**See also:** `src/voussoir/agent/plugins.py` for the loader implementation;
`src/voussoir/agent/bootstrap.py` for `bind_agent_registry`;
`tests/agent/test_plugins.py` and `tests/agent/test_plugin_allow_list.py` for
the invariant tests; `examples/05_delegate_plugin/` for a runnable plugin example.
