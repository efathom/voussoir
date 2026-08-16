# Architecture

voussoir is opinionated about a few things and unopinionated about most others.
This document explains the design choices — what was picked, what was considered,
and why.

---

## Design intent

voussoir makes two foundational bets:

1. **Default-runnable.** `Agent("name").run("task")` works in five lines with no
   config file, no external services, and no model API key beyond a single env var.
   The safe path is the default path.

2. **Highly extensible per business domain.** Every architectural concern is a
   `Protocol` injected via a lightweight DI container. Finance, healthcare, devops,
   and support teams plug in their own LLM providers, memory backends, tools,
   guardrails, and skills without touching framework code.

These two goals sit in tension — defaults that are too aggressive get in the way;
defaults that are too thin leave domains to reinvent the same security plumbing.
voussoir resolves this with **layered composability**: start with zero config, swap
in one component at a time, and the security invariants remain enforced at every
layer.

### The multi-paradigm intent

voussoir ships one DSL in v1.0: single-agent-first with hierarchical delegation as
the natural extension. The framework is designed so that a "hello world" agent,
a research crew of three collaborating agents, and a production deployment with
A2A remote peers all use the **same core abstractions** — `Agent`, `Tool`,
`Container`, `Guardrail`, `Authorizer`, `CredentialBroker`. No second set of classes
for the multi-agent case. Complexity scales continuously.

### "The safe path is the default path"

Every extension surface in voussoir defaults to the secure behavior:

- Tools default to `READ_PUBLIC` capability (lowest trust), not "trust everything."
- The default `Authorizer` is the fail-closed `DenyByDefaultAuthorizer` — every
  tool call denies until a concrete authorizer grants it (zero-trust by default).
- A tool with `auth_requirement` declared but no `CredentialBroker` bound fails
  loudly at the first call rather than silently proceeding without credentials.
- A run with `UNTRUSTED` content in its taint set cannot call `EXFILTRATION` tools
  at all — no flag to bypass it, no opt-in required.

The framework defends what it can deterministically; users extend from that
foundation.

---

## The Hollywood Principle

> "Don't call us, we'll call you."

voussoir uses a **custom ~150-line DI container** rather than a full-featured
framework like `dependency-injector`. The design choice here is deliberate: the
container's job is to resolve Protocols to implementations at run time, not to
generate boilerplate or define declarative component graphs.

```python
container = Container()
container.bind(ILLMProvider, AnthropicProvider(model="claude-opus-4-7"))
container.bind(IMemoryStore, SQLiteMemoryStore("./memory.db"))
container.bind(Guardrail, FinanceComplianceGuardrail(), name="finance")
container.bind(Validator, ToolUseFaithfulness(), scope=Scope.RUN)

agent = Agent("researcher", container=container)
```

User code **implements Protocols**; the framework calls them back. User code never
imports `StandardExecutor` to invoke it directly — the Agent calls into it through
the `ToolExecutor` Protocol, resolved by the container.

### Container scopes

| Scope | Lifetime | Typical use |
|---|---|---|
| `SINGLETON` (default) | one instance per process | LLM client, OTEL tracer, memory store |
| `RUN` | one instance per `agent.run()` call | conversation session, trace context, per-run validators |
| `TRANSIENT` | new instance every resolve | request-scoped DTOs |

### Default container

`default_container()` returns a pre-bound container that makes the five-line demo
work: Anthropic provider (if `ANTHROPIC_API_KEY` is set), else OpenAI (if
`OPENAI_API_KEY` is set), else OpenRouter (if `OPENROUTER_API_KEY` is set),
in-memory store, default guardrail chain (profile "off" — args schema check
only), console OTel exporter, and `StandardExecutor`. Every binding is
overridable.

### Protocols for extension

Every major concern is a Protocol:

- `ILLMProvider` — LLM backend
- `IMemoryStore` / `ISessionStore` — storage tiers
- `Tool` — any callable tool (local or remote)
- `ToolExecutor` — execution strategy
- `Guardrail` — soft policy screen
- `Authorizer` — user-level authorization (Axis 2 auth)
- `CredentialBroker` — tool-level credential resolution (Axis 1 auth)
- `Middleware` — before/after run hooks
- `Validator` — cascade gate logic

Protocols are declared in `voussoir.protocols` (re-exported from their home
modules). ctxforge Protocols (`ILLMProvider`, `IMemoryStore`, etc.) are
**imported and re-exported**, never redefined — an implementation that satisfies a
ctxforge Protocol satisfies the voussoir one automatically.

---

## Agent and run

### The public contract

```python
class Agent(Generic[In, Out]):
    name: str
    description: str
    model: str | None
    instructions: str | list[str] | None
    tools: list[Tool] = []
    delegates: list[Agent | AgentRef] | None = None
    cascade: RequestCascade | None = None
    middleware: list[Middleware] = []
    container: Container | None = None
    skills: list[str] | None = None
    allowed_capabilities: Capability = Capability.READ_PUBLIC | Capability.READ_PRIVATE

    async def run(self, input: In, *, principal: Principal | None = None,
                  **kwargs) -> AgentResult[Out]: ...
    def     run_sync(self, input: In, **kwargs) -> AgentResult[Out]: ...
    def     stream(self, input: In, **kwargs) -> AsyncIterator[AgentEvent]: ...
```

`In` and `Out` default to `str` for the "five-line demo" path; advanced users
supply Pydantic models for typed structured IO.

Guardrails are not an `Agent.__init__` kwarg — they are bound on the container
(typically via `bind_default_guardrails(container, profile=...)`) and resolved
by the executor automatically. See [Custom Guardrails](extending.md#custom-guardrail)
in the extending guide.

### The run loop

```
Agent.run(input, *, principal):
  1. Open AgentContext — allocate trace, container scope=RUN, OTel root span
  2. Apply pre-middleware (logging, tracing, retry policies, budget enforcement)
  3. Guardrail chain: screen_input(input)
     → deterministic chain first; LLM judge only on AMBIGUOUS
  4. Enrich context: skill activation, memory recall
  5. Loop until stop condition:
     a. LLM call → token stream → AgentEvent.token
     b. If tool_call requested:
          - Resolve tool via registry
          - Authorizer.authorize(principal, tool, args)      [Axis 2]
          - Capability check (tool.capability vs agent.allowed_capabilities)
          - Taint check (UNTRUSTED in taint → EXFILTRATION blocked)
          - Guardrail chain: screen_tool_call(args)
          - CredentialBroker.resolve(tool.auth_requirement)  [Axis 1]
          - StandardExecutor.invoke(tool, args)
          - Tag output trust; add to run taint set
          - Guardrail chain: screen_tool_output(result)
     c. If delegation requested:
          - Cascade policy: try SAS first, validate, escalate on failure
          - Sub-agent runs with child container scope (inherits singletons,
            fresh RUN scope, clamped capabilities)
     d. If final answer: validate against Out schema; break
     e. Policy: max_steps / time budget / token budget / cost budget
  6. Apply post-middleware (commit memory, finalize spans, emit metrics)
  7. Close AgentContext
```

### AgentResult

```python
class AgentResult(BaseModel, Generic[Out]):
    output: Out
    trace_id: str
    steps: list[Step]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    duration_ms: float
    delegation_chain: list[str]
    cascade_history: list[CascadeOutcome]
    guardrail_decisions: list[GuardrailDecision]
    authz_decisions: list[AuthzDecision]
    finish_reason: Literal["completed", "max_steps", "blocked", "error"]
```

### Stop conditions

```python
class AgentPolicy(BaseModel):
    max_steps: int = 25
    max_duration_s: float = 300.0
    max_input_tokens: int = 200_000
    max_output_tokens: int = 8_000
    max_cost_usd: float = 1.00
    on_violation: Literal["error", "summarize_and_stop"] = "summarize_and_stop"
```

`summarize_and_stop` returns the best partial answer rather than raising, so
callers always get a `AgentResult` even on budget exhaustion.

### AgentContext

`AgentContext` is the per-run state bag threaded through the run loop and into every
guardrail, middleware, and tool call:

- `session_id`, `run_id`, `trace_id` — identifiers
- `principal: Principal` — who initiated this run (propagated to sub-agents + A2A)
- `taint: set[Trust]` — accumulated taint from tool outputs
- `steps: list[Step]` — accumulated step log
- `authz_decisions: list[AuthzDecision]` — accumulated authz records
- `container` — the run-scoped DI container

`ToolContext` is a narrower view injected into each tool call, carrying `principal`,
`credentials` (populated by the broker), `allowed_capabilities`, and `taint`.

---

## Hierarchical delegation

### Centralized coordinator

```python
researcher = Agent("researcher", tools=[web_search])
writer     = Agent("writer",     instructions="Concise blog posts.")
reviewer   = Agent("reviewer",   instructions="Style + factual review.")

lead = Agent(
    "lead",
    delegates=[researcher, writer, reviewer],
    instructions="Coordinate research → draft → review.",
)
result = await lead.run("Write a 500-word post on the 2026 agent framework landscape.")
```

The lead agent has a `delegate_to(name=..., task=...)` synthetic tool in its
registry. The coordinator resolves the name to a sub-agent, runs it with a child
container scope, and feeds the structured result back as an observation tagged with
`provenance=delegated:<name>`.

Sub-agents **do not call each other directly** — all routing goes through the parent
coordinator. This eliminates the inter-agent coordination failure class: cross-agent
misalignment and cascading communication errors drop away when the topology is
strictly tree-shaped.

### Capability clamping on delegation

When a parent delegates to a child, the child's effective capabilities are the
**intersection** of parent and child declared capabilities:

```python
clamped = parent.allowed_capabilities & child.allowed_capabilities
```

A parent with only `READ_PUBLIC` cannot grant a child `WRITE_PRIVATE` even if the
child declared it. This is enforced in `Agent._make_delegate_invoker` before the
child's tool list is materialized. If clamping would leave a toolless child (where
the child had tools), the framework raises `PolicyViolationError(CAPABILITY_CLAMPED_EMPTY)`
immediately — loud failure at delegate-synthesis time, not at first call.

### Request Cascading

```python
from voussoir.agent import RequestCascade, ToolUseFaithfulness

agent = Agent(
    "smart_lead",
    cascade=RequestCascade(
        verifier=ToolUseFaithfulness(),
        escalation=multi_agent_lead,
        max_attempts=2,
    ),
)
```

Run sequence:

1. Run as a single agent (SAS).
2. `verifier.validate(result)` → PASS / FAIL / AMBIGUOUS.
3. On FAIL or AMBIGUOUS (within `max_attempts`), escalate to the MAS.
4. The escalation reuses the SAS's accumulated context as input.

This pattern saves ~20% cost vs always running MAS, while recovering MAS accuracy
on the hard cases.

### Principal propagation through the chain

`principal` flows from the top-level `agent.run()` call into every sub-agent run
and every A2A outbound call. Principals are never elevated mid-run — a sub-task
that needs more privilege fails with `AuthorizationError` and surfaces for a
human-in-the-loop step rather than silently escalating.

---

## Capabilities and Trust

### `Capability` — hard invariant

```python
class Capability(IntFlag):
    NONE          = 0
    READ_PUBLIC   = 1 << 0    # public web, public docs
    READ_PRIVATE  = 1 << 1    # internal corpus, internal APIs
    WRITE_PRIVATE = 1 << 2    # write to internal stores
    EXFILTRATION  = 1 << 3    # send email, post external, render image-with-URL
```

Every `@tool`-decorated function declares its capability. Every agent declares
`allowed_capabilities`. At every tool call, `StandardExecutor` checks that
`tool.capability` is a subset of `agent.allowed_capabilities`. This is not a
guardrail (soft policy) — it is a **hard invariant** enforced inline in the
executor. There is no flag to bypass it.

### `Trust` — taint propagation

```python
class Trust(StrEnum):
    SYSTEM    = "system"    # framework-generated; safe
    USER      = "user"      # the human caller's input; treated as instructions
    INTERNAL  = "internal"  # from internal trusted source
    UNTRUSTED = "untrusted" # from external source; never executed as instruction
```

Tool output trust is tagged automatically from the tool's capability:

| Tool capability | Output tagged |
|---|---|
| `READ_PUBLIC` | `UNTRUSTED` |
| `READ_PRIVATE` | `INTERNAL` |
| `WRITE_PRIVATE` | `INTERNAL` |
| `EXFILTRATION` | `INTERNAL` (sends; doesn't receive) |

Tags propagate into `AgentContext.taint` (a `set[Trust]`). The second hard invariant
lives here: **if `UNTRUSTED` is in the run's taint set, no `EXFILTRATION` tool may
run**. This prevents prompt-injection attacks from reaching exfiltration channels —
an attacker who injects content via a web-search result cannot cause the agent to
send that content to an email tool on the next step.

Both invariants are tested by a home-grown adversarial corpus (the Lethal Trifecta,
see below) in `tests/security/`.

---

## Guardrails (soft policy)

### Architecture

```
input
  → guardrail chain (stage="input")
  → run loop
       → tool_call guardrail chain (stage="tool_call")   [after authz + capability checks]
       → executor.invoke
       → tool_output guardrail chain (stage="tool_output")
  → guardrail chain (stage="output")
  → cascade gate
```

Hard invariants (capability mask, taint-vs-exfiltration gate) run **before** the
soft guardrail chain at the `tool_call` stage. If the hard invariant fires, the soft
chain never runs.

### The `Guardrail` Protocol

```python
class Guardrail(Protocol):
    name: str
    stage: Literal["input", "tool_call", "tool_output", "output"]
    async def screen(self, payload: GuardrailPayload, ctx: AgentContext) -> GuardrailVerdict: ...
```

Verdicts: `ALLOW` (pass through), `BLOCK` (deny with reason), `REWRITE` (substitute
content), `AMBIGUOUS` (escalate to LLM judge).

### DefaultGuardrailChain

The chain runs all guardrails for the current stage in declared order. The first
non-`ALLOW` verdict short-circuits the chain — if a guardrail returns `BLOCK`,
subsequent guardrails are skipped.

`REWRITE` applies the rewritten content and re-screens once. A second non-`ALLOW`
on the rewritten content is treated as `BLOCK`.

### Built-in deterministic guardrails

| Guardrail | Stage | Behavior |
|---|---|---|
| `InputLengthCap` | `input` | BLOCK if content exceeds 100,000 chars (configurable) |
| `PromptInjectionHeuristic` | `input` | Regex: BLOCK on "ignore previous instructions" family |
| `PIIDetector` | `input`, `output` | Regex: REWRITE email/phone/SSN/credit-card with `[REDACTED]` |
| `URLAllowlist` | `input`, `tool_output` | BLOCK if URL outside declared allowlist |
| `ArgsSchemaCheck` | `tool_call` | Pydantic re-validation of tool args (defense-in-depth) |
| `ArgsSizeCap` | `tool_call` | BLOCK if serialized args exceed 64 KiB |
| `ToolOutputSizeCap` | `tool_output` | BLOCK if output exceeds 1 MiB |
| `ExfilPatternScan` | `output` | BLOCK on suspicious URL/image/blob patterns in final output |

### Guardrail profiles

```python
bind_default_guardrails(container, profile="off")       # ArgsSchemaCheck only (default)
bind_default_guardrails(container, profile="standard")  # + injection + length + size caps
bind_default_guardrails(container, profile="strict")    # + PII + URLAllowlist
```

### LLMGuardrailJudge — AMBIGUOUS fallback

When a deterministic guardrail returns `AMBIGUOUS`, `LLMGuardrailJudge` sends the
payload to a small LLM (default: the container's bound provider) with a binary
ALLOW/BLOCK prompt. The judge wraps any `Guardrail` and is transparent to the chain
— from the chain's perspective, the wrapped guardrail never returns `AMBIGUOUS`.

### The Lethal Trifecta corpus

The framework ships a corpus of approximately 30 adversarial attacks in
`tests/security/lethal_trifecta/`. These represent the three attack classes that
dominate prompt-injection literature:

1. **Prompt injection via tool output** — attacker-controlled content in a web-search
   result that tries to overwrite the agent's instructions.
2. **Goal hijacking** — "ignore your previous instructions and do X instead."
3. **Exfiltration via channel abuse** — injected instructions that try to cause the
   agent to call an exfiltration tool with stolen content.

All 30 attacks are blocked at 100% in CI, tested as a parametrized suite in
`tests/security/lethal_trifecta/`. The deterministic chain blocks the majority;
the taint invariant blocks the exfiltration class regardless of guardrail state.

### Extending guardrails

See [Extending → Custom Guardrail](extending.md#custom-guardrail) for how to add
your own `Guardrail` implementation and register it on the container.

---

## Two-axis auth

Authentication and authorization are **orthogonal concerns** addressed by separate,
composable systems.

| Axis | Question | Answered by |
|---|---|---|
| **Axis 1 — AuthN** | How does the tool prove its identity to the backend service it calls? | `CredentialBroker` Protocol + `AuthRequirement` on tools |
| **Axis 2 — AuthZ** | What is the user behind this agent run allowed to do or see? | `Principal` model + `Authorizer` Protocol |

### Axis 1 — CredentialBroker (tool authentication)

```python
class CredentialBroker(Protocol):
    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials: ...
    async def refresh(self, creds: Credentials) -> Credentials: ...
```

Tools declare their credential requirements on the `@tool` decorator:

```python
@tool(
    capability=Capability.READ_PRIVATE,
    auth_requirement=AuthRequirement(auth_type=AuthType.OAUTH2, service="atlassian",
                                     scopes=["read:jira-work"]),
)
async def jira_search(query: str, ctx: ToolContext) -> list[dict]:
    creds = ctx.credentials        # Bearer token injected by broker before invoke
    ...
```

Six built-in brokers cover the common cases:

| Broker | Resolves from |
|---|---|
| `EnvCredentialBroker` | `${SERVICE}_API_KEY` / `${SERVICE}_TOKEN` env vars |
| `FileCredentialBroker` | Mounted secret files (K8s `/var/run/secrets/…`) |
| `MTLSCredentialBroker` | Service cert + key paths |
| `KeychainCredentialBroker` | macOS Keychain / Linux Secret Service / Windows Credential Store |
| `OAuth2CredentialBroker` | OAuth2 with refresh-token loop, optional Keychain persistence |
| `ChainedCredentialBroker` | Ordered list of brokers; first hit wins |

Credential lifecycle invariants:

- **Never logged.** Credentials are scrubbed from structured logs; only `auth_type`
  and `service` are emitted.
- **Never visible to the LLM.** The broker injects credentials into the HTTP
  transport (`ctx.credentials`), not into the prompt or any LLM-visible field.
- **Auto-refresh.** On 401/403, the executor calls `broker.refresh()` once and
  retries. Beyond that, `AuthenticationFailedError` propagates.
- **Expiry buffered.** Brokers refresh 60 seconds before nominal expiry.

### Axis 2 — Principal + Authorizer (user authorization)

```python
class Principal(BaseModel):
    user_id: str
    email: str | None = None
    auth_method: Literal["sso", "service", "api_key", "anonymous"] = "service"
    roles: list[str] = []
    teams: list[str] = []
    domains: list[str] = []
    classification: Literal["public", "internal", "confidential", "restricted"] = "internal"
    attributes: dict[str, Any] = {}
    issued_at: datetime
    expires_at: datetime | None = None
```

`Principal` is passed to `agent.run(input, principal=...)` and flows through the
entire call chain: into sub-agents, into `ToolContext`, and across A2A boundaries.

```python
class Authorizer(Protocol):
    name: str
    async def authorize(
        self,
        principal: Principal,
        tool: Tool,
        args: BaseModel,
        ctx: ToolContext,
    ) -> AuthzDecision: ...
```

`AuthzDecision` carries one of `ALLOW`, `DENY`, or `MASK`. `DENY` raises
`PolicyViolationError(AUTHZ_DENIED)` immediately. `MASK` records a list of
dotted field paths that are redacted from the tool output before it reaches the
LLM — the original values never enter the model context.

Five built-in authorizers:

| Authorizer | Mechanism |
|---|---|
| `DenyByDefaultAuthorizer` | Default — fail-closed; denies every tool call until granted |
| `AllowAllAuthorizer` | Permissive "hello world" — logs that no authz is enforced |
| `RoleAuthorizer` | RBAC — tool declares `roles=[...]`; principal must have one |
| `DomainAuthorizer` | `principal.domains ∩ tool.domains` non-empty |
| `ChainedAuthorizer` | Composes multiple authorizers; first non-ALLOW wins |

Tools declare their authz hints on the decorator:

```python
@tool(
    capability=Capability.READ_PRIVATE,
    roles=["incident-responder"],
    domains=["sre", "platform"],
    auth_requirement=AuthRequirement(auth_type=AuthType.OAUTH2, service="atlassian",
                                     scopes=["read:jira-work"]),
)
async def jira_search(query: str, ctx: ToolContext) -> list[dict]: ...
```

### Ordering inside StandardExecutor.invoke

At every tool call, the executor applies this sequence:

1. `Authorizer.authorize(principal, tool, args)` — DENY short-circuits; MASK records fields
2. Capability mask check (hard invariant)
3. Taint check (hard invariant)
4. Guardrail chain (`stage="tool_call"`)
5. `CredentialBroker.resolve(tool.auth_requirement)` — only if requirement is declared
6. `tool.invoke(args, ctx)` — credentials available at `ctx.credentials`
7. On 401/403: `broker.refresh()` + retry once
8. Apply `masked_fields` to output if decision was MASK
9. Guardrail chain (`stage="tool_output"`)

Authorization is the first check — a denied user never reaches capability or taint
logic.

### Where Principal comes from

Three legitimate entry points:

- **Direct invocation / CLI.** Caller passes `principal=Principal(...)` to
  `agent.run()`. If omitted, defaults to a `system` principal with empty roles.
  Production deployments should bind `RequiredPrincipalSource()` on the container
  to fail-closed on missing principal.
- **A2A inbound.** The A2A router requires Bearer JWT. The signature is verified
  against the configured JWKS; claims are mapped to `Principal` by a
  `JWTPrincipalMapper` (configurable Protocol). Default mapper handles standard
  claims (`sub`, `email`, `groups`).
- **A2A outbound.** When delegating to a remote A2A peer, voussoir forwards the
  caller's `Principal` as a downstream JWT via `PrincipalForwarder` (container-bound).
  Default is same-org passthrough; cross-org deployments can mint a fresh
  short-lived token.

---

## A2A peer protocol

A2A (Agent-to-Agent) is a native feature, not a plugin.

### AgentCard publication

```python
from voussoir.a2a import make_a2a_router

agent = Agent("researcher", container=default_container(), ...)
router = make_a2a_router(agent, endpoint="https://research.internal.example.com/a2a")
# Mount router on your FastAPI app; it publishes:
# GET /.well-known/agent-card.json
# GET /.well-known/jwks.json
# POST /a2a
```

The AgentCard describes the agent's capabilities, endpoint, input/output schemas,
and supported auth methods. Cards are JWS-signed with the publisher's key so that
discovery clients can verify authenticity before trusting capability advertisements.

### Discovery and remote delegation

```python
remote = await AgentRef.discover("https://research.internal.example.com/")
lead   = Agent("lead", delegates=[local_writer, remote])
```

`discover_card` fetches the AgentCard, verifies the JWS signature, and validates
capabilities. The resulting `AgentRef` satisfies the same `IDelegate` Protocol as a
local sub-agent — the coordinator does not distinguish between local and remote
peers.

### Transport

Remote agent calls use JSON-RPC 2.0 over HTTP with Bearer JWT authentication.
Trace context propagates via the W3C `traceparent` header so spans from remote
agents nest correctly under the parent's `delegation.dispatch` span.

Retry semantics: exponential backoff on transient errors, configurable via
`RetryMiddleware`. A2A peers are circuit-broken independently.

---

## OpenTelemetry

voussoir uses OpenTelemetry as its first-class observability layer. The
instrumentation library name is `"voussoir"`.

### Span hierarchy

```
agent.run                                attrs: agent_name, model, finish_reason
├── guardrail.input                      attrs: verdict, reason, n_guardrails
├── reason.<turn_n>                      one per LLM reasoning turn
│   ├── llm.complete                     attrs: gen_ai.system, gen_ai.request.model,
│   │                                           gen_ai.usage.input_tokens,
│   │                                           gen_ai.usage.output_tokens,
│   │                                           cost_usd, duration_ms
│   ├── tool.call.<tool_name>            attrs: tool_name, capability, trust_in, trust_out
│   │   ├── authz.check                  attrs: authorizer_name, decision, reason
│   │   ├── capability.check             attrs: required, allowed, passed
│   │   ├── taint.check                  attrs: tool_caps, run_taint, passed
│   │   ├── guardrail.tool_call          attrs: verdict, reason
│   │   ├── executor.invoke
│   │   └── guardrail.tool_output        attrs: verdict, reason, trust_inferred
│   └── delegation.dispatch.<child>      attrs: child_agent, delegate_kind
│       └── agent.run                    (recursive child span)
├── cascade.validate                     attrs: validator_name, verdict
└── guardrail.output                     attrs: verdict, reason
```

`gen_ai.*` attributes follow the OpenTelemetry Semantic Conventions for generative
AI systems, so spans appear correctly in compatible backends (Honeycomb, Datadog,
Phoenix, Langfuse, etc.).

### Metrics

Ten metric handles emitted per agent run:

| Metric | Kind | Attributes |
|---|---|---|
| `voussoir.tokens.in` | counter | model, agent_name |
| `voussoir.tokens.out` | counter | model, agent_name |
| `voussoir.cost_usd` | counter | model, agent_name, delegation_depth |
| `voussoir.duration_ms` | histogram | agent_name |
| `voussoir.tool_calls.count` | counter | tool_name, capability, success |
| `voussoir.guardrail.decisions` | counter | stage, verdict, guardrail_name |
| `voussoir.authz.decisions` | counter | decision, authorizer_name, tool_name |
| `voussoir.capability.denials` | counter | tool_name, required_capability |
| `voussoir.taint.exfil_blocks` | counter | tool_name |
| `voussoir.cascade.escalations` | counter | — |

### Exporter configuration

- **No env vars set (default):** `ConsoleSpanExporter` — spans printed at
  `agent.run` close. Good for local development.
- **`OTEL_EXPORTER_OTLP_ENDPOINT` set:** `OTLPSpanExporter` + `OTLPMetricExporter`
  over HTTP/protobuf. No code change required.
- **Custom provider:** bind `TracerProvider` or `MeterProvider` on the container
  before `Agent.__init__`. `configure_otel(container)` detects an already-bound
  provider and skips default setup.

The default `TracerProvider` is initialized **lazily** on first `Agent.__init__`,
so test suites that never construct an agent pay zero OTel overhead. pytest fixtures
use `NullTelemetrySink` to suppress all OTel I/O.

Resource attributes: `service.name = "voussoir"`, `service.version = <pkg version>`.
Override with `OTEL_SERVICE_NAME`.

---

## Cascade gates

A cascade gate is a `Validator` that inspects an `AgentResult` and returns
`PASS`, `FAIL`, or `AMBIGUOUS`. Gates are used by Request Cascading (see
[Hierarchical delegation](#request-cascading)) to decide whether to accept a
single-agent result or escalate to the MAS.

### The `Validator` Protocol

```python
class Validator(Protocol):
    async def validate(
        self, input: Any, result: AgentResult, ctx: AgentContext
    ) -> Decision: ...
```

`Decision` is `PASS`, `FAIL`, or `AMBIGUOUS`. `AMBIGUOUS` triggers `AmbiguousFallback`,
which wraps the validator with an `LLMJudge` to convert the ambiguity to a binary
decision.

### ToolUseFaithfulness — built-in verifier

The built-in `ToolUseFaithfulness` verifier checks that the agent's final output is
grounded in the tool results it observed — claims not traceable to a tool output are
flagged. It uses a combination of structured overlap checks (deterministic, fast) and
an LLM judge for the ambiguous middle ground.

### Custom verifiers

Any class satisfying `Validator` can be passed as the `verifier` in
`RequestCascade(verifier=...)`. Common domain patterns:

- **Schema validation** — verify the output matches a stricter Pydantic model than
  the agent's `Out` type.
- **Factual grounding** — check that every claim in the output cites a tool result.
- **Domain compliance** — check that the output passes a domain-specific rule set
  (e.g., finance disclaimers, medical liability language).

---

## Where the framework is going

### v1.x roadmap (incremental)

- **PyPI publish + ctxforge release.** Shipping both `voussoir` and its sibling
  `ctxforge` dependency to PyPI so `pip install voussoir` works without a sibling
  checkout. Until then, install from git, or clone `ctxforge` next to `voussoir`
  (the CI workflow and Dockerfile already pin a known-good `ctxforge` commit).
- **Public docs hosting.** ReadTheDocs / GitHub Pages. Docs build locally today:
  `make docs-build && make docs-serve`.
- **Cost/perf benchmarks vs LangGraph/CrewAI.** Preliminary benchmarks pending.
- **OPA + Masking authorizers** (v1.x). `OPAAuthorizer` bridges to an Open Policy
  Agent server for policy-as-code authorization. `MaskingAuthorizer` adds
  column-level field redaction based on declared schema sensitivity. Both deferred
  from v1.0 because they require a running OPA server and a richer tool schema for
  column-level declarations.
- **Synthetic-tool refactor** (v1.x). The tool registry's synthetic-tool path
  (used internally for `delegate_to` and `request_cascade`) needs unification with
  the `@tool` decorator path. Currently functional but inconsistent.
- **`.mcp.json` ecosystem compatibility** (v1.x). Auto-read `.mcp.json` from the
  project root so teams that maintain that file for their IDE / Claude Code setup
  get instant voussoir compatibility without migration.
- **Plugin activation via config** (v1.x). Explicit `plugins:` list in
  `voussoir.yaml` to control which installed domain packs are active for a given
  agent, rather than "installed = active."

### v2.x exploration

- **DualStream execution.** A persistent sandboxed Python kernel where tools become
  injected functions operating on live runtime objects (CaveAgent-style). Same
  `ToolExecutor` Protocol — the `DualStreamExecutor` implementation would be a
  drop-in swap. Requires a security audit pass before release.
- **Memory-layer taint propagation.** Coordinating with ctxforge owners to carry
  `Trust` tags through memory recall so that a retrieved memory item tagged
  `UNTRUSTED` at store time remains `UNTRUSTED` when recalled into a later run.
- **`max_cascade_depth > 1` in the streaming-cascade path.** Currently
  streaming cascades are depth=1 only (single escalation). Lifting this requires
  careful span bookkeeping and is deferred until there is a concrete use case.
- **mTLS-bound tokens for A2A** (RFC 8705). Stronger peer authentication than
  Bearer JWT alone. v1.0 ships Bearer JWT; mTLS-bound tokens are the natural v2
  upgrade for high-security deployments.

---

## Reading further

- **[Getting Started](getting-started.md)** — five ascending examples, from the
  five-line demo to a multi-agent A2A pipeline.
- **[Extending](extending.md)** — seven how-to recipes covering custom tools,
  guardrails, authorizers, brokers, memory backends, and more.
- **[API Reference](api/voussoir.md)** — every public Protocol and class generated
  from source docstrings.
