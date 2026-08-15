# Changelog

All notable changes to voussoir. The format follows [Keep a Changelog](https://keepachangelog.com/).
Voussoir follows semver from v1.0.0 onward.

## v1.3.0 — 2026-08-15

Enterprise-production hardening release. Closes the gaps between the
documented "safe by default" posture and the actual runtime defaults,
and adds resilience/observability that previously required manual wiring.

### Security

- **`default_container()` now binds the `standard` guardrail profile** (length
  caps, prompt-injection heuristic, args schema + size caps, exfil scan)
  instead of an empty chain. `strict` (PII + URL allow-list) remains opt-in.
- **Added `DenyByDefaultAuthorizer` and made it the default** — the default
  `Authorizer` is now fail-closed: tool calls deny until an operator binds a
  concrete grant (Role / Domain / Chained). `AllowAllAuthorizer` remains
  available for dev/experiments.
- **A2A hardening** — `make_a2a_router` now enforces a per-peer sliding-window
  rate limit (`rate_limit_per_minute`, default 60) and a request-body size cap
  (`max_request_bytes`, default 1 MB), and exposes a `/.well-known/health`
  liveness probe.

### Reliability

- **LLM resilience** — `AnthropicLLMProvider` now applies a request timeout
  (`timeout_s`, default 60 s) and SDK retries (`max_retries`, default 2) on
  transient 429/5xx failures.
- **Bounded tool concurrency** — tool dispatch is capped by a process-wide
  semaphore (`VOUSSOIR_MAX_CONCURRENT_TOOLS`, default 8) instead of unbounded
  `asyncio.gather`.

### Observability

- **Metrics are now exported** — `configure_otel()` installs a `MeterProvider`
  (OTLP or console) alongside the `TracerProvider`, so the metric handles in
  `voussoir.observability.metrics` reach a backend. Wired into
  `default_container()`.
- **JSON logging via env** — `VOUSSOIR_LOG_FORMAT=json` and
  `VOUSSOIR_LOG_LEVEL` now drive `configure_logging()`; `default_container()`
  no longer hard-codes the dev format.

### Correctness

- **Per-model cost pricing** — new `voussoir.llm.pricing` table (Anthropic +
  OpenAI) replaces the hardcoded `$10/$30 per 1M` estimate in run-level cost,
  budget checks, and span attributes.
- **Real tokenizer fallback** — `count_tokens` uses tiktoken (cl100k_base) when
  installed (`voussoir[tokenizers]`) and a char-based estimate otherwise,
  replacing the word-count stub.

### Packaging

- **Base install is now importable** — `PyJWT` and `cryptography` moved from
  the `[a2a]` extra to base dependencies (previously `pip install voussoir`
  could not `import voussoir`).
- `types-pyyaml` moved to the `dev` extra; `black` pinned to `24.4.2` to match
  pre-commit; `tokenizers` extra added for accurate token counting.

### Config

- **`voussoir.yaml` env interpolation** — `${VAR}` / `${VAR:-default}`
  placeholders expand from the environment at load time (unset vars without a
  default raise `KeyError`).

### Data layer

- **SQLite backend** now uses WAL mode, a busy timeout, and runs all sqlite
  calls in `asyncio.to_thread` so the event loop is never blocked.

### Repository

- Added `.gitignore`, `.pre-commit-config.yaml`, GitHub Actions CI (test
  matrix 3.12/3.13 + lint/typecheck), Dependabot, PR/issue templates,
  `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and a production
  `Dockerfile` (multi-stage, non-root, pinned `ctxforge` sibling) + `.dockerignore`.

### Tests

- Added an autouse dummy-API-key fixture so the suite runs without secrets;
  live tests skip on the dummy key. Optional-dep tests
  (qdrant / sentence-transformers) now `importorskip` instead of erroring.

---

## v1.2.1 — 2026-05-24

Housekeeping patch. Repairs four examples that were broken by API
hardening landed in earlier releases but never propagated to the
`examples/` tree. No source code changes; no CHANGELOG impact on
downstream consumers.

### Fixed

- **`examples/02_research_agent/main.py`**: stale since v1.0.4 E3 added
  `IMemoryStore` to the `default_container()` freeze list.
  `bind_sqlite_memory` rebinds `IMemoryStore` and so was rejected.
  Example now constructs an unfrozen `Container()` with the same
  defaults `default_container()` wires (memory, session, telemetry,
  authz, executor, guardrail chain, key provider, embedder, LLM) minus
  the freeze, then calls `bind_sqlite_memory` to tier-up to SQLite.
- **`examples/04_a2a_peer/caller.py`**: stale since Phase 4.5a P0 #8
  removed the implicit `httpx.AsyncClient` from bare `AgentRef(card)`.
  The caller now wraps `ref` in `async with ref:` so `.delegate()` has
  an open client.
- **`examples/04_a2a_peer/README.md`**: documents the
  `VOUSSOIR_A2A_ALLOWED_ISSUERS=lead` env var the publisher needs
  (Phase 4.5a P0 #2: default-deny on inbound JWT issuer).
- **`examples/05_observability/console_exporter_demo.py`** and
  **`streaming_cascade_demo.py`**: stale since `AgentContext.open`
  began resolving `ISessionStore` from the container. Both demos used
  bare `Container()`; they now use `default_container()`.

### Verified end-to-end

All four examples + `01_hello_agent`, `03_multi_agent_research`,
`04_validator_judge` were run against the live Anthropic API and
completed successfully. `02b_research_agent_yase`, `05_delegate_plugin`,
and `05_observability/otlp_phoenix_demo` were skipped (require docker /
pip install / phoenix collector respectively).

---

## v1.2.0 — 2026-05-24

Additive minor release. Two surface additions impost requires: a
cooperative interrupt primitive on `ToolContext`, and a runtime-container
factory in `voussoir.testing` that mirrors `make_container` without the
production freeze list.

No breaking changes. v1.1.0 back-compat aliases (`ToolExecutor`,
`InboundJWTVerificationNotConfiguredError`) remain in place; their
removal is deferred.

### Added

- **`voussoir.InterruptRequest`** — `VoussoirError` subclass carrying
  `kind: str` and `payload: dict[str, Any]`. Raised by
  `ToolContext.interrupt(kind, payload)` to signal that a tool needs
  external input. Propagates through `Agent.run` via the existing
  `on_error` middleware fanout — runtime layers (e.g. impost) catch it
  in a custom `Middleware.on_error` to persist the interrupt and
  resume the agent later.
- **`voussoir.IInterruptable`** — runtime-checkable Protocol:
  `async def interrupt(kind, payload) -> Mapping`. `ToolContext`
  structurally satisfies it.
- **`ToolContext.interrupt(kind, payload)`** — async method, default
  impl raises `InterruptRequest`. Replay-aware runtimes intercept the
  exception via middleware and re-enter the tool with a cached return
  value, so the same call appears to return the operator-supplied
  payload after resume.
- **`voussoir.testing.make_runtime_container(...)`** — companion to
  `make_container` for in-process runtimes that host voussoir agents.
  Pre-binds `ILLMProvider`, `IToolExecutor`, `IGuardrailChain`,
  `ITelemetrySink` (overridable via kwargs) plus `IMemoryStore` /
  `ISessionStore` / `KeyProvider` defaults. Built from a fresh
  `Container()` (no freeze) so runtime layers can rebind any Protocol.
  Accepts `extra_bindings: Mapping[type, Any]` for additional
  Protocol overrides.

### Migration

No action required. New surface is opt-in: agents that never call
`ctx.interrupt(...)` see no behavioral change. Runtime layers wanting
durable HITL adopt the pattern documented in `docs/extending.md`.

---

## v1.1.0 — 2026-05-23

Extensibility minor release. Aligns voussoir's three core extension
surfaces (executor, guardrail chain, middleware) on Protocol-based
DI; ships OpenAI tool-calling support via a per-provider adapter
layer; ships the public `voussoir.testing` subpackage for downstream
library authors; closes the v1.0.4 E2 outbound RS256 half-fix.

Eleven F-tasks across three tranches: DI alignment (F1+F2+F4),
provider portability (F5+F6+F7), A2A security cleanup (F8+F9), and
ecosystem (F10+F11+F12). F3 (Container.resolve @overload to drop the
13× `# type: ignore[type-abstract]` cluster) was investigated and
deferred to v1.2 — the experimental `TypeForm[T]` approach worked
but the runtime dep + mypy `enable_incomplete_feature` cost wasn't
worth it for this release.

### Breaking changes

- **`ToolExecutor` → `IToolExecutor` rename** (F1). The Protocol name
  aligns with `IMemoryStore` / `IGuardrailChain` / `ITelemetrySink`.
  Old name is kept as a module-level alias in `voussoir.executors`
  for one cycle; removed in v1.2.0.
- **`IToolExecutor` joins the freeze list (6 → 7 keys)** (F1).
  `default_container()` now freezes the executor binding alongside
  `Authorizer / KeyProvider / ITelemetrySink / ILLMProvider /
  IMemoryStore / ISessionStore`. Plugin code calling
  `c.bind(IToolExecutor, MyExecutor())` on a `default_container()`
  result raises `RuntimeError`. Override via the new
  `Agent(executor=…)` init kwarg or `agent.run(executor=…)` per-run
  kwarg.
- **`IGuardrailChain` Protocol as the container binding key**
  (F2). `bind_default_guardrails(c, profile=…)` now binds under
  `IGuardrailChain` (was the concrete `DefaultGuardrailChain` class).
  `AgentContext.guardrail_chain` field annotation widened to
  `IGuardrailChain`. Existing tests that did
  `c.bind(DefaultGuardrailChain, …)` must migrate to
  `c.bind(IGuardrailChain, …)`. `DefaultGuardrailChain` is unchanged
  as a concrete impl. Not added to the freeze list — profile
  switching depends on rebind.
- **`InboundJWTVerificationNotConfiguredError` → `JWTKeyNotConfiguredError`
  rename** (F8). Covers both inbound + outbound key-missing failures.
  Old name kept as a module-level alias in `voussoir.a2a.keys` and
  re-exported from `voussoir.a2a` for one cycle; removed in v1.2.0.
- **`KeyProvider` Protocol gains `jwt_signing_key(alg) -> Any`**
  (F8). External `KeyProvider` implementations must add the method
  or `isinstance(provider, KeyProvider)` fails structurally.
  `EnvKeyProvider` returns `jwt_secret()` for HS* and raises
  `JWTKeyNotConfiguredError("…no asymmetric signing key…")` for
  RS/ES/PS.
- **`Middleware` Protocol `ctx: AgentContext`** (F12). All four
  hooks (`before_run`, `after_step`, `after_run`, `on_error`) now
  type `ctx` as `AgentContext` instead of `Any` (and `step: Step`,
  `result: AgentResult[Any]`). External middleware impls that typed
  `ctx: Any` still satisfy structurally; impls that opt into the
  narrower type get type-checked attribute access.

### Added

- **`voussoir.testing` public subpackage** (F10). Four free callables
  for downstream library authors: `make_container(llm=…)` (fresh
  unfrozen `Container` with the standard test bindings),
  `stub_llm(content=…, finish_reason=…, tool_calls=…, raw_response=…,
  name=…, side_effect=…)` (mocked `ILLMProvider`), `multi_turn_llm(turns)`
  (ordered-response mocked `ILLMProvider`), `make_key_provider(jwt_secret=…)`
  (deterministic-or-ephemeral `KeyProvider`). `pip install voussoir[testing]`
  installs the recommended `pytest>=8.1` + `pytest-asyncio>=0.23` toolchain;
  the helpers themselves only need `unittest.mock`. Heavy/optional deps
  are lazy-imported inside helper bodies — `import voussoir.testing`
  does NOT pull `keyring`, `fastapi`, or `uvicorn` into `sys.modules`.
- **`IToolExecutor` Protocol** (F1) at `voussoir.executors.IToolExecutor`,
  re-exported from top-level `voussoir`. `@runtime_checkable`; defines
  `name: str` + async `invoke(tool, args, ctx) -> Any`.
- **`IGuardrailChain` Protocol** (F2) at `voussoir.guardrails.IGuardrailChain`,
  re-exported from top-level `voussoir`. `@runtime_checkable`; defines
  async `screen(payload, ctx) -> GuardrailVerdict` + `count() -> int`.
- **`Agent(executor=…, guardrail_chain=…)` init kwargs + matching
  `Agent.run` / `Agent.stream` per-run kwargs** (F4). Precedence:
  run-kwarg > init-kwarg > container resolve > fallback
  `StandardExecutor()` / empty `DefaultGuardrailChain([])`. Closes
  the DI alignment: all three extension surfaces (Tool, IToolExecutor,
  IGuardrailChain) now follow the same Protocol-typed, container-
  resolvable, init-overridable, run-overridable pattern.
- **OpenAI tool-calling support** (F5+F6+F7).
  `voussoir.agent.turn_adapter.ToolCallAdapter` Protocol with three
  methods (`serialize_tool`, `extract_tool_calls`,
  `build_tool_result_message`); two concrete impls (`AnthropicToolCalls`,
  `OpenAIToolCalls`); `adapter_for(llm)` selects via `llm.name`.
  OpenAI's `{type: "function", function: {…}}` envelope, JSON-string
  `arguments`, and `role="function"` + `function_call={"tool_call_id": …}`
  tool-result construction are all handled. Live OpenAI tool extraction
  still requires a ctxforge-side fix to populate `LLMResponse.raw_response`;
  the adapter shape is correct + tested via mocks.

### Changed

- **`Agent`'s hardcoded `StandardExecutor()` instantiations replaced
  by `self._resolve_executor(…)`** (F4) in `_run_normal`,
  `_run_with_cascade`, and `stream`. Defensive fallback to
  `StandardExecutor()` remains as the bottom rung of the precedence
  ladder.
- **`dispatch.py` + `turn.py` `executor` parameter annotation widened
  from `StandardExecutor` to `IToolExecutor`** (F4). Annotation-only;
  no behavior change.
- **`AgentContext.guardrail_chain` field annotation widened to
  `IGuardrailChain`** (F2).
- **`bind_default_guardrails` binding key changed from
  `DefaultGuardrailChain` to `IGuardrailChain`** (F2).
- **`tool_turn_prepare` + `accumulate_outcomes` route through the
  `ToolCallAdapter`** (F5). `PreparedTurn.adapter: ToolCallAdapter | None`
  carries the selected adapter from prep into dispatch.
- **`LoggingMiddleware` body upgraded from `getattr(ctx, "run_id", None)`
  / `getattr(ctx, "trace_id", None)` to direct attribute access** (F12).
  The Any-typed param had hidden the fact that these attrs are
  guaranteed by the Protocol.
- **`tests/imports/test_layering.py::_agent_imports_in` skips
  `if TYPE_CHECKING:` blocks** (F12). The Middleware Protocol now
  references `AgentContext` / `Step` / `AgentResult` under TYPE_CHECKING
  to break a runtime cycle.

### Fixed

- **Outbound JWT signing uses alg-correct key material** (F9).
  `AgentRef._issue_jwt` now calls `provider.jwt_signing_key(alg)`
  instead of `provider.jwt_secret()`. Closes the v1.0.4 E2 half-fix:
  inbound verification routed through `jwt_verification_key(alg)` but
  outbound signing still called `jwt_secret()` unconditionally. For
  RS256 outbound on `EnvKeyProvider`, PyJWT now never sees symmetric
  bytes — `JWTKeyNotConfiguredError` raises before the encode call
  with operator-actionable guidance.
- **Anthropic-only tool-calling stopgap removed** (F7). The `F-1
  stopgap` block in `_run_normal` (raised `NotImplementedError` for
  `effective_tools` + non-Anthropic LLM) is gone. Unsupported-provider
  errors now route uniformly through `adapter_for(llm)` and name both
  supported providers in the message.

### Removed

- Nothing removed in v1.1.0. Two renames (`ToolExecutor` → `IToolExecutor`,
  `InboundJWTVerificationNotConfiguredError` → `JWTKeyNotConfiguredError`)
  keep back-compat aliases for one cycle (removed in v1.2.0).

### Deferred to v1.2

- **F3 — `Container.resolve` `@overload` to drop `# type: ignore[type-abstract]`
  cluster.** Investigated via `TypeForm[T]` (PEP 747, mypy
  `enable_incomplete_feature` flag); achieved 109-of-109 site cleanup
  with full type inference preserved. Reverted due to experimental-
  feature risk + runtime `typing_extensions>=4.14` pin. Will revisit
  once PEP 747 lands in Python 3.14 stdlib `typing`.

### Reference commits

- `3161f9a` F12 — Middleware Protocol `ctx: AgentContext` + concrete impls
- `716b81e` F11 — delegate conftest fixtures to `voussoir.testing` + extend `stub_llm`
- `6d36281` F10 — `voussoir.testing` public subpackage + lazy A2A imports
- `cdb2a48` F9 — `AgentRef._issue_jwt` uses `jwt_signing_key(alg)`
- `9cdada2` F8 — `KeyProvider.jwt_signing_key` Protocol + `JWTKeyNotConfiguredError` rename
- `258a8b8` F7 — remove Anthropic-only tool-calling stopgap
- `dc5739b` F6 — `OpenAIToolCalls` adapter + dispatch
- `1f2cbbc` F5 — `ToolCallAdapter` Protocol + extract `AnthropicToolCalls`
- `67549d5` F4 — `Agent.__init__` + `run`/`stream` `executor=` + `guardrail_chain=` overrides
- `217b378` F2 — `IGuardrailChain` Protocol + container binding key change
- `7d43199` F1 — `IToolExecutor` Protocol + container binding + freeze list 6 → 7

## v1.0.4 — 2026-05-23

Round-3 review patch — 9 fixes spanning security, errors, perf, test
hygiene, and type discipline. Synthesized from five specialized
parallel reviewers (architecture, A2A/cascade, errors, test hygiene,
type discipline, perf) audit of v1.0.3.

### Breaking changes

- **`voussoir.a2a.discover_card`** still uses `timeout_s` (renamed in
  v1.0.3). No new kwarg breakage in v1.0.4.
- **`KeyProvider` Protocol** gains `jwt_verification_key(alg)` method
  (E2). External `KeyProvider` implementations must add this method
  or `isinstance(provider, KeyProvider)` will fail structurally.
- **`default_container()` freeze list extended to 6 keys** (E3). Was
  `Authorizer / KeyProvider / ITelemetrySink` (v1.0.2 D6); now also
  `ILLMProvider / IMemoryStore / ISessionStore`. Callers using
  `bind_sqlite_memory` / `bind_postgres_memory` on a
  `default_container()` result now hit `RuntimeError` — build a
  fresh `Container()` and call the helper before any freeze.

### Security

- **CRITICAL — AgentCard expiry now enforced** (E2 / C3). AgentCard
  gained an optional `exp: int | None = None` field +
  `card_from_agent(ttl_s=...)` ergonomic helper. `_fetch_and_verify`
  sets `verify_exp=True` and catches `ExpiredSignatureError` BEFORE
  the generic `InvalidTokenError` clause, re-raising as
  `CardVerificationError("Agent card expired (exp=...)")`. Cards
  without `exp` still verify (back-compat).
- **CRITICAL — Inbound JWT alg/key dispatch** (E2 / C4). The publisher
  unconditionally passed `provider.jwt_secret()` (symmetric bytes) as
  the verification key regardless of algorithm, so
  `VOUSSOIR_A2A_JWT_ALGORITHM=RS256` silently rejected every peer
  request with a generic 401 (PyJWT raised `InvalidKeyError`).
  Added `KeyProvider.jwt_verification_key(alg)` Protocol method +
  `InboundJWTVerificationNotConfiguredError`. `EnvKeyProvider`
  returns `jwt_secret()` for HS* and raises the new error for RS256.
  Publisher catches the misconfig → HTTP 500 (server-side error),
  distinct from 401 (client auth failure). No key material leaks
  through the 500 body.
- **CRITICAL — `jwks_uri` propagation** (E2 / C5). Published cards
  carried `jwks_uri=None`; consumer fallback was
  `{card_endpoint_base}/.well-known/jwks.json` which broke for
  non-root-mounted routers. `make_a2a_router` now derives `jwks_uri`
  via `urlunsplit(scheme, netloc, "/.well-known/jwks.json", "", "")`
  (origin-only) and threads it into `card_from_agent`. Cards mounted
  at `/api/v1/a2a` now advertise the correct
  `https://host/.well-known/jwks.json`.
- **Container freeze extension** (E3). Three additional
  security-critical bindings (`ILLMProvider`, `IMemoryStore`,
  `ISessionStore`) now freeze in `default_container()` to prevent
  plugin-driven swaps. Threat parity with the original D6 trio.

### Agent / streaming

- **Streaming output-guardrail `is_blocked` captured** (E4). Both
  simple-path and tool-path output-stage guardrail blocks now
  capture `is_blocked` from `apply_guardrail_verdict` and set
  `finish_reason = "blocked"` when True. Previously the bool was
  discarded; `finish_reason` stayed `"completed"`; the v1.0.2 D7
  cascade-on-completed gate fired the cascade against the blocked
  placeholder text, emitting a spurious `cascade_failed` event.

### Errors

- **`VoussoirError` base class** (E6). New `voussoir.errors.VoussoirError`
  is the shared ancestor of all voussoir-domain exception families
  (`AuthError`, `DelegationError`, `CardVerificationError`,
  `PolicyViolationError`, `NoCardSigningKeyError`,
  `InboundJWTVerificationNotConfiguredError`). Downstream callers can
  now `except VoussoirError:` at application boundaries. Top-level
  `voussoir.__init__.__all__` re-exports `VoussoirError`,
  `PolicyViolationError`, `PolicyViolation`, `AuthError`.
- **Auth retry path surfaces permanent failures clearly** (E6).
  `StandardExecutor` now catches `MissingCredentialError` from
  `broker.refresh()` AND a second `AuthenticationFailedError` from
  the retry `tool.invoke`, re-raising as `AuthenticationFailedError`
  with "permanently"/"after refresh"/"may have been revoked" framing
  (`from exc` preserves causation). Previously the raw error
  propagated to `_dispatch_one`'s except Exception clause as a
  `TOOL_ERROR: ...` string handed to the LLM, indistinguishable from
  a transient tool failure.
- **`PolicyViolationError` docstring corrected** (E6). Said "budget
  breached" but the class also raises for `CAPABILITY_DENIED`,
  `TAINT_EXFILTRATION`, `AUTHZ_DENIED`, `STREAMING_NOT_SUPPORTED`,
  `DELEGATE_NOT_FOUND`, `CAPABILITY_CLAMPED_EMPTY`. New docstring
  organises all 11 variants into Budget / Security / Dispatch groups
  and documents `.violation` as the StrEnum branching field.

### Performance

- **`_dispatch_one` skips Pydantic audit when guardrail chain is
  empty** (E5). Previously the executor unconditionally constructed
  `GuardrailPayload` + called `chain.screen()` + built
  `GuardrailVerdict` + `GuardrailDecision` + bumped the
  `GUARDRAIL_DECISIONS` OTel counter for BOTH `tool_call` AND
  `tool_output` stages on every tool call -- even with no chain
  configured (the common case). ~3 Pydantic constructions × 2 stages
  × O(tool_calls) of pure waste. Now wrapped in
  `if chain.count():` guards mirroring `agent.py`'s input/output
  pattern.

### Public API

- **Concrete authorizers + brokers surfaced** (E7). The four concrete
  `Authorizer` impls (`AllowAllAuthorizer`, `RoleAuthorizer`,
  `DomainAuthorizer`, `ChainedAuthorizer`) and six concrete
  `CredentialBroker` impls (`Env`/`File`/`MTLS`/`Keychain`/`OAuth2`/
  `Chained`) are now exported from `voussoir.auth.__all__`. Previously
  the framework's startup warning told users to bind `RoleAuthorizer`
  but the name hit `ImportError` when imported from `voussoir.auth`
  directly -- users had to discover the sub-package paths.
- **Broken example fixed** (E1). `examples/04_a2a_peer/caller.py`
  imported `AgentRef` from `voussoir.a2a.discovery` (wrong module);
  fixed to use the top-level re-export `voussoir.a2a.AgentRef`.

### Type discipline

- **`disallow_subclassing_any` scoped** (E9 / C2). The global
  override hid 4 real strict-mode errors in `memory.adapter` +
  `memory.backends.sqlite` + `memory.backends.qdrant` AND silently
  permitted any future untyped-subclass mistake anywhere in the
  codebase. Scoped via `[[tool.mypy.overrides]]` to exactly the 3
  memory modules that subclass ctxforge Protocols. The other 108
  source files now genuinely check.
- **`FinishReason` Literal deduplicated** (E9 / I9). Was duplicated
  across `agent.py` + `result.py` + a `str`-typed field on
  `WireAgentResult`. Canonical home is now `voussoir.agent.result`
  (PEP 695 `type` statement with class-level docstring matching the
  `StepKind` precedent). `WireAgentResult.finish_reason` narrowed
  from `str` to `FinishReason`. Re-exported from
  `voussoir.agent.__all__`.
- **`AgentResult.to_wire()` `@overload`** (E9 / I10). Was typed
  `AgentResult[Out] | Any` (the `Any` erased all return-type
  narrowing). Now: `profile="public" -> WireAgentResult`,
  `profile="trusted" -> AgentResult[Out]`. Callers get precise
  typing on both branches.

### Test infrastructure

- **`reset_warn_once_flags` autouse fixture** (E8 / I4). The
  module-global `_warned` flag in `voussoir.auth.authorizers.allow_all`
  and `_fallback_warned` in `voussoir.executors.standard` never reset
  between tests, making warn-once assertions execution-order-
  dependent. The fixture resets both flags before+after every test
  AND re-binds the module-level structlog loggers to fresh proxies
  so `capture_logs()` can intercept (defeats the
  `cache_logger_on_first_use` lazy-proxy freeze).
- **Private test helpers migrated to global fixtures** (E8 / I8).
  `tests/agent/test_agent.py` + `tests/agent/test_cascade.py` had
  private `_stub_llm` / `_container` helpers predating the Phase 3.5
  global fixtures. They built bare `Container()` without
  `NullTelemetrySink` or `EnvKeyProvider(allow_ephemeral=True)`,
  diverging from production-shaped wiring. 13 tests migrated; private
  helpers deleted.

### Tests

- 1006 → 1044 passing (+38 across 9 functional tasks). 6 skips
  (all pre-existing infrastructure / coverage / model-weight guards).

### Reference SHAs

| Task | SHA | Description |
|------|-----|-------------|
| E1 | 88e98ad | repair broken AgentRef import in 04_a2a_peer |
| E2 | 74d16cb | A2A security cluster -- exp + JWT alg dispatch + jwks_uri |
| E3 | fb7b29a | freeze ILLMProvider + IMemoryStore + ISessionStore |
| E4 | 5b7d96b | capture is_blocked from output guardrail in stream paths |
| E5 | b05fc6d | _dispatch_one skips Pydantic audit when chain empty |
| E6 | 77fd7d4 | VoussoirError base + auth retry MissingCredentialError handling |
| E7 | 24bf9d4 | surface concrete authorizers + brokers in voussoir.auth.__all__ |
| E8 | b284595 | autouse reset of warn-once flags + migrate private test helpers |
| E9 | 14593f8 | scope mypy strict subclassing + dedup FinishReason + overload to_wire |
| E10 | (current) | release ceremony |

## v1.0.3 — 2026-05-21

Housekeeping patch — three leftover items the v1.0.2 round-2 cycle
flagged as out-of-scope.

### Breaking changes

- **`voussoir.a2a.discover_card(timeout=...)` renamed to `timeout_s=`** to
  silence pre-existing ruff ASYNC109 (async function with a `timeout`
  kwarg) and match voussoir's naming convention (`ttl_s`,
  `refresh_buffer_s`). Behaviour unchanged — the body still passes the
  value to `httpx.AsyncClient(timeout=...)` for actual enforcement.
  Callers must rename the kwarg; passing `timeout=...` now raises
  `TypeError: unexpected keyword argument 'timeout'`.

### Fixes

- **`tests/test_phase3_exit::test_coverage_floor_phase3_packages` no
  longer fails on `--no-cov` runs.** A stale `.coverage` file from a
  prior partial run defeated the existing `if not cov_file.exists():
  pytest.skip(...)` guard. New `tests/conftest.py::pytest_configure`
  hook unlinks `.coverage` at session start when pytest-cov is NOT
  collecting (detected via `config.option.cov_source`). `make ci`'s
  `--cov=...` path is untouched; the test now correctly skips on
  `--no-cov` and runs only under coverage collection.

- **Simple-path streaming no longer reports `tokens_in=0` /
  `tokens_out=0`** (D7 leftover). `ILLMProvider.stream()` yields `str`
  chunks with no usage info, but `count_tokens()` is on the protocol.
  New private helper `_estimate_stream_tokens(llm, messages, full)`
  uses `llm.count_tokens()` to estimate both directions client-side.
  Best-effort: returns `(0, 0)` on exception OR if `count_tokens`
  returns a non-int (handles `MagicMock(spec=ILLMProvider)` in tests).
  Provider's tokenizer may differ slightly from voussoir's
  `count_tokens` -- numbers are approximate, but far better than 0.

### Tests

- Test count 1003 → 1005 passing; 3 → 6 skipped (incl. the now-cleanly-
  skipping `test_phase3_exit::test_coverage_floor_phase3_packages` and
  three pre-existing yase/sentence-transformers infrastructure skips).
- 4 new tests in `tests/agent/test_stream_simple_path_tokens.py`:
  helper correctness, simple-path uses estimate end-to-end, MagicMock
  non-int tolerated, exception path returns `(0, 0)`.

### Reference SHAs

| Task | SHA | Description |
|------|-----|-------------|
| housekeeping | a452ef3 | coverage flake + ASYNC109 + simple-path stream tokens |
| release | (current) | version bump + CHANGELOG + tag |

## v1.0.2 — 2026-05-21

Round-2 review patch release. 4 security tightenings + 1 critical
delegation-bypass fix + 1 stream parity bundle + 2 doc/perf cleanups
+ test-suite improvements.

### Security

- **CRITICAL — Lethal Trifecta delegation bypass closed** (D5).
  Sub-agent UNTRUSTED taint now propagates back to the parent through
  AgentResult.taint + accumulate_outcomes' new merge loop. Before this
  fix a parent agent could "launder" UNTRUSTED input through delegation
  and then call EXFILTRATION tools unblocked.
- **MCP tool descriptions screened for prompt injection** at
  registration time (D4). A compromised MCP server can put
  "ignore previous instructions ..." into tool.description and have
  voussoir bake it into the LLM system prompt; MCPTool.__init__ now
  runs the same regex set as PromptInjectionHeuristic via the new
  public voussoir.guardrails.builtin.injection.find_injection_pattern
  helper.
- **Container.freeze() locks security-critical bindings against
  plugin rebind** (D6). default_container() freezes Authorizer,
  KeyProvider, and ITelemetrySink. A voussoir.delegates plugin that
  tries c.bind(Authorizer, HostileAuthorizer()) now raises
  RuntimeError. child() containers inherit the frozen set.
- **@tool requires explicit capability=** (D3). Capability.NONE
  default silently bypassed the cap mask (0 & X == 0). Decoration
  now raises ValueError with an actionable error message naming the
  real Capability enum members.
- **_FallbackAllowAllAuthorizer emits a one-time
  authz_fallback_used structlog warning** when hand-rolled containers
  fall back (D3). default_container() bindings still log
  authz_unenforced as before; this catches the previously-silent
  fallback path.

### Auth model

- **Principal / AuthRequirement / Credentials** now use
  `extra="forbid"` (D2). Typos like `Principal(roels=[...])` raise
  ValidationError instead of silently dropping the field.

### Agent / stream parity (D7)

- `Agent.stream` `while True:` replaced with bounded
  `for step_idx in range(policy.max_steps + 1):` + the same
  `policy.check` budget gate `Agent.run` uses. Tokens, duration, cost
  now accumulate across turns in the tool-using path.
- `stream` calls `_run_setup` with `skill_content=...` (previously
  missing); agents with declared skills now get skill text in their
  streaming-mode prompt.
- `stream`'s outer try catches `BaseException` (was `Exception`), so
  `asyncio.CancelledError` and `KeyboardInterrupt` reach the
  `on_error` middleware fanout.
- The post-`done` AgentResult is constructed once and reused by the
  cascade gate + after_run hook (previously two zeroed-out copies).
- Cascade gate now requires `finish_reason == "completed"`; budget-
  violation runs no longer produce spurious `cascade_failed` events
  on empty output.

### Performance (D9)

- `@tool` `_wants_ctx` flag cached once at decoration time instead
  of calling `inspect.signature` per tool invoke.
- `ToolRegistry.describe()` memoizes the tool-descriptor list and
  invalidates on `register()`. Caller-must-not-mutate the returned
  list (documented in the docstring).
- ToolContext deep-copy claim investigated; 2.3μs per construction
  is below the threshold. No code change.

### Internal correctness (D8)

- `_dispatch_one` reads the cached guardrail chain from
  `ctx.guardrail_chain` instead of resolving from `ctx.container`
  per tool call. A mid-run rebind of the container's chain no
  longer creates asymmetric enforcement.
- `DefaultGuardrailChain.count()` public helper replaces the private
  `_by_stage` access agent.py was reaching into.
- `self._guardrail_chain` renamed to `self.guardrail_chain`
  (consistency with neighboring public attributes); the
  `self._has_guardrail_chain` boolean cache dropped (gates now use
  `if self.guardrail_chain.count():`).

### Documentation (D1)

- README install URLs pinned to `@v1.0.2` (was `@v1.0.0`); stale
  `~900 tests as of v1.0` updated; "PyPI publish deferred to v1.0.1"
  preamble removed (already shipped).
- `docs/architecture.md` public-contract block fixed: `cascade_policy`
  → `cascade`, `expose_a2a` removed (publish via `make_a2a_router`),
  `guardrails=` removed from Agent kwargs (container-bound; explained
  in a new note pointing at extending.md).
- Two `@tool(auth=...)` snippets corrected to `auth_requirement=`.
- `docs/getting-started.md` broken `examples/02_tool_demo` reference
  fixed to point at the real `examples/02_research_agent`.
- `docs/extending.md` @tool recipe documents `sensitive_args=[...]`
  (logging-only redaction, added in v1.0.1).
- `examples/01_hello_agent/main.py` no longer crashes — `Agent`
  construction now passes `container=default_container()` as
  Phase 4.5a requires.

### Tests

- Wall-clock-flaky `test_phase4c_exit::test_exit_1_concurrent_dispatch_wall_clock`
  replaced with deterministic asyncio.Event barrier (D10). Renamed to
  `test_exit_1_concurrent_dispatch_runs_in_parallel`. Sequential
  dispatch now produces a clear timeout-driven failure.
- Added coverage for the `tool_output REWRITE → re-screen still
  fails` branch in dispatch.py (D10).
- ~35 new tests across the 10 D-commits.

### Internal refactor (post-v1.0.1)

- Extracted `_run_setup` + `_safe_hook_fanout` helpers from
  `Agent._run_normal` (89dcc2f). Shared with the streaming path
  (D7 reuses `_run_setup`).

### Reference SHAs

| Task | SHA | Description |
|------|-----|-------------|
| post-v1.0.1 | 89dcc2f | extract _run_setup + _safe_hook_fanout helpers |
| D1 | 7738d1d | scrub stale + broken refs across docs/example |
| D2 | af0de60 | extra="forbid" on Principal/AuthRequirement/Credentials |
| D3 | d98e377 | warn-once on authz fallback + @tool requires explicit capability |
| D4 | 3e37284 | MCP tool description prompt-injection screen |
| D5 | abbb7bf | sub-agent UNTRUSTED taint propagates to parent |
| D6 | 18c0612 | Container.freeze() for security-critical keys |
| D7 | 8b948ad | close 4 stream() parity bugs vs _run_normal |
| D8 | 8e7faab | cache guardrail chain through ctx + count() helper |
| D9 | 1dc18d8 | cache _wants_ctx + registry describe() |
| D10 | 17842ea | fix wall-clock concurrency flake + cover REWRITE re-screen branch |
| D11 | (current) | release ceremony |

## v1.0.1 — 2026-05-21

**Cleanup patch from comprehensive v1.0 code review.** Five specialized
reviewers (architecture, security, code quality, concurrency, DX) audited the
v1.0 codebase end-to-end; this release lands the top 11 findings. No new
features. Test count: 908 → 943.

### Fixes

- **`fix(docs)` quickstart actually runs** — the README's 5-line quickstart
  used bare `Container()`, which has no bindings and crashed with
  `LookupError: No binding registered for ISessionStore` on first
  `agent.run()`. Switched to `default_container()` (now also exported from
  the top-level `voussoir` namespace so the import stays single-segment).
- **`fix(docs)` `extending.md` recipes** — 7 recipes used
  `from ctxforge import Container` (ImportError) and one used
  `finish_reason="stop"` (Pydantic ValidationError). Corrected.
- **`fix(polish)` `ConfigDict(extra="forbid")` on `ToolContext` +
  `RequestCascade`** — the two models still used dict-form `model_config`
  without `extra="forbid"`, silently accepting typo kwargs. Now matches
  the rest of the codebase.
- **`fix(executor)` `AUTHZ_DENIED` error names the firing authorizer** —
  ChainedAuthorizer denials previously required inspecting
  `AgentResult.authz_decisions` to know which policy fired; the error
  message now prepends `[<authorizer_name>]`.
- **`fix(executor)` dispatch.py comment corrected** — `dispatch_tool_calls`
  claimed `gather` "only ever propagates CancelledError", but
  `PolicyViolationError` is deliberately re-raised by `_dispatch_one`.
- **`fix(auth) BREAKING:` `SameOrgPassthroughForwarder` mints verifiable
  JWTs** — the forwarder was emitting tokens with only `{sub, iat, exp}`;
  the receiver requires `iss`/`aud`/`nbf` via `options={"require": [...]}`
  and rejected every forwarder-minted token with
  `MissingRequiredClaimError`. Constructor now requires `issuer: str`;
  `forward()` now requires `audience: str`. `AgentRef.delegate` passes
  `audience=self.card.name` automatically. Round-trip integration test
  added.
- **`fix(security)` `_redact_masked_fields` handles nested structures** —
  MASK previously only redacted top-level keys; nested dicts and lists
  silently leaked. Now recurses through dicts, lists, tuples, and Pydantic
  BaseModels.
- **`fix(guardrails)` `URLAllowlist` blocks IPv6 literals** — the URL regex
  `[\w.-]+` didn't match bracketed IPv6 hosts; `https://[::1]/admin`
  silently bypassed the allowlist. Now matches both DNS/IPv4 and IPv6
  forms; operators write `::1` (unbracketed) in their allowlist.
- **`fix(executor)` `@tool(sensitive_args=[...])` redacts secrets from
  INFO logs** — `tool.invoke` log entries unconditionally serialized
  `args.model_dump()`, leaking passwords / connection strings / API keys.
  Decorator now accepts a `sensitive_args` list; the executor replaces
  matching args with `"[REDACTED]"` in the log line (the tool function
  still receives the real values).
- **`fix(auth)` `OAuth2CredentialBroker` serializes concurrent refresh** —
  N concurrent tool calls hitting an expired token fired N parallel
  refresh-token grants; standard OAuth2 servers invalidate the
  refresh_token on first use, breaking the agent run at every expiry
  boundary. Added `asyncio.Lock` with double-checked-locking pattern.

### Refactor

- **`refactor` closes two private-cross-boundary imports** —
  `voussoir.guardrails.judge` imported the private
  `voussoir.agent.validators._parse_judge_verdict` (a layering inversion);
  the function was 3 lines and is now inlined in `LLMGuardrailJudge`.
  `voussoir.cli.cmd_validate` imported the private
  `voussoir.agent.agent_builder._parse_capability_list`; promoted to
  public `voussoir.tools.parse_capability_list`.
- **`refactor(agent)` extract `apply_guardrail_verdict`** — Agent.run and
  Agent.stream had ~90 lines of duplicated `BLOCK`/`REWRITE` check-and-
  apply logic across 5 sites. Consolidated into one helper in
  `dispatch.py` that records the audit decision and returns
  `(new_content, is_blocked)`.

### Features

- **`feat(middleware)` `after_step`/`after_run`/`on_error` hooks now fire** —
  the `Middleware` Protocol declared four lifecycle hooks but only
  `before_run` was actually invoked. `LoggingMiddleware.after_step` /
  `.after_run` / `.on_error` were dead code; `BudgetMiddleware` /
  `RetryMiddleware` were partially inert. All four hooks now fire in
  both `Agent.run` and `Agent.stream`. Hook exceptions are isolated:
  a faulty middleware logs a warning but cannot break the run loop.
  `LoggingMiddleware` now emits `agent.step` / `agent.run.end` /
  `agent.run.error` events.

### Breaking changes (v1.0 → v1.0.1)

- `SameOrgPassthroughForwarder.__init__` gains required `issuer: str`
  kwarg; `forward()` gains required `audience: str` kwarg. Acceptable
  because the v1.0 forwarder was 100% non-functional — nobody can be
  depending on the broken signature.

### Tooling

- agent.py LOC cap bumped 925 → 1025 (T9 wires the missing middleware
  hooks; +101 LOC); current size 1012 after the T10 guardrail-helper
  extraction (-4 LOC). The +97 net is the price of honoring the
  Middleware Protocol contract.

## v1.0.0 — 2026-05-21

**Voussoir 1.0 — production-stable agent framework.** Stable public API governed
by semver. Shipped across Phases 0–6 (~21 weeks of development). PyPI publish
deferred to v1.0.1; install via
`pip install git+https://github.com/your-org/voussoir@v1.0.0`.

Highlights across phases:

- **Phase 0–3:** Container + Agent + run loop + cascade gates + soft-policy
  guardrail Protocol. Single-agent and hierarchical delegation. MCP and A2A
  native (A2A spec compliant wire format, JWT auth, AgentCard discovery).
- **Phase 4:** Plugin loader (entry-point based) + cascade depth budgets +
  ctxforge memory layer (session store + semantic retrieval).
- **Phase 4.5:** Two-tranche architectural review: 25 P0/P1 hardening fixes
  (wire security, key management, AgentRef lifecycle, streaming-cascade gate)
  and 15 P2 polish fixes (module renames, stream/run parity, docstring sweep).
- **Phase 5 (Guardrails + OTel):** 8 deterministic built-in guardrails;
  30-attack Lethal Trifecta corpus blocked at 100%; OpenTelemetry promoted to
  a base dependency with `gen_ai.*` span conventions; A2A wire redaction via
  `WireAgentResult`.
- **Phase 6 (Auth + CLI + Docs + v1.0):** Two-axis auth model
  (`Authorizer` + `CredentialBroker` Protocols); 6 built-in
  `CredentialBroker` implementations (env, file, mTLS, keychain, OAuth2,
  chained); 4 built-in `Authorizer` implementations (allow-all, role, domain,
  chained); A2A `Principal` forwarding via JWT; `voussoir` CLI (`new` /
  `run` / `validate` / `doctor`); public documentation site
  (mkdocs-material + mkdocstrings).

Note: DualStream, OPA authorizer, and masking authorizer are v1.x / v2.x
roadmap items and did not ship in v1.0.0. Benchmark suite and PyPI publish
are deferred to v1.0.1.

Tags shipped before v1.0.0 (in chronological order):

- v0.0.1-phase0
- v0.1.0-phase1
- v0.2.0-phase2
- v0.3.0-phase3
- v0.3.5-phase35
- v0.4.0a-phase4a
- v0.4.0b-phase4b
- v0.4.0c-phase4c
- v0.4.0d-phase4d
- v0.4.0e-hardening
- v0.4.0f-polish
- v0.5.0a-phase5a
- v0.5.0b-phase5b
- v0.5.0c-phase5c
- v0.6.0a-phase6a
- v0.6.0b-phase6b

For migration notes from pre-v1 builds, see the per-phase entries below.

---

## v0.6.0b-phase6b — 2026-05-21

Phase 6 Tranche B — CLI + starter templates.

### Added
- **`voussoir` console script** registered via `pyproject.toml [project.scripts]`. Click group with four subcommands.
- **`voussoir new <name> [--template minimal|research]`** — scaffolds a starter project. Templates ship inside the wheel under `voussoir/templates/` and are loaded via `importlib.resources`; `{{project_name}}` placeholders substituted via `str.replace()`. Refuses to overwrite non-empty target directories.
- **`voussoir.templates.minimal`** — single-agent yaml + 5-line `main.py` + README. The smallest runnable voussoir project.
- **`voussoir.templates.research`** — lead agent + researcher + writer + `@tool web_search` stub. Pure-Python wire-up in `main.py` (yaml carries the topology; tools are attached via `Agent(tools=[...])` because `AgentBuilder._LAYERABLE_FIELDS` intentionally excludes tool callables).
- **`voussoir run <agent>`** — loads voussoir.yaml via `bind_agent_registry`, looks up the named agent in `AgentRegistry`, reads input from `--input <text>` or stdin (`--input -`), streams events to stdout/stderr.
- **`voussoir validate [--config]`** — lints voussoir.yaml without running anything: Pydantic validation surfaces extra-field errors; delegate references are cross-checked against declared agent names; `allowed_capabilities` strings are parsed via `agent_builder._parse_capability_list`; missing API keys per model are reported.
- **`voussoir doctor`** — environment health check: Python version (≥3.12), both LLM API keys, ctxforge importability + version, OpenTelemetry SDK, and optional extras (`mcp`, `fastapi/uvicorn/PyJWT/cryptography` for a2a, `keyring`). Exits 0 only if all required items are present.

### Dependencies
- Added `click>=8.0` to base deps (CLI is user-facing, not a dev tool).

### Test infrastructure
- `tests/cli/` (new) — 22 tests across `test_main.py`, `test_new.py`, `test_run.py`, `test_validate.py`, `test_doctor.py`. All exercise the click CliRunner; none require an LLM API key (stub container patched in `test_run.py`).

### Side fixes
- `tests/auth/test_brokers.py` keychain stub backends now subclass `keyring.backend.KeyringBackend` properly (latent A7 bug surfaced once `keyring` was installed via `pip install -e .` to wire up the console script).
- `.pre-commit-config.yaml` excludes `src/voussoir/templates/*.yaml` from `check-yaml` (placeholders aren't valid yaml; templates are validated at scaffold-time).

### Exit gate
- `tests/test_phase6b_cli_exit.py` — 5 invariants locking the CLI surface.

## v0.6.0a-phase6a — 2026-05-21

Phase 6 Tranche A — auth concrete bindings. F-3 carry-over from Phase 5 now real.

### Added

- **`voussoir.auth.AuthzDecision`** — Pydantic record for Authorizer verdicts (ALLOW/DENY/MASK + reason + masked_fields + authorizer_name).
- **`voussoir.auth` error types** — `AuthError` (base), `MissingCredentialError`, `AuthenticationFailedError`.
- **`voussoir.auth.default_principal()`** — shared helper returning `Principal(user_id="system")`.
- **`voussoir.auth.brokers.*`** — 6 built-in CredentialBroker impls: `EnvCredentialBroker`, `FileCredentialBroker`, `MTLSCredentialBroker`, `KeychainCredentialBroker`, `OAuth2CredentialBroker`, `ChainedCredentialBroker`.
- **`voussoir.auth.authorizers.*`** — 4 built-in Authorizer impls: `AllowAllAuthorizer`, `RoleAuthorizer`, `DomainAuthorizer`, `ChainedAuthorizer`.
- **`voussoir.auth.a2a`** — `JWTPrincipalMapper` + `DefaultJWTPrincipalMapper`; `PrincipalForwarder` + `SameOrgPassthroughForwarder`.
- **`Agent.run(principal=...)` and `Agent.stream(principal=...)`** — new keyword-only argument; `Agent.delegate(...)` carries `parent_ctx.principal` to the child run.
- **`AgentContext.principal` + `AgentContext.authz_decisions`** — first-class context fields.
- **`ToolContext.principal` + `ToolContext.credentials` + `ToolContext.container` + `ToolContext.authz_decisions`** — populated by dispatch + executor.
- **`AgentResult.authz_decisions: list[AuthzDecision]`** — per-run audit log, snapshotted from `AgentContext` at every result-construction site.
- **`@tool(roles=, domains=, auth_requirement=)`** — decorator kwargs read by the new authorizers + broker.
- **`StandardExecutor.invoke` enforces §12.1 ordering** — AUTHZ → CAPABILITY → TAINT → AUTHN (broker.resolve) → tool.invoke → 401/403 retry via `broker.refresh` → MASK redaction.
- **OTel `authz.check` span** — child of `tool.call`, sibling of `capability.check` and `taint.check`. Attributes: `authorizer_name`, `decision`, `reason`.
- **`voussoir.authz.decisions` counter** in `voussoir.observability.metrics`, labels: `decision`, `authorizer_name`, `tool_name`.
- **`make_a2a_router(principal_mapper=...)`** — optional kwarg; default `DefaultJWTPrincipalMapper` maps verified JWT claims into `Principal` (sub→user_id, email→email, groups→roles, plus custom claims).
- **`AgentRef.delegate`** mints the outbound Bearer JWT via `PrincipalForwarder` when one is bound on `parent_ctx.container`; falls back to the existing `_issue_jwt` path otherwise.
- **default_container binding**: `AllowAllAuthorizer` bound on the `Authorizer` Protocol; emits a one-time `authz_unenforced` warning to nudge production deployments to bind a real Authorizer.

### Changed

- **`Authorizer.authorize(...)` Protocol** return type sharpened: `bool` → `AuthzDecision`. (Phase 0 stub was the only blast radius; no production implementations existed.)
- **`tools/mcp.py` annotations** — Phase 5 B7 carry-forward: `int` → `Capability` at three sites (line 45, 79, 96). Mypy strict didn't flag these (IntFlag is int subtype) but they were stale.

### Dependencies

- Added `keyring>=24.0` to base deps (for `KeychainCredentialBroker`).

### Tests

- 868 passing, 8 skipped (~10 new tests this tranche for the exit gate alone, ~90 total).
- Security suite (`tests/security/`) + a2a suite (`tests/a2a/`) preserved: 138/138.

### Exit gate

- `tests/test_phase6a_auth_exit.py` — 10 invariants locking the shipped surface.

### Reference SHAs

| Task | SHA | Description |
|------|-----|-------------|
| A1 | b0b5096 | AuthzDecision + error types + sharpened Authorizer Protocol |
| A2 | dab280a | Principal field propagation through Agent.run/stream/delegate |
| A3 | 99177fd | @tool decorator roles/domains/auth_requirement kwargs |
| A4 | 76960f1 | Inline authz + broker integration in StandardExecutor |
| A5 | 2e06919 | AgentResult.authz_decisions accumulation + WireAgentResult rules |
| A6 | 23b6e87 | OTel authz.check span + voussoir.authz.decisions metric |
| A7 | 13e3abc | Six built-in CredentialBroker implementations |
| A8 | 4351840 | Four built-in Authorizer implementations + default container binding |
| A9 | 5c8515a | A2A JWTPrincipalMapper + PrincipalForwarder + wire-up |
| A10 | (current) | Tranche A exit gate + CHANGELOG |

## v0.5.0c-phase5c — 2026-05-18

Phase 5 Tranche C — OTel SDK promotion + A2A wire redaction + streaming cascade
gate + env-var overrides + examples. Closes Phase 5. The `[observability]` extra
is dissolved; OTel is now a base dependency. All Phase 4b/4c/3.5 carry-overs
are resolved.

### Breaking changes

1. **`opentelemetry-sdk`, `opentelemetry-api`, `opentelemetry-exporter-otlp-proto-http`
   are now base dependencies** (C1, 8869ef2). Was `[observability]` extra.
   ~10 transitive deps added.
2. **`_NoOpTracer` deleted from `voussoir.observability.tracer`** (C2, b58dccd).
   Direct callers (none expected outside voussoir) switch to
   `voussoir.observability.span(...)` or `trace.get_tracer(...)`.
3. **`POST /a2a` response shape: `AgentResult.model_dump()` → `WireAgentResult.model_dump()`**
   (C6, 9f5b833). Restore full dump via `make_a2a_router(wire_profile="trusted")`.
4. **`Agent.stream` no longer raises `STREAMING_NOT_SUPPORTED` for
   `max_cascade_depth=1` cascades** (C7, 41abf6a). Consumers now see
   `cascade_passed`/`cascade_failed` events after `done`.
   `max_cascade_depth > 1` still raises.
5. **`_PKG_VERSION` → `PKG_VERSION` in `voussoir.observability.tracer`**
   (C4, 47f9ab6). Public-named for cross-module import.

### Added

- `voussoir.observability.span(name, **attrs)` — canonical context manager across
  voussoir instrumentation.
- `voussoir.observability.get_meter(name)`.
- `voussoir.observability.configure_otel(c)` — lazy default setup.
- `voussoir.observability.metrics` — 9 centralized metric handles (`TOKENS_IN`,
  `TOKENS_OUT`, `COST_USD`, `DURATION_MS`, `TOOL_CALLS`, `GUARDRAIL_DECISIONS`,
  `CAPABILITY_DENIALS`, `TAINT_EXFIL_BLOCKS`, `CASCADE_ESCALATIONS`).
- 7 span categories instrumented across `agent.py` / `turn.py` / `dispatch.py` /
  `executors/standard.py`: `agent.run`, `reason.<n>`, `llm.complete` (with `gen_ai.*`
  attrs), `tool.call.<name>`, `capability.check` + `taint.check`,
  `guardrail.{input,output,tool_call,tool_output}`, `delegation.dispatch.<name>`,
  `cascade.validate`.
- `voussoir.a2a.WireAgentResult` public re-export.
- `voussoir.WireAgentResult` top-level re-export.
- `AgentResult.to_wire(profile="public"|"trusted")` method.
- `make_a2a_router(wire_profile=...)` kwarg.
- `cascade_passed` and `cascade_failed` `AgentEvent` kinds.
- `VOUSSOIR_AGENT_<NAME>_<FIELD>` env-var overrides applied by `bind_agent_registry`
  (whitelisted fields only).
- `AgentRegistry.list_names()` public method.
- `examples/05_observability/` — 3 demos + README:
  - `console_exporter_demo.py` — Tier 0 ConsoleSpanExporter default.
  - `otlp_phoenix_demo.py` — Tier 1 OTLP HTTP → local Phoenix.
  - `streaming_cascade_demo.py` — streaming + single-pass cascade gate events.

### Disable OTel

- `OTEL_SDK_DISABLED=true` (official OTel env var) or
- `VOUSSOIR_OTEL_DISABLED=1` (voussoir convenience).
- `tests/conftest.py` autouses `VOUSSOIR_OTEL_DISABLED=1` so the test suite stays quiet.

### Tests

- `tests/test_phase5c_otel_exit.py` — 8 exit-gate invariants per spec §6 + §7.
- Total: 768 passing, 6 skipped (was 760 at the C8 baseline).

### Architectural-review findings closed

- Phase 4b carry-over: A2A wire leaks lineage (C6).
- Phase 4c carry-over: streaming cascade gate (C7).
- Phase 3.5 carry-over: env-var overrides (C8).
- Phase 3.5 carry-over: ITracer Protocol — **dropped**; direct OTel SDK as base
  dep makes the indirection unnecessary.

### Reference SHAs

| Task | SHA | Description |
|------|-----|-------------|
| C1 | 8869ef2 | OTel SDK promoted to base deps |
| C2 | b58dccd | Rewrite observability/tracer.py — _NoOpTracer deleted |
| C3 | (spans) | ~15 span sites across agent / turn / dispatch / standard |
| C4 | 47f9ab6 | Metrics emission + PKG_VERSION rename |
| C5 | (env) | VOUSSOIR_OTEL_DISABLED + pytest conftest autouse |
| C6 | 9f5b833 | A2A wire redaction — WireAgentResult + to_wire() |
| C7 | 41abf6a | Streaming cascade gate lift — cascade_passed/failed events |
| C8 | (env-override) | VOUSSOIR_AGENT_<NAME>_<FIELD> env-var overrides |
| C9 | (current) | examples/05_observability/ + exit gate + CHANGELOG |

## v0.5.0b-phase5b — 2026-05-17

Phase 5 Tranche B — Soft-policy guardrail chain. Ships the full
`voussoir.guardrails` subsystem: typed protocol, `DefaultGuardrailChain`
with per-stage dispatch, 8 deterministic built-in guardrails, `LLMGuardrailJudge`
composer, `bind_default_guardrails` (3 profiles), and the 30-attack Lethal
Trifecta security corpus (100% blocked). Also closes 3 framework gaps surfaced
by the corpus.

### Breaking changes

1. **`voussoir.guardrails.Decision` → `GuardrailVerdict`.** The old `Decision`
   name is gone; import `GuardrailVerdict` from `voussoir.guardrails`. (B1, 123edf7)
2. **`@tool` schemas now reject extra fields (`extra="forbid"`)** — surfaced by
   B7 Lethal Trifecta corpus (gap 2). Previously, unknown extra fields were silently
   dropped by Pydantic. Now they cause a schema-validation failure caught by the
   dispatch layer and surfaced as a chain-level `BLOCK` in the audit log.
3. **`Agent.run` / `Agent.stream` populate `AgentResult.guardrail_decisions`**
   when a chain is bound. Runs without a chain bound see an empty list.

### Added

- `voussoir.guardrails.GuardrailPayload` — typed payload per stage (stage, content,
  tool_name, tool_args, capability). (B1, 123edf7)
- `voussoir.guardrails.GuardrailVerdict` — ephemeral return value from `screen()`
  carrying `verdict`, `reason`, and `rewrite`. (B1, 123edf7)
- `voussoir.guardrails.DefaultGuardrailChain` — chains multiple guardrails with
  per-stage dispatch; first BLOCK/REWRITE wins. (B2, fb578d4)
- `voussoir.guardrails.bind_default_guardrails` — three profiles (`off`, `standard`,
  `strict`) that wire a `DefaultGuardrailChain` into a `Container`. (B2, fb578d4)
- 8 built-in deterministic guardrails (`voussoir.guardrails.builtin.*`):
  `InputLengthCap`, `ArgsSizeCap`, `ToolOutputSizeCap`, `PromptInjectionHeuristic`,
  `ArgsSchemaCheck`, `PIIDetector`, `ExfilPatternScan`, `URLAllowlist`. (B3, 1272fe0)
- `voussoir.guardrails.builtin.PromptInjectionHeuristic` is now dual-stage (B8):
  accepts a `stage` kwarg (default `"input"`); the standard profile registers one
  instance at `input` and one at `tool_output` to cover attacker-controlled tool
  output as well as direct user input.
- `voussoir.guardrails.LLMGuardrailJudge` — AMBIGUOUS-fallback composer that wraps
  a primary guardrail and defers to an LLM judge when the primary returns AMBIGUOUS.
  (B4, cc0551d)
- `voussoir.agent.dispatch.verdict_to_record` — public helper that converts an
  ephemeral `GuardrailVerdict` to an audit-log `GuardrailDecision`. (B5, a408219)
- `voussoir.agent.delegation.clamp_tools` — public free function replacing the
  private `Agent._tools_after_clamp` method. (B5, a408219)
- `AgentContext.guardrail_decisions: list[GuardrailDecision]` field — accumulated
  during the run and copied to `AgentResult.guardrail_decisions` on completion.
- `tests/security/lethal_trifecta/` — 30-attack corpus blocked at 100% (B7, 4dffb06):
  6 attack categories × 5 attacks each — exfil-after-read, prompt injection via
  tool output, args-schema injection, PII leakage, A2A malicious card, sub-agent
  taint propagation.

### Framework gaps closed (surfaced by B7 corpus)

- **Gap 1 — dual-stage injection heuristic in standard profile**: `PromptInjectionHeuristic`
  now configurable via `stage` kwarg; standard profile registers instances at both
  `input` and `tool_output`. Test-local `_ToolOutputInjectionHeuristic` workaround
  removed from corpus.
- **Gap 2 — `@tool` schemas set `extra="forbid"`**: `_schema_from_signature` in
  `voussoir/tools/decorator.py` now passes `__config__=ConfigDict(extra="forbid")`
  to `create_model`. Extra fields are explicitly rejected rather than silently dropped.
- **Gap 3 — dispatch wraps args instantiation in try/except**: `dispatch.py` now
  catches `pydantic.ValidationError` at args instantiation and returns a `BLOCK`
  `ToolCallOutcome` with a `GuardrailDecision` record, instead of propagating a hard
  exception to the caller.

### Tests

- `tests/security/lethal_trifecta/test_prompt_injection.py` — 8 attacks, now uses
  `bind_default_guardrails(profile="standard")` (test-local workaround removed).
- `tests/security/lethal_trifecta/test_args_injection.py` — attacks 01 and 02 now
  assert `BLOCK` (framework actively rejects extras and wrong-type args).
- `tests/test_phase5b_guardrails_exit.py` — 8 exit-gate invariants per spec §5.
- Total: 732 passing, 6 skipped (was 724 at the B7 baseline).

### Architectural-review findings closed

- F-7: `Decision` rename → `GuardrailVerdict`. (B1)
- F-13: `voussoir.guardrails` stub honesty — the stub is now a full implementation
  (real chain + 8 built-ins + LLMJudge + 3 profiles). (B1–B4)

### Reference SHAs

| Task | SHA | Description |
|------|-----|-------------|
| B1 | 123edf7 | Decision → GuardrailVerdict + Protocol sharpening |
| B2 | fb578d4 | DefaultGuardrailChain + bind_default_guardrails |
| B3 | 1272fe0 | 8 built-in deterministic guardrails |
| B4 | cc0551d | LLMGuardrailJudge AMBIGUOUS-fallback composer |
| B5 | a408219 | Wire chain into Agent.run 4 stages |
| B6 | cef9af5 | REWRITE semantics — lock tests |
| B7 | 4dffb06 | Lethal Trifecta corpus — 30 attacks |
| B8 | (current) | 3 framework gap closes + exit gate + CHANGELOG |

## v0.5.0a-phase5a — 2026-05-17

Phase 5 Tranche A — Type primitives + hard invariants. Locks the security model
foundation: `Capability` as `IntFlag`, `Trust` StrEnum, three new `PolicyViolation`
variants, `StandardExecutor` capability-mask + taint-gate enforcement,
`Agent.allowed_capabilities`, and sub-agent capability clamping.

### Breaking changes

1. **`Tool.capability: int` → `Capability` (now an `IntFlag`).** Numerically
   equivalent to the prior int constants; call sites using `Capability.X` keep
   working. Bare-int call sites become mypy errors under strict mode. (A1, bccd205)
2. **`Agent.__init__` accepts new `allowed_capabilities: Capability` kwarg** with
   default `READ_PUBLIC | READ_PRIVATE`. Tools declaring `EXFILTRATION` are blocked
   unless the agent's mask explicitly includes it. (A5, 5c56732)
3. **`Agent.stream` and `Agent.run` initialize `AgentContext.allowed_capabilities`**
   from `self.allowed_capabilities` at run-start. Direct `AgentContext()`
   constructions default to fail-closed `Capability.NONE`. (A5, 5c56732)
4. **`ToolContext` gains `allowed_capabilities: Capability` (default `NONE`) and
   `taint: set[Trust]` (default empty) fields.** Direct test constructors that
   bypass `Agent.run` must supply `allowed_capabilities` explicitly to dispatch
   any non-NONE-capability tool. (A4, 53cf23e)
5. **`@tool` decorator accepts `output_trust: Trust | None = None` kwarg.** (A4)
6. **Sub-agent dispatch is now capability-clamped.** When a parent dispatches a
   local `Agent` delegate, the child runs with
   `parent.allowed_capabilities & child.allowed_capabilities`. Tools whose
   capabilities don't fit the clamped mask are filtered out.
   `PolicyViolationError(CAPABILITY_CLAMPED_EMPTY)` raises fail-loud in the
   parent's run when clamping yields an empty registry. (A6, 881e895)
7. **`_dispatch_one` no longer swallows `PolicyViolationError` as `TOOL_ERROR`
   text**; hard security denials propagate to the run loop. (A5)

### Added

- `voussoir.guardrails.Trust` StrEnum: `SYSTEM`/`USER`/`INTERNAL`/`UNTRUSTED`. (A2, 2968f50)
- `AgentContext.taint: set[Trust]` — run-level taint accumulator. (A2)
- `AgentContext.allowed_capabilities: Capability` — per-run capability mask. (A5)
- `voussoir.PolicyViolation.{CAPABILITY_DENIED, TAINT_EXFILTRATION,
  CAPABILITY_CLAMPED_EMPTY}` enum variants. (A3, b36d381)
- `StandardExecutor.invoke` enforces capability mask + Trust-vs-EXFILTRATION gate
  inline; tool outputs tag the run's taint set via the Capability→Trust mapping
  (or per-tool `output_trust` override). (A4)
- `voussoir.AgentConfig.allowed_capabilities: list[str] | None` for yaml config. (A5)
- `voussoir.agent.agent_builder._parse_capability_list` helper. (A5)
- `Agent._tools_after_clamp` method + `Agent._with_container` overrides for clamping. (A6)
- `tests/security/` directory: hard-invariant tests + Lethal Trifecta corpus skeleton.

### Tests

- `tests/security/test_capability_invariants.py` — capability mask enforcement.
- `tests/security/test_taint_invariants.py` — taint accumulation + EXFIL gate.
- `tests/security/test_clamping.py` — sub-agent clamping, fail-loud on empty.
- `tests/security/lethal_trifecta/test_exfil_after_read.py` — 3 EXFIL-after-READ_PUBLIC
  attack tests (Tranche A skeleton; full 30-attack corpus lands in B7).
- `tests/test_phase5a_security_exit.py` — 7 exit gate invariants (one per spec §4 rule).
- Total: 627 passing, 6 skipped (was 617 at the A6 baseline).

### Closed (architectural review)

- F-4: `Capability` is now `IntFlag`.
- Tranche C #1: Sub-agent capability clamping.

## v0.4.0f-polish — 2026-05-16

Polish release (Tranche B of Phase 4.5). Closes 15 P2 polish findings from
the post-Phase-4d code review. **7 user-visible breaking changes**;
several internal moves. Pre-1.0 — no compat shims for renames.

### Breaking changes

1. **`voussoir.a2a.discover` → `voussoir.a2a.discover_card`.** Same
   signature. The old name suggested "discover everything" but the
   function returns a single AgentCard. `AgentRef.discover`
   classmethod keeps its name.
2. **`AgentRef` moved to `voussoir.a2a.agent_ref`.** Top-level
   `voussoir.AgentRef` and `voussoir.a2a.AgentRef` re-exports
   unchanged. Update direct submodule imports.
3. **`voussoir._logging` → `voussoir.observability.logging_setup`.**
   `get_logger` and `configure_logging` keep their signatures. The
   module used to be underscore-prefixed but was imported by 14
   call sites; moving it out of the private-by-name namespace.
4. **`AgentCard.inline_jwk` field removed.** The verifier stopped
   consulting it in v0.4.0e (Tranche A P0 #1). Construct cards with
   `jwks_uri=` only.
5. **`register_agent` parameter `agent` → `delegate`** and type
   widened from `Agent` to `IDelegate`. Positional callers
   unchanged; kwarg callers (`register_agent(c, agent=...)`) must
   update to `delegate=`. No kwarg callers existed in the repo.
6. **`Step` / `AgentEvent` / `CascadeOutcome` / `GuardrailDecision` /
   `AgentResult` now `extra="forbid"`.** Typos like
   `Step(kind=..., naem="x")` now raise `ValidationError` instead of
   silently constructing a malformed Step.
7. **`voussoir.middleware.chain.Middleware` → `voussoir.middleware.protocol.Middleware`.**
   Matches the naming convention used by `executors/protocol.py`,
   `tools/protocol.py`, etc. Top-level `voussoir.middleware.Middleware`
   re-export unchanged.

Type narrowing (mostly invisible at runtime):

- **`Agent(delegates=...)` input narrowed** from
  `list[Agent | str | IDelegate]` to `list[IDelegate | str]`. Runtime
  unchanged (`Agent` IS `IDelegate`). Type-checked callers see the
  narrowing.

### Internal moves (no public-API impact for typical consumers)

- **`agent.py` shrinks from 795 → 706 LOC.** The shared per-turn body
  of `_run_normal` and `stream` moves to `voussoir.agent.turn` as a
  two-phase API: `tool_turn_prepare` (llm.chat + parse response) and
  `tool_turn_dispatch` (dispatch tool calls + accumulate outcomes).
  The two-phase split keeps `Agent.stream`'s real-time semantics
  intact: `tool_started` / `delegation_started` events fire BETWEEN
  the two phases, so consumers observe tool starts before the tool
  body executes.
- **Stream event helpers extracted** to `voussoir.agent.stream_events`
  (`token_event`, `done_event`, `pre_dispatch_events`,
  `post_dispatch_events`). One-way arrow: `agent.py → stream_events.py`.
- **`AgentRef` class** lives in its own `voussoir.a2a.agent_ref`
  module; `discovery.py` keeps `discover_card` + card-verification
  helpers.
- **`Middleware` Protocol** lives in `voussoir.middleware.protocol`
  (renamed from `chain.py`).
- **`RequestCascade.model_rebuild()`** moved from
  `voussoir/agent/__init__.py` module-top to a lazy
  `_ensure_cascade_rebuilt()` helper in `cascade.py`, fired from
  `RequestCascade.__init__` (before Pydantic validation). The
  helper uses `RequestCascade.__pydantic_complete__` as the
  idempotency signal — no module-level mutable state.
- **`Agent.stream` `temperature` bug fix.** Pre-4.5b, `stream`
  passed `temperature=None` to `llm.chat`, ignoring
  `self.temperature`. Now both `.run()` and `.stream()` honour the
  configured temperature.

### Tests

- `tests/conftest.py` — autouse `reset_dispatch_contextvars` fixture
  + `make_mini_delegate` factory fixture.
- `tests/a2a/test_agent_ref_failure_modes.py` — consolidated
  `DelegationError` coverage (7 tests, replaces
  `test_delegation_errors.py` which is deleted).
- `tests/agent/test_run_stream_parity.py` — verifies `run()` and
  `stream()` produce equivalent final output for the same fixture.
- `tests/agent/test_result_strictness.py` — 5 typo-rejection tests
  for the strict result models.
- `tests/test_docstring_sweep.py` — enforces non-empty docstrings on
  every name in `__all__` going forward.
- `tests/agent/test_dispatch_contextvar_reset.py` — autouse fixture
  positive cases + registration check.
- 15 exit criteria in `tests/test_phase45b_polish_exit.py`.
- Total: ~583 passing tests (was 554 at the v0.4.0e baseline).

### Migration guide

| Old | New |
|---|---|
| `voussoir.a2a.discover(url)` | `voussoir.a2a.discover_card(url)` |
| `from voussoir.a2a.discovery import AgentRef` | `from voussoir.a2a.agent_ref import AgentRef` (or `from voussoir import AgentRef`) |
| `from voussoir._logging import get_logger` | `from voussoir.observability.logging_setup import get_logger` |
| `AgentCard(..., inline_jwk={...})` | `AgentCard(..., jwks_uri=...)` (drop the field) |
| `register_agent(c, agent=my_agent)` | `register_agent(c, delegate=my_agent)` (positional callers unaffected) |
| `Step(kind="llm_call", naem="x")` (typo silently constructed) | Fix the typo |
| `from voussoir.middleware.chain import Middleware` | `from voussoir.middleware.protocol import Middleware` (or `from voussoir.middleware import Middleware`) |

## v0.4.0e-hardening — 2026-05-13

Hardening release (Tranche A of Phase 4.5). Closes 25 P0+P1 findings from
the post-Phase-4d code review. **14 user-visible breaking changes**;
3 internal moves. The migration guide at the bottom of this entry summarizes
the upgrade path.

### Breaking changes

**Wire format (A2A peers):**

1. **JSON-RPC `result` shape now `WireAgentResult`** (`output`, `finish_reason`,
   `tokens_in`, `tokens_out`, `duration_ms`). Was: full `AgentResult.model_dump()`
   including `steps`, `delegation_chain`, `trace_id`. Peers parsing the old
   shape break; internal trace and tool args no longer leak to remote callers.
2. **AgentCards must publish `jwks_uri`.** `inline_jwk` is no longer honored
   — accepting attacker-supplied keys was a forgery vector.
3. **Card endpoints must be HTTPS.** Non-HTTPS endpoints rejected at issue
   (`card_from_agent`) and at consumption (`discover()`). Loopback
   (`127.0.0.1`, `localhost`) is allowed for local dev / smoke tests.

**A2A auth:**

4. **JWT `iss` claim is validated** against `KeyProvider.expected_issuers()`
   (env `VOUSSOIR_A2A_ALLOWED_ISSUERS`, comma-separated). Tokens without
   `iss` or with disallowed `iss` are rejected. Empty allow-list is
   default-deny — prod operators must set the env var.
5. **JWT algorithm allow-list hardcoded** to `{HS256, HS384, HS512, RS256}`.
   `VOUSSOIR_A2A_JWT_ALGORITHM` no longer accepts arbitrary algorithms.
6. **Required JWT claim set**: `iss`, `exp`, `aud`, `nbf`, `iat`, `sub`.
   Tokens missing any of these are rejected.

**Key handling:**

7. **`EnvKeyProvider(allow_ephemeral=False)` is the new default.** Calling
   `.jwt_secret()` without `VOUSSOIR_A2A_JWT_SECRET` raises with an
   actionable message. `allow_ephemeral=True` (dev mode) emits a
   structured warning via the `voussoir.a2a.keys` logger; the previous
   stderr print of the generated secret is removed (it leaked into Docker
   / k8s log aggregation).

**Agent surface:**

8. **`Agent.__init__` requires `container=` explicitly.** The lazy
   `default_container()` fallback in the `container` property is removed.
9. **`Agent.__init__` raises `ValueError` immediately** when a string
   delegate is declared without an `AgentRegistry` bound on the container.
   Was: failed at first `.run()` deep in a traceback.
10. **`bind_agent_registry(load_plugins=True)` is the new default.** Plugins
    are gated by `allowed_plugin_names` (set / None). Pass `load_plugins=False`
    to fully suppress.

**`AgentRef` lifecycle:**

11. **`AgentRef(card)` no longer owns an httpx client.** Requires either
    `async with AgentRef(card) as ref:` (lazy construct + close on exit) or
    `AgentRef(card, http_client=external)` (caller-owned). Bare construct
    then `.delegate()` raises with a clear hint at both patterns. Pre-4.5a
    behaviour leaked connection pools in long-lived agents.

**Error types** (mild — only matters if catching by exact type):

12. **`NamedDelegate.delegate()` raises `PolicyViolation.DELEGATE_NOT_FOUND`**
    instead of `MAX_STEPS`. Catch-by-enum code adjusts.
13. **`AgentRef.delegate()` raises `voussoir.a2a.errors.DelegationError`
    subclasses** (`RemoteUnreachable`, `RemoteAuthFailed`,
    `RemoteProtocolError`, `RemoteMalformed`) instead of
    `PolicyViolationError(MAX_STEPS)`. End-user `DELEGATION_REFUSED`
    wrapping is unchanged.

**Streaming:**

14. **`Agent.stream()` raises `PolicyViolationError(STREAMING_NOT_SUPPORTED)`**
    when `self.cascade is not None`. Pre-4.5a silently bypassed the cascade
    gate — a policy violation that looked like a working stream. Use
    `.run()` for cascade-gated execution.

### Internal moves (no public-API impact for typical consumers)

- `voussoir.middleware.builtin.BudgetMiddleware` →
  `voussoir.agent.middleware.BudgetMiddleware`. The whole `builtin.py`
  module is gone; `LoggingMiddleware` and `RetryMiddleware` also moved
  to `voussoir.agent.middleware`. `voussoir.agent` re-exports all three.
  `voussoir.middleware` keeps only the agent-agnostic `Middleware` Protocol.
- Dispatch helpers (`ToolCallOutcome`, `dispatch_tool_calls`,
  `accumulate_outcomes`, `make_delegate_invoker`, `parent_ctx_var`,
  `last_sub_result_var`) moved from `voussoir.agent.agent` to a new
  `voussoir.agent.dispatch` module. Cross-boundary names are public
  (no underscore) per the framework's module-extraction rule.
- `BufferedTelemetrySink.merge_into` removed. Use
  `voussoir.agent.telemetry.merge_buffered_telemetry_into_result(result, sink.records)`.
  `voussoir.observability` now has zero imports from `voussoir.agent.*`.

### New top-level re-exports

- `voussoir.AgentRef`
- `voussoir.make_a2a_router`
- `voussoir.serve_a2a`
- `voussoir.load_delegate_plugins`, `voussoir.ENTRY_POINT_GROUP` (carried
  over from v0.4.0d)

### New enum variants

- `PolicyViolation.DELEGATE_NOT_FOUND`
- `PolicyViolation.STREAMING_NOT_SUPPORTED`

### Concurrency / lifecycle hardening

- `dispatch_tool_calls` switched from `asyncio.as_completed` to
  `asyncio.gather` inside a `try/finally` that cancels unfinished tasks.
  Caller cancellation no longer leaks in-flight tool tasks.
- `_dispatch_one` narrowed `except BaseException` → `except Exception`,
  letting `KeyboardInterrupt` / `SystemExit` propagate while
  `asyncio.CancelledError` is still explicitly handled.

### Tests

- ≥40 new tests across `tests/a2a/test_wire_model.py`,
  `test_jwt_validation.py`, `test_card_signing.py`, `test_key_provider.py`,
  `test_agent_ref_lifecycle.py`, `test_delegation_errors.py`,
  `tests/agent/test_plugin_allow_list.py`, `test_dispatch_cancellation.py`,
  `test_dispatch_keyboardinterrupt.py`, `test_stream_cascade_raises.py`,
  `test_named_delegate_error_type.py`, `test_agent_init_validates.py`,
  `tests/imports/test_layering.py`.
- 14 exit criteria in `tests/test_phase45a_hardening_exit.py`.
- Phase 1 / 3 / 4a / 4b / 4c / 4d exit suites adapted to the new
  contracts; all prior assertions kept their behavioural intent.
- Total: ~554 passing tests (was 497 at the v0.4.0d-phase4d baseline).

### Migration guide

| Old | New |
|---|---|
| `Agent("x")` | `Agent("x", container=default_container())` |
| `AgentRef(card)` | `async with AgentRef(card) as ref:` _or_ `AgentRef(card, http_client=...)` |
| `bind_agent_registry(c)` | Same — plugins now load by default (allow-list via `allowed_plugin_names=`) |
| `bind_agent_registry(c, load_plugins=True)` | Same — but consider `allowed_plugin_names={...}` for strict filtering |
| Reading `body["result"]["steps"]` from A2A response | Steps no longer on the wire; track locally via `agent.run()` result |
| Mint JWT without `iss` / `nbf` / `iat` / `sub` | Always set all six required claims |
| `EnvKeyProvider()` in tests | `EnvKeyProvider(allow_ephemeral=True)` |
| `EnvKeyProvider()` in prod without `VOUSSOIR_A2A_JWT_SECRET` | Set the env var |
| `card_from_agent(agent, endpoint="http://...")` | Use HTTPS (or `127.0.0.1`/`localhost` for dev) |
| `buf.merge_into(result)` | `merge_buffered_telemetry_into_result(result, buf.records)` |
| `from voussoir.middleware.builtin import BudgetMiddleware` | `from voussoir.agent.middleware import BudgetMiddleware` (or `from voussoir.agent import BudgetMiddleware`) |
| `except PolicyViolationError as exc: if exc.violation is MAX_STEPS: ...` (for NamedDelegate misses) | `if exc.violation is PolicyViolation.DELEGATE_NOT_FOUND: ...` |
| `except PolicyViolationError` from `AgentRef.delegate` | `except DelegationError` (or one of the four subclasses) |
| `Agent("x", cascade=...).stream(...)` | Use `.run()` — streaming-cascade lands in Phase 5 |
