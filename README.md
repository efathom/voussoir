# voussoir

voussoir is a Python framework for building LLM agents that work securely out of the
box — no configuration ceremony required. You get a `Container` that holds your tools,
credentials, and guardrails, and an `Agent` that runs against it. Security is structural:
every tool carries a `Capability` enum and a `Trust` taint that the executor enforces
inline — there is no opt-in flag to forget, no skip path to exploit. The default guardrail
chain blocks the 30-attack Lethal Trifecta corpus (indirect prompt injection, reflection
exfiltration, system-prompt extraction) at 100%. OpenTelemetry traces and metrics emit
on every run without any setup. Agents compose into hierarchies via local delegation or
the A2A peer protocol; capabilities clamp on delegation so sub-agents can never exceed
their grant. The result is a framework where the safe path is the default path, and the
extension points (custom `Authorizer`, `CredentialBroker`, `Guardrail`, `Cascade`) are
clean Protocol interfaces you drop in without touching framework internals.

## Quick start

```python
import asyncio
from voussoir import Agent, default_container

async def main():
    agent = Agent("hello", container=default_container())
    result = await agent.run("Say hi.")
    print(result.output)

asyncio.run(main())
```

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) and run. That's it.

> `default_container()` wires the LLM provider (from your `ANTHROPIC_API_KEY` /
> `OPENAI_API_KEY` env), default memory + session stores, the fail-closed
> `DenyByDefaultAuthorizer` (bind a concrete authorizer to grant tool access),
> the `standard` guardrail chain, and a no-op telemetry sink. For explicit
> control over bindings, construct a bare `Container()` and
> `container.bind(Protocol, impl)` yourself.

## Install

Install from git:

```bash
pip install git+https://github.com/efathom/voussoir@v1.3.0
```

Or with `uv`:

```bash
uv pip install git+https://github.com/efathom/voussoir@v1.3.0
```

Optional extras:

```bash
pip install 'voussoir[a2a]@git+https://github.com/efathom/voussoir@v1.3.0'   # A2A peer protocol
pip install 'voussoir[mcp]@git+https://github.com/efathom/voussoir@v1.3.0'   # MCP tool adapter
pip install 'voussoir[all]@git+https://github.com/efathom/voussoir@v1.3.0'   # everything
```

## Highlights

- **Two-axis security model.** `Capability` enum + `Trust` taint mark every tool; the
  executor enforces them inline — no opt-in, no skip flag. Soft policies layer on top
  via a composable `Guardrail` chain (the `standard` profile is bound by default:
  length caps, injection heuristic, exfil scan).
- **30-attack Lethal Trifecta corpus.** Indirect prompt injection, exfiltration via
  reflection, system-prompt extraction — all blocked by the default guardrails at 100%.
  Run the corpus yourself: `pytest tests/security/lethal_trifecta/`.
- **A2A peer protocol.** `make_a2a_router(agent)` exposes any voussoir agent as a
  JSON-RPC peer; `AgentRef(url)` delegates to a remote peer. Bearer-JWT verified;
  Principal forwarded downstream.
- **OpenTelemetry by default.** Every run emits `gen_ai.*`-conformant spans for
  `agent.run`, `tool.call`, `authz.check`, `capability.check`, and `taint.check`, plus
  9 metric handles (token counts, costs, denials, escalations). No opt-in needed.
- **Soft-policy guardrail chain.** Composable `Guardrail` Protocol with input and output
  stages and an LLM judge for AMBIGUOUS cases. 8 built-in screens (input length, PII,
  URL allowlist, exfil pattern, and more).
- **Hierarchical delegation.** Lead agents delegate to sub-agents (local or A2A);
  capabilities clamp on delegation; Principal forwards through the chain.
- **Cascade gates.** Validate-then-judge cascade for tool-use faithfulness, output
  schema conformance, or any custom verifier you supply.
- **CredentialBroker + Authorizer Protocols.** 6 built-in brokers (env, file, mTLS,
  keychain, OAuth2-with-refresh, chained) + 5 authorizers (deny-by-default, allow-all,
  role, domain, chained). Implement custom impls without touching framework internals.

## Documentation

- **[Getting Started](docs/getting-started.md)** — 5 progressively-deeper examples
- **[Extending](docs/extending.md)** — 7 how-to recipes for the public Protocols
- **[Architecture](docs/architecture.md)** — framework design for v1.0
- **[API Reference](docs/api/voussoir.md)** — auto-generated from source

Build the docs locally:

```bash
make docs-build && make docs-serve
```

## CLI

```bash
voussoir new my-agent              # scaffold a starter project
voussoir run my-agent --input "ok" # run an agent from voussoir.yaml
voussoir validate                  # lint voussoir.yaml
voussoir doctor                    # environment health check
```

## Repository layout

```
src/voussoir/        — library code
docs/                — public docs (mkdocs-material)
docs/superpowers/    — internal specs and plans (not shipped)
examples/            — runnable demos
tests/               — pytest suite (~950 tests as of v1.0.1)
```

## Status

- **v1.3.0** — current stable; production-ready; semver from v1.0.0 onwards.
- Python ≥ 3.12 required.
- License: [Apache-2.0](LICENSE).
