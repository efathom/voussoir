# Getting Started

This guide walks through five progressively-deeper examples — from a single agent
chatting back at you to a multi-agent peer-protocol setup. Each example is a
complete, runnable script; the `examples/` directory in the repo has each one
ready to `python main.py`.

Before you start, set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) in your
environment. That is the only required setup — voussoir does not need config
files or service registration.

```bash
pip install voussoir
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 1. Single agent (no tools)

The minimum viable voussoir program: one agent, one prompt, one response.

```python
# examples/01_hello_agent/main.py
import asyncio
from voussoir.container.defaults import default_container
from voussoir import Agent

async def main() -> None:
    agent = Agent(
        name="greeter",
        instructions="You are a brief, friendly assistant.",
        container=default_container(),
    )
    result = await agent.run("Say hi in three words.")
    print(f"output:        {result.output}")
    print(f"tokens_in:     {result.tokens_in}")
    print(f"tokens_out:    {result.tokens_out}")
    print(f"cost_usd:      ${result.cost_usd:.6f}")
    print(f"duration_ms:   {result.duration_ms:.1f}")
    print(f"finish_reason: {result.finish_reason}")

asyncio.run(main())
```

`default_container()` wires all framework defaults in one call: it detects your
API key, binds the matching LLM provider (Anthropic when `ANTHROPIC_API_KEY` is
set, OpenAI otherwise), and registers an in-memory session and memory store.
Non-security-critical bindings can be overridden after the call; six
security-critical bindings (`Authorizer`, `KeyProvider`, `ITelemetrySink`,
`ILLMProvider`, `IMemoryStore`, `ISessionStore`) are **frozen** to prevent
plugin-driven swaps. To upgrade the memory tier, build a fresh `Container()`
and call `bind_sqlite_memory(container, path="memory.db")` on it before the
default freezes apply — see `voussoir.container.defaults` for the canonical
composition pattern. Every `Agent` requires an explicit `container=` argument;
the framework refuses to construct one silently so configuration errors surface
at startup, not buried inside `agent.run()`.

`agent.run()` returns an `AgentResult[str]`. The `output` field holds the final
text the model produced. `tokens_in` and `tokens_out` are the raw token counts
from the LLM response. `cost_usd` is a coarse estimate derived from token counts.
`finish_reason` is `"completed"` on a clean run; other values — `"max_steps"`,
`"blocked"`, `"error"` — indicate the agent stopped early. Every run also emits
OpenTelemetry spans automatically; connect an OTLP-compatible collector and you
get distributed traces with no extra code.

**Where to go next:** Add a Python function the agent can call as a tool.

---

## 2. Agent with one tool

Tools are async Python functions decorated with `@tool`. The `capability` tag
declares what the function is allowed to do; the framework enforces it inline
before the tool body runs.

```python
# examples/02_research_agent/main.py
import asyncio
from voussoir import Agent
from voussoir.container.defaults import default_container
from voussoir.tools import Capability, tool

@tool(capability=Capability.READ_PUBLIC, name="get_weather")
async def get_weather(city: str) -> str:
    """Return a weather report for the given city."""
    # In production replace this with a real HTTP call.
    return f"It is 72°F and sunny in {city}."

async def main() -> None:
    agent = Agent(
        name="weather-assistant",
        instructions="You are a helpful weather assistant. Use get_weather to answer questions.",
        tools=[get_weather],
        container=default_container(),
    )
    result = await agent.run("What is the weather in Tokyo?")
    print(result.output)
    print("steps:", [(s.kind, s.name) for s in result.steps])

asyncio.run(main())
```

`@tool` generates a Pydantic input schema from the function's type annotations, so
`city: str` becomes a required string field — the LLM cannot pass an integer or
omit the argument without triggering a validation error. The `capability` tag maps
to voussoir's `Capability` enum (`READ_PUBLIC`, `READ_PRIVATE`, `WRITE`, `EXECUTE`,
etc.). Before each tool invocation the `StandardExecutor` checks the tag against
the agent's `allowed_capabilities` mask; if the capability is not in the mask, the
tool call is rejected without reaching the function body. This makes capability
enforcement deterministic and auditable — no prompt engineering required.

`result.steps` is a list of `Step` records, one per LLM call and tool invocation.
After the agent calls `get_weather`, the steps list typically contains three
entries: `("llm_call", "chat")` for the initial reasoning step, `("tool_call",
"get_weather")` for the actual invocation, and `("llm_call", "chat")` for the
final response turn that incorporates the tool's output. This trace is available
synchronously on the result — no sidecar required.

**Where to go next:** Compose multiple agents so a lead delegates to specialists.

---

## 3. Hierarchical delegation

A lead agent can delegate subtasks to sub-agents. Declare each sub-agent as a
`delegate`; voussoir synthesizes a `delegate_to_<name>` tool automatically. Sub-agent
capabilities are clamped to the lead's grant — a sub-agent cannot exceed the
permissions its parent holds.

```python
# examples/03_multi_agent_research/main.py
import asyncio
from voussoir import Agent
from voussoir.container.defaults import default_container

async def main() -> None:
    container = default_container()

    researcher = Agent(
        name="researcher",
        description="Surveys a topic and returns 3-5 bullet points.",
        instructions=(
            "You are a research assistant. Given a topic, produce 3-5 bullet "
            "points capturing the most important recent developments. Be concise."
        ),
        model="claude-haiku-4-5-20251001",
        container=container,
    )
    writer = Agent(
        name="writer",
        description="Turns research notes into a polished short paragraph.",
        instructions=(
            "You are a writer. Given research notes, produce a single concise "
            "paragraph (120 words or fewer) suitable for an executive summary."
        ),
        model="claude-haiku-4-5-20251001",
        container=container,
    )
    lead = Agent(
        name="lead",
        description="Coordinates research then writing.",
        instructions=(
            "You are a lead agent. Use delegate_to_researcher to gather research "
            "notes, then use delegate_to_writer to produce a final paragraph. "
            "Return only the writer's output."
        ),
        delegates=[researcher, writer],
        model="claude-haiku-4-5-20251001",
        container=container,
    )

    result = await lead.run(
        "Topic: 2026 trends in agentic LLM frameworks. "
        "Coordinate a research-then-writing pipeline."
    )
    print("output:           ", result.output)
    print("delegation_chain: ", result.delegation_chain)
    print("cost_usd:         ", f"${result.cost_usd:.6f}")
    print("finish_reason:    ", result.finish_reason)

asyncio.run(main())
```

Each sub-agent is a full `Agent` instance with its own `instructions`, `model`, and
`container`. Sharing a single `default_container()` across agents is the normal
pattern — they each open a child scope during a run so their session state stays
isolated. The `delegates` list on the lead causes voussoir to synthesize one
`delegate_to_researcher` and one `delegate_to_writer` tool; the lead's LLM
discovers these tools the same way it discovers ordinary `@tool` functions.
Delegation depth is capped at `max_delegation_depth=3` by default, preventing
runaway recursive delegation chains.

`result.delegation_chain` records every agent that participated in the run, in
order. `cost_usd` is the sum of tokens consumed by the lead and all sub-agents —
you get one number for the whole pipeline. The individual sub-agent runs are also
reflected in `result.steps` as `("delegation", "researcher")` and
`("delegation", "writer")` entries alongside the lead's own LLM calls.

**Where to go next:** Gate tool outputs through a cascade of validators.

---

## 4. Cascade gates

A `RequestCascade` wraps a `verifier` — a chain of validators that inspect each
run's output and return `PASS`, `FAIL`, or `AMBIGUOUS`. `ToolUseFaithfulness`
is a deterministic validator that checks whether the agent claimed tool invocations
it did not actually make. `LLMJudge` is called only when the primary validator
returns `AMBIGUOUS`, so the LLM judge runs only when needed.

```python
# examples/04_validator_judge/main.py
import asyncio
from voussoir import Agent
from voussoir.agent.cascade import RequestCascade
from voussoir.agent.validators import AmbiguousFallback, LLMJudge, ToolUseFaithfulness
from voussoir.container.defaults import default_container

async def main() -> None:
    c = default_container()

    cascade = RequestCascade(
        verifier=AmbiguousFallback(
            primary=ToolUseFaithfulness(),
            judge=LLMJudge(
                "the output answers the user's question accurately and concisely",
                container=c,
            ),
        )
    )

    lead = Agent(
        name="lead",
        description="Answers factual questions with optional tool-use claims.",
        instructions=(
            "You answer factual questions. Be concise. Important style note: "
            "always preface your final answer with a hedge phrase like "
            "'I think I used' or 'I might have invoked' an internal lookup, "
            "even if you did not actually call a tool. This is part of the "
            "demo's narration style."
        ),
        cascade=cascade,
        model="claude-haiku-4-5-20251001",
        container=c,
    )

    result = await lead.run("What is the boiling point of water in Celsius?")
    print("output:           ", result.output)
    print("cost_usd:         ", f"${result.cost_usd:.6f}")
    print(
        "validator_steps:  ",
        [s.name for s in result.steps if s.kind == "validator_call"],
    )
    print("cascade_history:  ", result.cascade_history)
    print("finish_reason:    ", result.finish_reason)

asyncio.run(main())
```

`AmbiguousFallback` implements a two-tier gate: `ToolUseFaithfulness` runs first
because it is fast and deterministic — it compares the tool calls the LLM
mentioned in its output against the tool calls recorded in `result.steps`. When
the agent hedges ("I might have invoked…") about a tool call that never happened,
the validator returns `AMBIGUOUS` rather than an outright `FAIL`. `AmbiguousFallback`
then delegates to `LLMJudge`, which performs a second LLM call to evaluate whether
the output actually satisfies the stated criterion. The judge's token cost is
merged into the run's `cost_usd` automatically via a scoped telemetry buffer.

`result.cascade_history` is a list of `CascadeOutcome` records — one per
validation attempt — capturing which agent ran, whether it escalated, and the
validator's reason. The `RequestCascade` accepts an optional `escalation` agent
that is invoked on `FAIL`; without one, a failed gate marks the result
`finish_reason="error"` and returns the last output. `result.steps` includes a
`("validator_call", ...)` entry for each validator that ran so you can audit the
gate logic in post.

**Where to go next:** Expose an agent as a peer-protocol HTTP service.

---

## 5. A2A peer protocol

voussoir implements the Agent-to-Agent (A2A) peer protocol over JSON-RPC.
`serve_a2a` wraps your agent in a FastAPI app and starts a uvicorn server.
`AgentRef.discover` fetches the remote agent's card and wraps it as a local
delegate — the lead agent treats a remote peer identically to a local sub-agent.

Run the server in one terminal and the client in a second terminal.

**Terminal 1 — publisher:**

```python
# examples/04_a2a_peer/publisher.py
import os
from voussoir import Agent, serve_a2a
from voussoir.container.defaults import default_container

def main() -> None:
    c = default_container()
    agent = Agent(
        name="researcher",
        description="Surveys a topic and produces a brief summary.",
        instructions=(
            "You are a research assistant. Given a topic, produce 3-5 bullet "
            "points capturing key recent developments. Be concise."
        ),
        model="claude-haiku-4-5-20251001",
        container=c,
    )
    print("Publishing 'researcher' on http://127.0.0.1:8765")
    print("Card: http://127.0.0.1:8765/.well-known/agent-card.json")
    serve_a2a(agent, host="127.0.0.1", port=8765)

if __name__ == "__main__":
    main()
```

**Terminal 2 — caller:**

```python
# examples/04_a2a_peer/caller.py
import asyncio
from voussoir import Agent, AgentRef
from voussoir.container.defaults import default_container

async def main() -> None:
    c = default_container()

    # Fetch the remote agent's card and wrap it as a delegate.
    ref = await AgentRef.discover("http://127.0.0.1:8765")
    print(f"Discovered: {ref.name} — {ref.description}")

    lead = Agent(
        name="lead",
        instructions=(
            "You orchestrate research. Use delegate_to_researcher to gather "
            "notes on the user's topic, then return a single concise paragraph "
            "summarizing them."
        ),
        delegates=[ref],
        model="claude-haiku-4-5-20251001",
        container=c,
    )
    result = await lead.run(
        "Topic: 2026 advances in agent-to-agent protocols. "
        "Get research notes and produce a final summary."
    )
    print()
    print("Final output:")
    print(result.output)
    print()
    print(f"delegation_chain: {result.delegation_chain}")
    print(f"cost_usd: ${result.cost_usd:.6f}")
    print(f"finish_reason: {result.finish_reason}")

    await ref.aclose()

asyncio.run(main())
```

!!! note "JWT shared secret"
    Before running either script, export a shared JWT secret so the publisher and
    caller can authenticate each other:

    ```bash
    export VOUSSOIR_A2A_JWT_SECRET=$(python -c \
      "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())")
    ```

    Set the same value in both terminals. The publisher's `serve_a2a` reads it from
    the environment; the caller's `AgentRef` signs outbound requests with it.

`serve_a2a` builds a FastAPI app with three routes: `GET
/.well-known/agent-card.json` (the agent's machine-readable identity card), `GET
/.well-known/jwks.json` (the publisher's signing keys), and `POST /a2a` (the
JSON-RPC invocation endpoint). All three are wired by `make_a2a_router`, which
`serve_a2a` calls internally. If you already have a FastAPI app, call
`make_a2a_router` directly and mount the returned `APIRouter` yourself.

`AgentRef.discover` performs an HTTP `GET` to `/.well-known/agent-card.json`,
parses the `AgentCard`, and wraps it in a thin delegate shim. From that point on
the lead agent treats `ref` exactly like a local `Agent` in its `delegates` list —
the same `delegate_to_researcher` synthetic tool is synthesized, the same
depth-cap enforcement applies, and the remote agent's cost rolls into
`result.cost_usd`. The A2A transport uses Bearer JWT authentication; the shared
secret approach shown here is appropriate for local development. Production
deployments should use `RS256` with a keypair stored outside the environment.

**Where to go next:**

- **[Extending](extending.md)** — Write custom Authorizers, CredentialBrokers, and Guardrails.
- **[Architecture](architecture.md)** — Why voussoir is shaped the way it is.
- **[API Reference](api/voussoir.md)** — Every public Protocol and class.
