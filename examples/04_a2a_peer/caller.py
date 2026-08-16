"""Phase 4b A2A example — caller.

Discovers the publisher's AgentCard, wraps it in an AgentRef, and uses it
as a delegate from a local 'lead' agent.
"""

from __future__ import annotations

import asyncio
import os

from ctxforge.llm.openrouter_provider import (
    OPENROUTER_BASE_URL,
    OpenRouterConfig,
    OpenRouterLLMProvider,
)

from voussoir.a2a import AgentRef
from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
from voussoir.agent.agent import Agent
from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
from voussoir.auth.protocol import Authorizer
from voussoir.container import Container
from voussoir.executors import IToolExecutor, StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain, IGuardrailChain
from voussoir.llm.fake_embedder import FakeEmbeddingProvider
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
from voussoir.protocols import IEmbeddingProvider, ILLMProvider, IMemoryStore, ISessionStore


def build_unfrozen_container() -> Container:
    """Mirror default_container's wiring minus the security-critical freeze,
    so AllowAllAuthorizer can be bound for this demo."""
    c = Container()
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    c.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]
    c.bind(KeyProvider, EnvKeyProvider(allow_ephemeral=True))  # type: ignore[type-abstract]
    c.bind(Authorizer, AllowAllAuthorizer())  # type: ignore[type-abstract]
    c.bind(IToolExecutor, StandardExecutor())  # type: ignore[type-abstract]
    c.bind(IGuardrailChain, DefaultGuardrailChain([]))  # type: ignore[type-abstract]
    c.bind(IEmbeddingProvider, FakeEmbeddingProvider())
    c.bind(
        ILLMProvider,
        OpenRouterLLMProvider(
            OpenRouterConfig(
                api_key=os.environ["OPENROUTER_API_KEY"],
                model="deepseek/deepseek-v4-flash-0731",
                base_url=os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
            )
        ),
    )  # type: ignore[type-abstract]
    return c


async def main() -> None:
    if not any(
        os.environ.get(k)
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    ):
        print("Set an LLM API key (ANTHROPIC_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY) to run the caller.")
        return
    if not os.environ.get("VOUSSOIR_A2A_JWT_SECRET"):
        print(
            "Set VOUSSOIR_A2A_JWT_SECRET to the same value as the publisher "
            "(see examples/04_a2a_peer/README.md)."
        )
        return

    c = build_unfrozen_container()
    ref = await AgentRef.discover("http://127.0.0.1:8765")
    print(f"Discovered: {ref.name} — {ref.description}")

    # Phase 4.5a P0 #8: bare AgentRef(card) has _http=None. Wrap in
    # `async with ref:` so the httpx client is open for .delegate() calls.
    async with ref:
        lead = Agent(
            name="lead",
            instructions=(
                "You orchestrate research. Use delegate_to_researcher to gather "
                "notes on the user's topic, then return a single concise paragraph "
                "summarizing them."
            ),
            delegates=[ref],
            model="deepseek/deepseek-v4-flash-0731",
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


if __name__ == "__main__":
    asyncio.run(main())
