"""Multi-agent research demo: lead → researcher → writer.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/03_multi_agent_research/main.py
"""

from __future__ import annotations

import asyncio
import os

from ctxforge.llm.openrouter_provider import (
    OPENROUTER_BASE_URL,
    OpenRouterConfig,
    OpenRouterLLMProvider,
)

from voussoir import Agent, Container
from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
from voussoir.auth.protocol import Authorizer
from voussoir.executors import IToolExecutor, StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain, IGuardrailChain
from voussoir.llm.fake_embedder import FakeEmbeddingProvider
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
from voussoir.protocols import IEmbeddingProvider, ILLMProvider, IMemoryStore, ISessionStore


def build_unfrozen_container() -> Container:
    """Mirror default_container's wiring without the security-critical freeze,
    so AllowAllAuthorizer can be bound for this demo."""
    container = Container()
    container.bind(IMemoryStore, InMemoryStore())
    container.bind(ISessionStore, InMemorySessionStore())
    container.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]
    container.bind(KeyProvider, EnvKeyProvider(allow_ephemeral=True))  # type: ignore[type-abstract]
    container.bind(Authorizer, AllowAllAuthorizer())  # type: ignore[type-abstract]
    container.bind(IToolExecutor, StandardExecutor())  # type: ignore[type-abstract]
    container.bind(IGuardrailChain, DefaultGuardrailChain([]))  # type: ignore[type-abstract]
    container.bind(IEmbeddingProvider, FakeEmbeddingProvider())
    container.bind(
        ILLMProvider,
        OpenRouterLLMProvider(
            OpenRouterConfig(
                api_key=os.environ["OPENROUTER_API_KEY"],
                model="deepseek/deepseek-v4-flash-0731",
                base_url=os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
            )
        ),
    )  # type: ignore[type-abstract]
    return container


async def main() -> None:
    container = build_unfrozen_container()

    researcher = Agent(
        name="researcher",
        description="Surveys recent reading on a topic and returns a brief summary.",
        instructions=(
            "You are a research assistant. Given a topic, produce 3-5 bullet "
            "points capturing the most important recent developments. Be "
            "concise; cite no sources (you do not have web access in this "
            "demo)."
        ),
        model="deepseek/deepseek-v4-flash-0731",
        container=container,
    )
    writer = Agent(
        name="writer",
        description="Turns research notes into a polished short paragraph.",
        instructions=(
            "You are a writer. Given research notes (passed as your task), "
            "produce a single concise paragraph (≤120 words) suitable for an "
            "executive summary."
        ),
        model="deepseek/deepseek-v4-flash-0731",
        container=container,
    )
    lead = Agent(
        name="lead",
        description="Coordinates research → writing.",
        instructions=(
            "You are a lead agent. Use delegate_to_researcher to gather "
            "research notes, then use delegate_to_writer to produce a final "
            "paragraph. Return only the writer's output."
        ),
        delegates=[researcher, writer],
        model="deepseek/deepseek-v4-flash-0731",
        container=container,
    )

    result = await lead.run(
        "Topic: 2026 trends in agentic LLM frameworks. " "Coordinate a research → writing pipeline."
    )
    print("output:           ", result.output)
    print("delegation_chain: ", result.delegation_chain)
    print("steps:            ", [(s.kind, s.name) for s in result.steps])
    print("tokens_in:        ", result.tokens_in)
    print("tokens_out:       ", result.tokens_out)
    print("cost_usd:         ", f"${result.cost_usd:.6f}")
    print("duration_ms:      ", f"{result.duration_ms:.1f}")
    print("finish_reason:    ", result.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
