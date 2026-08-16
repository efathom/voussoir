"""Research agent with yase-backed RAG tool.

Demonstrates the canonical voussoir-yase wiring:

  YaseClient (HTTP) → YaseRetriever (RAG) → make_yase_search_tool (Tool)
                                            → Agent(tools=[...])

The agent calls yase_search() like any other tool when it needs to ground
its answer in the yase corpus.

Run:
    docker compose -f examples/02b_research_agent_yase/docker-compose.yml up -d
    export ANTHROPIC_API_KEY=sk-ant-...
    export YASE_URL=http://localhost:8000
    python examples/02b_research_agent_yase/main.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
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
from voussoir.memory.backends.yase.client import YaseClient
from voussoir.memory.backends.yase.retriever import YaseRetriever
from voussoir.memory.backends.yase.tool import make_yase_search_tool
from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
from voussoir.protocols import IEmbeddingProvider, ILLMProvider, IMemoryStore, ISessionStore


def build_unfrozen_container() -> Container:
    """Mirror default_container's wiring without the fail-closed authorizer."""
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
    yase_url = os.environ.get("YASE_URL")
    if not yase_url:
        print("YASE_URL not set — start yase first (see README) and rerun.", file=sys.stderr)
        sys.exit(1)
    try:
        async with httpx.AsyncClient(timeout=2.0) as h:
            await h.get(f"{yase_url}/health")
    except httpx.ConnectError:
        print(
            f"yase not reachable at {yase_url}; is `docker compose up` running?",
            file=sys.stderr,
        )
        sys.exit(1)

    container = build_unfrozen_container()

    yase_client = YaseClient(
        base_url=yase_url,
        api_key=os.environ.get("YASE_API_KEY"),
    )
    yase_retriever = YaseRetriever(client=yase_client)
    yase_tool = make_yase_search_tool(yase_retriever)

    agent = Agent(
        name="researcher",
        instructions=(
            "You are a research assistant. When asked a factual question, "
            "use the yase_search tool to retrieve relevant passages from "
            "the corpus, then answer using only those passages and cite "
            "them by their [N] markers."
        ),
        model="deepseek/deepseek-v4-flash-0731",
        tools=[yase_tool],
        container=container,
    )
    result = await agent.run("What is voussoir?")
    print("output:        ", result.output)
    print("steps:         ", [(s.kind, s.name) for s in result.steps])
    print("tokens_in:     ", result.tokens_in)
    print("tokens_out:    ", result.tokens_out)
    print("finish_reason: ", result.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
