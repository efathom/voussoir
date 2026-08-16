"""Research agent demo — MCP tool + SQLite memory, no docker required.

Run:
    python examples/02_research_agent/main.py

Requires:
    ANTHROPIC_API_KEY set
    `pip install -e ".[mcp,mem-sqlite]"` already done

Note (v1.0.4 E3): ``default_container()`` freezes ``IMemoryStore`` to
prevent plugin-driven history poisoning. ``bind_sqlite_memory`` rebinds
``IMemoryStore`` and therefore must be called on a fresh ``Container()``
that owns the binding before any freeze is applied. This example wires
the bindings ``default_container`` would have set up, minus the freeze.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from ctxforge.llm.openrouter_provider import (
    OPENROUTER_BASE_URL,
    OpenRouterConfig,
    OpenRouterLLMProvider,
)

from voussoir import Agent, Container
from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
from voussoir.auth.protocol import Authorizer
from voussoir.container.defaults import bind_sqlite_memory
from voussoir.executors import IToolExecutor, StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain, IGuardrailChain
from voussoir.llm.fake_embedder import FakeEmbeddingProvider
from voussoir.mcp.client import MCPClient
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
from voussoir.protocols import IEmbeddingProvider, ILLMProvider, IMemoryStore, ISessionStore
from voussoir.tools.mcp import MCPTool

FAKE_SERVER = str(
    Path(__file__).resolve().parent.parent.parent / "tests" / "mcp" / "fake_mcp_server.py"
)


def build_unfrozen_container() -> Container:
    """Mirror default_container's wiring without the v1.0.4 E3 freeze."""
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
    bind_sqlite_memory(container, path=str(Path(__file__).parent / "memory.db"))

    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        tools = await MCPTool.from_server(session, server_name="fake")

        agent = Agent(
            name="researcher",
            instructions="You are a research assistant.",
            model="deepseek/deepseek-v4-flash-0731",
            tools=tools,
            container=container,
        )
        result = await agent.run("Use the echo tool to repeat: 'hello voussoir'.")
        print("output:        ", result.output)
        print("steps:         ", [(s.kind, s.name) for s in result.steps])
        print("tokens_in:     ", result.tokens_in)
        print("tokens_out:    ", result.tokens_out)
        print("finish_reason: ", result.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
