"""Research agent demo — MCP tool + SQLite memory, no docker required.

Run:
    python examples/02_research_agent/main.py

Requires:
    OPENROUTER_API_KEY set
    `pip install -e ".[mcp,mem-sqlite]"` already done

Everything this example needs to override — the LLM, the memory store, the
embedder, the authorizer — is passed straight to `default_container()`. Those
keys are frozen once it returns, so passing them at construction is the
supported way to change them; there is no post-hoc `bind()` to reach for.

This used to be ~25 lines of hand-wiring that re-created default_container's
bindings "minus the freeze", because there was no other way to get a SQLite
memory store or a permissive authorizer into an otherwise-default container.
A side effect was that the hand-built container bound an EMPTY guardrail chain,
so the demo silently ran with no guardrails at all.
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

from voussoir import Agent
from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
from voussoir.container.defaults import default_container
from voussoir.llm.fake_embedder import FakeEmbeddingProvider
from voussoir.mcp.client import MCPClient
from voussoir.memory.backends.sqlite import SQLiteMemoryStore
from voussoir.tools.mcp import MCPTool

FAKE_SERVER = str(
    Path(__file__).resolve().parent.parent.parent / "tests" / "mcp" / "fake_mcp_server.py"
)
MODEL = "deepseek/deepseek-v4-flash-0731"


async def main() -> None:
    embedder = FakeEmbeddingProvider()
    container = default_container(
        llm=OpenRouterLLMProvider(
            OpenRouterConfig(
                api_key=os.environ["OPENROUTER_API_KEY"],
                model=MODEL,
                base_url=os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
            )
        ),
        embedder=embedder,
        memory_store=SQLiteMemoryStore(
            path=str(Path(__file__).parent / "memory.db"),
            embedder=embedder,
        ),
        # Demo only — grants every tool call. Production wants a RoleAuthorizer,
        # DomainAuthorizer or ChainedAuthorizer here.
        authorizer=AllowAllAuthorizer(),
    )

    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        tools = await MCPTool.from_server(session, server_name="fake")

        agent = Agent(
            name="researcher",
            instructions="You are a research assistant.",
            model=MODEL,
            tools=tools,
            container=container,
        )
        result = await agent.run("Use the echo tool to repeat: 'hello voussoir'.")
        print("output:        ", result.output)
        print("steps:         ", [(s.kind, s.name) for s in result.steps])
        print("tokens_in:     ", result.tokens_in)
        print("tokens_out:    ", result.tokens_out)
        print("cost_usd:      ", f"${result.cost_usd:.6f}")
        print("finish_reason: ", result.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
