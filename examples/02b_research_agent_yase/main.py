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

from voussoir import Agent
from voussoir.container.defaults import default_container
from voussoir.memory.backends.yase.client import YaseClient
from voussoir.memory.backends.yase.retriever import YaseRetriever
from voussoir.memory.backends.yase.tool import make_yase_search_tool


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

    container = default_container()

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
        model="claude-haiku-4-5-20251001",
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
