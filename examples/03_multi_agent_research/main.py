"""Multi-agent research demo: lead → researcher → writer.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/03_multi_agent_research/main.py
"""

from __future__ import annotations

import asyncio

from voussoir import Agent
from voussoir.container.defaults import default_container


async def main() -> None:
    container = default_container()

    researcher = Agent(
        name="researcher",
        description="Surveys recent reading on a topic and returns a brief summary.",
        instructions=(
            "You are a research assistant. Given a topic, produce 3-5 bullet "
            "points capturing the most important recent developments. Be "
            "concise; cite no sources (you do not have web access in this "
            "demo)."
        ),
        model="claude-haiku-4-5-20251001",
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
        model="claude-haiku-4-5-20251001",
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
        model="claude-haiku-4-5-20251001",
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
