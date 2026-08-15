"""Phase 4b A2A example — caller.

Discovers the publisher's AgentCard, wraps it in an AgentRef, and uses it
as a delegate from a local 'lead' agent.
"""

from __future__ import annotations

import asyncio
import os

from voussoir.a2a import AgentRef
from voussoir.agent.agent import Agent
from voussoir.container.defaults import default_container


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the caller.")
        return
    if not os.environ.get("VOUSSOIR_A2A_JWT_SECRET"):
        print(
            "Set VOUSSOIR_A2A_JWT_SECRET to the same value as the publisher "
            "(see examples/04_a2a_peer/README.md)."
        )
        return

    c = default_container()
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


if __name__ == "__main__":
    asyncio.run(main())
