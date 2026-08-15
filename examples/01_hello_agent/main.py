"""Hello-world agent demo.

Run: python examples/01_hello_agent/main.py

Requires ANTHROPIC_API_KEY in the environment.
"""

import asyncio

from voussoir import Agent, default_container


async def main() -> None:
    agent = Agent(
        name="greeter",
        container=default_container(),
        instructions="You are a brief, friendly assistant.",
    )
    result = await agent.run("Say hi in three words.")
    print(f"output:        {result.output}")
    print(f"tokens_in:     {result.tokens_in}")
    print(f"tokens_out:    {result.tokens_out}")
    print(f"cost_usd:      ${result.cost_usd:.6f}")
    print(f"duration_ms:   {result.duration_ms:.1f}")
    print(f"finish_reason: {result.finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
