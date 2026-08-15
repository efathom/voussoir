"""{{project_name}} -- minimal voussoir example.

Set ANTHROPIC_API_KEY then `python main.py`.
"""

import asyncio

from voussoir import Agent
from voussoir.container.defaults import default_container


async def main() -> None:
    container = default_container()
    agent = Agent(name="{{project_name}}", container=container)
    result = await agent.run("Say hi in five words or fewer.")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
