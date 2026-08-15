"""{{project_name}} -- research multi-agent example.

Builds the three-agent topology directly in Python so that tools imported
from tools.py are wired to the researcher at construction time.  voussoir.yaml
ships as documentation of the intended topology; it is not loaded at runtime
because the yaml path does not support tool registration.
"""

import asyncio

from tools import web_search

from voussoir import Agent
from voussoir.container.defaults import default_container


async def main() -> None:
    container = default_container()

    researcher = Agent(
        name="researcher",
        container=container,
        instructions="You research topics. Use web_search.",
        tools=[web_search],
    )
    writer = Agent(
        name="writer",
        container=container,
        instructions="You write polished, concise prose from a research brief.",
    )
    lead = Agent(
        name="{{project_name}}",
        container=container,
        instructions=(
            "You are the lead. Delegate research tasks to `researcher` "
            "and writing tasks to `writer`."
        ),
        delegates=[researcher, writer],
    )

    result = await lead.run(
        "Research the history of arch bridges, then write a 5-sentence summary."
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
