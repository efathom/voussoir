"""Run an agent with OTel spans printed to the console (Tier 0 default).

No env vars set → voussoir.observability.tracer.configure_otel() installs a
BatchSpanProcessor + ConsoleSpanExporter on first Agent.__init__. Spans are
printed to stdout at run-end.
"""

from __future__ import annotations

import asyncio

from voussoir import Agent, default_container


async def main() -> None:
    c = default_container()
    a = Agent(name="demo", container=c)
    result = await a.run("Say hello in three words.")
    print(f"\n=== Final output: {result.output} ===")


if __name__ == "__main__":
    asyncio.run(main())
