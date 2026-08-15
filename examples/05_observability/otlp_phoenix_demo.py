"""Run an agent with OTel spans exported via OTLP HTTP to a local Phoenix instance.

Tier 1: OTEL_EXPORTER_OTLP_ENDPOINT triggers OTLPSpanExporter inside
voussoir.observability.tracer.configure_otel(). Phoenix accepts OTLP HTTP at
http://localhost:6006/v1/traces by default.

Start Phoenix first:
    docker run -p 6006:6006 arizephoenix/phoenix:latest
"""

from __future__ import annotations

import asyncio
import os

from voussoir import Agent, Container


async def main() -> None:
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
    os.environ.setdefault("OTEL_SERVICE_NAME", "voussoir-demo")

    c = Container()
    a = Agent(name="demo", container=c)
    result = await a.run("What is 2+2? Reply with the number only.")
    print(f"\n=== Final output: {result.output} ===")
    print("Spans should appear in Phoenix at http://localhost:6006")


if __name__ == "__main__":
    asyncio.run(main())
