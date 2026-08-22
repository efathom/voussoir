"""Streaming agent + single-pass cascade gate (Phase 5 C7).

When a cascade is configured with max_cascade_depth=1, Agent.stream emits
`cascade_passed` or `cascade_failed` events AFTER the `done` event. This
example uses a trivial always-pass validator so the cascade_passed event
fires; switch to a failing validator to see cascade_failed instead.
"""

from __future__ import annotations

import asyncio
from typing import Any

from voussoir import Agent, default_container
from voussoir.agent.cascade import Decision, RequestCascade


class _AlwaysPassValidator:
    name = "demo-pass"

    async def validate(self, result: Any, *, task: str) -> Decision:
        del result, task
        return Decision.PASS


async def main() -> None:
    cascade = RequestCascade(
        verifier=_AlwaysPassValidator(),
        max_attempts=1,
        max_cascade_depth=1,
    )
    c = default_container()
    a = Agent(name="demo", container=c, cascade=cascade)

    async for ev in a.stream("Say hi briefly."):
        print(f"event: {ev.kind} | payload: {ev.payload}")


if __name__ == "__main__":
    asyncio.run(main())
