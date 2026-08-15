"""Phase 4d canonical example: a delegate exposed via the voussoir.delegates
entry-point group.

A real plugin would parametrize its instructions / tools / model; this
example keeps the factory minimal so authors can compare diffs.
"""

from __future__ import annotations

from voussoir import Agent
from voussoir.container import Container


def make_plugin_agent(c: Container) -> Agent:
    """voussoir.delegates entry point — return an Agent (any IDelegate works).

    The container `c` is the host application's; resolve any dependencies
    (ILLMProvider, KeyProvider, your own bindings) through it rather than
    constructing them inline.
    """
    return Agent(
        name="example_plugin_agent",
        description="Canonical Phase 4d example delegate (returns a greeting).",
        instructions="You are a friendly greeter. Reply in one short sentence.",
        container=c,
    )
