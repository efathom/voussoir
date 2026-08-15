"""AgentRegistry: container-singleton mapping name → Agent.

Lookup is at delegate-tool-call time (lazy resolution); not at
Agent.__init__. Lives in its own module so it can be imported by
agent.py without forming a cycle — Agent is referenced here only
via TYPE_CHECKING.

bind_agent_registry (the yaml-driven helper) lives in bootstrap.py,
not here, so registry.py stays free of runtime deps on agent_builder.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from voussoir.container import Container

if TYPE_CHECKING:
    from voussoir.agent.delegate import IDelegate


class AgentRegistry:
    """Container-singleton mapping `name -> IDelegate` for delegation lookup.

    Use this when you want a parent Agent to dispatch sub-tasks by name
    (e.g. delegate_to(\"writer\", task=...)). Registration happens up-front
    (via `register_agent`, `bind_agent_registry`, or plugin entry-points);
    lookup is lazy at the delegate-tool-call site so child agents can be
    declared in any order.
    """

    def __init__(self) -> None:
        self._agents: dict[str, IDelegate] = {}

    def add(self, delegate: IDelegate) -> None:
        if delegate.name in self._agents:
            raise ValueError(f"agent {delegate.name!r} already registered")
        self._agents[delegate.name] = delegate

    def replace(self, delegate: IDelegate) -> None:
        """Override an already-registered delegate (used by yaml override path)."""
        self._agents[delegate.name] = delegate

    def get(self, name: str) -> IDelegate:
        try:
            return self._agents[name]
        except KeyError:
            known = ", ".join(sorted(self._agents)) or "<none>"
            raise KeyError(f"agent {name!r} not registered; known agents: {known}") from None

    def list_names(self) -> list[str]:
        """Use this when you need to enumerate registered agent names (e.g.
        env-var override matching in `bind_agent_registry`)."""
        return list(self._agents)

    def has(self, name: str) -> bool:
        return name in self._agents

    def names(self) -> list[str]:
        return sorted(self._agents)


def register_agent(c: Container, delegate: IDelegate) -> None:
    """Convenience: ensure an AgentRegistry exists on `c`, then add `delegate`.

    Use this when you need to add a delegate to the registry from Python code
    (instead of via `bind_agent_registry` config loading or plugin entry-points).
    Param widened in v0.4.0f from `Agent` to `IDelegate` — `Agent` IS `IDelegate`,
    so existing positional calls are unaffected; kwarg callers
    (`register_agent(c, agent=...)`) must update to `delegate=`.
    """
    if not c.has(AgentRegistry):
        c.bind(AgentRegistry, AgentRegistry())
    c.resolve(AgentRegistry).add(delegate)
