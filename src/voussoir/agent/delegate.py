"""IDelegate Protocol — uniform surface for local, named, remote, and plugin delegates.

Phase 4a introduces the Protocol and one concrete impl (NamedDelegate).
Phase 4b's AgentRef extends this module via HTTP/JSON-RPC. Phase 4d
adds the `voussoir.delegates` entry-point group: an installed Python
package advertises an IDelegate factory that `load_delegate_plugins`
adds to the registry (see voussoir.agent.plugins).

The caller (Agent's synthetic-tool invoker) owns depth-cap checks and
post-run bookkeeping (cost aggregation via _last_sub_result_var,
DELEGATION_REFUSED wrapping, delegation_chain propagation). Implementations
own only kind-specific execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from voussoir.agent.context import AgentContext
    from voussoir.agent.result import AgentResult


@runtime_checkable
class IDelegate(Protocol):
    """A delegate that an Agent can hand a task to during its run loop.

    Implementations:
      - `Agent` itself — local execution; satisfies IDelegate via `.delegate()`.
      - `NamedDelegate(name)` — lazy registry lookup, then dispatches to the
        registered IDelegate's `.delegate()`.
      - `RemoteDelegate` / `AgentRef` — Phase 4b (HTTP via JSON-RPC).
      - Any IDelegate factory advertised by an installed package via the
        `voussoir.delegates` entry-point group (Phase 4d). See
        `voussoir.agent.plugins.load_delegate_plugins`.
    """

    name: str
    description: str

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        """Execute as a sub-agent. Implementations handle their own scoping
        (child container for local agents, HTTP call for remote, etc.).

        Raises:
          PolicyViolationError: if the implementation hits its own limits
            (e.g. local Agent's max_steps, missing registry binding for
            NamedDelegate). Caller converts to DELEGATION_REFUSED.
        """
        ...


class NamedDelegate:
    """Lazy-resolve a delegate by name at delegate() call time.

    The parent agent doesn't need to know the target Agent at construction.
    `register_agent` / `bind_agent_registry` (Phase 3.5) populate the
    registry; NamedDelegate reads it at invocation.

    Description is "" because the registered Agent's description isn't
    known at construction. Users who want a description should pass the
    Agent instance directly rather than the name string.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        # Deferred imports to keep delegate.py free of runtime back-edges
        # to registry.py / policy.py at module load time.
        from voussoir.agent.policy import PolicyViolation, PolicyViolationError
        from voussoir.agent.registry import AgentRegistry

        if not parent_ctx.container.has(AgentRegistry):
            raise PolicyViolationError(
                PolicyViolation.DELEGATE_NOT_FOUND,
                "AgentRegistry not bound on container; register agents via "
                "register_agent(c, agent) before delegating by name.",
            )
        try:
            target = parent_ctx.container.resolve(AgentRegistry).get(self.name)
        except KeyError as exc:
            raise PolicyViolationError(
                PolicyViolation.DELEGATE_NOT_FOUND, str(exc.args[0])
            ) from None
        return await target.delegate(task, parent_ctx=parent_ctx)
