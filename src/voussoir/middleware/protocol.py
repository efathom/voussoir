"""Middleware Protocol for cross-cutting concerns around an Agent run.

Use this when you need to observe or shape an agent's lifecycle without
modifying the Agent class — budgets, retries, structured logging, etc.
Built-ins live in voussoir.agent.middleware.

v1.1.0 F12: ctx narrowed from Any to AgentContext on all four hooks so
mypy strict can verify that middleware bodies only access real context
attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from voussoir.agent.context import AgentContext
    from voussoir.agent.result import AgentResult, Step


@runtime_checkable
class Middleware(Protocol):
    """Hook protocol for cross-cutting concerns around an Agent run.

    Use this when you need to observe or shape an agent's lifecycle
    without modifying the Agent class — budgets, retries, structured
    logging, etc. Built-ins live in voussoir.agent.middleware.

    v1.1.0 F12: ctx is typed AgentContext (was Any) on all four hooks.
    """

    async def before_run(self, ctx: AgentContext, input: Any) -> Any | None:
        """Called once before the first LLM turn. Raise to abort the run."""
        ...

    async def after_step(self, ctx: AgentContext, step: Step) -> None:
        """Called after each step (llm_call, tool_call, validator_call,
        delegation) with the just-completed Step. Cannot mutate the step.
        """
        ...

    async def after_run(self, ctx: AgentContext, result: AgentResult[Any]) -> Any:
        """Called once after the run completes, success or failure. Final
        result is observable but not mutable.
        """
        ...

    async def on_error(self, ctx: AgentContext, exc: BaseException) -> Any:
        """Called when an uncaught exception escapes the run.
        Implementations observe / log / record; the exception still
        propagates to the caller. Re-raising a different exception from
        here is undefined; do not rely on it.
        """
        ...
