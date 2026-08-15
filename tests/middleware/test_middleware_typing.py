"""Locks Middleware ctx: AgentContext typing (v1.1.0 F12).

The Middleware Protocol lives in voussoir.middleware (lower layer) and may
only import voussoir.agent.* under TYPE_CHECKING. At runtime its forward
references are strings. get_type_hints() requires a localns with the
forward-reference targets when inspecting Protocol hooks; concrete middleware
hooks live in voussoir.agent.* which can import AgentContext directly, so
no localns is needed there.
"""

from __future__ import annotations

import typing


def _resolved_ctx_annotation(fn: object, localns: dict[str, object] | None = None) -> object:
    """Return the resolved annotation for the 'ctx' param of `fn`."""
    hints = typing.get_type_hints(fn, localns=localns)
    return hints["ctx"]


def test_middleware_protocol_ctx_typed_as_agentcontext() -> None:
    from voussoir.agent.context import AgentContext
    from voussoir.agent.result import AgentResult, Step
    from voussoir.middleware.protocol import Middleware

    ns: dict[str, object] = {
        "AgentContext": AgentContext,
        "AgentResult": AgentResult,
        "Step": Step,
    }

    for hook_name in ("before_run", "after_step", "after_run", "on_error"):
        hook = getattr(Middleware, hook_name)
        ctx_ann = _resolved_ctx_annotation(hook, localns=ns)
        assert (
            ctx_ann is AgentContext
        ), f"Middleware.{hook_name} ctx is typed {ctx_ann!r}, expected AgentContext"


def test_concrete_middlewares_use_agentcontext() -> None:
    """Built-in middleware impls' ctx params are typed AgentContext."""
    from voussoir.agent.context import AgentContext
    from voussoir.agent.middleware import (
        BudgetMiddleware,
        LoggingMiddleware,
        RetryMiddleware,
    )

    for cls in (LoggingMiddleware, RetryMiddleware, BudgetMiddleware):
        for hook_name in ("before_run", "after_step", "after_run", "on_error"):
            hook = getattr(cls, hook_name, None)
            if hook is None:
                continue  # Some middlewares may only implement a subset.
            ctx_ann = _resolved_ctx_annotation(hook)
            assert ctx_ann is AgentContext, f"{cls.__name__}.{hook_name} ctx is typed {ctx_ann!r}"
