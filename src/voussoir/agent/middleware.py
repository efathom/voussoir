"""Agent-coupled built-in middlewares.

Phase 4.5a Task 1: relocated from voussoir.middleware.builtin to restore
one-way layering — voussoir.middleware now contains only the Middleware
Protocol (Agent-agnostic). All three built-in middlewares type-reference
AgentPolicy / AgentEvent / AgentResult so they belong Agent-side.

The Middleware Protocol itself remains in voussoir.middleware.protocol and
is implemented (structurally) by these classes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from voussoir.agent.context import AgentContext
from voussoir.agent.policy import AgentPolicy, PolicyViolation
from voussoir.agent.result import AgentEvent, AgentResult
from voussoir.observability.logging_setup import get_logger

_log = get_logger("voussoir.middleware")

T = TypeVar("T")


class LoggingMiddleware:
    """Emits structured logs at agent.run.start / agent.step / agent.run.end / agent.run.error."""

    async def before_run(self, ctx: AgentContext, input: Any) -> Any | None:
        _log.info(
            "agent.run.start",
            run_id=ctx.run_id,
            trace_id=ctx.trace_id,
            input_preview=str(input)[:200],
        )
        return None

    async def after_step(self, ctx: AgentContext, step: AgentEvent) -> None:
        _log.info(
            "agent.step",
            run_id=ctx.run_id,
            kind=step.kind,
            payload=step.payload,
        )

    async def after_run(self, ctx: AgentContext, result: AgentResult[Any]) -> AgentResult[Any]:
        _log.info(
            "agent.run.end",
            run_id=ctx.run_id,
            finish_reason=result.finish_reason,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            duration_ms=result.duration_ms,
        )
        return result

    async def on_error(self, ctx: AgentContext, exc: BaseException) -> Any:
        _log.error(
            "agent.run.error",
            run_id=ctx.run_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return exc


class RetryMiddleware:
    """Retries transient errors with exponential backoff.

    Usage note: ``RetryMiddleware`` is a *wrapper*, not a per-step hook.
    Its core capability is ``run_with_retry(call, retryable=...)``, which
    wraps a *single callable* (e.g. ``agent.run``) with exponential-backoff
    retry logic.  The Middleware Protocol hooks (``before_run``, ``after_step``,
    ``after_run``, ``on_error``) are no-ops in this class — that is intentional.
    Retry semantics span the *entire* run, not individual steps, so they cannot
    be expressed as in-loop hooks without breaking the loop contract.

    Typical usage::

        retry = RetryMiddleware(max_attempts=3)
        result = await retry.run_with_retry(
            lambda: agent.run("task"),
            retryable=(TransientLLMError,),
        )
    """

    def __init__(self, *, max_attempts: int = 3, base_delay_s: float = 0.5) -> None:
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s

    async def run_with_retry(
        self,
        call: Callable[[], Awaitable[T]],
        *,
        retryable: tuple[type[BaseException], ...],
    ) -> T:
        last: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                return await call()
            except retryable as exc:
                last = exc
                if attempt < self.max_attempts - 1:
                    await asyncio.sleep(self.base_delay_s * (2**attempt))
        assert last is not None
        raise last

    async def before_run(self, ctx: AgentContext, input: Any) -> Any | None:
        return None

    async def after_step(self, ctx: AgentContext, step: Any) -> None:
        return None

    async def after_run(self, ctx: AgentContext, result: Any) -> Any:
        return result

    async def on_error(self, ctx: AgentContext, exc: BaseException) -> Any:
        return exc


class BudgetMiddleware:
    """Wraps the agent's running counters in AgentPolicy.check() at step boundaries."""

    def __init__(self, *, policy: AgentPolicy) -> None:
        self.policy = policy

    def check(
        self,
        ctx: AgentContext,
        *,
        steps: int,
        duration_s: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> PolicyViolation | None:
        return self.policy.check(
            steps=steps,
            duration_s=duration_s,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )

    async def before_run(self, ctx: AgentContext, input: Any) -> Any | None:
        return None

    async def after_step(self, ctx: AgentContext, step: Any) -> None:
        # Budget checks require running token/cost totals, which are
        # maintained by the Agent loop (not available on the step object).
        # The primary budget enforcement is in Agent._run_normal's per-step
        # self.policy.check() call. This hook is intentionally a no-op.
        return None

    async def after_run(self, ctx: AgentContext, result: Any) -> Any:
        return result

    async def on_error(self, ctx: AgentContext, exc: BaseException) -> Any:
        return exc
