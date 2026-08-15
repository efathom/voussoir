from datetime import UTC, datetime

import pytest

from voussoir.agent.middleware import (
    BudgetMiddleware,
    LoggingMiddleware,
    RetryMiddleware,
)
from voussoir.agent.policy import AgentPolicy, PolicyViolation
from voussoir.agent.result import AgentEvent, AgentResult
from voussoir.observability.logging_setup import configure_logging


def _ensure_stderr_routing() -> None:
    # configure_logging(force=True) clobbers caplog's handler, so we read stderr
    # directly via capsys instead of caplog.
    configure_logging(level="DEBUG", format="dev")


async def test_logging_middleware_logs_each_lifecycle(capsys):
    _ensure_stderr_routing()
    mw = LoggingMiddleware()
    ctx = type("MockCtx", (), {"run_id": "r1", "trace_id": "t1"})()

    await mw.before_run(ctx, "hello")
    step = AgentEvent(
        kind="tool_started",
        payload={"tool": "echo"},
        span_id="s1",
        timestamp=datetime.now(UTC),
    )
    await mw.after_step(ctx, step)
    result = AgentResult[str](
        output="hi",
        trace_id="t1",
        steps=[],
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.0001,
        duration_ms=120.5,
        delegation_chain=[],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )
    await mw.after_run(ctx, result)

    err = capsys.readouterr().err
    assert "agent.run.start" in err
    assert "agent.step" in err
    assert "agent.run.end" in err
    assert "run_id=r1" in err


async def test_logging_middleware_on_error_logs_exception(capsys):
    _ensure_stderr_routing()
    mw = LoggingMiddleware()
    ctx = type("MockCtx", (), {"run_id": "r1", "trace_id": "t1"})()
    exc = RuntimeError("boom")

    result = await mw.on_error(ctx, exc)

    err = capsys.readouterr().err
    assert "agent.run.error" in err
    assert "boom" in err
    # on_error returns the exception (LoggingMiddleware doesn't suppress).
    assert result is exc


async def test_retry_succeeds_on_third_attempt():
    mw = RetryMiddleware(max_attempts=3, base_delay_s=0.0)
    attempts = {"n": 0}

    async def call() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await mw.run_with_retry(call, retryable=(ConnectionError,))
    assert result == "ok"
    assert attempts["n"] == 3


async def test_retry_raises_after_max_attempts():
    mw = RetryMiddleware(max_attempts=2, base_delay_s=0.0)
    attempts = {"n": 0}

    async def call() -> str:
        attempts["n"] += 1
        raise ConnectionError("always")

    with pytest.raises(ConnectionError, match="always"):
        await mw.run_with_retry(call, retryable=(ConnectionError,))
    assert attempts["n"] == 2


async def test_retry_does_not_retry_non_retryable():
    mw = RetryMiddleware(max_attempts=3, base_delay_s=0.0)
    attempts = {"n": 0}

    async def call() -> str:
        attempts["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        await mw.run_with_retry(call, retryable=(ConnectionError,))
    assert attempts["n"] == 1  # didn't retry


async def test_budget_middleware_returns_violation_when_exceeded():
    mw = BudgetMiddleware(policy=AgentPolicy(max_cost_usd=0.10))
    ctx = type("MockCtx", (), {})()
    v = mw.check(ctx, steps=1, duration_s=1.0, tokens_in=10, tokens_out=10, cost_usd=0.20)
    assert v == PolicyViolation.MAX_COST


async def test_budget_middleware_returns_none_when_within():
    mw = BudgetMiddleware(policy=AgentPolicy())
    ctx = type("MockCtx", (), {})()
    v = mw.check(ctx, steps=1, duration_s=1.0, tokens_in=10, tokens_out=10, cost_usd=0.0)
    assert v is None
