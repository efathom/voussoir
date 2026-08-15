"""AmbiguousFallback: only invokes judge on primary AMBIGUOUS."""

from __future__ import annotations

import pytest

from voussoir.agent.cascade import Decision
from voussoir.agent.result import AgentResult
from voussoir.agent.validators import AmbiguousFallback


def _make_result() -> AgentResult[str]:
    return AgentResult(
        output="x",
        trace_id="t",
        steps=[],
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        duration_ms=0.0,
        delegation_chain=[],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )


class _FixedValidator:
    def __init__(self, name: str, decision: Decision) -> None:
        self.name = name
        self._decision = decision
        self.calls = 0

    async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self._decision


@pytest.mark.asyncio
async def test_primary_pass_skips_judge() -> None:
    primary = _FixedValidator("primary", Decision.PASS)
    judge = _FixedValidator("judge", Decision.FAIL)
    fb = AmbiguousFallback(primary, judge)
    decision = await fb.validate(_make_result(), task="t")
    assert decision is Decision.PASS
    assert primary.calls == 1
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_primary_fail_skips_judge() -> None:
    primary = _FixedValidator("primary", Decision.FAIL)
    judge = _FixedValidator("judge", Decision.PASS)
    fb = AmbiguousFallback(primary, judge)
    decision = await fb.validate(_make_result(), task="t")
    assert decision is Decision.FAIL
    assert primary.calls == 1
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_primary_ambiguous_consults_judge_pass() -> None:
    primary = _FixedValidator("primary", Decision.AMBIGUOUS)
    judge = _FixedValidator("judge", Decision.PASS)
    fb = AmbiguousFallback(primary, judge)
    decision = await fb.validate(_make_result(), task="t")
    assert decision is Decision.PASS
    assert primary.calls == 1
    assert judge.calls == 1


@pytest.mark.asyncio
async def test_primary_ambiguous_consults_judge_fail() -> None:
    primary = _FixedValidator("primary", Decision.AMBIGUOUS)
    judge = _FixedValidator("judge", Decision.FAIL)
    fb = AmbiguousFallback(primary, judge)
    decision = await fb.validate(_make_result(), task="t")
    assert decision is Decision.FAIL


def test_name_includes_inner_validator_names() -> None:
    primary = _FixedValidator("p", Decision.PASS)
    judge = _FixedValidator("j", Decision.PASS)
    fb = AmbiguousFallback(primary, judge)
    assert "p" in fb.name
    assert "j" in fb.name
