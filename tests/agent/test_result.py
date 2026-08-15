from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from voussoir.agent.result import AgentEvent, AgentResult, Step


def test_agent_result_minimal():
    r = AgentResult[str](
        output="hello",
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
    assert r.output == "hello"
    assert r.tokens_in == 10
    assert r.delegation_chain == []


def test_agent_result_finish_reason_literal():
    with pytest.raises(ValidationError):
        AgentResult[str](
            output="x",
            trace_id="t1",
            steps=[],
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            duration_ms=0.0,
            delegation_chain=[],
            cascade_history=[],
            guardrail_decisions=[],
            finish_reason="bogus",  # not in the Literal set
        )


def test_agent_event_kinds():
    valid_kinds = [
        "token",
        "tool_started",
        "tool_finished",
        "delegation_started",
        "delegation_finished",
        "cascade_escalated",
        "guardrail_triggered",
        "checkpoint",
        "error",
        "done",
    ]
    for kind in valid_kinds:
        e = AgentEvent(kind=kind, payload={}, span_id="s1", timestamp=datetime.now(UTC))
        assert e.kind == kind


def test_agent_event_invalid_kind_rejected():
    with pytest.raises(ValidationError):
        AgentEvent(kind="bogus", payload={}, span_id="s1", timestamp=datetime.now(UTC))


def test_step_records_tool_call():
    s = Step(kind="tool_call", name="web_search", duration_ms=42.0, payload={"q": "x"})
    assert s.name == "web_search"
