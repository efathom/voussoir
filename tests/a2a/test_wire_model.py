"""Phase 4.5a P0 #3 — WireAgentResult redacts leaky fields."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from voussoir.a2a.wire import WireAgentResult
from voussoir.agent.result import AgentResult, Step


def _make_full_result() -> AgentResult[str]:
    return AgentResult[str](
        output="user-visible answer",
        trace_id="internal-trace-uuid",
        steps=[
            Step(
                kind="tool_call",
                name="secret_tool",
                duration_ms=1.0,
                payload={
                    "args": {"api_key": "should-not-leak"},
                    "output_preview": "internal output up to 200 chars",
                },
            )
        ],
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        duration_ms=12.0,
        delegation_chain=["downstream_agent", "another_internal"],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )


def test_wire_model_redacts_steps_chain_traceid() -> None:
    """from_agent_result drops trace_id, steps, delegation_chain."""
    wire = WireAgentResult.from_agent_result(_make_full_result())
    serialized = wire.model_dump()
    assert wire.output == "user-visible answer"
    assert wire.finish_reason == "completed"
    assert set(serialized.keys()) == {
        "output",
        "finish_reason",
        "tokens_in",
        "tokens_out",
        "duration_ms",
    }
    blob = json.dumps(serialized)
    assert "should-not-leak" not in blob
    assert "internal-trace-uuid" not in blob
    assert "downstream_agent" not in blob
    assert "secret_tool" not in blob


def test_wire_model_preserves_summary_fields() -> None:
    wire = WireAgentResult.from_agent_result(_make_full_result())
    assert wire.tokens_in == 100
    assert wire.tokens_out == 50
    assert wire.duration_ms == 12.0


def test_wire_model_extra_fields_forbidden() -> None:
    """Pydantic extra='forbid' so accidental field addition is caught."""
    with pytest.raises(ValidationError):
        WireAgentResult(
            output="x",
            finish_reason="completed",
            tokens_in=1,
            tokens_out=1,
            duration_ms=1.0,
            extra_field="boom",  # type: ignore[call-arg]
        )
