"""Phase 4.5b B5 — agent/result.py models reject extra fields.

Pre-B5 these models silently accepted extra fields (Pydantic's default
extra='ignore'), turning typos like Step(naem='x', ...) into latent
bugs. B5 added ConfigDict(extra='forbid') to match the strictness of
the a2a/ models.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from voussoir.agent.result import (
    AgentEvent,
    AgentResult,
    CascadeOutcome,
    GuardrailDecision,
    Step,
)


def test_step_rejects_typo() -> None:
    """Step.model_validate with a typo'd field name raises ValidationError."""
    with pytest.raises(ValidationError):
        Step.model_validate(
            {
                "kind": "llm_call",
                "name": "step1",
                "duration_ms": 0.0,
                "naem": "x",  # typo
            }
        )


def test_agent_event_rejects_typo() -> None:
    """AgentEvent rejects an extra field."""
    with pytest.raises(ValidationError):
        AgentEvent.model_validate(
            {
                "kind": "token",
                "payload": {},
                "span_id": "s1",
                "timestamp": datetime.now(UTC).isoformat(),
                "extra_field": True,
            }
        )


def test_agent_result_rejects_typo() -> None:
    """AgentResult rejects an extra field."""
    with pytest.raises(ValidationError):
        AgentResult.model_validate(
            {
                "output": "ok",
                "trace_id": "t",
                "steps": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "duration_ms": 0.0,
                "delegation_chain": [],
                "cascade_history": [],
                "guardrail_decisions": [],
                "finish_reason": "completed",
                "extra_field_that_should_not_be_here": True,
            }
        )


def test_cascade_outcome_rejects_typo() -> None:
    """CascadeOutcome rejects an extra field."""
    with pytest.raises(ValidationError):
        CascadeOutcome.model_validate(
            {
                "attempted": "agent_a",
                "escalated": False,
                "reason": "ok",
                "extra": "no",
            }
        )


def test_guardrail_decision_rejects_typo() -> None:
    """GuardrailDecision rejects an extra field."""
    with pytest.raises(ValidationError):
        GuardrailDecision.model_validate(
            {
                "name": "g1",
                "stage": "input",
                "decision": "ALLOW",
                "reason": "",
                "extra": "no",
            }
        )
