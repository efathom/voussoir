"""Locks the Guardrail Protocol + GuardrailVerdict / GuardrailPayload shapes (B1).

Verifies the Tranche B rename of voussoir.guardrails.Decision → GuardrailVerdict
(closes architectural-review F-7), the sharpened Protocol signature, and the
voussoir.agent.result.GuardrailDecision audit-log record gaining a rewrite field.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voussoir.agent.result import GuardrailDecision
from voussoir.guardrails import Guardrail, GuardrailPayload, GuardrailVerdict, Trust
from voussoir.tools import Capability


def test_guardrail_verdict_shape():
    v = GuardrailVerdict(verdict="BLOCK", reason="length")
    assert v.verdict == "BLOCK"
    assert v.rewrite is None


def test_guardrail_verdict_rewrite_field():
    v = GuardrailVerdict(verdict="REWRITE", rewrite="cleaned", reason="pii")
    assert v.rewrite == "cleaned"


def test_guardrail_verdict_extra_forbidden():
    with pytest.raises(ValidationError):
        GuardrailVerdict(verdict="ALLOW", bogus="x")


def test_guardrail_payload_input_stage():
    p = GuardrailPayload(stage="input", content="hi")
    assert p.tool_name is None
    assert p.capability is None
    assert p.trust is None


def test_guardrail_payload_tool_output_stage():
    p = GuardrailPayload(
        stage="tool_output",
        content="...",
        tool_name="fetch",
        trust=Trust.UNTRUSTED,
        capability=Capability.READ_PUBLIC,
    )
    assert p.trust == Trust.UNTRUSTED


def test_guardrail_payload_extra_forbidden():
    with pytest.raises(ValidationError):
        GuardrailPayload(stage="input", content="hi", bogus="x")


def test_guardrail_decision_record_has_rewrite_field():
    """The audit-log record gains a rewrite field for REWRITE verdicts."""
    d = GuardrailDecision(
        name="pii",
        stage="output",
        decision="REWRITE",
        reason="email redacted",
        rewrite="see [REDACTED] above",
    )
    assert d.rewrite == "see [REDACTED] above"


def test_guardrail_decision_rewrite_default_none():
    """Record's rewrite field defaults to None for non-REWRITE decisions."""
    d = GuardrailDecision(name="ok", stage="input", decision="ALLOW")
    assert d.rewrite is None


def test_decision_alias_removed():
    """Phase 5 renames voussoir.guardrails.Decision → GuardrailVerdict; no back-compat alias."""
    with pytest.raises(ImportError):
        from voussoir.guardrails import Decision  # noqa: F401


def test_guardrail_protocol_runtime_checkable():
    """A class with name, stage, and async screen() satisfies the Protocol."""

    class _StubGuardrail:
        name = "stub"
        stage = "input"

        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            del payload, ctx
            return GuardrailVerdict(verdict="ALLOW")

    assert isinstance(_StubGuardrail(), Guardrail)
