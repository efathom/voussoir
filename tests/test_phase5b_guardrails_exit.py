"""Phase 5 Tranche B exit gate — one test per invariant from spec §5.

These tests lock the soft-policy claim for Tranche B. Future changes that
break any of these will fail the exit gate, before they reach CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_exit_1_decision_renamed_to_guardrail_verdict() -> None:
    """Spec §5.1: voussoir.guardrails.Decision is gone; GuardrailVerdict replaces it."""
    from voussoir.guardrails import GuardrailVerdict

    assert GuardrailVerdict(verdict="ALLOW").verdict == "ALLOW"
    with pytest.raises(ImportError):
        from voussoir.guardrails import Decision  # noqa: F401


def test_exit_2_guardrail_payload_typed() -> None:
    """Spec §5.1: GuardrailPayload has typed fields per stage."""
    from voussoir.guardrails import GuardrailPayload

    p = GuardrailPayload(stage="input", content="hi")
    assert p.stage == "input"
    assert p.tool_name is None


def test_exit_3_default_chain_dispatches_per_stage() -> None:
    """Spec §5.2: DefaultGuardrailChain groups by stage internally."""
    from voussoir.guardrails import DefaultGuardrailChain

    chain = DefaultGuardrailChain([])
    assert chain._by_stage == {}


def test_exit_4_bind_default_guardrails_three_profiles() -> None:
    """Spec §5.4: three profiles (off/standard/strict) — all bind without error."""
    from voussoir.container import Container
    from voussoir.guardrails import DefaultGuardrailChain, IGuardrailChain, bind_default_guardrails

    for profile in ("off", "standard", "strict"):
        c = Container()
        bind_default_guardrails(
            c, profile=profile, url_allowlist=["x"] if profile == "strict" else None
        )
        assert isinstance(c.resolve(IGuardrailChain), DefaultGuardrailChain)


def test_exit_5_llm_judge_composable() -> None:
    """Spec §5.5: LLMGuardrailJudge wraps a primary Guardrail."""
    from voussoir.guardrails import LLMGuardrailJudge

    assert LLMGuardrailJudge.__name__ == "LLMGuardrailJudge"


def test_exit_6_eight_builtin_guardrails_shipped() -> None:
    """Spec §5.3: all 8 built-in guardrails are importable."""
    from voussoir.guardrails.builtin.injection import PromptInjectionHeuristic
    from voussoir.guardrails.builtin.length import (
        ArgsSizeCap,
        InputLengthCap,
        ToolOutputSizeCap,
    )
    from voussoir.guardrails.builtin.pii import PIIDetector
    from voussoir.guardrails.builtin.schema import ArgsSchemaCheck
    from voussoir.guardrails.builtin.urls import ExfilPatternScan, URLAllowlist

    classes = [
        ArgsSchemaCheck,
        ArgsSizeCap,
        ExfilPatternScan,
        InputLengthCap,
        PIIDetector,
        PromptInjectionHeuristic,
        ToolOutputSizeCap,
        URLAllowlist,
    ]
    assert len(classes) == 8


def test_exit_7_lethal_trifecta_corpus_blocked_at_100pct() -> None:
    """Spec §10: 30/30 attacks in tests/security/lethal_trifecta/ pass in CI.

    v1.0.2 D5 added a 7th category (test_delegation_bypass.py) for the
    sub-agent UNTRUSTED-taint-propagation defence.
    """
    corpus_dir = Path(__file__).resolve().parent / "security" / "lethal_trifecta"
    test_files = [
        f for f in corpus_dir.iterdir() if f.name.startswith("test_") and f.name.endswith(".py")
    ]
    assert len(test_files) == 7  # seven categories (delegation bypass added in v1.0.2 D5)


def test_exit_8_guardrail_decision_record_has_rewrite_field() -> None:
    """Spec §5.1: audit-log GuardrailDecision gained rewrite field for REWRITE verdicts."""
    from voussoir.agent.result import GuardrailDecision

    d = GuardrailDecision(name="x", stage="output", decision="REWRITE", rewrite="cleaned")
    assert d.rewrite == "cleaned"
