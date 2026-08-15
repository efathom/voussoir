"""Locks LLMGuardrailJudge AMBIGUOUS-fallback composer (Phase 5 Task B4).

Pass-through behavior for non-AMBIGUOUS primaries; LLM-backed forced-binary
verdict for AMBIGUOUS; fail-closed on malformed LLM responses.
"""

from __future__ import annotations

from typing import Literal

import pytest

from voussoir.guardrails import (
    GuardrailPayload,
    GuardrailVerdict,
    LLMGuardrailJudge,
)


class _StubGuardrail:
    """Returns whatever verdict the test seeds. Stage is parameterized."""

    name = "stub"

    def __init__(
        self,
        verdict: GuardrailVerdict,
        *,
        stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_output",
    ) -> None:
        self._verdict = verdict
        self.stage: Literal["input", "tool_call", "tool_output", "output"] = stage

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del payload, ctx
        return self._verdict


async def test_judge_passes_allow_through(make_container, stub_llm):
    """When primary returns ALLOW, judge returns it unchanged (no LLM call)."""
    primary = _StubGuardrail(GuardrailVerdict(verdict="ALLOW"))
    judge = LLMGuardrailJudge(primary, container=make_container(stub_llm()))
    v = await judge.screen(
        GuardrailPayload(stage="tool_output", content="hi", tool_name="t"), ctx=None
    )
    assert v.verdict == "ALLOW"


async def test_judge_passes_block_through(make_container, stub_llm):
    """When primary returns BLOCK, judge returns it unchanged (no LLM call)."""
    primary = _StubGuardrail(GuardrailVerdict(verdict="BLOCK", reason="bad"))
    judge = LLMGuardrailJudge(primary, container=make_container(stub_llm()))
    v = await judge.screen(
        GuardrailPayload(stage="tool_output", content="hi", tool_name="t"), ctx=None
    )
    assert v.verdict == "BLOCK"


async def test_judge_passes_rewrite_through(make_container, stub_llm):
    """When primary returns REWRITE, judge returns it unchanged (no LLM call)."""
    primary = _StubGuardrail(GuardrailVerdict(verdict="REWRITE", rewrite="cleaned"))
    judge = LLMGuardrailJudge(primary, container=make_container(stub_llm()))
    v = await judge.screen(
        GuardrailPayload(stage="tool_output", content="hi", tool_name="t"), ctx=None
    )
    assert v.verdict == "REWRITE"
    assert v.rewrite == "cleaned"


async def test_judge_routes_ambiguous_to_llm_pass_maps_to_allow(make_container, stub_llm):
    """LLM says PASS → judge returns ALLOW."""
    llm = stub_llm(content="PASS")
    primary = _StubGuardrail(GuardrailVerdict(verdict="AMBIGUOUS", reason="not sure"))
    judge = LLMGuardrailJudge(primary, container=make_container(llm))
    v = await judge.screen(
        GuardrailPayload(stage="tool_output", content="...", tool_name="t"), ctx=None
    )
    assert v.verdict == "ALLOW"
    assert "llm_judge" in v.reason


async def test_judge_routes_ambiguous_to_llm_fail_maps_to_block(make_container, stub_llm):
    """LLM says FAIL → judge returns BLOCK."""
    llm = stub_llm(content="FAIL")
    primary = _StubGuardrail(GuardrailVerdict(verdict="AMBIGUOUS"))
    judge = LLMGuardrailJudge(primary, container=make_container(llm))
    v = await judge.screen(
        GuardrailPayload(stage="tool_output", content="...", tool_name="t"), ctx=None
    )
    assert v.verdict == "BLOCK"


async def test_judge_malformed_llm_response_defaults_to_block(make_container, stub_llm):
    """Anything other than exact 'PASS' → BLOCK (fail-closed)."""
    llm = stub_llm(content="maybe? hard to say...")
    primary = _StubGuardrail(GuardrailVerdict(verdict="AMBIGUOUS"))
    judge = LLMGuardrailJudge(primary, container=make_container(llm))
    v = await judge.screen(
        GuardrailPayload(stage="tool_output", content="...", tool_name="t"), ctx=None
    )
    assert v.verdict == "BLOCK"


def test_judge_name_combines_primary_name(make_container):
    primary = _StubGuardrail(GuardrailVerdict(verdict="ALLOW"))
    judge = LLMGuardrailJudge(primary, container=make_container())
    assert judge.name == "stub+llm_judge"


def test_judge_stage_mirrors_primary(make_container):
    primary = _StubGuardrail(GuardrailVerdict(verdict="ALLOW"), stage="input")
    judge = LLMGuardrailJudge(primary, container=make_container())
    assert judge.stage == "input"


def test_judge_validates_prompt_template_placeholders(make_container):
    """A custom prompt missing any required placeholder raises ValueError at construction."""
    primary = _StubGuardrail(GuardrailVerdict(verdict="ALLOW"))
    with pytest.raises(ValueError, match="prompt_template must include placeholder"):
        LLMGuardrailJudge(
            primary,
            container=make_container(),
            prompt_template="missing placeholders",
        )


async def test_judge_emits_telemetry_on_ambiguous(make_container, stub_llm):
    """On the AMBIGUOUS → LLM path, judge calls ITelemetrySink.record_llm_call."""
    from voussoir.observability.sink import (
        InMemoryTelemetrySink,
        ITelemetrySink,
    )

    llm = stub_llm(content="PASS")
    sink = InMemoryTelemetrySink()
    c = make_container(llm)
    c.bind(ITelemetrySink, sink)  # type: ignore[type-abstract]

    primary = _StubGuardrail(GuardrailVerdict(verdict="AMBIGUOUS", reason="hedged"))
    judge = LLMGuardrailJudge(primary, container=c)
    await judge.screen(
        GuardrailPayload(stage="tool_output", content="...", tool_name="t"),
        ctx=None,
    )

    records = sink.records
    assert len(records) == 1
    rec = records[0]
    assert rec.name == "stub+llm_judge"
    assert rec.payload["primary_name"] == "stub"
    assert rec.payload["primary_stage"] == "tool_output"


async def test_judge_does_not_emit_telemetry_on_pass_through(make_container):
    """When primary returns non-AMBIGUOUS, judge skips the LLM (and the telemetry)."""
    from voussoir.observability.sink import (
        InMemoryTelemetrySink,
        ITelemetrySink,
    )

    sink = InMemoryTelemetrySink()
    c = make_container()
    c.bind(ITelemetrySink, sink)  # type: ignore[type-abstract]

    primary = _StubGuardrail(GuardrailVerdict(verdict="ALLOW"))
    judge = LLMGuardrailJudge(primary, container=c)
    await judge.screen(
        GuardrailPayload(stage="tool_output", content="ok", tool_name="t"),
        ctx=None,
    )

    assert sink.records == []
