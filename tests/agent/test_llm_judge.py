"""LLMJudge: PASS/FAIL parsing, prompt customization, cost emission."""

from __future__ import annotations

import pytest

from voussoir.agent.cascade import Decision
from voussoir.agent.result import AgentResult
from voussoir.agent.validators import (
    DEFAULT_LLM_JUDGE_PROMPT,
    LLMJudge,
    _parse_judge_verdict,
)
from voussoir.observability.sink import (
    InMemoryTelemetrySink,
    ITelemetrySink,
)


def _make_result(output: str = "the answer") -> AgentResult[str]:
    return AgentResult(
        output=output,
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


def test_parse_pass() -> None:
    assert _parse_judge_verdict("PASS") is Decision.PASS
    assert _parse_judge_verdict("  pass  ") is Decision.PASS
    assert _parse_judge_verdict("Pass") is Decision.PASS


def test_parse_fail_for_anything_else() -> None:
    for s in ["FAIL", "fail", "AMBIGUOUS", "unsure", "", "PASS — looks fine"]:
        assert _parse_judge_verdict(s) is Decision.FAIL, f"expected FAIL for {s!r}"


def test_default_prompt_has_required_placeholders() -> None:
    for ph in ("{task}", "{output}", "{criterion}"):
        assert ph in DEFAULT_LLM_JUDGE_PROMPT


def test_prompt_template_missing_placeholder_raises(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    with pytest.raises(ValueError, match="placeholder"):
        LLMJudge("X", container=c, prompt_template="bad: {task} {output}")  # no criterion


@pytest.mark.asyncio
async def test_llm_judge_pass(make_container, stub_llm) -> None:
    c = make_container(stub_llm(content="PASS"))
    judge = LLMJudge("output is non-empty", container=c)
    decision = await judge.validate(_make_result(), task="test task")
    assert decision is Decision.PASS


@pytest.mark.asyncio
async def test_llm_judge_fail(make_container, stub_llm) -> None:
    c = make_container(stub_llm(content="FAIL — not good"))
    judge = LLMJudge("output is good", container=c)
    decision = await judge.validate(_make_result(), task="test task")
    assert decision is Decision.FAIL


@pytest.mark.asyncio
async def test_llm_judge_emits_cost_via_sink(make_container, stub_llm) -> None:
    c = make_container(stub_llm(content="PASS", input_tokens=12, output_tokens=3))
    sink = InMemoryTelemetrySink()
    c.bind(ITelemetrySink, sink)  # type: ignore[type-abstract]
    judge = LLMJudge("anything", container=c)
    await judge.validate(_make_result(), task="t")
    judge_records = [r for r in sink.records if r.name == "llm_judge"]
    assert len(judge_records) == 1
    rec = judge_records[0]
    assert rec.tokens_in == 12
    assert rec.tokens_out == 3
    assert rec.cost_usd > 0  # cost_estimate property derives from tokens


@pytest.mark.asyncio
async def test_llm_judge_prompt_contains_task_output_criterion(make_container, stub_llm) -> None:
    """Verify the prompt sent to the LLM includes all three pieces of context."""
    llm = stub_llm(content="PASS")
    c = make_container(llm)
    judge = LLMJudge("output is well-formed", container=c)
    await judge.validate(_make_result(output="hello world"), task="answer the question")
    # llm.chat was called once with a single user message containing the prompt
    assert llm.chat.await_count == 1
    call_args = llm.chat.await_args
    messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages")
    assert messages is not None and len(messages) == 1
    prompt = messages[0].content
    assert "answer the question" in prompt
    assert "hello world" in prompt
    assert "output is well-formed" in prompt


@pytest.mark.asyncio
async def test_llm_judge_custom_prompt_template(make_container, stub_llm) -> None:
    llm = stub_llm(content="PASS")
    c = make_container(llm)
    custom = "JUDGE-CUSTOM: task={task} output={output} criterion={criterion}"
    judge = LLMJudge("X", container=c, prompt_template=custom)
    await judge.validate(_make_result(output="Y"), task="Z")
    prompt = llm.chat.await_args.args[0][0].content
    assert prompt.startswith("JUDGE-CUSTOM:")
    assert "task=Z" in prompt
    assert "output=Y" in prompt
    assert "criterion=X" in prompt
