"""Locks accumulate_outcomes' sub-agent taint merge (v1.0.2 D5).

Without this merge, a sub-agent that reads UNTRUSTED content "launders" its
taint — the parent gets a clean ctx.taint back and can call EXFILTRATION
tools unblocked (Lethal Trifecta delegation bypass).
"""

from __future__ import annotations

from collections.abc import Callable

from ctxforge.protocols.llm import ChatMessage

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import ToolCallOutcome, accumulate_outcomes
from voussoir.agent.result import AgentResult, Step
from voussoir.agent.turn_adapter import AnthropicToolCalls
from voussoir.container import Container
from voussoir.guardrails.trust import Trust


def _make_sub_result(taint: set[Trust] | None = None) -> AgentResult[str]:
    return AgentResult[str](
        output="sub",
        trace_id="trace-sub",
        steps=[],
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        duration_ms=0.0,
        delegation_chain=["sub-agent"],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
        taint=taint if taint is not None else set(),
    )


async def test_accumulate_outcomes_merges_sub_agent_taint(
    make_container: Callable[..., Container],
) -> None:
    """A ToolCallOutcome with sub_result.taint={UNTRUSTED} merges into parent ctx.taint.

    Lethal Trifecta delegation bypass guard: a sub-agent that reads UNTRUSTED
    content must propagate that taint to the parent. Without this merge a
    parent could "launder" taint through delegation and call EXFILTRATION
    tools unblocked.
    """
    c = make_container()
    async with await AgentContext.open(
        container=c, run_id="r", session_id="s", user_id="u"
    ) as parent_ctx:
        assert parent_ctx.taint == set()
        sr = _make_sub_result(taint={Trust.UNTRUSTED})
        outcome = ToolCallOutcome(
            tc={"id": "tD", "name": "delegate_to_helper", "arguments": {"task": "go"}},
            output_str="sub",
            sub_result=sr,
            duration_ms=1.0,
            error=None,
        )
        steps: list[Step] = []
        messages: list[ChatMessage] = []
        accumulate_outcomes(
            [outcome], ctx=parent_ctx, steps=steps, messages=messages, adapter=AnthropicToolCalls()
        )
        assert Trust.UNTRUSTED in parent_ctx.taint


async def test_accumulate_outcomes_no_sub_result_no_taint_change(
    make_container: Callable[..., Container],
) -> None:
    """Outcomes without sub_result do not touch ctx.taint."""
    c = make_container()
    async with await AgentContext.open(
        container=c, run_id="r", session_id="s", user_id="u"
    ) as parent_ctx:
        outcome = ToolCallOutcome(
            tc={"id": "tA", "name": "echo_tool", "arguments": {"text": "A"}},
            output_str="echo:A",
            sub_result=None,
            duration_ms=1.0,
            error=None,
        )
        steps: list[Step] = []
        messages: list[ChatMessage] = []
        accumulate_outcomes(
            [outcome], ctx=parent_ctx, steps=steps, messages=messages, adapter=AnthropicToolCalls()
        )
        assert parent_ctx.taint == set()


async def test_accumulate_outcomes_taint_union_not_replacement(
    make_container: Callable[..., Container],
) -> None:
    """Merge is union — pre-existing parent taint is preserved alongside sub taint."""
    c = make_container()
    async with await AgentContext.open(
        container=c, run_id="r", session_id="s", user_id="u"
    ) as parent_ctx:
        parent_ctx.taint.add(Trust.INTERNAL)
        sr = _make_sub_result(taint={Trust.UNTRUSTED})
        outcome = ToolCallOutcome(
            tc={"id": "tD", "name": "delegate_to_helper", "arguments": {"task": "go"}},
            output_str="sub",
            sub_result=sr,
            duration_ms=1.0,
            error=None,
        )
        steps: list[Step] = []
        messages: list[ChatMessage] = []
        accumulate_outcomes(
            [outcome], ctx=parent_ctx, steps=steps, messages=messages, adapter=AnthropicToolCalls()
        )
        assert Trust.INTERNAL in parent_ctx.taint
        assert Trust.UNTRUSTED in parent_ctx.taint
