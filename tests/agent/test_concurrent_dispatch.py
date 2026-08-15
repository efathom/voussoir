"""Concurrent tool-call dispatch — _dispatch_one + dispatch_tool_calls."""

from __future__ import annotations

import asyncio

import pytest
from ctxforge.protocols.llm import ChatMessage

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import (
    ToolCallOutcome,
    _dispatch_one,
    accumulate_outcomes,
    dispatch_tool_calls,
)
from voussoir.agent.result import AgentResult, Step
from voussoir.agent.turn_adapter import AnthropicToolCalls
from voussoir.executors.standard import StandardExecutor
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability
from voussoir.tools.registry import ToolRegistry


@tool(capability=Capability.READ_PRIVATE, name="echo_tool", description="Echo input back")
async def _echo_tool(text: str) -> str:
    return f"echo:{text}"


@tool(capability=Capability.READ_PRIVATE, name="raise_tool", description="Always raises")
async def _raise_tool(text: str) -> str:
    raise ValueError(f"kaboom:{text}")


@tool(capability=Capability.READ_PRIVATE, name="sleep_tool", description="Async sleep")
async def _sleep_tool(text: str) -> str:
    await asyncio.sleep(0.2)
    return f"slept:{text}"


def _registry(*tools_to_register) -> ToolRegistry:
    r = ToolRegistry()
    r.register_many(list(tools_to_register))
    return r


@pytest.mark.asyncio
async def test_dispatch_one_success_path(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        outcome = await _dispatch_one(
            {"id": "t1", "name": "echo_tool", "arguments": {"text": "hi"}},
            registry=_registry(_echo_tool),
            executor=StandardExecutor(),
            ctx=ctx,
        )
    assert outcome.output_str == "echo:hi"
    assert outcome.error is None
    assert outcome.sub_result is None
    assert outcome.duration_ms >= 0


@pytest.mark.asyncio
async def test_dispatch_one_captures_exception_as_tool_error(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        outcome = await _dispatch_one(
            {"id": "t1", "name": "raise_tool", "arguments": {"text": "x"}},
            registry=_registry(_raise_tool),
            executor=StandardExecutor(),
            ctx=ctx,
        )
    assert "kaboom:x" in outcome.output_str
    assert outcome.output_str.startswith("TOOL_ERROR:")
    assert isinstance(outcome.error, ValueError)
    assert outcome.sub_result is None


@pytest.mark.asyncio
async def test_dispatch_one_reraises_cancelled_error(make_container, stub_llm) -> None:
    """asyncio.CancelledError must propagate, not be swallowed as TOOL_ERROR."""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        task = asyncio.create_task(
            _dispatch_one(
                {"id": "t1", "name": "sleep_tool", "arguments": {"text": "x"}},
                registry=_registry(_sleep_tool),
                executor=StandardExecutor(),
                ctx=ctx,
            )
        )
        await asyncio.sleep(0.05)  # let the task start, then cancel
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_dispatch_tool_calls_preserves_declared_order(make_container, stub_llm) -> None:
    """A, B, C with B sleeping; outcomes return in declared (A, B, C) order."""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        outcomes = await dispatch_tool_calls(
            [
                {"id": "tA", "name": "echo_tool", "arguments": {"text": "A"}},
                {"id": "tB", "name": "sleep_tool", "arguments": {"text": "B"}},
                {"id": "tC", "name": "echo_tool", "arguments": {"text": "C"}},
            ],
            registry=_registry(_echo_tool, _sleep_tool),
            executor=StandardExecutor(),
            ctx=ctx,
        )
    assert [o.tc["id"] for o in outcomes] == ["tA", "tB", "tC"]
    assert outcomes[0].output_str == "echo:A"
    assert outcomes[1].output_str == "slept:B"
    assert outcomes[2].output_str == "echo:C"


@pytest.mark.asyncio
async def test_dispatch_tool_calls_runs_concurrently(make_container, stub_llm) -> None:
    """Two gated tools must both enter before either can return — proves the
    dispatch_tool_calls batch actually runs concurrently.

    Pre-B1 this used wall-clock thresholds (`elapsed < 0.35s`) which was
    flaky on loaded CI. B1 swapped to asyncio.Event gating: both tools must
    signal `started_X` before any of them is allowed past `release.wait()`,
    so a sequential dispatcher would deadlock at the first `started_a` /
    `release.wait()` boundary.
    """
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    release = asyncio.Event()

    @tool(capability=Capability.READ_PRIVATE, name="gated_a", description="Gated tool A")
    async def _gated_a(text: str) -> str:
        started_a.set()
        await release.wait()
        return f"a:{text}"

    @tool(capability=Capability.READ_PRIVATE, name="gated_b", description="Gated tool B")
    async def _gated_b(text: str) -> str:
        started_b.set()
        await release.wait()
        return f"b:{text}"

    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        dispatch_task = asyncio.create_task(
            dispatch_tool_calls(
                [
                    {"id": "t1", "name": "gated_a", "arguments": {"text": "1"}},
                    {"id": "t2", "name": "gated_b", "arguments": {"text": "2"}},
                ],
                registry=_registry(_gated_a, _gated_b),
                executor=StandardExecutor(),
                ctx=ctx,
            )
        )
        # If dispatch is sequential, started_b never fires (because gated_a
        # blocks on `release.wait()`), and the wait_for below times out.
        await asyncio.wait_for(started_a.wait(), timeout=2.0)
        await asyncio.wait_for(started_b.wait(), timeout=2.0)
        # Both tools entered before either could finish — concurrent dispatch confirmed.
        release.set()
        outcomes = await dispatch_task
    assert len(outcomes) == 2
    assert outcomes[0].output_str == "a:1"
    assert outcomes[1].output_str == "b:2"


def _make_sub_result(output: str = "sub", cost: float = 0.001) -> AgentResult[str]:
    return AgentResult[str](
        output=output,
        trace_id="sub-t",
        steps=[],
        tokens_in=10,
        tokens_out=5,
        cost_usd=cost,
        duration_ms=12.0,
        delegation_chain=["sub-agent"],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )


@pytest.mark.asyncio
async def test_accumulate_outcomes_appends_step_per_tool(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        outcomes = [
            ToolCallOutcome(
                tc={"id": "tA", "name": "echo_tool", "arguments": {"text": "A"}},
                output_str="echo:A",
                sub_result=None,
                duration_ms=1.2,
                error=None,
            ),
            ToolCallOutcome(
                tc={"id": "tB", "name": "echo_tool", "arguments": {"text": "B"}},
                output_str="echo:B",
                sub_result=None,
                duration_ms=0.9,
                error=None,
            ),
        ]
        steps: list[Step] = []
        messages: list[ChatMessage] = []
        d_in, d_out, d_cost = accumulate_outcomes(
            outcomes,
            ctx=ctx,
            steps=steps,
            messages=messages,
            adapter=AnthropicToolCalls(),
        )
    assert d_in == 0 and d_out == 0 and d_cost == 0.0
    assert [s.kind for s in steps] == ["tool_call", "tool_call"]
    assert [s.name for s in steps] == ["echo_tool", "echo_tool"]
    assert len(messages) == 2
    assert all(m.role == "function" for m in messages)
    assert messages[0].function_call == {"tool_use_id": "tA"}
    assert messages[1].function_call == {"tool_use_id": "tB"}


@pytest.mark.asyncio
async def test_accumulate_outcomes_aggregates_sub_result(make_container, stub_llm) -> None:
    """When outcome.sub_result is set, a delegation Step is appended and
    tokens/cost deltas reflect the sub-agent's contribution."""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        sr = _make_sub_result(cost=0.002)
        outcomes = [
            ToolCallOutcome(
                tc={"id": "tD", "name": "delegate_to_helper", "arguments": {"task": "go"}},
                output_str='<delegate_response from="helper" trust="untrusted">sub</delegate_response>',
                sub_result=sr,
                duration_ms=15.0,
                error=None,
            ),
        ]
        steps: list[Step] = []
        messages: list[ChatMessage] = []
        d_in, d_out, d_cost = accumulate_outcomes(
            outcomes,
            ctx=ctx,
            steps=steps,
            messages=messages,
            adapter=AnthropicToolCalls(),
        )
    assert d_in == 10
    assert d_out == 5
    assert d_cost == 0.002
    assert [s.kind for s in steps] == ["tool_call", "delegation"]
    assert steps[1].name == "helper"
    assert steps[1].payload["delegation_chain"] == ["sub-agent"]


@pytest.mark.asyncio
async def test_accumulate_outcomes_error_branch_in_step_payload(make_container, stub_llm) -> None:
    """An outcome with `error` set produces a tool_call Step whose
    output_preview contains 'TOOL_ERROR:'."""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        outcomes = [
            ToolCallOutcome(
                tc={"id": "tE", "name": "raise_tool", "arguments": {"text": "x"}},
                output_str="TOOL_ERROR: ValueError('kaboom:x')",
                sub_result=None,
                duration_ms=0.3,
                error=ValueError("kaboom:x"),
            ),
        ]
        steps: list[Step] = []
        messages: list[ChatMessage] = []
        accumulate_outcomes(
            outcomes, ctx=ctx, steps=steps, messages=messages, adapter=AnthropicToolCalls()
        )
    assert "TOOL_ERROR" in steps[0].payload["output_preview"]
    # Function-message still appended (so the LLM sees the failure on its next turn).
    assert messages[0].content.startswith("TOOL_ERROR:")


# --- Workstream A test set 2: end-to-end through Agent.run ---


from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from ctxforge.protocols.llm import ILLMProvider, LLMResponse  # noqa: E402

from voussoir.agent.agent import Agent  # noqa: E402
from voussoir.protocols import ILLMProvider as ILLMProviderProto  # noqa: E402


def _llm_tool_use(tool_calls: list[dict], content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_use",
        raw_response={"tool_calls": tool_calls},
    )


def _llm_text(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="end_turn",
        raw_response=None,
    )


@pytest.mark.asyncio
async def test_e2e_two_tool_calls_collected_into_steps(make_container, stub_llm) -> None:
    """End-to-end: LLM emits two tool_use blocks in one turn; both Steps
    appear in result.steps in declared order; final output reflects both
    tool results visible to the LLM on its next turn."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [
                    {"id": "tA", "name": "echo_tool", "arguments": {"text": "alpha"}},
                    {"id": "tB", "name": "echo_tool", "arguments": {"text": "beta"}},
                ]
            ),
            _llm_text("done with both"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[_echo_tool], container=c)
    result = await agent.run("test")
    assert result.output == "done with both"
    tool_call_steps = [s for s in result.steps if s.kind == "tool_call"]
    assert [s.payload["args"]["text"] for s in tool_call_steps] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_e2e_partial_failure_does_not_short_circuit(make_container, stub_llm) -> None:
    """One tool raises, the other succeeds; the next chat() call sees BOTH
    function-message replies (success content + TOOL_ERROR string)."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    captured_second_call_messages: list = []

    async def chat(messages, **kwargs):
        if llm.chat.await_count == 1:
            return _llm_tool_use(
                [
                    {"id": "tOK", "name": "echo_tool", "arguments": {"text": "good"}},
                    {"id": "tBAD", "name": "raise_tool", "arguments": {"text": "bad"}},
                ]
            )
        captured_second_call_messages.extend(messages)
        return _llm_text("saw both")

    llm.chat = AsyncMock(side_effect=chat)
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[_echo_tool, _raise_tool], container=c)
    result = await agent.run("test")
    assert result.output == "saw both"
    # Second chat must include two function-role messages (one per tool call).
    function_msgs = [m for m in captured_second_call_messages if m.role == "function"]
    assert len(function_msgs) == 2
    contents = [m.content for m in function_msgs]
    assert any("echo:good" in c_ for c_ in contents)
    assert any("TOOL_ERROR" in c_ for c_ in contents)


@pytest.mark.asyncio
async def test_e2e_concurrent_delegations_each_capture_own_sub_result(
    make_container, stub_llm
) -> None:
    """Two parallel delegate_to_* calls each capture their own sub-agent's
    AgentResult. delegation_chain on the lead's result includes both
    delegate names; tokens_in/out aggregates both sub-agents' contributions."""
    # Lead LLM: one turn with two delegate_to_*, then final text.
    # With concurrent dispatch, both helpers' chat() calls fire BEFORE
    # lead's second turn — so side_effect order is:
    #   1. lead's tool_use turn
    #   2-3. two helper chat() calls (concurrent, order undefined)
    #   4. lead's final text turn
    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"
    lead_llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [
                    {"id": "tH1", "name": "delegate_to_helper1", "arguments": {"task": "t1"}},
                    {"id": "tH2", "name": "delegate_to_helper2", "arguments": {"task": "t2"}},
                ]
            ),
            _llm_text("helper1 result"),
            _llm_text("helper2 result"),
            _llm_text("both helpers done"),
        ]
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)
    helper1 = Agent("helper1", instructions="", container=c)
    helper2 = Agent("helper2", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper1, helper2], container=c)
    result = await lead.run("test")
    assert result.output == "both helpers done"
    # delegation_chain must contain both helper names.
    assert "helper1" in result.delegation_chain
    assert "helper2" in result.delegation_chain
    # Both delegation Steps present.
    delegation_steps = [s for s in result.steps if s.kind == "delegation"]
    assert {s.name for s in delegation_steps} == {"helper1", "helper2"}
