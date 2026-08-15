"""Phase 4c exit criteria — gates the v0.4.0c-phase4c tag."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent.agent import Agent
from voussoir.agent.result import AgentEvent
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability


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


@tool(capability=Capability.READ_PRIVATE, name="echo_tool", description="echo")
async def _echo_tool(text: str) -> str:
    return f"echo:{text}"


@tool(capability=Capability.READ_PRIVATE, name="raise_tool", description="raises")
async def _raise_tool(text: str) -> str:
    raise ValueError(f"kaboom:{text}")


# Exit 1 — concurrent dispatch barrier proof (was wall-clock; v1.0.2 D10).
@pytest.mark.asyncio
async def test_exit_1_concurrent_dispatch_runs_in_parallel(make_container, stub_llm) -> None:
    """Two tool calls dispatched concurrently both reach a shared barrier
    before either completes — proves true parallelism without a flaky
    wall-clock cap. If dispatch is ever made sequential, the first tool
    sets started_count to 1 then waits forever for the barrier (the
    second tool can't start until the first returns), and the test fails
    via the asyncio.wait_for timeout."""
    both_started = asyncio.Event()
    started_count = 0
    lock = asyncio.Lock()

    @tool(capability=Capability.READ_PRIVATE, name="barrier_a", description="barrier a")
    async def barrier_a(text: str) -> str:
        nonlocal started_count
        async with lock:
            started_count += 1
            if started_count >= 2:
                both_started.set()
        # Wait for the other tool to also reach the barrier. Sequential
        # dispatch would hang here until wait_for trips the timeout.
        await asyncio.wait_for(both_started.wait(), timeout=2.0)
        return f"a:{text}"

    @tool(capability=Capability.READ_PRIVATE, name="barrier_b", description="barrier b")
    async def barrier_b(text: str) -> str:
        nonlocal started_count
        async with lock:
            started_count += 1
            if started_count >= 2:
                both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2.0)
        return f"b:{text}"

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [
                    {"id": "t1", "name": "barrier_a", "arguments": {"text": "1"}},
                    {"id": "t2", "name": "barrier_b", "arguments": {"text": "2"}},
                ]
            ),
            _llm_text("both done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[barrier_a, barrier_b], container=c)
    result = await agent.run("test")
    assert "both done" in result.output
    # Both tools must have started and reached the barrier — proves concurrent dispatch.
    assert started_count == 2
    assert both_started.is_set()
    # Critical: both tool calls must have completed WITHOUT TOOL_ERROR. Under
    # sequential dispatch the first tool would hit asyncio.wait_for(timeout=2.0)
    # → TimeoutError → "TOOL_ERROR: ..." in its output preview. Asserting on
    # error-free outputs is what makes this test a true regression catcher for
    # any future change that breaks concurrent dispatch.
    tool_steps = [s for s in result.steps if s.kind == "tool_call"]
    assert len(tool_steps) == 2
    for step in tool_steps:
        preview = step.payload.get("output_preview", "")
        assert "TOOL_ERROR" not in preview, f"tool {step.name} errored: {preview!r}"


# Exit 2 — mixed-failure tool dispatch.
@pytest.mark.asyncio
async def test_exit_2_mixed_failure_does_not_short_circuit(make_container, stub_llm) -> None:
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [
                    {"id": "tOK", "name": "echo_tool", "arguments": {"text": "ok"}},
                    {"id": "tBAD", "name": "raise_tool", "arguments": {"text": "bad"}},
                ]
            ),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[_echo_tool, _raise_tool], container=c)
    result = await agent.run("test")
    tool_steps = [s for s in result.steps if s.kind == "tool_call"]
    assert len(tool_steps) == 2


# Exit 3 — Agent.stream emits tool_started + tool_finished.
@pytest.mark.asyncio
async def test_exit_3_stream_emits_tool_events(make_container, stub_llm) -> None:
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use([{"id": "t1", "name": "echo_tool", "arguments": {"text": "hi"}}]),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[_echo_tool], container=c)
    events = [e async for e in agent.stream("test")]
    kinds = [e.kind for e in events]
    assert "tool_started" in kinds and "tool_finished" in kinds


# Exit 4 — full delegation pair.
@pytest.mark.asyncio
async def test_exit_4_stream_delegation_full_pair(make_container, stub_llm) -> None:
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [{"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "go"}}]
            ),
            _llm_text("helper"),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    events = [e async for e in lead.stream("test")]
    kinds = [e.kind for e in events]
    for k in ("tool_started", "delegation_started", "tool_finished", "delegation_finished"):
        assert k in kinds, f"missing {k}"


# Exit 5 — refused delegation omits delegation_finished.
@pytest.mark.asyncio
async def test_exit_5_stream_refused_delegation_no_finished(make_container, stub_llm) -> None:
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [{"id": "tU", "name": "delegate_to_unknown", "arguments": {"task": "x"}}]
            ),
            _llm_text("saw refusal"),
        ]
    )
    from voussoir.agent.registry import AgentRegistry

    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    # Phase 4.5a P1 #23: bind empty registry so unresolvable-by-name still
    # surfaces as a runtime DELEGATION_REFUSED (not as Agent.__init__ raise).
    c.bind(AgentRegistry, AgentRegistry())
    lead = Agent("lead", instructions="", delegates=["unknown"], container=c)
    events = [e async for e in lead.stream("test")]
    kinds = [e.kind for e in events]
    assert "delegation_started" in kinds
    assert "delegation_finished" not in kinds
    tf = next(e for e in events if e.kind == "tool_finished")
    assert "DELEGATION_REFUSED" in tf.payload["output_preview"]


# Exit 6 — simple-agent fast path unchanged.
@pytest.mark.asyncio
async def test_exit_6_stream_simple_agent_path_unchanged(make_container, stub_llm) -> None:
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"

    async def _stream_gen():
        yield "hello "
        yield "world"

    llm.stream = MagicMock(return_value=_stream_gen())
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("simple", container=c)
    events = [e async for e in agent.stream("test")]
    kinds = [e.kind for e in events]
    assert all(k in ("token", "done") for k in kinds)
    assert kinds[-1] == "done"


# Exit 7 — done is always last; StopAsyncIteration on next.
@pytest.mark.asyncio
async def test_exit_7_done_is_last(make_container, stub_llm) -> None:
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [{"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "go"}}]
            ),
            _llm_text("helper"),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    stream = lead.stream("test")
    events: list[AgentEvent] = []
    async for ev in stream:
        events.append(ev)
    assert events[-1].kind == "done"


# Exit 8 — Agent.stream works with AgentRef (Phase 4b).
@pytest.mark.asyncio
async def test_exit_8_stream_with_agent_ref(make_container, stub_llm, fresh_env) -> None:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "lead")
    server_c = make_container(stub_llm(content="from remote"))
    remote = Agent("remote_helper", description="r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(remote, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)

        llm = MagicMock(spec=ILLMProvider)
        llm.name = "anthropic"
        llm.chat = AsyncMock(
            side_effect=[
                _llm_tool_use(
                    [{"id": "t1", "name": "delegate_to_remote_helper", "arguments": {"task": "x"}}]
                ),
                _llm_text("done"),
            ]
        )
        caller_c = make_container(llm)
        caller_c.bind(ILLMProviderProto, llm)
        caller_c.bind(KeyProvider, server_c.resolve(KeyProvider))  # type: ignore[type-abstract]

        lead = Agent("lead", instructions="", delegates=[ref], container=caller_c)
        events = [e async for e in lead.stream("test")]

    kinds = [e.kind for e in events]
    assert "delegation_started" in kinds and "delegation_finished" in kinds


# Exit 9 — Phase 3 regression.
def test_exit_9_phase3_regression() -> None:
    import tests.test_phase3_exit as p3

    for name in (
        "test_delegation_chain_populates",
        "test_cascade_pass_returns_sas_no_escalation",
    ):
        assert hasattr(p3, name), f"Phase 3 exit {name!r} missing"


# Exit 10 — Phase 3.5 regression.
def test_exit_10_phase35_regression() -> None:
    import tests.test_phase35_exit as p35

    for name in (
        "test_exit_1_validator_protocol_passes_task",
        "test_exit_4_judge_cost_in_agent_result",
    ):
        assert hasattr(p35, name), f"Phase 3.5 exit {name!r} missing"


# Exit 11 — Phase 4a regression.
def test_exit_11_phase4a_regression() -> None:
    import tests.test_phase4a_exit as p4a

    for name in (
        "test_exit_1_agent_satisfies_idelegate",
        "test_exit_3_local_agent_delegation_e2e",
    ):
        assert hasattr(p4a, name), f"Phase 4a exit {name!r} missing"


# Exit 12 — Phase 4b regression.
def test_exit_12_phase4b_regression() -> None:
    import tests.test_phase4b_exit as p4b

    for name in (
        "test_exit_3_jsonrpc_happy_path",
        "test_exit_6_agent_ref_round_trip",
        "test_exit_7_mixed_delegates",
    ):
        assert hasattr(p4b, name), f"Phase 4b exit {name!r} missing"
