"""Agent.stream tool-call + delegation event sequencing (Phase 4c)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent.agent import Agent
from voussoir.agent.result import AgentEvent
from voussoir.protocols import ILLMProvider as ILLMProviderProto


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


def _make_chat_llm(*responses: LLMResponse) -> MagicMock:
    """Build a MagicMock provider with side_effect-driven chat. Note: no
    .stream method — tool-using tests don't exercise it."""
    p = MagicMock(spec=ILLMProvider)
    p.name = "anthropic"
    p.chat = AsyncMock(side_effect=list(responses))
    return p


async def _collect(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [ev async for ev in stream]


# --- Workstream B test set 1: tool-call event sequencing ---


@pytest.mark.asyncio
async def test_stream_with_tool_call_emits_started_finished(make_container, stub_llm) -> None:
    """delegate-free tool call: tool_started → tool_finished → token → done."""
    from voussoir.tools.decorator import tool
    from voussoir.tools.protocol import Capability

    @tool(capability=Capability.READ_PRIVATE, name="echo", description="echo")
    async def echo(text: str) -> str:
        return f"echo:{text}"

    llm = _make_chat_llm(
        _llm_tool_use([{"id": "t1", "name": "echo", "arguments": {"text": "hi"}}]),
        _llm_text("done"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[echo], container=c)
    events = await _collect(agent.stream("test"))
    kinds = [e.kind for e in events]
    assert "tool_started" in kinds
    assert "tool_finished" in kinds
    assert kinds[-1] == "done"
    # tool_started and tool_finished pair via tool_call_id.
    started = next(e for e in events if e.kind == "tool_started")
    finished = next(e for e in events if e.kind == "tool_finished")
    assert started.payload["tool_call_id"] == finished.payload["tool_call_id"]


@pytest.mark.asyncio
async def test_stream_with_delegation_emits_full_pair(make_container, stub_llm) -> None:
    """Successful delegation:
    tool_started → delegation_started → tool_finished → delegation_finished
    → token → done."""
    llm = _make_chat_llm(
        _llm_tool_use([{"id": "tH", "name": "delegate_to_helper", "arguments": {"task": "go"}}]),
        _llm_text("helper said hi"),
        _llm_text("done"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    events = await _collect(lead.stream("test"))
    kinds = [e.kind for e in events]
    # Required sequence subset (other token/done events may interleave).
    for k in ("tool_started", "delegation_started", "tool_finished", "delegation_finished"):
        assert k in kinds, f"missing {k} in {kinds}"
    # Pairing.
    ds = next(e for e in events if e.kind == "delegation_started")
    df = next(e for e in events if e.kind == "delegation_finished")
    assert ds.payload["delegate_name"] == "helper"
    assert df.payload["delegate_name"] == "helper"
    assert ds.payload["tool_call_id"] == df.payload["tool_call_id"]
    assert kinds[-1] == "done"


@pytest.mark.asyncio
async def test_stream_delegation_refused_no_delegation_finished(make_container, stub_llm) -> None:
    """Unknown name in delegates → DELEGATION_REFUSED. Event invariant:
    delegation_started fires, delegation_finished does NOT."""
    llm = _make_chat_llm(
        _llm_tool_use([{"id": "tU", "name": "delegate_to_unknown", "arguments": {"task": "x"}}]),
        _llm_text("saw refusal"),
    )
    from voussoir.agent.registry import AgentRegistry

    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(AgentRegistry, AgentRegistry())  # Phase 4.5a P1 #23
    lead = Agent("lead", instructions="", delegates=["unknown"], container=c)
    events = await _collect(lead.stream("test"))
    kinds = [e.kind for e in events]
    assert "delegation_started" in kinds
    assert "delegation_finished" not in kinds
    # The refusal message must appear in tool_finished's output_preview.
    tf = next(e for e in events if e.kind == "tool_finished")
    assert "DELEGATION_REFUSED" in tf.payload["output_preview"]


@pytest.mark.asyncio
async def test_stream_two_tool_calls_declared_order(make_container, stub_llm) -> None:
    """Two tool_use blocks in one turn: tool_started events in declared
    order, tool_finished events in declared order."""
    from voussoir.tools.decorator import tool
    from voussoir.tools.protocol import Capability

    @tool(capability=Capability.READ_PRIVATE, name="echo", description="echo")
    async def echo(text: str) -> str:
        return f"echo:{text}"

    llm = _make_chat_llm(
        _llm_tool_use(
            [
                {"id": "tA", "name": "echo", "arguments": {"text": "A"}},
                {"id": "tB", "name": "echo", "arguments": {"text": "B"}},
            ]
        ),
        _llm_text("both done"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[echo], container=c)
    events = await _collect(agent.stream("test"))
    started_ids = [e.payload["tool_call_id"] for e in events if e.kind == "tool_started"]
    finished_ids = [e.payload["tool_call_id"] for e in events if e.kind == "tool_finished"]
    assert started_ids == ["tA", "tB"]
    assert finished_ids == ["tA", "tB"]


@pytest.mark.asyncio
async def test_stream_done_is_last_event(make_container, stub_llm) -> None:
    """Across simple + tool-using paths, done is the last event."""
    # Simple agent.
    simple_llm = MagicMock(spec=ILLMProvider)
    simple_llm.name = "anthropic"

    async def _stream_gen():
        yield "hello "
        yield "world"

    simple_llm.stream = MagicMock(return_value=_stream_gen())
    c1 = make_container(simple_llm)
    c1.bind(ILLMProviderProto, simple_llm)
    simple = Agent("simple", container=c1)
    events_simple = await _collect(simple.stream("hi"))
    assert events_simple[-1].kind == "done"

    # Tool-using agent.
    llm = _make_chat_llm(_llm_text("just text"))
    c2 = make_container(llm)
    c2.bind(ILLMProviderProto, llm)
    from voussoir.tools.decorator import tool
    from voussoir.tools.protocol import Capability

    @tool(capability=Capability.READ_PRIVATE, name="t", description="t")
    async def t(x: str) -> str:
        return x

    tool_agent = Agent("tool_agent", tools=[t], container=c2)
    events_tool = await _collect(tool_agent.stream("hi"))
    assert events_tool[-1].kind == "done"


# --- Workstream B test set 2: invariants + remote AgentRef + tool errors ---


@pytest.mark.asyncio
async def test_stream_span_id_constant_across_events(make_container, stub_llm) -> None:
    llm = _make_chat_llm(
        _llm_tool_use([{"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "go"}}]),
        _llm_text("helper"),
        _llm_text("done"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    events = await _collect(lead.stream("test"))
    span_ids = {e.span_id for e in events}
    assert len(span_ids) == 1


@pytest.mark.asyncio
async def test_stream_timestamps_monotonic(make_container, stub_llm) -> None:
    llm = _make_chat_llm(
        _llm_tool_use([{"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "go"}}]),
        _llm_text("helper"),
        _llm_text("done"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    events = await _collect(lead.stream("test"))
    timestamps = [e.timestamp for e in events]
    for prev, cur in zip(timestamps, timestamps[1:], strict=False):
        assert cur >= prev, f"timestamps not monotonic: {prev} → {cur}"


@pytest.mark.asyncio
async def test_stream_token_only_for_simple_agent_path(make_container, stub_llm) -> None:
    """Simple agent (no tools, no delegates) — only token + done events."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"

    async def _stream_gen():
        yield "hello "
        yield "world"

    llm.stream = MagicMock(return_value=_stream_gen())
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("simple", container=c)
    events = await _collect(agent.stream("hi"))
    kinds = [e.kind for e in events]
    assert all(k in ("token", "done") for k in kinds), f"unexpected kinds: {kinds}"
    assert kinds.count("done") == 1
    assert kinds[-1] == "done"


@pytest.mark.asyncio
async def test_stream_token_event_per_assistant_turn_in_tool_path(make_container, stub_llm) -> None:
    """Tool-using path: each non-empty-content turn emits one token; the
    final turn's token immediately precedes `done`."""
    llm = _make_chat_llm(
        _llm_tool_use(
            [{"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "go"}}],
            content="thinking",
        ),
        _llm_text("helper"),
        _llm_text("final"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    events = await _collect(lead.stream("test"))
    # Two assistant turns produce two `token` events; final precedes done.
    kinds = [e.kind for e in events]
    token_indices = [i for i, k in enumerate(kinds) if k == "token"]
    assert len(token_indices) == 2
    assert kinds[token_indices[-1] + 1] == "done"


@pytest.mark.asyncio
async def test_stream_tool_error_in_finished_payload(make_container, stub_llm) -> None:
    """A tool that raises produces tool_finished with error set and
    output_preview containing TOOL_ERROR."""
    from voussoir.tools.decorator import tool
    from voussoir.tools.protocol import Capability

    @tool(capability=Capability.READ_PRIVATE, name="raise_tool", description="raises")
    async def raise_tool(text: str) -> str:
        raise ValueError(f"bad:{text}")

    llm = _make_chat_llm(
        _llm_tool_use([{"id": "t1", "name": "raise_tool", "arguments": {"text": "x"}}]),
        _llm_text("seen"),
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", instructions="", tools=[raise_tool], container=c)
    events = await _collect(agent.stream("test"))
    tf = next(e for e in events if e.kind == "tool_finished")
    assert "bad:x" in tf.payload["error"]
    assert "TOOL_ERROR" in tf.payload["output_preview"]


@pytest.mark.asyncio
async def test_stream_with_remote_agent_ref_delegate_emits_events(
    make_container, stub_llm, fresh_env
) -> None:
    """A Phase 4b AgentRef in lead.delegates streams the full delegation
    event pair, with delegate_name matching the remote agent's name."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router

    # Phase 4.5a P0 #2: lead's Agent.run sets agent_name="lead" which becomes
    # the JWT iss; the server's expected_issuers must include it.
    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "lead")
    # Server side: a remote helper.
    server_c = make_container(stub_llm(content="from remote"))
    remote_helper = Agent("remote_helper", description="r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(remote_helper, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)

        # Lead-side: tool-use turn + final.
        lead_llm = _make_chat_llm(
            _llm_tool_use(
                [{"id": "t1", "name": "delegate_to_remote_helper", "arguments": {"task": "x"}}]
            ),
            _llm_text("done"),
        )
        caller_c = make_container(lead_llm)
        caller_c.bind(ILLMProviderProto, lead_llm)
        caller_c.bind(KeyProvider, server_c.resolve(KeyProvider))  # type: ignore[type-abstract]

        lead = Agent("lead", instructions="", delegates=[ref], container=caller_c)
        events = await _collect(lead.stream("test"))

    kinds = [e.kind for e in events]
    assert "delegation_started" in kinds
    assert "delegation_finished" in kinds
    ds = next(e for e in events if e.kind == "delegation_started")
    df = next(e for e in events if e.kind == "delegation_finished")
    assert ds.payload["delegate_name"] == "remote_helper"
    assert df.payload["delegate_name"] == "remote_helper"
