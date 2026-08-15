from unittest.mock import MagicMock

from ctxforge.protocols.llm import ILLMProvider

from voussoir import Container
from voussoir.agent import Agent, AgentEvent
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.protocols import IMemoryStore, ISessionStore


async def _streaming_chunks(*texts: str):
    for t in texts:
        yield t


def _streaming_llm(*chunks: str) -> ILLMProvider:
    p = MagicMock(spec=ILLMProvider)
    p.stream = MagicMock(return_value=_streaming_chunks(*chunks))
    return p


def _container(llm: ILLMProvider) -> Container:
    c = Container()
    c.bind(ILLMProviderProto, llm)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    return c


async def test_stream_with_delegates_now_emits_delegation_events():
    """Phase 4c lifts the prior Tranche 3.5a limitation: streaming with
    delegates DOES emit delegation_started / delegation_finished events.
    Kept here (rather than only in test_streaming_delegation.py) as the
    explicit regression marker showing the contract changed at Phase 4c."""
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="tool_use",
                raw_response={
                    "tool_calls": [
                        {"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "go"}}
                    ]
                },
            ),
            LLMResponse(
                content="helper",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
            LLMResponse(
                content="done",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
        ]
    )
    helper = Agent(name="helper", container=_container(llm))
    agent = Agent(name="lead", container=_container(llm), delegates=[helper])
    events: list[AgentEvent] = []
    async for ev in agent.stream("hi"):
        events.append(ev)
    kinds = [e.kind for e in events]
    assert "delegation_started" in kinds
    assert "delegation_finished" in kinds
    assert kinds[-1] == "done"


async def test_stream_yields_token_events_then_done():
    llm = _streaming_llm("hello", " ", "world")
    agent = Agent(name="t", container=_container(llm))
    events: list[AgentEvent] = []
    async for ev in agent.stream("hi"):
        events.append(ev)
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["token", "token", "token"]
    assert kinds[-1] == "done"
    assert events[0].payload == {"text": "hello"}
    assert events[1].payload == {"text": " "}
    assert events[2].payload == {"text": "world"}


async def test_stream_done_event_includes_full_output():
    llm = _streaming_llm("a", "b")
    agent = Agent(name="t", container=_container(llm))
    events = [e async for e in agent.stream("x")]
    done = events[-1]
    assert done.kind == "done"
    assert done.payload["output"] == "ab"
