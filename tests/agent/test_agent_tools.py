from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Container
from voussoir.agent import Agent
from voussoir.agent.policy import AgentPolicy
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.protocols import IMemoryStore, ISessionStore
from voussoir.tools import Capability, tool


@tool(capability=Capability.READ_PUBLIC)
async def echo(text: str) -> str:
    """Echo the input."""
    return f"echoed: {text}"


@tool(capability=Capability.READ_PUBLIC)
async def upper(text: str) -> str:
    """Return uppercased text."""
    return text.upper()


def _container_with_tool_calling_llm() -> Container:
    """LLM that calls `echo` once, then returns a final answer."""
    p = MagicMock(spec=ILLMProvider)
    p.name = "anthropic"
    responses = [
        LLMResponse(
            content="",
            model="stub",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            finish_reason="tool_use",
            raw_response={
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "name": "echo",
                        "arguments": {"text": "hi"},
                    }
                ]
            },
        ),
        LLMResponse(
            content="final: echoed: hi",
            model="stub",
            input_tokens=20,
            output_tokens=8,
            total_tokens=28,
            finish_reason="end_turn",
        ),
    ]
    p.chat = AsyncMock(side_effect=responses)
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    return c


async def test_agent_invokes_tool_then_returns_final():
    agent = Agent(name="t", tools=[echo], container=_container_with_tool_calling_llm())
    result = await agent.run("ask")
    assert result.output == "final: echoed: hi"
    assert any(s.kind == "tool_call" and s.name == "echo" for s in result.steps)
    # Aggregate tokens summed across both LLM calls:
    assert result.tokens_in == 30
    assert result.tokens_out == 13


async def test_agent_multi_turn_tool_dialogue():
    """3-turn dialogue: tool A → tool B → final answer.

    Verifies that across multiple tool-call round-trips, (1) each tool runs,
    (2) the accumulating messages list carries prior assistant + function
    turns into the next LLM call, and (3) tokens/steps aggregate correctly.
    """
    p = MagicMock(spec=ILLMProvider)
    p.name = "anthropic"
    responses = [
        # Turn 1: ask for echo
        LLMResponse(
            content="",
            model="stub",
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            finish_reason="tool_use",
            raw_response={
                "tool_calls": [{"id": "tc_1", "name": "echo", "arguments": {"text": "first"}}]
            },
        ),
        # Turn 2: ask for upper (different tool, simulating chained reasoning)
        LLMResponse(
            content="",
            model="stub",
            input_tokens=12,
            output_tokens=5,
            total_tokens=17,
            finish_reason="tool_use",
            raw_response={
                "tool_calls": [
                    {"id": "tc_2", "name": "upper", "arguments": {"text": "echoed: first"}}
                ]
            },
        ),
        # Turn 3: final
        LLMResponse(
            content="all done: ECHOED: FIRST",
            model="stub",
            input_tokens=14,
            output_tokens=7,
            total_tokens=21,
            finish_reason="end_turn",
        ),
    ]
    p.chat = AsyncMock(side_effect=responses)
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())

    agent = Agent(name="multi", tools=[echo, upper], container=c)
    result = await agent.run("kick it off")

    # All three LLM calls happened.
    assert p.chat.call_count == 3
    assert result.finish_reason == "completed"
    assert result.output == "all done: ECHOED: FIRST"

    # Both tools ran (in order).
    tool_steps = [s for s in result.steps if s.kind == "tool_call"]
    assert [s.name for s in tool_steps] == ["echo", "upper"]

    # Tokens summed across all 3 LLM calls (10+12+14, 4+5+7).
    assert result.tokens_in == 36
    assert result.tokens_out == 16

    # Verify the message accumulation: turn-3's `messages` includes the
    # turn-1 + turn-2 assistant/function pairs, not just the original user input.
    turn3_messages = p.chat.call_args_list[2].kwargs["messages"]
    roles = [m.role for m in turn3_messages]
    # Expected sequence: [user, assistant(tc_1), function(echo), assistant(tc_2), function(upper)]
    assert roles == ["user", "assistant", "function", "assistant", "function"]
    # The two function messages carry the actual tool outputs back to the LLM.
    function_msgs = [m for m in turn3_messages if m.role == "function"]
    assert function_msgs[0].content == "echoed: first"
    assert function_msgs[1].content == "ECHOED: FIRST"


async def test_agent_max_cost_reports_max_cost_finish_reason():
    """Budget breach by cost should surface as finish_reason='max_cost', not collapsed to max_steps."""
    p = MagicMock(spec=ILLMProvider)
    p.name = "anthropic"
    # 4000 output tokens at claude-opus-4-7 pricing ($75/1M) = $0.30, which
    # breaches the $0.10 cap. Stays under max_output_tokens (default 8_000) so
    # the cost rule is the one that fires.
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="stub",
            input_tokens=0,
            output_tokens=4_000,
            total_tokens=4_000,
            finish_reason="tool_use",
            raw_response={
                "tool_calls": [{"id": "x", "name": "echo", "arguments": {"text": "loop"}}]
            },
        )
    )
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    agent = Agent(
        name="costly",
        # Priced in voussoir.llm.pricing: claude-opus-4-7 output is $75/1M,
        # so 4000 output tokens → $0.30 > $0.10 cap. Stays under
        # max_output_tokens (default 8_000) so the cost rule fires, not tokens.
        model="claude-opus-4-7",
        tools=[echo],
        policy=AgentPolicy(max_cost_usd=0.10, max_steps=100),
        container=c,
    )
    result = await agent.run("go")
    assert result.finish_reason == "max_cost"


async def test_agent_max_steps_short_circuits():
    """LLM keeps requesting tool calls; policy stops at max_steps."""
    p = MagicMock(spec=ILLMProvider)
    p.name = "anthropic"
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="stub",
            input_tokens=5,
            output_tokens=2,
            total_tokens=7,
            finish_reason="tool_use",
            raw_response={
                "tool_calls": [
                    {
                        "id": "tc_loop",
                        "name": "echo",
                        "arguments": {"text": "loop"},
                    }
                ]
            },
        )
    )
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    agent = Agent(
        name="loopy",
        tools=[echo],
        policy=AgentPolicy(max_steps=3, on_violation="summarize_and_stop"),
        container=c,
    )
    result = await agent.run("go")
    assert result.finish_reason == "max_steps"


async def test_agent_with_tools_and_unsupported_provider_raises_clear_error():
    """F7: adapter_for raises a clear NotImplementedError for providers not yet
    supported. The error names the failing provider and lists supported ones.
    """
    p = MagicMock(spec=ILLMProvider)
    p.name = "cohere"
    p.chat = AsyncMock()
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())

    agent = Agent(name="cross-provider", tools=[echo], container=c)
    with pytest.raises(NotImplementedError) as excinfo:
        await agent.run("go")
    msg = str(excinfo.value).lower()
    assert "cohere" in msg
    assert "anthropic" in msg
    assert "openai" in msg
    assert "openrouter" in msg


async def test_agent_without_tools_works_on_non_anthropic_provider():
    """A toolless agent must run end-to-end against any provider — plain
    chat completion is provider-agnostic. Tool-calling is what required
    a per-provider adapter (v1.1.0 F5-F7)."""
    p = MagicMock(spec=ILLMProvider)
    p.name = "openai"
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content="hello from openai",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            finish_reason="end_turn",
        )
    )
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())

    agent = Agent(name="toolless", container=c)
    result = await agent.run("hi")
    assert result.output == "hello from openai"
