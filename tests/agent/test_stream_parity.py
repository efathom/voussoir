"""Locks 4 stream() parity bugs vs _run_normal (v1.0.2 D7).

1. while True has no policy.check budget gate -- infinite loop risk.
2. _run_setup called without skill_content -- streaming agents with skills
   silently get empty skill content in prompts.
3. except Exception (should be BaseException) -- asyncio.CancelledError +
   KeyboardInterrupt bypass on_error middleware fanout.
4. pseudo_result duplicated between cascade gate + after_run hook -- two
   AgentResult constructions for the same data.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent import Agent
from voussoir.agent.policy import AgentPolicy
from voussoir.tools import Capability, tool

# ------- Bug 1: stream respects max_steps budget -------


@tool(capability=Capability.READ_PUBLIC)
async def _no_op_tool() -> str:
    """Trivial tool used to keep the LLM in a tool-use loop."""
    return "ok"


@pytest.mark.asyncio
async def test_stream_respects_max_steps(make_container) -> None:
    """A tool-loop LLM that never emits finish_reason=stop must be capped
    by policy.max_steps. Before this fix `while True` looped forever.

    Wrap in asyncio.wait_for so a regression hangs the test with a clear
    timeout rather than wedging the whole suite.
    """
    # LLM that ALWAYS asks for a tool call, never stops.
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="tool_use",
            raw_response={"tool_calls": [{"id": "t1", "name": "_no_op_tool", "arguments": {}}]},
        )
    )

    container = make_container(llm)
    agent = Agent(
        name="loop",
        container=container,
        tools=[_no_op_tool],
        policy=AgentPolicy(max_steps=3),
    )

    async def _drain() -> int:
        n = 0
        async for _ev in agent.stream("hi"):
            n += 1
        return n

    # If the bug regresses, this will hang; the tight timeout converts it
    # into a clear failure mode.
    n = await asyncio.wait_for(_drain(), timeout=5.0)
    # Must complete (not infinite). Concrete cap: at most a handful of
    # llm.chat calls — policy.max_steps=3 plus startup/teardown events.
    assert n < 100, f"stream emitted {n} events for max_steps=3 -- looks unbounded"
    # And the LLM must have been called bounded by max_steps + 1.
    call_count = llm.chat.call_count
    assert (
        call_count <= 4
    ), f"llm.chat called {call_count} times for max_steps=3 -- budget gate missing"


# ------- Bug 2: _run_setup receives skill_content -------


def test_stream_calls_run_setup_with_skill_content() -> None:
    """stream() must pass skill_content= to _run_setup, like _run_normal.

    Without the kwarg, agents declaring `skills=[...]` get no skill text
    in their streaming-mode prompt. Static guarantee via AST inspection.
    """
    source = inspect.getsource(Agent.stream)
    assert "skill_content" in source, (
        "Agent.stream does not pass skill_content= to _run_setup; "
        "agents with skills get empty prompts in streaming mode."
    )


# ------- Bug 3: except BaseException, not Exception -------


def test_stream_catches_base_exception_not_just_exception() -> None:
    """stream's outer try must catch BaseException so asyncio.CancelledError
    + KeyboardInterrupt fire on_error middleware fanout.

    asyncio.CancelledError derives from BaseException (not Exception) in
    Python 3.8+, so `except Exception` silently skips the fanout.
    """
    source = inspect.getsource(Agent.stream)
    assert "except BaseException" in source, (
        "Agent.stream uses `except Exception`; asyncio.CancelledError + "
        "KeyboardInterrupt skip the on_error middleware fanout."
    )


# ------- Bug 4: pseudo_result deduped -------


def test_stream_pseudo_result_not_duplicated() -> None:
    """Agent.stream must construct the post-done AgentResult at most once.

    Before the fix the cascade gate built one AgentResult and the
    after_run hook fanout built a second identical one.
    """
    source = inspect.getsource(Agent.stream)
    count = source.count("AgentResult(") + source.count("AgentResult[str](")
    assert count <= 1, (
        f"Agent.stream constructs AgentResult in {count} places; "
        "the cascade gate + after_run hook should share one instance."
    )
