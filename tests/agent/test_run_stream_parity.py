"""Phase 4.5b B4b — run() and stream() produce equivalent results on the same fixture.

Locks the contract that the two paths are behaviourally consistent after
the B4b helper extraction. Both paths now call the same `tool_turn` for
the per-turn body; this test catches regressions where one path's
surrounding logic drifts from the other.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent import Agent


def _streaming_chunks(text: str):
    """Async iterator that yields `text` as a single chunk — for the
    simple-agent fast path in Agent.stream, which uses llm.stream()."""

    async def _gen():
        yield text

    return _gen()


@pytest.mark.asyncio
async def test_run_and_stream_match_on_simple_task(make_container) -> None:
    """Same agent, same task: run() and stream() produce the same final
    output. With no tools or delegates, this exercises the simple-agent
    fast path in stream (llm.stream) and the bare chat path in run
    (llm.chat). The two paths use different LLM methods, so we wire both
    on the mock with matching content; the contract is that the FINAL
    OUTPUT is identical, not that internal calls match.
    """
    expected = "final answer"

    # Mock supporting both chat (for run) and stream (for stream).
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        return_value=LLMResponse(
            content=expected,
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
        )
    )
    llm.stream = MagicMock(return_value=_streaming_chunks(expected))

    c_run = make_container(llm)
    agent_run = Agent("x", container=c_run)
    result = await agent_run.run("hello")

    # Rebuild llm so stream() doesn't reuse the consumed generator.
    llm.stream = MagicMock(return_value=_streaming_chunks(expected))
    c_stream = make_container(llm)
    agent_stream = Agent("x", container=c_stream)
    events = []
    async for ev in agent_stream.stream("hello"):
        events.append(ev)

    assert result.output == expected
    done_events = [ev for ev in events if ev.kind == "done"]
    assert len(done_events) == 1
    assert done_events[0].payload["output"] == result.output
