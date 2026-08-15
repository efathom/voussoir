from datetime import datetime
from unittest.mock import MagicMock

from voussoir.agent.result import AgentEvent
from voussoir.mcp.streaming import progress_to_agent_event, run_with_progress


def test_progress_to_agent_event_builds_a_tool_progress_event():
    ev = progress_to_agent_event(
        tool_name="fake.echo",
        progress=0.5,
        total=1.0,
        message="halfway",
        span_id="span-1",
    )
    assert isinstance(ev, AgentEvent)
    assert ev.kind == "tool_progress"
    assert ev.payload["tool"] == "fake.echo"
    assert ev.payload["progress"] == 0.5
    assert ev.payload["total"] == 1.0
    assert ev.payload["message"] == "halfway"
    assert ev.span_id == "span-1"
    assert isinstance(ev.timestamp, datetime)


async def test_run_with_progress_invokes_callback_on_each_progress():
    """Wraps a session.call_tool that fires its progress_callback twice."""
    fake_session = MagicMock()

    async def fake_call_tool(name, arguments=None, progress_callback=None, **_kw):
        if progress_callback is not None:
            await progress_callback(0.25, 1.0, "starting")
            await progress_callback(0.75, 1.0, "almost there")
        result = MagicMock()
        result.isError = False
        result.content = [MagicMock(type="text", text="done")]
        return result

    fake_session.call_tool = fake_call_tool

    events: list[AgentEvent] = []
    text = await run_with_progress(
        session=fake_session,
        tool_name="search",
        arguments={"q": "x"},
        on_event=events.append,
        span_id="span-x",
    )

    assert text == "done"
    assert len(events) == 2
    assert all(e.kind == "tool_progress" for e in events)
    assert events[0].payload["progress"] == 0.25
    assert events[1].payload["progress"] == 0.75
