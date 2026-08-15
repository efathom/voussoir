"""Phase 4.5a — dispatch_tool_calls cancels in-flight tasks on caller cancel."""

from __future__ import annotations

import asyncio

import pytest

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import dispatch_tool_calls
from voussoir.executors.standard import StandardExecutor
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability
from voussoir.tools.registry import ToolRegistry


@tool(capability=Capability.READ_PRIVATE, name="slow", description="long sleep")
async def _slow(text: str) -> str:
    await asyncio.sleep(1.0)  # long enough that we cancel before it finishes
    return f"slept:{text}"


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register_many([_slow])
    return r


@pytest.mark.asyncio
async def test_cancel_propagates_to_dispatch(make_container, stub_llm) -> None:
    """Cancelling the caller of dispatch_tool_calls causes CancelledError
    to propagate out (not be swallowed)."""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        task = asyncio.create_task(
            dispatch_tool_calls(
                [
                    {"id": "tA", "name": "slow", "arguments": {"text": "A"}},
                    {"id": "tB", "name": "slow", "arguments": {"text": "B"}},
                ],
                registry=_registry(),
                executor=StandardExecutor(),
                ctx=ctx,
            )
        )
        await asyncio.sleep(0.1)  # let dispatch begin
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_no_pending_tasks_remain_after_cancel(make_container, stub_llm) -> None:
    """After dispatch_tool_calls is cancelled, no inner _dispatch_one tasks
    remain in asyncio.all_tasks()."""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        task = asyncio.create_task(
            dispatch_tool_calls(
                [
                    {"id": "t1", "name": "slow", "arguments": {"text": "1"}},
                    {"id": "t2", "name": "slow", "arguments": {"text": "2"}},
                ],
                registry=_registry(),
                executor=StandardExecutor(),
                ctx=ctx,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Give the event loop a tick to drain cancelled subtasks.
        await asyncio.sleep(0.05)
        current = asyncio.current_task()
        leaked = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        assert leaked == [], f"leaked tasks: {leaked}"
