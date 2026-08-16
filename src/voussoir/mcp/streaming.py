"""MCP progress notification → AgentEvent.tool_progress adapter.

Phase 2.5 surfaces progress notifications from MCP `tools/call` as
streaming AgentEvents. The Agent.stream() integration that wires this
into the run loop is a Phase 3 follow-up — Phase 2 only ships the
adapter and proves the callback fires.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mcp import ClientSession

from voussoir.agent.result import AgentEvent

OnEvent = Callable[[AgentEvent], None]


def progress_to_agent_event(
    *,
    tool_name: str,
    progress: float,
    total: float | None,
    message: str | None,
    span_id: str,
) -> AgentEvent:
    """Build a `tool_progress` AgentEvent from raw MCP progress fields."""
    return AgentEvent(
        kind="tool_progress",
        payload={
            "tool": tool_name,
            "progress": progress,
            "total": total,
            "message": message,
        },
        span_id=span_id,
        timestamp=datetime.now(UTC),
    )


async def run_with_progress(
    *,
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
    on_event: OnEvent,
    span_id: str,
) -> str:
    """Call an MCP tool with a progress callback that emits AgentEvents.

    Returns the stringified text content of the call result. Errors raise.
    """

    async def _cb(progress: float, total: float | None, message: str | None) -> None:
        on_event(
            progress_to_agent_event(
                tool_name=tool_name,
                progress=progress,
                total=total,
                message=message,
                span_id=span_id,
            )
        )

    result = await session.call_tool(tool_name, arguments=arguments, progress_callback=_cb)
    if result.is_error:
        raise RuntimeError(f"MCP tool {tool_name} failed")
    # The Union (TextContent | ImageContent | AudioContent | ResourceLink |
    # EmbeddedResource) only has `.text` on TextContent — narrow with getattr
    # so mypy doesn't trip on the union.
    parts: list[str] = []
    for b in result.content:
        if getattr(b, "type", None) == "text":
            text = getattr(b, "text", None)
            if text is not None:
                parts.append(text)
    return "".join(parts)
