"""Tests for ToolContext.interrupt -- the default impl raises InterruptRequest (v1.2 G2)."""

from __future__ import annotations

from typing import Any

import pytest

from voussoir import IInterruptable, InterruptRequest
from voussoir.tools.protocol import ToolContext


def make_ctx() -> ToolContext:
    return ToolContext(run_id="r1", span_id="s1")


@pytest.mark.asyncio
async def test_tool_context_interrupt_raises_interrupt_request() -> None:
    ctx = make_ctx()
    with pytest.raises(InterruptRequest) as exc_info:
        await ctx.interrupt("needs_review", {"item": 42})
    assert exc_info.value.kind == "needs_review"
    assert exc_info.value.payload == {"item": 42}


@pytest.mark.asyncio
async def test_tool_context_interrupt_passes_payload_through_unchanged() -> None:
    ctx = make_ctx()
    payload: dict[str, Any] = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    with pytest.raises(InterruptRequest) as exc_info:
        await ctx.interrupt("x", payload)
    assert exc_info.value.payload == payload


def test_tool_context_satisfies_iinterruptable_protocol() -> None:
    ctx = make_ctx()
    assert isinstance(ctx, IInterruptable)


@pytest.mark.asyncio
async def test_tool_context_interrupt_accepts_empty_payload() -> None:
    ctx = make_ctx()
    with pytest.raises(InterruptRequest) as exc_info:
        await ctx.interrupt("ping", {})
    assert exc_info.value.kind == "ping"
    assert exc_info.value.payload == {}
