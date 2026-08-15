"""Phase 4.5a — _dispatch_one does NOT swallow KeyboardInterrupt (P1 #10)."""

from __future__ import annotations

import pytest

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import _dispatch_one
from voussoir.executors.standard import StandardExecutor
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability
from voussoir.tools.registry import ToolRegistry


@tool(capability=Capability.READ_PRIVATE, name="kbi", description="raises KBI")
async def _kbi(text: str) -> str:
    raise KeyboardInterrupt("user pressed Ctrl-C")


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register_many([_kbi])
    return r


@pytest.mark.asyncio
async def test_keyboardinterrupt_propagates_not_swallowed(make_container, stub_llm) -> None:
    """When a tool body raises KeyboardInterrupt, _dispatch_one re-raises
    rather than wrapping as TOOL_ERROR. (Pre-Phase-4.5a it caught
    BaseException including KeyboardInterrupt/SystemExit.)"""
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        ctx.allowed_capabilities = Capability.READ_PRIVATE
        with pytest.raises(KeyboardInterrupt):
            await _dispatch_one(
                {"id": "t1", "name": "kbi", "arguments": {"text": "x"}},
                registry=_registry(),
                executor=StandardExecutor(),
                ctx=ctx,
            )
