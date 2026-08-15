"""Locks IToolExecutor Protocol + container binding (v1.1.0 F1)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from voussoir.executors import IToolExecutor, StandardExecutor
from voussoir.tools import Tool, ToolContext


def test_itoolexecutor_is_runtime_checkable():
    assert isinstance(StandardExecutor(), IToolExecutor)


def test_back_compat_tool_executor_alias():
    """ToolExecutor (old name) still importable + identical to IToolExecutor."""
    from voussoir.executors import ToolExecutor

    assert ToolExecutor is IToolExecutor


def test_custom_executor_satisfies_protocol():
    class _CustomExec:
        name = "custom"

        async def invoke(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> Any:
            return "custom-output"

    assert isinstance(_CustomExec(), IToolExecutor)


def test_default_container_binds_itoolexecutor():
    from voussoir.container.defaults import default_container

    c = default_container()
    resolved = c.resolve(IToolExecutor)
    assert isinstance(resolved, StandardExecutor)


def test_default_container_freezes_itoolexecutor():
    """v1.1.0 F1: hostile plugin can't rebind IToolExecutor."""
    from voussoir.container.defaults import default_container

    c = default_container()

    class _Hostile:
        name = "hostile"

        async def invoke(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> Any:
            return "hostile"

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(IToolExecutor, _Hostile())
