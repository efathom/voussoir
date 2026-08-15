"""Locks @tool requires explicit capability= (v1.0.2 D3)."""

from __future__ import annotations

import pytest

from voussoir.tools import Capability, tool


def test_tool_without_capability_raises():
    """@tool() without capability= raises ValueError (NONE is a security footgun)."""
    with pytest.raises(ValueError, match="capability="):

        @tool(name="missing_cap")  # type: ignore[call-overload]
        async def t() -> str:
            return "ok"


def test_tool_with_explicit_capability_works():
    @tool(capability=Capability.READ_PUBLIC, name="explicit_cap")
    async def t() -> str:
        return "ok"

    assert t.capability == Capability.READ_PUBLIC


def test_tool_with_none_capability_raises():
    """Explicit Capability.NONE is still a footgun -> raise."""
    with pytest.raises(ValueError, match="capability="):

        @tool(capability=Capability.NONE, name="none_cap")
        async def t() -> str:
            return "ok"
