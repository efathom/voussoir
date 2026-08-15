"""Phase 4d — AgentRegistry now stores IDelegate, not only Agent.

This test locks the widening contract: a custom IDelegate impl (no Agent
inheritance) round-trips through registry.add / .get / .has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from voussoir.agent.registry import AgentRegistry

if TYPE_CHECKING:
    from voussoir.agent.context import AgentContext
    from voussoir.agent.result import AgentResult


class _MiniDelegate:
    """Minimal IDelegate implementation — no Agent inheritance."""

    name = "mini"
    description = "tiny standalone idelegate"

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        raise NotImplementedError("not exercised by this test")


def test_registry_accepts_non_agent_idelegate() -> None:
    r = AgentRegistry()
    d = _MiniDelegate()
    r.add(d)
    assert r.has("mini")
    assert r.get("mini") is d
    assert r.names() == ["mini"]


def test_registry_replace_widens_to_idelegate() -> None:
    r = AgentRegistry()
    r.add(_MiniDelegate())
    replacement = _MiniDelegate()
    r.replace(replacement)
    assert r.get("mini") is replacement


def test_registry_duplicate_add_raises() -> None:
    """Pre-existing behavior preserved: add() rejects duplicate names."""
    r = AgentRegistry()
    r.add(_MiniDelegate())
    with pytest.raises(ValueError, match="already registered"):
        r.add(_MiniDelegate())
