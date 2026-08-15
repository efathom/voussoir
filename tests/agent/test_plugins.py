"""Unit tests for voussoir.agent.plugins.load_delegate_plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from structlog.testing import capture_logs

from voussoir.agent.plugins import ENTRY_POINT_GROUP, load_delegate_plugins
from voussoir.agent.registry import AgentRegistry

if TYPE_CHECKING:
    from voussoir.agent.context import AgentContext
    from voussoir.agent.result import AgentResult


@dataclass
class _FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint."""

    name: str
    loader: Callable[[], Any]

    def load(self) -> Any:
        return self.loader()


def _patch_entry_points(monkeypatch, eps: list[_FakeEntryPoint]) -> None:
    """Patch importlib.metadata.entry_points to return our fake list."""
    import importlib.metadata as md

    def fake_entry_points(*, group: str | None = None) -> list[_FakeEntryPoint]:
        if group == ENTRY_POINT_GROUP:
            return eps
        return []

    monkeypatch.setattr(md, "entry_points", fake_entry_points)
    import voussoir.agent.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "entry_points", fake_entry_points)


class _MiniDelegate:
    name = "mini"
    description = ""

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        raise NotImplementedError


def _make_mini(c) -> _MiniDelegate:
    return _MiniDelegate()


def test_entry_point_group_constant_value() -> None:
    assert ENTRY_POINT_GROUP == "voussoir.delegates"


def test_load_returns_empty_when_no_entry_points(make_container, stub_llm, monkeypatch) -> None:
    _patch_entry_points(monkeypatch, [])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    out = load_delegate_plugins(c, registry=registry)
    assert out == []
    assert registry.names() == []


def test_load_registers_factory_returned_idelegate(make_container, stub_llm, monkeypatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="ep1", loader=lambda: _make_mini)])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    out = load_delegate_plugins(c, registry=registry)
    assert out == ["mini"]
    assert registry.has("mini")


def test_load_registers_factory_returned_agent(make_container, stub_llm, monkeypatch) -> None:
    """Sanity: a factory returning a real Agent also lands in the registry."""
    from voussoir.agent.agent import Agent

    def make_agent(c):
        return Agent("plug_agent", description="d", container=c)

    _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="ep1", loader=lambda: make_agent)])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    out = load_delegate_plugins(c, registry=registry)
    assert out == ["plug_agent"]
    assert registry.has("plug_agent")


def test_load_skips_factory_that_raises_during_load(make_container, stub_llm, monkeypatch) -> None:
    """ep.load() raising any Exception → warning logged, skip; no raise out."""

    def bad_load():
        raise ImportError("plugin package broken")

    _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="bad_ep", loader=bad_load)])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)
    assert out == []
    assert any(log.get("entry_point") == "bad_ep" for log in cap_logs)


def test_load_skips_factory_that_raises_during_call(make_container, stub_llm, monkeypatch) -> None:
    """factory(c) raising → warning logged, skip; subsequent plugins still load."""

    def bad_factory(c):
        raise RuntimeError("DI miss")

    def good_factory(c):
        return _MiniDelegate()

    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(name="bad_ep", loader=lambda: bad_factory),
            _FakeEntryPoint(name="good_ep", loader=lambda: good_factory),
        ],
    )
    c = make_container(stub_llm())
    registry = AgentRegistry()
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)
    assert out == ["mini"]  # only the good one
    assert registry.has("mini")
    assert any(log.get("entry_point") == "bad_ep" for log in cap_logs)


def test_load_skips_non_idelegate_return(make_container, stub_llm, monkeypatch) -> None:
    """Factory returning a non-IDelegate → warning, skip."""

    def bad_return(c):
        return 42  # not an IDelegate

    _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="weird_ep", loader=lambda: bad_return)])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)
    assert out == []
    assert any(
        log.get("entry_point") == "weird_ep" and log.get("event") == "plugin_returned_non_idelegate"
        for log in cap_logs
    )


def test_load_skips_on_name_collision(make_container, stub_llm, monkeypatch) -> None:
    """Pre-registered name → plugin loses; existing entry preserved."""

    pre_existing = _MiniDelegate()
    registry = AgentRegistry()
    registry.add(pre_existing)

    _patch_entry_points(monkeypatch, [_FakeEntryPoint(name="dup_ep", loader=lambda: _make_mini)])
    c = make_container(stub_llm())
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)
    assert out == []
    assert registry.get("mini") is pre_existing  # existing entry survived
    assert any(log.get("event") == "plugin_name_collides" for log in cap_logs)


def test_top_level_reexports() -> None:
    """Phase 4d public surface is reachable from voussoir.agent."""
    import voussoir.agent as agent_pkg

    assert agent_pkg.ENTRY_POINT_GROUP == "voussoir.delegates"
    assert callable(agent_pkg.load_delegate_plugins)


def test_load_continues_after_one_plugin_fails(make_container, stub_llm, monkeypatch) -> None:
    """Two entry points; first ep.load raises, second succeeds → second loads."""

    def bad_load():
        raise SyntaxError("boom")

    def good_factory(c):
        return _MiniDelegate()

    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(name="bad_ep", loader=bad_load),
            _FakeEntryPoint(name="good_ep", loader=lambda: good_factory),
        ],
    )
    c = make_container(stub_llm())
    registry = AgentRegistry()
    out = load_delegate_plugins(c, registry=registry)
    assert out == ["mini"]
    assert registry.has("mini")
