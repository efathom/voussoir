"""Phase 4.5a — plugin entry-point allow-list (P0 #4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from structlog.testing import capture_logs

from voussoir.agent.plugins import ENTRY_POINT_GROUP, load_delegate_plugins
from voussoir.agent.registry import AgentRegistry


@dataclass
class _FakeEP:
    name: str
    loader: Callable[[], Any]

    def load(self) -> Any:
        return self.loader()


def _patch_eps(monkeypatch, eps: list[_FakeEP]) -> None:
    import voussoir.agent.plugins as plugins_mod

    def fake(*, group: str | None = None):
        return eps if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr(plugins_mod, "entry_points", fake)


def test_allowed_names_none_loads_all_with_info_log(
    make_container, stub_llm, monkeypatch, make_mini_delegate
) -> None:
    """Phase 4.5a default: allowed_names=None loads every plugin and emits
    an INFO log per loaded plugin so operators can audit what got loaded."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("ep_a", lambda: (lambda c: make_mini_delegate("a"))),
            _FakeEP("ep_b", lambda: (lambda c: make_mini_delegate("b"))),
        ],
    )
    c = make_container(stub_llm())
    registry = AgentRegistry()
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)
    assert sorted(out) == ["a", "b"]
    info_events = [log for log in cap_logs if log.get("event") == "plugin_loaded"]
    assert len(info_events) == 2
    assert {log["delegate_name"] for log in info_events} == {"a", "b"}


def test_allowed_names_set_filters_to_listed_plugins_only(
    make_container, stub_llm, monkeypatch, make_mini_delegate
) -> None:
    """allowed_names={'a'} loads 'a' but skips 'b' with a warning."""
    _patch_eps(
        monkeypatch,
        [
            _FakeEP("ep_a", lambda: (lambda c: make_mini_delegate("a"))),
            _FakeEP("ep_b", lambda: (lambda c: make_mini_delegate("b"))),
        ],
    )
    c = make_container(stub_llm())
    registry = AgentRegistry()
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry, allowed_names={"a"})
    assert out == ["a"]
    assert registry.has("a")
    assert not registry.has("b")
    skipped = [log for log in cap_logs if log.get("event") == "plugin_not_in_allowed_names"]
    assert any(log["delegate_name"] == "b" for log in skipped)


def test_allowed_names_empty_set_loads_nothing(
    make_container, stub_llm, monkeypatch, make_mini_delegate
) -> None:
    """allowed_names=set() is a strict deny-all."""
    _patch_eps(monkeypatch, [_FakeEP("ep_a", lambda: (lambda c: make_mini_delegate("a")))])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    out = load_delegate_plugins(c, registry=registry, allowed_names=set())
    assert out == []
    assert not registry.has("a")
