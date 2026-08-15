"""Phase 4d exit criteria — gates the v0.4.0d-phase4d tag."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from structlog.testing import capture_logs


# Exit 1 — load_delegate_plugins exists with the documented signature.
def test_exit_1_load_delegate_plugins_signature() -> None:
    from voussoir.agent.plugins import AgentRegistry, load_delegate_plugins
    from voussoir.container import Container

    sig = inspect.signature(load_delegate_plugins)
    params = sig.parameters
    # Phase 4.5a P0 #4 added the allowed_names kwarg; the original
    # (c, *, registry) contract is preserved as a prefix.
    assert list(params)[:2] == ["c", "registry"]
    assert params["c"].annotation in (Container, "Container")
    assert params["registry"].annotation in (AgentRegistry, "AgentRegistry")
    assert params["registry"].kind is inspect.Parameter.KEYWORD_ONLY


# Exit 2 — entry-point group constant.
def test_exit_2_entry_point_group_constant() -> None:
    from voussoir.agent.plugins import ENTRY_POINT_GROUP

    assert ENTRY_POINT_GROUP == "voussoir.delegates"


# Exit 3 — AgentRegistry accepts a non-Agent IDelegate.
def test_exit_3_registry_widened_to_idelegate() -> None:
    from voussoir.agent.registry import AgentRegistry

    class _MD:
        name = "md"
        description = ""

        async def delegate(self, task, *, parent_ctx):
            raise NotImplementedError

    r = AgentRegistry()
    r.add(_MD())  # widened: previously declared agent: Agent
    assert r.has("md")


# Exit 4 — bind_agent_registry kwarg default is True (Phase 4.5a P1 #22 flipped).
def test_exit_4_bind_agent_registry_load_plugins_default_true() -> None:
    from voussoir.agent.bootstrap import bind_agent_registry

    sig = inspect.signature(bind_agent_registry)
    assert "load_plugins" in sig.parameters
    assert sig.parameters["load_plugins"].default is True


@dataclass
class _FakeEP:
    name: str
    loader: Callable[[], Any]

    def load(self) -> Any:
        return self.loader()


def _patch_eps(monkeypatch, eps: list[_FakeEP]) -> None:
    import voussoir.agent.plugins as plugins_mod
    from voussoir.agent.plugins import ENTRY_POINT_GROUP

    def fake(*, group: str | None = None):
        return eps if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr(plugins_mod, "entry_points", fake)


# Exit 5 — failure semantics: loader never raises on factory raise.
def test_exit_5_factory_raise_does_not_propagate(make_container, stub_llm, monkeypatch) -> None:
    from voussoir.agent.plugins import load_delegate_plugins
    from voussoir.agent.registry import AgentRegistry

    def bad(c):
        raise RuntimeError("nope")

    _patch_eps(monkeypatch, [_FakeEP("bad", lambda: bad)])
    c = make_container(stub_llm())
    registry = AgentRegistry()
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)  # must not raise
    assert out == []
    assert any(log.get("entry_point") == "bad" for log in cap_logs)


# Exit 6 — collision: plugin loses; existing entry preserved.
def test_exit_6_collision_plugin_loses(make_container, stub_llm, monkeypatch) -> None:
    from voussoir.agent.plugins import load_delegate_plugins
    from voussoir.agent.registry import AgentRegistry

    class _MD:
        name = "dup"
        description = ""

        async def delegate(self, task, *, parent_ctx):
            raise NotImplementedError

    existing = _MD()
    registry = AgentRegistry()
    registry.add(existing)

    def factory(c):
        return _MD()  # new instance, same name "dup"

    _patch_eps(monkeypatch, [_FakeEP("dup_ep", lambda: factory)])
    c = make_container(stub_llm())
    with capture_logs() as cap_logs:
        out = load_delegate_plugins(c, registry=registry)
    assert out == []
    assert registry.get("dup") is existing
    assert any(log.get("event") == "plugin_name_collides" for log in cap_logs)


# Exit 7 — end-to-end: plugin delegate appears in lead.run delegation_chain.
@pytest.mark.asyncio
async def test_exit_7_e2e_plugin_in_delegation_chain(make_container, stub_llm, monkeypatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    from voussoir.agent.agent import Agent
    from voussoir.agent.bootstrap import bind_agent_registry
    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    def tool_use(tcs):
        return LLMResponse(
            content="",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="tool_use",
            raw_response={"tool_calls": tcs},
        )

    def text(content):
        return LLMResponse(
            content=content,
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
            raw_response=None,
        )

    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"
    lead_llm.chat = AsyncMock(
        side_effect=[
            tool_use([{"id": "t1", "name": "delegate_to_plug_helper", "arguments": {"task": "x"}}]),
            text("helper finished"),
            text("done"),
        ]
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)

    def make_plug_helper(container):
        return Agent("plug_helper", description="d", container=container)

    _patch_eps(monkeypatch, [_FakeEP("ep1", lambda: make_plug_helper)])
    bind_agent_registry(c, load_plugins=True)
    lead = Agent("lead", instructions="", delegates=["plug_helper"], container=c)
    result = await lead.run("test")
    assert "plug_helper" in result.delegation_chain


# Exit 8 — re-exports.
def test_exit_8_reexports() -> None:
    import voussoir.agent as agent_pkg

    assert agent_pkg.ENTRY_POINT_GROUP == "voussoir.delegates"
    assert callable(agent_pkg.load_delegate_plugins)


# Exit 9 — prior-phase exit suites still importable.
def test_exit_9_prior_phase_exit_suites_importable() -> None:
    import tests.test_phase4a_exit as p4a
    import tests.test_phase4b_exit as p4b
    import tests.test_phase4c_exit as p4c
    import tests.test_phase35_exit as p35

    for mod, names in (
        (p35, ("test_exit_1_validator_protocol_passes_task",)),
        (p4a, ("test_exit_1_agent_satisfies_idelegate", "test_exit_3_local_agent_delegation_e2e")),
        (p4b, ("test_exit_3_jsonrpc_happy_path", "test_exit_6_agent_ref_round_trip")),
        (p4c, ("test_exit_3_stream_emits_tool_events", "test_exit_4_stream_delegation_full_pair")),
    ):
        for name in names:
            assert hasattr(mod, name), f"{mod.__name__}.{name} missing"


# Exit 10 — example package layout.
def test_exit_10_example_package_layout() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples" / "05_delegate_plugin"
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "example_plugin" / "__init__.py").exists()
    assert (root / "README.md").exists()
    # Entry-point declaration in pyproject.toml.
    pyproj = (root / "pyproject.toml").read_text()
    assert '[project.entry-points."voussoir.delegates"]' in pyproj
    assert "example_plugin_agent" in pyproj
