"""End-to-end: plugin-registered delegate participates in Agent.run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent.agent import Agent
from voussoir.agent.bootstrap import bind_agent_registry
from voussoir.agent.plugins import ENTRY_POINT_GROUP
from voussoir.protocols import ILLMProvider as ILLMProviderProto


@dataclass
class _FakeEntryPoint:
    name: str
    loader: Callable[[], Any]

    def load(self) -> Any:
        return self.loader()


def _patch_entry_points(monkeypatch, eps: list[_FakeEntryPoint]) -> None:
    import importlib.metadata as md

    import voussoir.agent.plugins as plugins_mod

    def fake_entry_points(*, group: str | None = None):
        return eps if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr(md, "entry_points", fake_entry_points)
    monkeypatch.setattr(plugins_mod, "entry_points", fake_entry_points)


def _llm_tool_use(tool_calls: list[dict], content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_use",
        raw_response={"tool_calls": tool_calls},
    )


def _llm_text(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="end_turn",
        raw_response=None,
    )


@pytest.mark.asyncio
async def test_plugin_delegate_participates_in_agent_run(
    make_container, stub_llm, monkeypatch
) -> None:
    """End-to-end: entry-point → factory → registry → NamedDelegate → run."""

    # Lead LLM: tool_use turn (delegate) + sub-agent chat + final text turn.
    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"
    lead_llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use(
                [{"id": "t1", "name": "delegate_to_plugin_helper", "arguments": {"task": "do it"}}]
            ),
            _llm_text("plugin helper finished"),
            _llm_text("done"),
        ]
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)

    def make_plugin_helper(container):
        return Agent("plugin_helper", description="from plugin", container=container)

    _patch_entry_points(
        monkeypatch, [_FakeEntryPoint(name="ep1", loader=lambda: make_plugin_helper)]
    )
    bind_agent_registry(c, load_plugins=True)

    lead = Agent("lead", instructions="", delegates=["plugin_helper"], container=c)
    result = await lead.run("test")

    assert result.output == "done"
    assert "plugin_helper" in result.delegation_chain
    delegation_steps = [s for s in result.steps if s.kind == "delegation"]
    assert len(delegation_steps) == 1
    assert delegation_steps[0].name == "plugin_helper"


@pytest.mark.asyncio
async def test_load_plugins_default_is_true(make_container, stub_llm, monkeypatch) -> None:
    """Phase 4.5a P1 #22: bind_agent_registry now defaults load_plugins=True;
    the allow-list gating (allowed_plugin_names) makes the new default safe."""

    called_with: dict = {}

    def fake_load_delegate_plugins(c, *, registry, allowed_names=None):
        called_with["allowed_names"] = allowed_names
        return []

    import voussoir.agent.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "load_delegate_plugins", fake_load_delegate_plugins)
    c = make_container(stub_llm())
    bind_agent_registry(c)  # default load_plugins=True
    assert "allowed_names" in called_with  # loader was invoked
    assert called_with["allowed_names"] is None  # default allow-all


@pytest.mark.asyncio
async def test_load_plugins_false_suppresses_loader(make_container, stub_llm, monkeypatch) -> None:
    """Explicit load_plugins=False skips the loader entirely."""

    called = False

    def fake_load_delegate_plugins(c, *, registry, allowed_names=None):
        nonlocal called
        called = True
        return []

    import voussoir.agent.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "load_delegate_plugins", fake_load_delegate_plugins)
    c = make_container(stub_llm())
    bind_agent_registry(c, load_plugins=False)
    assert called is False
