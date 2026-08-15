"""Locks VOUSSOIR_AGENT_<NAME>_<FIELD> env-var override behavior (Phase 5 Task C8).

Reuses Phase 3.5's AgentBuilder.apply_config(...) path; the env-var loop runs
LAST in bind_agent_registry so it takes precedence over yaml + plugin
registration. Whitelisted fields: MODEL, TEMPERATURE, SYSTEM_PROMPT, MAX_STEPS,
MAX_CASCADE_DEPTH, ALLOWED_CAPABILITIES.
"""

from __future__ import annotations

import structlog

from voussoir.agent import bind_agent_registry, register_agent
from voussoir.agent.registry import AgentRegistry
from voussoir.tools import Capability


def _agent(c, *, name: str, model: str = "claude-opus-4-7"):
    from voussoir import Agent

    return Agent(name=name, container=c, model=model)


def test_env_override_changes_model(monkeypatch, make_container):
    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, _agent(c, name="my-agent"))
    monkeypatch.setenv("VOUSSOIR_AGENT_MY_AGENT_MODEL", "claude-haiku-4-5")
    bind_agent_registry(c, load_plugins=False)
    registry = c.resolve(AgentRegistry)
    rebuilt = registry.get("my-agent")
    assert rebuilt.model == "claude-haiku-4-5"


def test_env_override_temperature_parses_float(monkeypatch, make_container):
    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, _agent(c, name="t"))
    monkeypatch.setenv("VOUSSOIR_AGENT_T_TEMPERATURE", "0.42")
    bind_agent_registry(c, load_plugins=False)
    registry = c.resolve(AgentRegistry)
    assert registry.get("t").temperature == 0.42


def test_env_override_allowed_capabilities_parses_pipe_list(monkeypatch, make_container):
    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, _agent(c, name="c8"))
    monkeypatch.setenv("VOUSSOIR_AGENT_C8_ALLOWED_CAPABILITIES", "READ_PUBLIC|EXFILTRATION")
    bind_agent_registry(c, load_plugins=False)
    rebuilt = c.resolve(AgentRegistry).get("c8")
    assert Capability.READ_PUBLIC in rebuilt.allowed_capabilities
    assert Capability.EXFILTRATION in rebuilt.allowed_capabilities
    assert Capability.READ_PRIVATE not in rebuilt.allowed_capabilities


def test_env_override_unknown_field_logs_warning(monkeypatch, make_container):
    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, _agent(c, name="u"))
    monkeypatch.setenv("VOUSSOIR_AGENT_U_GIBBERISH", "x")
    with structlog.testing.capture_logs() as captured:
        bind_agent_registry(c, load_plugins=False)
    assert any("env_override_unknown_field" in str(r.get("event", "")) for r in captured)


def test_env_override_unknown_agent_logs_warning(monkeypatch, make_container):
    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, _agent(c, name="known"))
    monkeypatch.setenv("VOUSSOIR_AGENT_NEVER_REGISTERED_MODEL", "x")
    with structlog.testing.capture_logs() as captured:
        bind_agent_registry(c, load_plugins=False)
    assert any("env_override_unknown_agent" in str(r.get("event", "")) for r in captured)


def test_env_override_bad_value_logs_and_skips(monkeypatch, make_container):
    """A non-float temperature env value logs a warning and skips."""
    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, _agent(c, name="bad"))
    monkeypatch.setenv("VOUSSOIR_AGENT_BAD_TEMPERATURE", "not-a-float")
    with structlog.testing.capture_logs() as captured:
        bind_agent_registry(c, load_plugins=False)
    assert any("env_override_value_parse_failed" in str(r.get("event", "")) for r in captured)
    # Agent unchanged (still has default temperature).
    rebuilt = c.resolve(AgentRegistry).get("bad")
    assert rebuilt.temperature is None
