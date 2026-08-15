"""bind_agent_registry: yaml override, define-from-scratch, missing yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voussoir.agent.agent import Agent
from voussoir.agent.bootstrap import bind_agent_registry
from voussoir.agent.registry import AgentRegistry, register_agent


def test_bind_with_no_yaml_is_noop(make_container, stub_llm) -> None:
    """No yaml present + no path passed → no-op apart from binding an
    empty registry on the container."""
    c = make_container(stub_llm())
    bind_agent_registry(c)
    # An empty registry was created lazily; safe either way.
    if c.has(AgentRegistry):
        assert c.resolve(AgentRegistry).names() == []


def test_bind_creates_registry_if_missing(make_container, stub_llm, tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents:\n  alpha:\n    system_prompt: 'hi'\n")
    c = make_container(stub_llm())
    bind_agent_registry(c, config_path=p)
    assert c.has(AgentRegistry)


def test_yaml_define_from_scratch(make_container, stub_llm, tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  researcher:
    system_prompt: |
      Research things.
    model: claude-sonnet-4-6
"""
    )
    c = make_container(stub_llm())
    bind_agent_registry(c, config_path=p)
    a = c.resolve(AgentRegistry).get("researcher")
    assert a.instructions is not None and "Research things" in a.instructions
    assert a.model == "claude-sonnet-4-6"


def test_yaml_overrides_python_registered(make_container, stub_llm, tmp_path: Path) -> None:
    c = make_container(stub_llm())
    register_agent(c, Agent("alpha", instructions="original", container=c, model="m1"))
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents:\n  alpha:\n    model: m2\n")
    bind_agent_registry(c, config_path=p)
    a = c.resolve(AgentRegistry).get("alpha")
    assert a.instructions == "original"  # untouched
    assert a.model == "m2"  # overridden


def test_yaml_define_without_system_prompt_raises(make_container, stub_llm, tmp_path: Path) -> None:
    c = make_container(stub_llm())
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents:\n  alpha:\n    model: m1\n")
    with pytest.raises(ValueError, match="system_prompt"):
        bind_agent_registry(c, config_path=p)


def test_yaml_extra_field_raises(make_container, stub_llm, tmp_path: Path) -> None:
    c = make_container(stub_llm())
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents:\n  alpha:\n    system_prompt: x\n    tools: [t]\n")
    with pytest.raises(ValidationError):
        bind_agent_registry(c, config_path=p)


def test_three_agent_round_trip(make_container, stub_llm, tmp_path: Path) -> None:
    """Two Python-registered + one yaml-only; one Python overridden by yaml."""
    c = make_container(stub_llm())
    register_agent(c, Agent("alpha", instructions="A", container=c, model="m1"))
    register_agent(c, Agent("beta", instructions="B", container=c))
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  alpha:
    model: m2
  gamma:
    system_prompt: |
      Gamma defined in yaml.
"""
    )
    bind_agent_registry(c, config_path=p)
    r = c.resolve(AgentRegistry)
    assert r.get("alpha").model == "m2"
    assert r.get("alpha").instructions == "A"
    assert r.get("beta").instructions == "B"
    gamma = r.get("gamma")
    assert gamma.instructions is not None and "Gamma defined in yaml" in gamma.instructions


def test_bind_with_explicit_missing_path_no_op(make_container, stub_llm, tmp_path: Path) -> None:
    """An explicit config_path that doesn't exist is treated as no-op so
    test setups don't have to short-circuit."""
    c = make_container(stub_llm())
    p = tmp_path / "does_not_exist.yaml"
    bind_agent_registry(c, config_path=p)
    # No raise; registry may or may not exist.


def test_discover_from_env_var(make_container, stub_llm, tmp_path: Path, monkeypatch) -> None:
    """$VOUSSOIR_CONFIG env var points at the file."""
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents:\n  alpha:\n    system_prompt: 'hi'\n")
    monkeypatch.setenv("VOUSSOIR_CONFIG", str(p))
    c = make_container(stub_llm())
    bind_agent_registry(c)
    assert c.resolve(AgentRegistry).has("alpha")


def test_discover_from_cwd(make_container, stub_llm, tmp_path: Path, monkeypatch) -> None:
    """./voussoir.yaml in the current working directory."""
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents:\n  beta:\n    system_prompt: 'hi'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    c = make_container(stub_llm())
    bind_agent_registry(c)
    assert c.resolve(AgentRegistry).has("beta")


def test_discover_from_project_root(make_container, stub_llm, tmp_path: Path, monkeypatch) -> None:
    """Walk up from cwd to a directory containing pyproject.toml; load that
    directory's voussoir.yaml."""
    project_root = tmp_path / "project"
    nested = project_root / "subdir"
    nested.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (project_root / "voussoir.yaml").write_text("agents:\n  gamma:\n    system_prompt: 'hi'\n")
    monkeypatch.chdir(nested)
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    c = make_container(stub_llm())
    bind_agent_registry(c)
    assert c.resolve(AgentRegistry).has("gamma")
