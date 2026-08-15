"""voussoir.yaml schema + loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from voussoir.config import AgentConfig, VoussoirConfig, load_voussoir_config


def test_empty_yaml_yields_empty_config(tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text("")
    cfg = load_voussoir_config(p)
    assert cfg.agents == {}


def test_minimal_agent(tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  researcher:
    model: claude-sonnet-4-6
    temperature: 0.3
"""
    )
    cfg = load_voussoir_config(p)
    assert "researcher" in cfg.agents
    a = cfg.agents["researcher"]
    assert a.model == "claude-sonnet-4-6"
    assert a.temperature == 0.3
    assert a.system_prompt is None


def test_define_from_scratch(tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  fact_checker:
    model: claude-haiku-4-5
    system_prompt: |
      You verify facts.
    delegates: [researcher]
    description: "Cross-references claims."
"""
    )
    cfg = load_voussoir_config(p)
    a = cfg.agents["fact_checker"]
    assert a.system_prompt is not None
    assert "verify facts" in a.system_prompt
    assert a.delegates == ["researcher"]
    assert a.description == "Cross-references claims."


def test_extra_field_rejected(tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  foo:
    model: m1
    tools: [some_tool]
"""
    )
    with pytest.raises(ValidationError):
        load_voussoir_config(p)


def test_extra_top_level_rejected(tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  foo:
    system_prompt: x
unknown_block: 42
"""
    )
    with pytest.raises(ValidationError):
        load_voussoir_config(p)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "voussoir.yaml"
    p.write_text("agents: {[}")
    with pytest.raises(yaml.YAMLError):
        load_voussoir_config(p)


def test_agent_config_all_fields_optional() -> None:
    cfg = AgentConfig()
    assert cfg.model is None
    assert cfg.temperature is None
    assert cfg.max_steps is None
    assert cfg.max_duration_s is None
    assert cfg.system_prompt is None
    assert cfg.description is None
    assert cfg.delegates is None
    assert cfg.skills is None


def test_voussoir_config_default_empty() -> None:
    cfg = VoussoirConfig()
    assert cfg.agents == {}
