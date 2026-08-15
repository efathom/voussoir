"""Locks `voussoir validate` (Phase 6 B4)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from voussoir.cli.main import main


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_validate_good_yaml_exits_zero(tmp_path: Path, monkeypatch: object) -> None:
    yaml_path = tmp_path / "voussoir.yaml"
    _write_yaml(
        yaml_path,
        "agents:\n  ok:\n    model: claude-opus-4-7\n    system_prompt: 'be brief'\n",
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    # Set the API key so the model-key check passes
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0, f"validate failed: {result.output}"


def test_validate_missing_yaml_exits_nonzero(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code != 0
    assert (
        "voussoir.yaml" in result.output.lower()
        or "no config" in result.output.lower()
        or "not found" in result.output.lower()
    )


def test_validate_invalid_yaml_structure_exits_nonzero(tmp_path: Path, monkeypatch: object) -> None:
    yaml_path = tmp_path / "voussoir.yaml"
    _write_yaml(yaml_path, "not: valid: yaml: structure: here\n")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code != 0


def test_validate_invalid_agent_config_exits_nonzero(tmp_path: Path, monkeypatch: object) -> None:
    """Pydantic ValidationError on extra fields should surface a clean error."""
    yaml_path = tmp_path / "voussoir.yaml"
    _write_yaml(
        yaml_path,
        "agents:\n  ok:\n    model: claude-opus-4-7\n    bogus_field: value\n",
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code != 0
    # Some sign of the validation issue
    assert (
        "bogus_field" in result.output
        or "extra" in result.output.lower()
        or "validation" in result.output.lower()
    )


def test_validate_delegate_reference_unresolved_exits_nonzero(
    tmp_path: Path, monkeypatch: object
) -> None:
    """`delegates: [other]` where 'other' isn't in the yaml should warn or fail."""
    yaml_path = tmp_path / "voussoir.yaml"
    _write_yaml(
        yaml_path,
        "agents:\n"
        "  lead:\n"
        "    model: claude-opus-4-7\n"
        "    delegates: [missing_subagent]\n",
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code != 0
    assert "missing_subagent" in result.output


def test_validate_invalid_capability_string_exits_nonzero(
    tmp_path: Path, monkeypatch: object
) -> None:
    yaml_path = tmp_path / "voussoir.yaml"
    _write_yaml(
        yaml_path,
        "agents:\n"
        "  x:\n"
        "    model: claude-opus-4-7\n"
        "    allowed_capabilities: [READ_PUBLIC, BOGUS_CAPABILITY]\n",
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code != 0
    assert "BOGUS_CAPABILITY" in result.output


def test_validate_missing_api_key_warns(tmp_path: Path, monkeypatch: object) -> None:
    """A claude- model without ANTHROPIC_API_KEY should at least warn (non-zero exit acceptable)."""
    yaml_path = tmp_path / "voussoir.yaml"
    _write_yaml(
        yaml_path,
        "agents:\n  x:\n    model: claude-opus-4-7\n    system_prompt: 'hi'\n",
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    # Either exit nonzero or print a warning; either way mention the key name
    assert "ANTHROPIC_API_KEY" in result.output


def test_validate_explicit_config_path(tmp_path: Path, monkeypatch: object) -> None:
    """--config <path> reads the file even when cwd has no voussoir.yaml."""
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)  # type: ignore[attr-defined]
    yaml_path = tmp_path / "elsewhere.yaml"
    _write_yaml(yaml_path, "agents:\n  ok:\n    model: claude-opus-4-7\n    system_prompt: hi\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--config", str(yaml_path)])
    assert result.exit_code == 0, f"validate failed: {result.output}"
    assert "elsewhere.yaml" in result.output


def test_validate_explicit_nonexistent_config_distinct_error(
    tmp_path: Path, monkeypatch: object
) -> None:
    """--config <missing-path> reports the bad path, not the generic 'no yaml found'."""
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert "config file not found" in result.output.lower() or "nope.yaml" in result.output
