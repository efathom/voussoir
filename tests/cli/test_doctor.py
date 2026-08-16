"""Locks `voussoir doctor` (Phase 6 B4)."""

from __future__ import annotations

from click.testing import CliRunner

from voussoir.cli.main import main


def test_doctor_reports_python_version(monkeypatch: object) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert "Python" in result.output or "python" in result.output


def test_doctor_reports_api_key_status(monkeypatch: object) -> None:
    """All three provider key statuses appear in output."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert "ANTHROPIC_API_KEY" in result.output
    assert "OPENAI_API_KEY" in result.output
    assert "OPENROUTER_API_KEY" in result.output


def test_doctor_reports_ctxforge_status(monkeypatch: object) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert "ctxforge" in result.output.lower()


def test_doctor_reports_otel_status(monkeypatch: object) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert "otel" in result.output.lower() or "opentelemetry" in result.output.lower()


def test_doctor_exit_zero_when_all_required_present(monkeypatch: object) -> None:
    """When required tools are importable and an API key is set, exit 0."""
    # ctxforge and otel are base deps, so they're always available.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0, f"doctor failed: {result.output}"


def test_doctor_exit_nonzero_when_no_api_key(monkeypatch: object) -> None:
    """No API key at all → exit 1 (one of them is required)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # type: ignore[attr-defined]
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code != 0
