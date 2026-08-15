"""Locks `voussoir new` scaffolding + 2 starter templates (Phase 6 B2)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from voussoir.cli.main import main


def test_new_minimal_scaffolds_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", "myproj"])
    assert result.exit_code == 0, f"`voussoir new myproj` failed: {result.output}"

    proj = tmp_path / "myproj"
    assert proj.exists() and proj.is_dir()
    assert (proj / "voussoir.yaml").exists()
    assert (proj / "main.py").exists()
    assert (proj / "README.md").exists()

    # {{project_name}} substitution: "myproj" appears in at least one file
    yaml_content = (proj / "voussoir.yaml").read_text()
    main_content = (proj / "main.py").read_text()
    assert "myproj" in yaml_content or "myproj" in main_content


def test_new_research_template(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", "rsch", "--template", "research"])
    assert result.exit_code == 0
    proj = tmp_path / "rsch"
    assert (proj / "tools.py").exists()
    assert (proj / "main.py").exists()
    assert (proj / "voussoir.yaml").exists()
    assert (proj / "README.md").exists()
    # {{project_name}} substitution applied across at least one research-template file
    yaml_content = (proj / "voussoir.yaml").read_text()
    main_content = (proj / "main.py").read_text()
    readme_content = (proj / "README.md").read_text()
    assert "rsch" in yaml_content or "rsch" in main_content or "rsch" in readme_content


def test_new_refuses_non_empty_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    (target / "blocker.txt").write_text("x")
    runner = CliRunner()
    result = runner.invoke(main, ["new", "existing"])
    assert result.exit_code != 0
    assert "exists" in result.output.lower() or "non-empty" in result.output.lower()


def test_new_empty_target_dir_is_ok(tmp_path: Path, monkeypatch) -> None:
    """An empty existing dir should be filled, not refused."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "empty"
    target.mkdir()
    runner = CliRunner()
    result = runner.invoke(main, ["new", "empty"])
    assert result.exit_code == 0
    assert (target / "voussoir.yaml").exists()


def test_new_unknown_template_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", "x", "--template", "bogus"])
    assert result.exit_code != 0
