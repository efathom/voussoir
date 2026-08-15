"""Phase 6 Tranche B exit gate -- CLI + starter templates (Phase 6 Task B5).

5 invariants locking the shipped CLI surface. If any fails, the CLI contract
has regressed.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

from click.testing import CliRunner


def test_exit_1_voussoir_command_exists() -> None:
    """`voussoir --version` runs (the click group is importable + wired)."""
    from voussoir.cli.main import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0, f"--version failed: {result.output}"
    assert "voussoir" in result.output.lower() or "version" in result.output.lower()


def test_exit_2_four_subcommands_registered() -> None:
    """--help lists all 4 subcommands."""
    from voussoir.cli.main import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("new", "run", "validate", "doctor"):
        assert cmd in result.output, f"--help missing subcommand '{cmd}'"


def test_exit_3_two_templates_shipped() -> None:
    """voussoir.templates.{minimal,research} are importable + contain the expected files."""
    templates = resources.files("voussoir.templates")
    minimal = templates / "minimal"
    research = templates / "research"

    assert minimal.is_dir(), "voussoir.templates.minimal not present"
    assert research.is_dir(), "voussoir.templates.research not present"

    minimal_names = {entry.name for entry in minimal.iterdir()}
    assert {
        "voussoir.yaml",
        "main.py",
        "README.md",
    } <= minimal_names, f"minimal template missing required files; found {minimal_names}"

    research_names = {entry.name for entry in research.iterdir()}
    assert {
        "voussoir.yaml",
        "main.py",
        "README.md",
        "tools.py",
    } <= research_names, f"research template missing required files; found {research_names}"


def test_exit_4_doctor_runs_without_crashing() -> None:
    """`voussoir doctor` produces output without raising. Exit code may be 0 or 1
    depending on which optional deps are installed; both are valid."""
    from voussoir.cli.main import main

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code in (0, 1), f"unexpected exit {result.exit_code}"
    assert result.output != ""
    # Doctor must emit Python + at least one API key line + ctxforge + OTel
    assert "Python" in result.output or "python" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "ctxforge" in result.output.lower()


def test_exit_5_pyproject_registers_console_script() -> None:
    """pyproject.toml's [project.scripts] table registers the voussoir entry point."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    scripts = pyproject["project"].get("scripts", {})
    assert "voussoir" in scripts, f"pyproject [project.scripts] missing 'voussoir' entry: {scripts}"
    assert scripts["voussoir"].startswith(
        "voussoir.cli.main:"
    ), f"voussoir entry point should target voussoir.cli.main:main; got {scripts['voussoir']!r}"
