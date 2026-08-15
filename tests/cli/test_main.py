"""Locks the voussoir CLI scaffold + 4-subcommand registration (Phase 6 B1)."""

from __future__ import annotations

from click.testing import CliRunner

from voussoir.cli.main import main


def test_voussoir_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    # Some version string is present
    assert "voussoir" in result.output.lower() or "0." in result.output or "1." in result.output


def test_voussoir_help_lists_four_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("new", "run", "validate", "doctor"):
        assert cmd in result.output, f"--help missing subcommand '{cmd}'"


def test_voussoir_stub_subcommands_invoke():
    """Each subcommand stub at least returns a non-error exit (or a clear 'not yet implemented' message).

    B1's stubs may exit 1 with a 'not yet implemented' message — that's fine. The point of
    this test is to verify the subcommands are wired into the click group, not that they
    do real work yet (B2-B4 implement bodies).
    """
    runner = CliRunner()
    for cmd in ("new", "run", "validate", "doctor"):
        result = runner.invoke(main, [cmd, "--help"])
        # --help on each subcommand must succeed; the click group has wired the command.
        assert (
            result.exit_code == 0
        ), f"`voussoir {cmd} --help` failed with exit={result.exit_code}: {result.output}"
