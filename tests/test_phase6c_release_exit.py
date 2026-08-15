"""Phase 6 Tranche C exit gate — v1.0 release readiness (Phase 6 Task C6).

Locks the v1.0.0 version bump invariants. These tests fail until the
release ceremony commit lands.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_README = Path(__file__).resolve().parents[1] / "README.md"
_CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
_MKDOCS = Path(__file__).resolve().parents[1] / "mkdocs.yml"


_EXPECTED_VERSION = "1.3.0"  # bumped from 1.2.1 in the enterprise-hardening release


def test_exit_1_version_matches_release_label() -> None:
    from voussoir import __version__

    assert __version__ == _EXPECTED_VERSION


def test_exit_2_pyproject_version_matches() -> None:
    data = tomllib.loads(_PYPROJECT.read_text())
    assert data["project"]["version"] == _EXPECTED_VERSION


def test_exit_3_production_stable_classifier() -> None:
    data = tomllib.loads(_PYPROJECT.read_text())
    classifiers = data["project"]["classifiers"]
    assert any(
        "Production/Stable" in c for c in classifiers
    ), f"expected Development Status :: 5 - Production/Stable, got {classifiers}"


def test_exit_4_readme_contains_quickstart_imports() -> None:
    readme = _README.read_text()
    # The 5-line quickstart from C2 (v1.0.1: switched to default_container())
    assert "from voussoir import Agent, default_container" in readme
    assert "default_container()" in readme


def test_exit_5_mkdocs_config_present() -> None:
    assert _MKDOCS.exists()


def test_exit_6_changelog_has_v1_0_0_entry() -> None:
    changelog = _CHANGELOG.read_text()
    assert "1.0.0" in changelog or "v1.0.0" in changelog


def test_exit_7_changelog_has_v1_0_1_entry() -> None:
    """v1.0.1 cleanup release entry exists above the v1.0.0 entry."""
    changelog = _CHANGELOG.read_text()
    assert "1.0.1" in changelog
    # Confirm ordering: 1.0.1 entry appears before 1.0.0
    idx_101 = changelog.find("1.0.1")
    idx_100 = changelog.find("## v1.0.0")
    assert idx_101 != -1 and idx_100 != -1
    assert idx_101 < idx_100, "expected v1.0.1 entry to appear above v1.0.0"


def test_exit_8_changelog_has_v1_0_2_entry() -> None:
    """v1.0.2 round-2 review patch entry exists above the v1.0.1 entry."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.0.2" in changelog
    # Confirm ordering: 1.0.2 entry appears before 1.0.1
    idx_102 = changelog.find("## v1.0.2")
    idx_101 = changelog.find("## v1.0.1")
    assert idx_102 != -1 and idx_101 != -1
    assert idx_102 < idx_101, "expected v1.0.2 entry to appear above v1.0.1"


def test_exit_9_changelog_has_v1_0_3_entry() -> None:
    """v1.0.3 housekeeping patch entry exists above the v1.0.2 entry."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.0.3" in changelog
    idx_103 = changelog.find("## v1.0.3")
    idx_102 = changelog.find("## v1.0.2")
    assert idx_103 != -1 and idx_102 != -1
    assert idx_103 < idx_102, "expected v1.0.3 entry to appear above v1.0.2"


def test_exit_10_changelog_has_v1_0_4_entry() -> None:
    """v1.0.4 Round-3 review patch entry exists above the v1.0.3 entry."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.0.4" in changelog
    idx_104 = changelog.find("## v1.0.4")
    idx_103 = changelog.find("## v1.0.3")
    assert idx_104 != -1 and idx_103 != -1
    assert idx_104 < idx_103, "expected v1.0.4 entry to appear above v1.0.3"


def test_exit_11_changelog_has_v1_1_0_entry() -> None:
    """v1.1.0 extensibility minor release entry exists above the v1.0.4 entry."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.1.0" in changelog
    idx_110 = changelog.find("## v1.1.0")
    idx_104 = changelog.find("## v1.0.4")
    assert idx_110 != -1 and idx_104 != -1
    assert idx_110 < idx_104, "expected v1.1.0 entry to appear above v1.0.4"


def test_exit_12_changelog_has_v1_2_0_entry() -> None:
    """v1.2.0 interrupts + runtime-container minor release entry exists above v1.1.0."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.2.0" in changelog
    idx_120 = changelog.find("## v1.2.0")
    idx_110 = changelog.find("## v1.1.0")
    assert idx_120 != -1 and idx_110 != -1
    assert idx_120 < idx_110, "expected v1.2.0 entry to appear above v1.1.0"


def test_exit_13_changelog_has_v1_2_1_entry() -> None:
    """v1.2.1 examples-housekeeping patch entry exists above v1.2.0."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.2.1" in changelog
    idx_121 = changelog.find("## v1.2.1")
    idx_120 = changelog.find("## v1.2.0")
    assert idx_121 != -1 and idx_120 != -1
    assert idx_121 < idx_120, "expected v1.2.1 entry to appear above v1.2.0"


def test_exit_14_changelog_has_v1_3_0_entry() -> None:
    """v1.3.0 enterprise-hardening release entry exists above v1.2.1."""
    changelog = _CHANGELOG.read_text()
    assert "## v1.3.0" in changelog
    idx_130 = changelog.find("## v1.3.0")
    idx_121 = changelog.find("## v1.2.1")
    assert idx_130 != -1 and idx_121 != -1
    assert idx_130 < idx_121, "expected v1.3.0 entry to appear above v1.2.1"
