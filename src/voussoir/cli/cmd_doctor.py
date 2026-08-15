"""voussoir doctor — environment health check (Phase 6 B4).

Use this CLI command to verify the runtime has what voussoir needs. Reports
✓/✗ per check with one-line remediation hints. Exit 0 only if all required
items are present (Python ≥3.12, an LLM API key, ctxforge, OTel SDK).
"""

from __future__ import annotations

import importlib
import os
import sys

import click


def _check(label: str, ok: bool, *, hint: str = "") -> bool:
    """Emit one status line and return ok."""
    marker = "✓" if ok else "✗"
    line = f"{marker} {label}"
    if not ok and hint:
        line += f"  — {hint}"
    click.echo(line)
    return ok


def _check_importable(module: str, label: str, *, hint: str = "") -> bool:
    """Try to import *module*; emit a ✓/✗ line labelled *label*, return success."""
    try:
        importlib.import_module(module)
        return _check(f"{label} importable", ok=True)
    except ImportError:
        fallback_hint = hint or f"pip install {module}"
        return _check(f"{label} importable", ok=False, hint=fallback_hint)


@click.command(name="doctor")
def doctor() -> None:
    """Report ✓/✗ for required + optional voussoir dependencies."""
    failed = 0

    # --- Required: Python version ------------------------------------------
    py = sys.version_info
    py_ok = (py.major, py.minor) >= (3, 12)
    _check(
        f"Python {py.major}.{py.minor}.{py.micro}",
        ok=py_ok,
        hint="need Python >= 3.12",
    )
    if not py_ok:
        failed += 1

    # --- Required: at least one LLM API key --------------------------------
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    _check("ANTHROPIC_API_KEY set", ok=has_anthropic, hint="export ANTHROPIC_API_KEY=sk-ant-...")
    _check("OPENAI_API_KEY set", ok=has_openai, hint="export OPENAI_API_KEY=sk-...")
    if not (has_anthropic or has_openai):
        failed += 1

    # --- Required: core framework deps -------------------------------------
    try:
        importlib.import_module("ctxforge")
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _pkg_version

        try:
            ctxforge_ver = _pkg_version("ctxforge")
        except PackageNotFoundError:
            ctxforge_ver = "unknown"
        _check(f"ctxforge {ctxforge_ver} importable", ok=True)
    except ImportError:
        _check(
            "ctxforge importable",
            ok=False,
            hint="re-install voussoir; ctxforge is a required base dep",
        )
        failed += 1

    if not _check_importable(
        "opentelemetry.sdk.trace",
        "opentelemetry-sdk",
        hint="re-install voussoir; opentelemetry-sdk is a required base dep",
    ):
        failed += 1

    # --- Optional extras ----------------------------------------------------
    _check_importable("mcp", "mcp (optional)", hint="pip install 'voussoir[mcp]'")
    _check_importable("fastapi", "fastapi (a2a optional)", hint="pip install 'voussoir[a2a]'")
    _check_importable("uvicorn", "uvicorn (a2a optional)", hint="pip install 'voussoir[a2a]'")
    _check_importable("jwt", "PyJWT (a2a optional)", hint="pip install 'voussoir[a2a]'")
    _check_importable(
        "cryptography", "cryptography (a2a optional)", hint="pip install 'voussoir[a2a]'"
    )
    _check_importable("keyring", "keyring (auth optional)", hint="pip install keyring")

    if failed > 0:
        raise click.ClickException(f"{failed} required check(s) failed")
