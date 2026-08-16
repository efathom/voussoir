"""voussoir validate — lint voussoir.yaml without running anything (Phase 6 B4).

Use this CLI command before deploying to catch config errors early. Returns
a list of issues by agent; exit 0 if all green, exit 1 with a line-by-line
report on failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from voussoir.config import load_voussoir_config
from voussoir.tools import parse_capability_list

# Models whose prefix implies a specific provider API key.
_MODEL_PREFIX_KEY: tuple[tuple[str, str], ...] = (
    ("claude-", "ANTHROPIC_API_KEY"),
    ("gpt-", "OPENAI_API_KEY"),
    ("o1-", "OPENAI_API_KEY"),
    ("o3-", "OPENAI_API_KEY"),
)


def _discover_config_path_simple(explicit: Path | None) -> Path | None:
    """Lightweight config discovery without a Container (safe before any keys are set).

    Lookup order mirrors bootstrap._discover_config_path:
      1. Explicit CLI argument.
      2. $VOUSSOIR_CONFIG env var.
      3. ./voussoir.yaml (cwd).
      4. Walk up from cwd looking for pyproject.toml, then <root>/voussoir.yaml.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None

    env = os.environ.get("VOUSSOIR_CONFIG")
    if env:
        p = Path(env)
        if p.exists():
            return p

    cwd_candidate = Path.cwd() / "voussoir.yaml"
    if cwd_candidate.exists():
        return cwd_candidate

    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            candidate = parent / "voussoir.yaml"
            if candidate.exists():
                return candidate
            break

    return None


def _model_needs_key(model: str) -> str | None:
    """Return the env-var name required by *model*, or None if no key is needed."""
    m = model.lower()
    for prefix, key in _MODEL_PREFIX_KEY:
        if m.startswith(prefix):
            return key
    # OpenRouter model slugs are "vendor/model" (e.g. "openai/gpt-4o-mini",
    # "anthropic/claude-sonnet-4"); native provider models never contain "/".
    if "/" in m:
        return "OPENROUTER_API_KEY"
    return None


@click.command(name="validate")
@click.option(
    "--config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to voussoir.yaml (default: auto-discovered from cwd / $VOUSSOIR_CONFIG).",
)
def validate(config: Path | None) -> None:
    """Lint voussoir.yaml; exit 0 if all checks pass."""
    path = _discover_config_path_simple(config)
    if path is None:
        if config is not None:
            raise click.ClickException(f"config file not found: {config}")
        raise click.ClickException(
            "no voussoir.yaml found — pass --config <path> or run from a project directory"
        )

    try:
        cfg = load_voussoir_config(path)
    except Exception as exc:  # yaml.YAMLError or pydantic.ValidationError
        raise click.ClickException(f"{path}: {exc}") from exc

    issues: list[str] = []

    agent_names = set(cfg.agents.keys())

    for name, agent_cfg in cfg.agents.items():
        # 1. Delegate cross-references
        for delegate_name in agent_cfg.delegates or []:
            if delegate_name not in agent_names:
                issues.append(
                    f"agent {name!r}: delegate {delegate_name!r} is not declared in this yaml"
                )

        # 2. allowed_capabilities string parsing
        if agent_cfg.allowed_capabilities is not None:
            try:
                parse_capability_list(agent_cfg.allowed_capabilities)
            except ValueError as exc:
                issues.append(f"agent {name!r}: invalid allowed_capabilities — {exc}")

        # 3. Missing API key for declared model
        if agent_cfg.model is not None:
            required_key = _model_needs_key(agent_cfg.model)
            if required_key is not None and not os.environ.get(required_key):
                issues.append(
                    f"agent {name!r}: model {agent_cfg.model!r} requires"
                    f" {required_key} but it is not set"
                )

    click.echo(f"checked {path}: {len(cfg.agents)} agent(s) declared")
    if issues:
        for issue in issues:
            click.echo(f"  ✗ {issue}")
        raise click.ClickException(f"{len(issues)} issue(s) found")
    click.echo("✓ all checks passed")
