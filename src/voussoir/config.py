"""voussoir.yaml schema + loader.

Phase 3.5 introduces the `agents:` block. Future top-level blocks
(providers, capabilities, etc.) extend VoussoirConfig.

`extra="forbid"` makes typos surface as ValidationError rather than
silent no-ops. yaml fields that are intentionally Python-only (cascade,
policy, tools, guardrails, middleware) cannot appear in yaml — they're
not in AgentConfig and the strict schema rejects them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Matches ${VAR} and ${VAR:-default}. Default may itself contain nested
# ${...} but not a bare `:-` or `}` (keep it simple).
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(text: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in a yaml document body.

    Lets operators keep secrets out of voussoir.yaml while still pinning other
    values there. Unset variables without a default raise KeyError so a typo
    surfaces at load time rather than silently loading a literal ``${...}``.
    """

    def _sub(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None or value == "":
            if default is None:
                raise KeyError(
                    f"voussoir.yaml references ${{{name}}} but the env var is unset "
                    "and no default was provided"
                )
            return default
        return value

    return _ENV_PATTERN.sub(_sub, text)


class AgentConfig(BaseModel):
    """Per-agent yaml entry. All fields optional; merged into AgentBuilder."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    temperature: float | None = None
    max_steps: int | None = None
    max_duration_s: float | None = None
    system_prompt: str | None = None  # → Agent(instructions=...)
    description: str | None = None
    delegates: list[str] | None = None  # name strings, lazy-resolved
    skills: list[str] | None = None
    allowed_capabilities: list[str] | None = None  # Phase 5 A5: yaml capability mask


class VoussoirConfig(BaseModel):
    """Top-level voussoir.yaml. Phase 3.5 ships only the `agents:` block."""

    model_config = ConfigDict(extra="forbid")

    agents: dict[str, AgentConfig] = Field(default_factory=dict)


def load_voussoir_config(path: Path) -> VoussoirConfig:
    """Read + validate yaml at `path`.

    ``${VAR}`` and ``${VAR:-default}`` placeholders in the file are expanded
    from the environment before parsing (see :func:`_interpolate_env`).

    Raises:
      yaml.YAMLError: malformed yaml.
      pydantic.ValidationError: schema violation (extra fields, wrong types).
      KeyError: an ``${VAR}`` placeholder references an unset env var.
    """
    raw = _interpolate_env(path.read_text())
    return VoussoirConfig.model_validate(yaml.safe_load(raw) or {})
