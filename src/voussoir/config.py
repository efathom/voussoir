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
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Matches ${VAR} and ${VAR:-default}. Default may itself contain nested
# ${...} but not a bare `:-` or `}` (keep it simple).
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str, *, path: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` inside one yaml string scalar.

    ``path`` is the dotted location in the document, used only to make the
    unset-variable error actionable.
    """

    def _sub(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None or value == "":
            if default is None:
                raise KeyError(
                    f"voussoir.yaml references ${{{name}}} at {path!r} but the env "
                    "var is unset and no default was provided"
                )
            return default
        return value

    return _ENV_PATTERN.sub(_sub, value)


def _interpolate_tree(node: Any, *, path: str = "$") -> Any:
    """Walk a parsed yaml document expanding ``${VAR}`` in every string scalar.

    Substituting into the raw document TEXT before parsing (the previous
    approach) meant an env value containing a newline, a colon, a quote or a
    leading ``-`` changed the document's STRUCTURE instead of filling a scalar
    — an operator pasting a multi-line secret got a confusing parse error at
    best and a silently altered config at worst (audit M15). Expanding after
    the parse makes a value a value, whatever characters it holds, and lets the
    unset-variable error name the yaml path it came from.
    """
    if isinstance(node, str):
        return _expand(node, path=path)
    if isinstance(node, dict):
        return {k: _interpolate_tree(v, path=f"{path}.{k}") for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate_tree(v, path=f"{path}[{i}]") for i, v in enumerate(node)]
    return node


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

    ``${VAR}`` and ``${VAR:-default}`` placeholders are expanded in every
    string scalar AFTER the yaml is parsed (see :func:`_interpolate_tree`), so
    an env value can contain newlines, colons or quotes without altering the
    document's structure.

    Raises:
      yaml.YAMLError: malformed yaml.
      pydantic.ValidationError: schema violation (extra fields, wrong types).
      KeyError: an ``${VAR}`` placeholder references an unset env var; the
        message names the yaml path where the placeholder appeared.
    """
    parsed = yaml.safe_load(path.read_text()) or {}
    return VoussoirConfig.model_validate(_interpolate_tree(parsed))
