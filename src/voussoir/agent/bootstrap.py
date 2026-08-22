"""bind_agent_registry — applies voussoir.yaml `agents:` overrides + definitions.

Lives in its own module (not registry.py) to keep import edges one-way:
agent.py and bootstrap.py both runtime-import registry.py and agent_builder.py,
but neither imports the other and registry.py imports neither.

Lookup order for the config file when `config_path` is omitted:
  1. c.config['voussoir_yaml_path'] (if container exposes a config dict)
  2. $VOUSSOIR_CONFIG env var
  3. ./voussoir.yaml (cwd)
  4. <project_root>/voussoir.yaml — walk up from cwd looking for pyproject.toml

Missing yaml is **not an error**. `bind_agent_registry` becomes a no-op
(apart from ensuring an AgentRegistry binding exists on the container);
Python-only registration still works.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

from voussoir.agent.agent import Agent
from voussoir.agent.agent_builder import AgentBuilder
from voussoir.agent.registry import AgentRegistry
from voussoir.config import AgentConfig, load_voussoir_config
from voussoir.container import Container
from voussoir.observability.logging_setup import get_logger

_ENV_OVERRIDE_FIELDS: dict[str, str] = {
    # ENV-VAR-FIELD : AgentConfig field name
    "MODEL": "model",
    "TEMPERATURE": "temperature",
    "SYSTEM_PROMPT": "system_prompt",
    "MAX_STEPS": "max_steps",
    "MAX_DURATION_S": "max_duration_s",
    # NOTE: every value here must be a real AgentConfig field. AgentConfig sets
    # extra="forbid", so a mapping to a non-existent field raises
    # pydantic.ValidationError from model_validate below. "MAX_CASCADE_DEPTH"
    # was mapped to a field that never existed, and the resulting error was not
    # caught by the parse guard — so a documented env override took down
    # `voussoir run` at startup instead of applying (audit M2). Cascade is a
    # Python-only construct (see voussoir.config's module docstring); it has no
    # yaml/env surface.
    "ALLOWED_CAPABILITIES": "allowed_capabilities",
}


def _env_value_for_field(field_name: str, raw: str) -> object:
    """Convert env-var string to the typed value AgentConfig expects."""
    if field_name == "temperature":
        return float(raw)
    if field_name == "max_duration_s":
        return float(raw)
    if field_name == "max_steps":
        return int(raw)
    if field_name == "allowed_capabilities":
        # Pipe-separated capability names: "READ_PUBLIC|EXFILTRATION"
        return [s.strip() for s in raw.split("|") if s.strip()]
    return raw  # model, system_prompt — keep as str


def _apply_env_overrides(c: Container, registry: AgentRegistry) -> None:
    """Apply `VOUSSOIR_AGENT_<NAME>_<FIELD>=VALUE` env overrides to registered agents.

    Use this when ops needs to tweak a yaml-defined or code-registered agent at
    deploy time without touching the source. NAME is the agent's name uppercased
    with hyphens replaced by underscores; FIELD is from `_ENV_OVERRIDE_FIELDS`.
    Unknown FIELD names log a structlog warning and are skipped.
    Reuses AgentBuilder.from_agent(...).apply_config(...) to rebuild.
    """
    log = get_logger(__name__)
    for env_key, env_value in os.environ.items():
        if not env_key.startswith("VOUSSOIR_AGENT_"):
            continue
        rest = env_key[len("VOUSSOIR_AGENT_") :]
        # Split on the LAST occurrence of an _ENV_OVERRIDE_FIELDS suffix so agent
        # names containing underscores (e.g. "my_agent") still parse.
        matched_field: str | None = None
        agent_key: str | None = None
        for env_field in _ENV_OVERRIDE_FIELDS:
            suffix = "_" + env_field
            if rest.endswith(suffix):
                matched_field = _ENV_OVERRIDE_FIELDS[env_field]
                agent_key = rest[: -len(suffix)]
                break
        if matched_field is None or agent_key is None:
            log.warning(
                "env_override_unknown_field",
                env_key=env_key,
                hint=f"valid suffixes: {sorted(_ENV_OVERRIDE_FIELDS)}",
            )
            continue
        # Match agent name (case-insensitive, treating _ and - as equivalent).
        agent_name_norm = agent_key.lower().replace("-", "_")
        matching = [
            n for n in registry.list_names() if n.lower().replace("-", "_") == agent_name_norm
        ]
        if not matching:
            log.warning(
                "env_override_unknown_agent",
                env_key=env_key,
                agent_key=agent_key,
                known_agents=registry.list_names(),
            )
            continue
        name = matching[0]
        existing = registry.get(name)
        if not isinstance(existing, Agent):
            log.warning(
                "env_override_non_agent_skipped",
                env_key=env_key,
                name=name,
                existing_type=type(existing).__name__,
            )
            continue
        try:
            typed_value = _env_value_for_field(matched_field, env_value)
        except (ValueError, TypeError) as exc:
            log.warning(
                "env_override_value_parse_failed",
                env_key=env_key,
                value=env_value,
                error=str(exc),
            )
            continue
        try:
            overrides = AgentConfig.model_validate({matched_field: typed_value})
        except ValidationError as exc:
            # Defence in depth for the class of bug audit M2 found: a mapping in
            # _ENV_OVERRIDE_FIELDS that names a field AgentConfig doesn't have.
            # A bad override must degrade to "skipped, with a warning" like
            # every other one, never take down process startup.
            log.warning(
                "env_override_rejected_by_schema",
                env_key=env_key,
                field=matched_field,
                error=str(exc),
            )
            continue
        updated = AgentBuilder.from_agent(existing).apply_config(overrides).build()
        registry.replace(updated)


def _discover_config_path(c: Container) -> Path | None:
    # 1. Container-bound override (some test harnesses set c.config).
    cfg = getattr(c, "config", None)
    if isinstance(cfg, dict):
        path = cfg.get("voussoir_yaml_path")
        if path is not None:
            p = Path(path)
            if p.exists():
                return p

    # 2. Env var
    env = os.environ.get("VOUSSOIR_CONFIG")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 3. cwd
    cwd = Path.cwd() / "voussoir.yaml"
    if cwd.exists():
        return cwd

    # 4. Walk up from cwd looking for pyproject.toml as project root marker.
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            candidate = parent / "voussoir.yaml"
            if candidate.exists():
                return candidate
            break

    return None


def bind_agent_registry(
    c: Container,
    *,
    config_path: Path | None = None,
    load_plugins: bool = True,
    allowed_plugin_names: set[str] | None = None,
) -> None:
    """Apply voussoir.yaml `agents:` overrides + definitions to the registry,
    then (optionally) load entry-point plugin delegates.

    Ensures an AgentRegistry exists on `c`. If `config_path` is None and
    no yaml file is found via the discovery order, the yaml step is a
    no-op (Python-only mode).

    Phase 4.5a P1 #22: `load_plugins` defaults to True now that plugin loading
    is gated by `allowed_plugin_names` (None = load all + INFO log each). Pass
    `load_plugins=False` to fully suppress; pass a set of names to allow-list.

    Phase 4.5a P1 #18: yaml override targeting a non-Agent IDelegate (e.g. a
    plugin-loaded AgentRef) logs a warning and skips, rather than raising
    TypeError. Plugin delegates are not yaml-overridable by design.

    Idempotent in the steady state: re-applying the same yaml is harmless.
    """
    if not c.has(AgentRegistry):
        c.bind(AgentRegistry, AgentRegistry())

    if config_path is not None:
        # Explicit nonexistent path → skip yaml. Phase 4d/4.5a still runs
        # plugins (if requested) so the plugin path isn't accidentally
        # suppressed by a missing yaml.
        resolved: Path | None = config_path if config_path.exists() else None
    else:
        resolved = _discover_config_path(c)

    if resolved is not None:
        cfg = load_voussoir_config(resolved)
        registry = c.resolve(AgentRegistry)
        for name, agent_cfg in cfg.agents.items():
            if registry.has(name):
                existing = registry.get(name)
                if not isinstance(existing, Agent):
                    # Phase 4.5a P1 #18: warn + skip instead of raising.
                    get_logger(__name__).warning(
                        "yaml_override_non_agent_skipped",
                        name=name,
                        existing_type=type(existing).__name__,
                        reason="yaml override only supports local Agent "
                        "(AgentBuilder.from_agent); plugin-only IDelegate cannot be cloned",
                    )
                    continue
                updated = AgentBuilder.from_agent(existing).apply_config(agent_cfg).build()
                registry.replace(updated)
            else:
                new_agent = AgentBuilder.from_config(name, agent_cfg, container=c).build()
                registry.add(new_agent)

    if load_plugins:
        from voussoir.agent.plugins import load_delegate_plugins

        load_delegate_plugins(
            c,
            registry=c.resolve(AgentRegistry),
            allowed_names=allowed_plugin_names,
        )

    # Phase 5 C8: VOUSSOIR_AGENT_<NAME>_<FIELD> env-var overrides apply LAST so
    # they take precedence over yaml and plugin-registered agents.
    _apply_env_overrides(c, c.resolve(AgentRegistry))
