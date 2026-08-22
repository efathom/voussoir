"""Lightweight builder for Agent.

Three entry points:
  - `AgentBuilder.lead(name, container=c, ...)` — Python kwargs for the
    top-level lead agent constructed inline in app code.
  - `AgentBuilder.from_agent(agent)` — pre-populate from an existing Agent
    (yaml override path).
  - `AgentBuilder.from_config(name, cfg, container=c)` — pre-populate from
    a yaml AgentConfig (yaml define-from-scratch path).

Then optionally `.apply_config(cfg)` to layer additional config on top
(idempotent: None fields in cfg don't erase existing values), and `.build()`.

Define-from-scratch builders (`from_config` only) require system_prompt /
instructions to be set by build time; from_agent and lead don't.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voussoir.agent.policy import AgentPolicy
from voussoir.container import Container
from voussoir.tools import Capability, parse_capability_list

if TYPE_CHECKING:
    from voussoir.agent.agent import Agent
    from voussoir.agent.cascade import RequestCascade
    from voussoir.agent.policy import AgentPolicy


# Fields that can be layered from a config object via apply_config.
# Anything not in this whitelist (cascade, policy, tools, guardrails,
# middleware) is Python-only and not exposed through yaml.
_LAYERABLE_FIELDS = (
    "model",
    "temperature",
    "max_steps",
    "max_duration_s",
    "system_prompt",  # mapped to `instructions` at build time
    "description",
    "delegates",
    "skills",
    "allowed_capabilities",  # Phase 5 A5: per-agent capability mask
)


def _policy_with_budget_overrides(
    base: AgentPolicy | None,
    *,
    max_steps: int | None,
    max_duration_s: float | None,
) -> AgentPolicy | None:
    """Layer yaml/env budget fields onto an AgentPolicy.

    `max_steps` and `max_duration_s` are `_LAYERABLE_FIELDS`, are validated by
    `AgentConfig`, and land in `self._fields` — but `build()` used to forward a
    fixed kwarg list to `Agent(...)` that never touched a policy, so a
    `voussoir.yaml` entry of `max_steps: 5` was accepted without complaint and
    the agent ran with the default 25 (audit M1).

    Returns None when there is nothing to say, so `build()` leaves `policy`
    out of its kwargs and `Agent.__init__` applies its own default.
    """
    if max_steps is None and max_duration_s is None:
        return base
    updates: dict[str, Any] = {}
    if max_steps is not None:
        updates["max_steps"] = max_steps
    if max_duration_s is not None:
        updates["max_duration_s"] = max_duration_s
    # An explicitly-passed policy keeps every other field it set; the config
    # layer only overrides the two budgets it actually carries.
    return (base or AgentPolicy()).model_copy(update=updates)


class AgentBuilder:
    """Fluent builder for constructing an `Agent` from config or scratch.

    Use this when you want partial Agent construction — e.g. starting from
    a yaml `AgentConfig` (`AgentBuilder.from_config(...)`), branching from
    an existing agent (`AgentBuilder.from_agent(...)`), or composing a lead
    agent step-by-step (`AgentBuilder.lead(...)`). Call `.build()` to
    materialize the `Agent`. Direct `Agent(...)` construction stays valid
    for simple cases; the builder is for the config-driven path.
    """

    def __init__(self, name: str, *, container: Container) -> None:
        self._name = name
        self._container = container
        self._fields: dict[str, Any] = {}
        # Set by from_config() — at build() time we enforce that some form
        # of instructions has been supplied. from_agent and lead don't set
        # this because their starting state may legitimately have no
        # instructions (Agent.__init__ accepts instructions=None).
        self._require_instructions = False

    @classmethod
    def lead(
        cls,
        name: str,
        *,
        container: Container,
        instructions: str | list[str] | None = None,
        delegates: list[Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[Any] | None = None,
        skills: list[str] | None = None,
        policy: AgentPolicy | None = None,
        cascade: RequestCascade | None = None,
        description: str = "",
        max_delegation_depth: int = 3,
        allowed_capabilities: Capability = Capability.READ_PUBLIC | Capability.READ_PRIVATE,
    ) -> AgentBuilder:
        """Pre-populate the builder for a top-level lead agent.

        Mirrors `Agent.__init__`'s public surface so callers can construct
        the entry-point agent inline without dropping into the bare
        `Agent(...)` constructor. Subsequent `.apply_config(cfg)` calls let
        yaml tune the lead even though it isn't in the registry.
        """
        b = cls(name, container=container)
        if instructions is not None:
            b._fields["instructions"] = instructions
        if delegates is not None:
            b._fields["delegates"] = delegates
        if model is not None:
            b._fields["model"] = model
        if temperature is not None:
            b._fields["temperature"] = temperature
        if tools is not None:
            b._fields["tools"] = tools
        if skills is not None:
            b._fields["skills"] = skills
        if policy is not None:
            b._fields["policy"] = policy
        if cascade is not None:
            b._fields["cascade"] = cascade
        if description:
            b._fields["description"] = description
        if max_delegation_depth != 3:
            b._fields["max_delegation_depth"] = max_delegation_depth
        _default_caps = Capability.READ_PUBLIC | Capability.READ_PRIVATE
        if allowed_capabilities != _default_caps:
            b._fields["allowed_capabilities"] = allowed_capabilities
        return b

    @classmethod
    def from_agent(cls, agent: Agent) -> AgentBuilder:
        """Pre-populate from an existing Agent's fields. Yaml override path."""
        container = agent._container  # noqa: SLF001  — intentional cross-package access
        if container is None:
            raise ValueError(
                f"AgentBuilder.from_agent({agent.name!r}): source agent has no "
                f"container; rebuild a fresh Agent with an explicit container "
                f"instead of going through the builder."
            )
        b = cls(agent.name, container=container)
        b._fields = {
            "instructions": agent.instructions,
            "delegates": list(agent.delegates),
            "model": agent.model,
            "temperature": agent.temperature,
            "tools": list(agent.tools),
            "skills": list(agent.skills),
            "policy": agent.policy,
            "cascade": agent.cascade,
            "description": agent.description,
            "max_delegation_depth": agent.max_delegation_depth,
            "allowed_capabilities": agent.allowed_capabilities,
        }
        return b

    @classmethod
    def from_config(cls, name: str, cfg: Any, *, container: Container) -> AgentBuilder:
        """Pre-populate from a yaml AgentConfig. Define-from-scratch path.

        Flags the builder as requiring instructions by build() time; if cfg
        has no system_prompt and no later apply_config supplies one, build()
        raises ValueError.
        """
        b = cls(name, container=container)
        b._require_instructions = True
        b.apply_config(cfg)
        return b

    def apply_config(self, cfg: Any | None) -> AgentBuilder:
        """Layer non-None fields from cfg onto current builder state.

        cfg=None is a no-op (lets callers blindly chain whether or not
        yaml has an entry for this agent).
        """
        if cfg is None:
            return self
        for field in _LAYERABLE_FIELDS:
            value = getattr(cfg, field, None)
            if value is None:
                continue
            if field == "allowed_capabilities":
                # cfg.allowed_capabilities is list[str]; parse to Capability flag.
                # parse_capability_list returns None only when input is None,
                # which we've already filtered, so the cast is safe.
                self._fields["allowed_capabilities"] = parse_capability_list(value)
            else:
                self._fields[field] = value
        return self

    def build(self) -> Agent:
        """Construct the Agent. Maps `system_prompt` → `instructions` if both
        not present (system_prompt always wins over instructions if both
        appeared, since it came from a config layer)."""
        # Late import to keep agent_builder.py importable from agent/__init__.py
        from voussoir.agent.agent import Agent

        fields = dict(self._fields)
        if "system_prompt" in fields:
            fields["instructions"] = fields.pop("system_prompt")

        if self._require_instructions and not fields.get("instructions"):
            raise ValueError(
                f"agent {self._name!r}: yaml define-from-scratch requires "
                f"system_prompt (or instructions via later apply_config)"
            )

        kwargs: dict[str, Any] = {"container": self._container}
        for k in (
            "instructions",
            "model",
            "temperature",
            "tools",
            "skills",
            "policy",
            "delegates",
            "cascade",
        ):
            if k in fields:
                kwargs[k] = fields[k]
        if "description" in fields:
            kwargs["description"] = fields["description"]
        if "max_delegation_depth" in fields:
            kwargs["max_delegation_depth"] = fields["max_delegation_depth"]
        if "allowed_capabilities" in fields:
            kwargs["allowed_capabilities"] = fields["allowed_capabilities"]
        policy = _policy_with_budget_overrides(
            kwargs.get("policy"),
            max_steps=fields.get("max_steps"),
            max_duration_s=fields.get("max_duration_s"),
        )
        if policy is not None:
            kwargs["policy"] = policy
        return Agent(self._name, **kwargs)
