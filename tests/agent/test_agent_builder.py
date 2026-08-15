"""AgentBuilder: from_agent, from_config, lead, apply_config, build."""

from __future__ import annotations

import pytest

from voussoir.agent.agent import Agent
from voussoir.agent.agent_builder import AgentBuilder
from voussoir.config import AgentConfig


def test_from_agent_round_trip(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    original = Agent("researcher", instructions="research things", container=c, model="m1")
    rebuilt = AgentBuilder.from_agent(original).build()
    assert rebuilt.name == "researcher"
    assert rebuilt.instructions == "research things"
    assert rebuilt.model == "m1"


def test_from_agent_apply_config_overrides_model(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    original = Agent("alpha", instructions="x", container=c, model="m1")
    cfg = AgentConfig(model="m2")
    rebuilt = AgentBuilder.from_agent(original).apply_config(cfg).build()
    assert rebuilt.model == "m2"
    assert rebuilt.instructions == "x"  # unchanged


def test_apply_config_none_field_leaves_existing(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    original = Agent("alpha", instructions="x", container=c, model="m1")
    cfg = AgentConfig(model=None, temperature=0.5)
    rebuilt = AgentBuilder.from_agent(original).apply_config(cfg).build()
    assert rebuilt.model == "m1"  # None in config does not erase
    assert rebuilt.temperature == 0.5


def test_from_config_define_from_scratch(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    cfg = AgentConfig(system_prompt="You research things.", model="m1")
    built = AgentBuilder.from_config("researcher", cfg, container=c).build()
    assert built.name == "researcher"
    assert built.instructions == "You research things."
    assert built.model == "m1"


def test_from_config_missing_system_prompt_raises(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    cfg = AgentConfig(model="m1")  # no system_prompt
    with pytest.raises(ValueError, match="system_prompt"):
        AgentBuilder.from_config("researcher", cfg, container=c).build()


def test_lead_classmethod_basic(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    researcher = Agent("researcher", instructions="", container=c)
    writer = Agent("writer", instructions="", container=c)
    lead = AgentBuilder.lead(
        "lead",
        container=c,
        instructions="Orchestrate.",
        delegates=[researcher, writer],
    ).build()
    assert lead.name == "lead"
    assert lead.instructions == "Orchestrate."
    # Task 9 widens delegates to list[Agent | str]; string-delegate tests
    # land in test_lazy_delegate_resolution.py.
    assert lead.delegates == [researcher, writer]


def test_lead_apply_config_layered_overrides(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    lead = (
        AgentBuilder.lead("lead", container=c, instructions="Orchestrate.", model="m1")
        .apply_config(AgentConfig(model="m2"))
        .build()
    )
    assert lead.model == "m2"


def test_apply_config_none_is_noop(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    lead = AgentBuilder.lead("lead", container=c, instructions="x").apply_config(None).build()
    assert lead.instructions == "x"


def test_agent_without_container_raises_at_init() -> None:
    """Phase 4.5a P1 #21: Agent('x') without container= raises at __init__.
    Pre-4.5a, the property lazily constructed default_container(); the new
    contract fails fast at construction so misuse surfaces immediately rather
    than at first .run() / .delegate()."""
    with pytest.raises(TypeError, match="requires container="):
        Agent("orphan", instructions="x")


def test_lead_passes_every_kwarg_through(make_container, stub_llm) -> None:
    """Every optional kwarg branch of AgentBuilder.lead is exercised."""
    from voussoir.agent.cascade import RequestCascade
    from voussoir.agent.policy import AgentPolicy

    c = make_container(stub_llm())
    helper = Agent("helper", instructions="", container=c)

    class _AlwaysPass:
        name = "always_pass"

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            from voussoir.agent.cascade import Decision

            return Decision.PASS

    cascade = RequestCascade(verifier=_AlwaysPass())
    policy = AgentPolicy()

    lead = AgentBuilder.lead(
        "lead",
        container=c,
        instructions="orchestrate",
        delegates=[helper],
        model="claude-haiku-4-5",
        temperature=0.25,
        tools=[],
        skills=["s1"],
        policy=policy,
        cascade=cascade,
        description="lead agent",
        max_delegation_depth=5,
    ).build()

    assert lead.name == "lead"
    assert lead.instructions == "orchestrate"
    assert lead.delegates == [helper]
    assert lead.model == "claude-haiku-4-5"
    assert lead.temperature == 0.25
    assert lead.tools == []
    assert lead.skills == ["s1"]
    assert lead.policy is policy
    assert lead.cascade is cascade
    assert lead.description == "lead agent"
    assert lead.max_delegation_depth == 5


def test_apply_config_system_prompt_satisfies_from_config(make_container, stub_llm) -> None:
    """from_config with no system_prompt can be salvaged by a later
    apply_config that supplies one."""
    c = make_container(stub_llm())
    base = AgentConfig(model="m1")  # no system_prompt
    layer = AgentConfig(system_prompt="Hi.")
    built = AgentBuilder.from_config("a", base, container=c).apply_config(layer).build()
    assert built.instructions == "Hi."
    assert built.model == "m1"
