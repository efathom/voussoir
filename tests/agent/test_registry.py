"""AgentRegistry + register_agent."""

from __future__ import annotations

import pytest

from voussoir.agent.agent import Agent
from voussoir.agent.registry import AgentRegistry, register_agent


def test_registry_add_get_has_names(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    r = AgentRegistry()
    a = Agent("alpha", instructions="", container=c)
    r.add(a)
    assert r.has("alpha")
    assert r.get("alpha") is a
    assert r.names() == ["alpha"]


def test_registry_duplicate_add_raises(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    r = AgentRegistry()
    r.add(Agent("alpha", instructions="", container=c))
    with pytest.raises(ValueError, match="already registered"):
        r.add(Agent("alpha", instructions="", container=c))


def test_registry_replace_overrides(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    r = AgentRegistry()
    first = Agent("alpha", instructions="v1", container=c)
    second = Agent("alpha", instructions="v2", container=c)
    r.add(first)
    r.replace(second)
    assert r.get("alpha") is second


def test_registry_get_missing_lists_known(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    r = AgentRegistry()
    r.add(Agent("alpha", instructions="", container=c))
    r.add(Agent("beta", instructions="", container=c))
    with pytest.raises(KeyError, match="alpha, beta"):
        r.get("gamma")


def test_registry_get_missing_with_empty_registry() -> None:
    r = AgentRegistry()
    with pytest.raises(KeyError, match="<none>"):
        r.get("anything")


def test_register_agent_helper_creates_registry_lazily(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    assert not c.has(AgentRegistry)
    register_agent(c, Agent("alpha", instructions="", container=c))
    assert c.has(AgentRegistry)
    assert c.resolve(AgentRegistry).get("alpha").name == "alpha"


def test_register_agent_helper_reuses_existing_registry(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    r = AgentRegistry()
    c.bind(AgentRegistry, r)
    register_agent(c, Agent("alpha", instructions="", container=c))
    assert c.resolve(AgentRegistry) is r
    assert r.has("alpha")
