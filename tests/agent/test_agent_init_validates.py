"""Phase 4.5a P1 #21 + #23 — Agent.__init__ fails loud on common foot-guns."""

from __future__ import annotations

import pytest

from voussoir.agent.agent import Agent


def test_no_container_raises() -> None:
    """Phase 4.5a P1 #21: Agent('x') without container= raises."""
    with pytest.raises(TypeError, match="requires container="):
        Agent("x")


def test_string_delegate_without_registry_raises(make_container, stub_llm) -> None:
    """Phase 4.5a P1 #23: Agent('x', delegates=['name']) without an
    AgentRegistry bound raises at construction (not at first run)."""
    c = make_container(stub_llm())
    with pytest.raises(ValueError, match="AgentRegistry"):
        Agent("x", delegates=["unresolvable"], container=c)


def test_string_delegate_with_registry_succeeds(make_container, stub_llm) -> None:
    """Sanity: with a registry bound, string delegates pass __init__ (the
    delegate name doesn't need to be in the registry yet — lazy resolution
    still applies at delegate-call time)."""
    from voussoir.agent.registry import AgentRegistry

    c = make_container(stub_llm())
    c.bind(AgentRegistry, AgentRegistry())
    agent = Agent("x", delegates=["future_name"], container=c)
    assert agent.name == "x"


def test_no_string_delegates_skips_registry_check(make_container, stub_llm) -> None:
    """Agent without string delegates does NOT require a registry."""
    c = make_container(stub_llm())
    agent = Agent("x", container=c)  # No delegates, no registry needed.
    assert agent.name == "x"
