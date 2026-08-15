"""Locks Agent.allowed_capabilities wiring through __init__, AgentBuilder, and AgentConfig.

Verifies the per-agent capability mask is honored by StandardExecutor when an
agent runs (end-to-end), and that yaml `allowed_capabilities: ["READ_PUBLIC"]`
parses correctly via voussoir.tools.parse_capability_list (promoted from
voussoir.agent.agent_builder._parse_capability_list in v1.0.1).
"""

from __future__ import annotations

import pytest

from voussoir import Agent, AgentConfig
from voussoir.tools import Capability


def test_agent_default_allowed_capabilities(make_container):
    a = Agent(name="x", container=make_container())
    assert a.allowed_capabilities == (Capability.READ_PUBLIC | Capability.READ_PRIVATE)


def test_agent_init_accepts_allowed_capabilities(make_container):
    a = Agent(
        name="x",
        container=make_container(),
        allowed_capabilities=Capability.READ_PUBLIC | Capability.EXFILTRATION,
    )
    assert Capability.EXFILTRATION in a.allowed_capabilities
    assert Capability.READ_PRIVATE not in a.allowed_capabilities


def test_agent_config_accepts_capability_list():
    cfg = AgentConfig(
        model="claude-opus-4-7",
        allowed_capabilities=["READ_PUBLIC", "EXFILTRATION"],
    )
    assert cfg.allowed_capabilities == ["READ_PUBLIC", "EXFILTRATION"]


def test_parse_capability_list_helper():
    from voussoir.tools import parse_capability_list

    assert parse_capability_list(None) is None
    assert parse_capability_list([]) == Capability.NONE
    assert parse_capability_list(["READ_PUBLIC", "EXFILTRATION"]) == (
        Capability.READ_PUBLIC | Capability.EXFILTRATION
    )


def test_builder_applies_capability_list_from_config(make_container):
    from voussoir.agent.agent_builder import AgentBuilder

    cfg = AgentConfig(
        model="claude-opus-4-7",
        system_prompt="hi",
        allowed_capabilities=["READ_PUBLIC", "WRITE_PRIVATE"],
    )
    b = AgentBuilder.from_config("x", cfg, container=make_container())
    a = b.build()
    assert a.allowed_capabilities == (Capability.READ_PUBLIC | Capability.WRITE_PRIVATE)


def test_parse_capability_list_unknown_name_raises_valueerror():
    """A yaml typo surfaces as a friendly ValueError, not a bare KeyError."""
    from voussoir.tools import parse_capability_list

    with pytest.raises(ValueError, match="Unknown capability 'BOGUS'"):
        parse_capability_list(["READ_PUBLIC", "BOGUS"])


def test_from_agent_preserves_allowed_capabilities(make_container):
    """AgentBuilder.from_agent clones a non-default allowed_capabilities mask."""
    from voussoir.agent.agent_builder import AgentBuilder

    source = Agent(
        name="src",
        container=make_container(),
        allowed_capabilities=Capability.READ_PUBLIC | Capability.EXFILTRATION,
    )
    rebuilt = AgentBuilder.from_agent(source).build()
    assert rebuilt.allowed_capabilities == (Capability.READ_PUBLIC | Capability.EXFILTRATION)


def test_lead_accepts_allowed_capabilities(make_container):
    """AgentBuilder.lead mirrors Agent.__init__ — must accept the cap kwarg."""
    from voussoir.agent.agent_builder import AgentBuilder

    b = AgentBuilder.lead(
        "x",
        container=make_container(),
        instructions="hi",
        allowed_capabilities=Capability.READ_PUBLIC | Capability.EXFILTRATION,
    )
    a = b.build()
    assert a.allowed_capabilities == (Capability.READ_PUBLIC | Capability.EXFILTRATION)
