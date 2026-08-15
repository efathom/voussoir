"""AgentCard + AuthMethod + card_from_agent."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voussoir.a2a.card import AgentCard, AuthMethod, card_from_agent


def test_auth_method_bearer_jwt() -> None:
    am = AuthMethod(type="bearer")
    assert am.type == "bearer"
    assert am.scheme == "JWT"


def test_auth_method_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        AuthMethod(type="basic")  # type: ignore[arg-type]


def test_agent_card_minimal() -> None:
    card = AgentCard(name="r", endpoint="https://h/a2a")
    assert card.name == "r"
    assert str(card.endpoint) == "https://h/a2a"
    assert card.version == "1.0.0"
    assert card.capabilities == []
    assert card.supports == ["json-rpc/2.0"]


def test_agent_card_full() -> None:
    card = AgentCard(
        name="researcher",
        description="Searches papers.",
        version="2.1.0",
        endpoint="https://h/a2a",
        capabilities=["web_search"],
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        auth=[AuthMethod(type="bearer")],
        supports=["json-rpc/2.0"],
        jwks_uri="https://h/.well-known/jwks.json",
    )
    assert card.capabilities == ["web_search"]
    assert card.jwks_uri is not None


def test_agent_card_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        AgentCard(
            name="x",
            endpoint="https://h/a2a",
            unknown_field="nope",  # type: ignore[call-arg]
        )


def test_agent_card_rejects_unknown_supports() -> None:
    with pytest.raises(ValidationError):
        AgentCard(
            name="x",
            endpoint="https://h/a2a",
            supports=["voussoir-rpc"],  # type: ignore[list-item]
        )


def test_agent_card_endpoint_must_be_url() -> None:
    with pytest.raises(ValidationError):
        AgentCard(name="x", endpoint="not-a-url")


def test_card_from_agent_basic(make_container, stub_llm) -> None:
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("researcher", description="r-desc", container=c)
    card = card_from_agent(agent, endpoint="https://h/a2a")  # type: ignore[arg-type]
    assert card.name == "researcher"
    assert card.description == "r-desc"
    assert str(card.endpoint) == "https://h/a2a"
    assert card.capabilities == []
    assert card.input_schema == {
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    }
    assert card.output_schema == {"type": "string"}
    assert len(card.auth) == 1
    assert card.auth[0].type == "bearer"


def test_card_from_agent_with_jwks_uri(make_container, stub_llm) -> None:
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("x", container=c)
    card = card_from_agent(
        agent,
        endpoint="https://h/a2a",  # type: ignore[arg-type]
        jwks_uri="https://h/.well-known/jwks.json",  # type: ignore[arg-type]
    )
    assert card.jwks_uri is not None
    assert str(card.jwks_uri) == "https://h/.well-known/jwks.json"
