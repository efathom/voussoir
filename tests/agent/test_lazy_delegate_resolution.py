"""Agent.delegates accepts name strings; resolution is lazy at invoke time."""

from __future__ import annotations

import pytest

from voussoir.agent.agent import Agent
from voussoir.agent.registry import register_agent


def _with_empty_registry(c):
    """Phase 4.5a P1 #23: Agent.__init__ now requires an AgentRegistry on
    the container when string delegates are present. Tests that exercise
    lazy resolution bind an empty registry up front."""
    from voussoir.agent.registry import AgentRegistry

    c.bind(AgentRegistry, AgentRegistry())
    return c


def test_str_str_collision_raises(make_container, stub_llm) -> None:
    c = _with_empty_registry(make_container(stub_llm()))
    with pytest.raises(ValueError, match="collision"):
        Agent("lead", instructions="", delegates=["foo", "foo"], container=c)


def test_str_agent_collision_raises(make_container, stub_llm) -> None:
    c = _with_empty_registry(make_container(stub_llm()))
    a = Agent("foo", instructions="", container=c)
    with pytest.raises(ValueError, match="collision"):
        Agent("lead", instructions="", delegates=[a, "foo"], container=c)


def test_string_delegate_accepted_at_construction(make_container, stub_llm) -> None:
    """Strings in delegates list are auto-wrapped into NamedDelegate;
    resolution is deferred to delegate-tool-call time. (Phase 4.5a P1 #23
    requires AgentRegistry to be bound at construction; name lookup is
    still lazy.)"""
    from voussoir.agent.delegate import NamedDelegate

    c = _with_empty_registry(make_container(stub_llm()))
    lead = Agent(
        "lead",
        instructions="",
        delegates=["unregistered_name"],
        container=c,
    )
    assert isinstance(lead.delegates[0], NamedDelegate)
    assert lead.delegates[0].name == "unregistered_name"


def test_mixed_agent_and_string_delegates_construct(make_container, stub_llm) -> None:
    from voussoir.agent.delegate import NamedDelegate

    c = _with_empty_registry(make_container(stub_llm()))
    y = Agent("y", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[y, "x"], container=c)
    assert lead.delegates[0] is y
    assert isinstance(lead.delegates[1], NamedDelegate)
    assert lead.delegates[1].name == "x"


@pytest.mark.asyncio
async def test_string_delegate_resolves_via_registry(make_container) -> None:
    """A registered string-delegate is resolvable at invoke time."""
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    # Lead's LLM returns a tool-use block for delegate_to_researcher, then a
    # final text response after the tool result.
    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"
    lead_llm.chat = AsyncMock(
        side_effect=[
            # Lead turn 1: tool_use → delegate_to_researcher
            LLMResponse(
                content="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="tool_use",
                raw_response={
                    "tool_calls": [
                        {
                            "id": "t1",
                            "name": "delegate_to_researcher",
                            "arguments": {"task": "research X"},
                        }
                    ]
                },
            ),
            # Researcher's single turn: a final text response
            LLMResponse(
                content="Researched X.",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
            # Lead turn 2: final text after tool result
            LLMResponse(
                content="Lead saw researcher's reply.",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
        ]
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)

    register_agent(c, Agent("researcher", instructions="research", container=c))
    lead = Agent("lead", instructions="", delegates=["researcher"], container=c)
    result = await lead.run("research X")
    assert "researcher's reply" in result.output
    assert "researcher" in result.delegation_chain


def test_missing_registry_raises_at_init(make_container, stub_llm) -> None:
    """Phase 4.5a P1 #23: pre-4.5a this scenario produced DELEGATION_REFUSED
    at first .run(). Now Agent.__init__ raises ValueError immediately so the
    foot-gun surfaces at construction, not six layers deep in a traceback."""
    c = make_container(stub_llm())
    with pytest.raises(ValueError, match="AgentRegistry"):
        Agent("lead", instructions="", delegates=["x"], container=c)


@pytest.mark.asyncio
async def test_unknown_name_returns_delegation_refused(make_container) -> None:
    """Registry present but name missing — refusal surfaces to the lead."""
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    captured_results: list[str] = []
    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"

    async def _chat(messages, **kwargs):
        if lead_llm.chat.await_count == 1:
            return LLMResponse(
                content="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="tool_use",
                raw_response={
                    "tool_calls": [
                        {
                            "id": "t1",
                            "name": "delegate_to_typo",
                            "arguments": {"task": "do X"},
                        }
                    ]
                },
            )
        for m in messages:
            if m.role == "function":
                captured_results.append(m.content)
        return LLMResponse(
            content="ok",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
            raw_response=None,
        )

    lead_llm.chat = AsyncMock(side_effect=_chat)
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)
    register_agent(c, Agent("researcher", instructions="r", container=c))

    lead = Agent("lead", instructions="", delegates=["typo"], container=c)
    await lead.run("test")
    assert any("DELEGATION_REFUSED" in s and "typo" in s for s in captured_results)
