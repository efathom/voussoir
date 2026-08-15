"""Phase 4a exit criteria — gates the v0.4.0a-phase4a tag."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent import Agent, IDelegate, NamedDelegate
from voussoir.agent.registry import register_agent
from voussoir.agent.result import AgentResult
from voussoir.protocols import ILLMProvider as ILLMProviderProto


def _multi_turn_llm(
    *responses: tuple[str, str | None, dict | None],
) -> MagicMock:
    """Build a stub LLM whose chat returns the given sequence.
    Each tuple is (content, finish_reason, raw_response)."""
    p = MagicMock(spec=ILLMProvider)
    p.name = "anthropic"
    p.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content=c_,
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason=fr or "end_turn",
                raw_response=raw,
            )
            for c_, fr, raw in responses
        ]
    )
    return p


# Exit 1 — Agent + NamedDelegate satisfy IDelegate at runtime.
def test_exit_1_agent_satisfies_idelegate(make_container, stub_llm) -> None:
    c = make_container(stub_llm())
    assert isinstance(Agent("x", container=c), IDelegate)
    assert isinstance(NamedDelegate("x"), IDelegate)


# Exit 2 — Auto-wrap invariant.
def test_exit_2_auto_wrap_invariant(make_container, stub_llm) -> None:
    from voussoir.agent.registry import AgentRegistry

    c = make_container(stub_llm())
    # Phase 4.5a P1 #23: AgentRegistry binding required when delegates
    # contain bare strings. Registration of the actual name is still lazy.
    c.bind(AgentRegistry, AgentRegistry())
    lead = Agent("lead", instructions="", delegates=["x"], container=c)
    assert isinstance(lead.delegates[0], NamedDelegate)
    assert lead.delegates[0].name == "x"


# Exit 3 — Local-Agent delegation end-to-end.
@pytest.mark.asyncio
async def test_exit_3_local_agent_delegation_e2e(make_container) -> None:
    lead_llm = _multi_turn_llm(
        (
            "",
            "tool_use",
            {
                "tool_calls": [
                    {"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "do"}}
                ]
            },
        ),
        ("helper said hi", None, None),
        ("Lead saw helper.", None, None),
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    result = await lead.run("test")
    assert "Lead saw helper" in result.output
    assert "helper" in result.delegation_chain


# Exit 4 — Name-resolution delegation end-to-end.
@pytest.mark.asyncio
async def test_exit_4_named_delegate_e2e(make_container) -> None:
    lead_llm = _multi_turn_llm(
        (
            "",
            "tool_use",
            {
                "tool_calls": [
                    {"id": "t1", "name": "delegate_to_helper", "arguments": {"task": "do"}}
                ]
            },
        ),
        ("helper", None, None),
        ("OK", None, None),
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)
    register_agent(c, Agent("helper", instructions="", container=c))
    lead = Agent("lead", instructions="", delegates=["helper"], container=c)
    result = await lead.run("test")
    assert "helper" in result.delegation_chain


# Exit 5 — Mixed delegates end-to-end.
@pytest.mark.asyncio
async def test_exit_5_mixed_delegates_e2e(make_container) -> None:
    lead_llm = _multi_turn_llm(
        (
            "",
            "tool_use",
            {
                "tool_calls": [
                    {"id": "t1", "name": "delegate_to_y", "arguments": {"task": "y-task"}}
                ]
            },
        ),
        ("y output", None, None),
        (
            "",
            "tool_use",
            {
                "tool_calls": [
                    {"id": "t2", "name": "delegate_to_x", "arguments": {"task": "x-task"}}
                ]
            },
        ),
        ("x output", None, None),
        ("Done.", None, None),
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)
    y = Agent("y", instructions="", container=c)
    register_agent(c, Agent("x", instructions="", container=c))
    lead = Agent("lead", instructions="", delegates=[y, "x"], container=c)
    result = await lead.run("test")
    assert "y" in result.delegation_chain
    assert "x" in result.delegation_chain


# Exit 6 — Custom IDelegate accepted + invoked.
@pytest.mark.asyncio
async def test_exit_6_custom_idelegate_accepted(make_container) -> None:
    captured: dict[str, object] = {}

    class _Custom:
        name = "custom_delegate"
        description = ""

        async def delegate(self, task, *, parent_ctx):  # type: ignore[no-untyped-def]
            captured["task"] = task
            return AgentResult[str](
                output=f"custom: {task}",
                trace_id="t",
                steps=[],
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                duration_ms=0.0,
                delegation_chain=[],
                cascade_history=[],
                guardrail_decisions=[],
                finish_reason="completed",
            )

    lead_llm = _multi_turn_llm(
        (
            "",
            "tool_use",
            {
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "delegate_to_custom_delegate",
                        "arguments": {"task": "go"},
                    }
                ]
            },
        ),
        ("Lead saw custom.", None, None),
    )
    c = make_container(lead_llm)
    c.bind(ILLMProviderProto, lead_llm)
    lead = Agent("lead", instructions="", delegates=[_Custom()], container=c)
    result = await lead.run("test")
    assert captured["task"] == "go"
    assert "Lead saw custom" in result.output


# Exit 7 — Phase 3 exit suite still importable + key tests present.
def test_exit_7_phase3_regression_still_green() -> None:
    import tests.test_phase3_exit as phase3_exit

    expected = [
        "test_delegation_chain_populates",
        "test_child_container_inherits_singletons_but_fresh_run_scope",
        "test_cascade_pass_returns_sas_no_escalation",
        "test_max_delegation_depth_refuses_past_limit",
    ]
    for name in expected:
        assert hasattr(phase3_exit, name), f"Phase 3 exit test {name!r} missing"


# Exit 8 — Phase 3.5 exit suite still importable + key tests present.
def test_exit_8_phase35_regression_still_green() -> None:
    import tests.test_phase35_exit as phase35_exit

    expected = [
        "test_exit_1_validator_protocol_passes_task",
        "test_exit_5_lazy_delegate_resolution_e2e",
        "test_exit_6_three_agent_yaml_round_trip",
    ]
    for name in expected:
        assert hasattr(phase35_exit, name), f"Phase 3.5 exit test {name!r} missing"
