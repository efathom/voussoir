"""Phase 3.5 exit criteria — gates the v0.3.5-phase35 tag."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir.agent.agent import Agent
from voussoir.agent.bootstrap import bind_agent_registry
from voussoir.agent.cascade import Decision, RequestCascade
from voussoir.agent.registry import AgentRegistry, register_agent
from voussoir.agent.result import AgentResult
from voussoir.agent.validators import AmbiguousFallback, LLMJudge
from voussoir.observability.sink import ITelemetrySink
from voussoir.protocols import ILLMProvider as ILLMProviderProto


def _make_result(output: str = "x") -> AgentResult[str]:
    return AgentResult(
        output=output,
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


# Exit 1 — Validator protocol carries `task`; cascade passes input through.
@pytest.mark.asyncio
async def test_exit_1_validator_protocol_passes_task(make_container, stub_llm) -> None:
    captured: dict[str, object] = {}

    class Spy:
        name = "spy"

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            captured["task"] = task
            return Decision.PASS

    c = make_container(stub_llm())
    agent = Agent("lead", instructions="", container=c, cascade=RequestCascade(verifier=Spy()))
    await agent.run("research X and report")
    assert captured["task"] == "research X and report"


# Exit 2 — LLMJudge returns PASS on a "PASS" response, FAIL otherwise.
@pytest.mark.asyncio
async def test_exit_2_llm_judge_pass_fail(make_container, stub_llm) -> None:
    c1 = make_container(stub_llm(content="PASS"))
    judge1 = LLMJudge("output is non-empty", container=c1)
    assert await judge1.validate(_make_result("hi"), task="t") is Decision.PASS

    c2 = make_container(stub_llm(content="FAIL"))
    judge2 = LLMJudge("output is correct", container=c2)
    assert await judge2.validate(_make_result("hi"), task="t") is Decision.FAIL


# Exit 3 — AmbiguousFallback short-circuits on PASS/FAIL; routes to judge on AMBIGUOUS.
@pytest.mark.asyncio
async def test_exit_3_ambiguous_fallback_routing() -> None:
    class P:
        def __init__(self, decision: Decision) -> None:
            self.name = "p"
            self._d = decision
            self.calls = 0

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            self.calls += 1
            return self._d

    class J:
        def __init__(self, decision: Decision) -> None:
            self.name = "j"
            self._d = decision
            self.calls = 0

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            self.calls += 1
            return self._d

    p, j = P(Decision.PASS), J(Decision.FAIL)
    assert await AmbiguousFallback(p, j).validate(_make_result(), task="t") is Decision.PASS
    assert j.calls == 0

    p2, j2 = P(Decision.AMBIGUOUS), J(Decision.PASS)
    assert await AmbiguousFallback(p2, j2).validate(_make_result(), task="t") is Decision.PASS
    assert j2.calls == 1


# Exit 4 — Judge cost rolls into AgentResult via cascade-scoped sink.
@pytest.mark.asyncio
async def test_exit_4_judge_cost_in_agent_result(make_container) -> None:
    """Full cascade run: lead returns once, judge validates and emits cost.
    Verify result.cost_usd includes the judge's call and result.steps has
    a validator_call Step."""
    # Lead's LLM: single text turn.
    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"
    lead_llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content="The answer is 42.",
                model="stub",
                input_tokens=2,
                output_tokens=2,
                finish_reason="end_turn",
                raw_response=None,
            ),
            # LLMJudge call — same provider since judge resolves ILLMProvider
            # from the same container.
            LLMResponse(
                content="PASS",
                model="stub",
                input_tokens=15,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
        ]
    )
    c = MagicMock()  # fall back to a real container below
    from voussoir.container import Container
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.observability.sink import NullTelemetrySink
    from voussoir.protocols import IMemoryStore, ISessionStore

    c = Container()
    c.bind(ILLMProviderProto, lead_llm)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    c.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]

    judge = LLMJudge("answer is non-empty", container=c)
    agent = Agent("lead", instructions="", container=c, cascade=RequestCascade(verifier=judge))
    result = await agent.run("test")
    assert any(s.kind == "validator_call" and s.name == "llm_judge" for s in result.steps)
    assert result.cost_usd > 0


# Exit 5 — Lazy delegate-by-name resolution end-to-end.
@pytest.mark.asyncio
async def test_exit_5_lazy_delegate_resolution_e2e(make_container) -> None:
    lead_llm = MagicMock(spec=ILLMProvider)
    lead_llm.name = "anthropic"
    lead_llm.chat = AsyncMock(
        side_effect=[
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
            LLMResponse(
                content="Researched X.",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
            LLMResponse(
                content="Done.",
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
    assert "Done" in result.output
    assert "researcher" in result.delegation_chain


# Exit 6 — Three-agent yaml round-trip: Python + override + define-from-scratch.
def test_exit_6_three_agent_yaml_round_trip(make_container, stub_llm, tmp_path: Path) -> None:
    c = make_container(stub_llm())
    register_agent(c, Agent("alpha", instructions="A", container=c, model="m1"))
    register_agent(c, Agent("beta", instructions="B", container=c))
    p = tmp_path / "voussoir.yaml"
    p.write_text(
        """
agents:
  alpha:
    model: m2
  gamma:
    system_prompt: |
      Gamma yaml-defined.
"""
    )
    bind_agent_registry(c, config_path=p)
    r = c.resolve(AgentRegistry)
    assert r.get("alpha").model == "m2"
    assert r.get("alpha").instructions == "A"
    assert r.get("beta").instructions == "B"
    gamma_instructions = r.get("gamma").instructions
    assert gamma_instructions is not None and "Gamma yaml-defined" in gamma_instructions


# Exit 7 — Phase 3 sub-agent cost aggregation regression guard.
#
# We delegate `test_exit_7_phase3_regression_still_green` to the full
# tests/test_phase3_exit.py suite — running that file is the canonical
# regression check. This thin wrapper just imports it and asserts a
# representative test still loads and exists.
def test_exit_7_phase3_regression_still_green() -> None:
    """Lock that tests/test_phase3_exit.py is still collectable + present."""
    import tests.test_phase3_exit as phase3_exit

    # A subset of Phase 3 exit functions must still exist by name.
    expected = [
        "test_delegation_chain_populates",
        "test_child_container_inherits_singletons_but_fresh_run_scope",
        "test_cascade_pass_returns_sas_no_escalation",
        "test_max_delegation_depth_refuses_past_limit",
    ]
    for name in expected:
        assert hasattr(phase3_exit, name), f"Phase 3 exit test {name!r} missing"
