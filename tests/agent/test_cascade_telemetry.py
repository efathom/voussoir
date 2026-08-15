"""Cascade scopes a buffered sink around validate(); records merge into result."""

from __future__ import annotations

import pytest

from voussoir.agent.agent import Agent
from voussoir.agent.cascade import Decision, RequestCascade
from voussoir.observability.sink import ITelemetrySink


@pytest.mark.asyncio
async def test_validator_emission_merges_into_agent_result(make_container, stub_llm) -> None:
    """A validator that calls sink.record_llm_call must see its tokens/cost
    rolled into the AgentResult returned by the agent."""

    captured_container: list = []

    class EmittingValidator:
        name = "spy"

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            sink = captured_container[0].resolve(ITelemetrySink)
            sink.record_llm_call(
                name="spy_judge",
                tokens_in=11,
                tokens_out=3,
                cost_usd=0.0042,
                duration_ms=33.0,
            )
            return Decision.PASS

    c = make_container(stub_llm())
    captured_container.append(c)
    agent = Agent(
        "lead",
        instructions="",
        container=c,
        cascade=RequestCascade(verifier=EmittingValidator()),
    )
    result = await agent.run("test input")
    assert any(s.name == "spy_judge" and s.kind == "validator_call" for s in result.steps)
    assert result.tokens_in >= 11
    assert result.tokens_out >= 3
    assert result.cost_usd >= 0.0042


@pytest.mark.asyncio
async def test_non_emitting_validator_no_change_to_result(make_container, stub_llm) -> None:
    """ToolUseFaithfulness-style validator that doesn't touch the sink
    leaves AgentResult.cost_usd / tokens unchanged from baseline."""

    class QuietValidator:
        name = "quiet"

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            return Decision.PASS

    c = make_container(stub_llm(input_tokens=2, output_tokens=2))
    agent = Agent(
        "lead",
        instructions="",
        container=c,
        cascade=RequestCascade(verifier=QuietValidator()),
    )
    result = await agent.run("test input")
    assert not any(s.kind == "validator_call" for s in result.steps)
