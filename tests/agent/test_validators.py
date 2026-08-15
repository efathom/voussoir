from voussoir.agent.cascade import Decision
from voussoir.agent.result import AgentResult, Step
from voussoir.agent.validators import ToolUseFaithfulness


def _result(*, output: str, tool_calls: list[str]) -> AgentResult[str]:
    return AgentResult[str](
        output=output,
        trace_id="t",
        steps=[
            Step(kind="tool_call", name=name, duration_ms=0.0, payload={}) for name in tool_calls
        ],
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        duration_ms=0.0,
        delegation_chain=[],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )


async def test_passes_when_no_claims_made():
    v = ToolUseFaithfulness()
    out = await v.validate(
        _result(output="A flat answer with no tool talk.", tool_calls=[]),
        task="t",
    )
    assert out == Decision.PASS


async def test_passes_when_explicit_marker_matches_actual_call():
    v = ToolUseFaithfulness()
    r = _result(output="I answered with [tool: search_web] data.", tool_calls=["search_web"])
    assert await v.validate(r, task="t") == Decision.PASS


async def test_fails_when_explicit_marker_unmatched():
    v = ToolUseFaithfulness()
    r = _result(output="See [tool: send_email] result.", tool_calls=["search_web"])
    assert await v.validate(r, task="t") == Decision.FAIL


async def test_passes_when_natural_language_claim_matches():
    v = ToolUseFaithfulness()
    r = _result(
        output="I called search_web and found three sources.",
        tool_calls=["search_web"],
    )
    assert await v.validate(r, task="t") == Decision.PASS


async def test_fails_when_natural_language_claim_unmatched():
    v = ToolUseFaithfulness()
    r = _result(
        output="I invoked the secret_admin_tool and got the answer.",
        tool_calls=["search_web"],
    )
    assert await v.validate(r, task="t") == Decision.FAIL


async def test_validator_has_name_attribute():
    v = ToolUseFaithfulness()
    assert v.name == "tool_use_faithfulness"


# Hedge-language AMBIGUOUS cases — defer to a downstream judge.
async def test_ambiguous_when_might_have_used():
    v = ToolUseFaithfulness()
    r = _result(
        output="I might have used the calculator to verify this.",
        tool_calls=[],
    )
    assert await v.validate(r, task="t") == Decision.AMBIGUOUS


async def test_ambiguous_when_may_have_called():
    v = ToolUseFaithfulness()
    r = _result(
        output="I may have called search_web for context.",
        tool_calls=["search_web"],
    )
    assert await v.validate(r, task="t") == Decision.AMBIGUOUS


async def test_ambiguous_when_think_i_invoked():
    v = ToolUseFaithfulness()
    r = _result(
        output="I think I invoked the lookup tool earlier.",
        tool_calls=[],
    )
    assert await v.validate(r, task="t") == Decision.AMBIGUOUS


async def test_ambiguous_when_believe_i_used():
    v = ToolUseFaithfulness()
    r = _result(
        output="I believe I used a calculator to check.",
        tool_calls=[],
    )
    assert await v.validate(r, task="t") == Decision.AMBIGUOUS


async def test_hedge_wins_over_definite_claims():
    """Hedge detection short-circuits before definite-claim matching."""
    v = ToolUseFaithfulness()
    r = _result(
        output="I called search_web. I might have used calculator too.",
        tool_calls=["search_web"],
    )
    # Without hedge: PASS (search_web claim matches). With hedge: AMBIGUOUS.
    assert await v.validate(r, task="t") == Decision.AMBIGUOUS
