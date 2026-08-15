import pytest

from voussoir.agent.policy import AgentPolicy, PolicyViolation, PolicyViolationError


def test_default_policy_values():
    p = AgentPolicy()
    assert p.max_steps == 25
    assert p.max_duration_s == 300.0
    assert p.max_input_tokens == 200_000
    assert p.max_cost_usd == 1.00
    assert p.on_violation == "summarize_and_stop"


def test_check_max_steps_under_limit():
    p = AgentPolicy(max_steps=5, on_violation="error")
    p.check(steps=4, duration_s=1.0, tokens_in=10, tokens_out=10, cost_usd=0.0)


def test_check_raises_when_max_steps_exceeded_and_on_violation_error():
    p = AgentPolicy(max_steps=3, on_violation="error")
    with pytest.raises(PolicyViolationError) as exc:
        p.check(steps=3, duration_s=1.0, tokens_in=10, tokens_out=10, cost_usd=0.0)
    assert exc.value.violation == PolicyViolation.MAX_STEPS


def test_check_returns_violation_when_summarize_and_stop():
    p = AgentPolicy(max_cost_usd=0.10, on_violation="summarize_and_stop")
    v = p.check(steps=1, duration_s=1.0, tokens_in=10, tokens_out=10, cost_usd=0.20)
    assert v == PolicyViolation.MAX_COST


def test_check_returns_none_when_no_violation():
    p = AgentPolicy()
    v = p.check(steps=1, duration_s=1.0, tokens_in=10, tokens_out=10, cost_usd=0.0)
    assert v is None
