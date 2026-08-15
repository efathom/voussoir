import pytest
from pydantic import ValidationError

from voussoir.agent.cascade import Decision, RequestCascade, Validator


def test_decision_enum_values():
    assert Decision.PASS.value == "pass"
    assert Decision.FAIL.value == "fail"
    assert Decision.AMBIGUOUS.value == "ambiguous"


def test_validator_is_runtime_checkable_protocol():
    class MyValidator:
        name = "my"

        async def validate(self, result, *, task):
            return Decision.PASS

    v = MyValidator()
    assert isinstance(v, Validator)


def test_request_cascade_requires_verifier():
    class _V:
        name = "v"

        async def validate(self, result, *, task):
            return Decision.PASS

    cascade = RequestCascade(verifier=_V())
    assert cascade.sas_attempt_first is True
    assert cascade.escalation is None
    assert cascade.max_attempts == 2


def test_request_cascade_rejects_max_attempts_below_one():
    class _V:
        name = "v"

        async def validate(self, result, *, task):
            return Decision.PASS

    with pytest.raises(ValidationError):
        RequestCascade(verifier=_V(), max_attempts=0)


async def test_runtime_validator_object_satisfies_protocol():
    """Phase 3 v1: AMBIGUOUS treated as FAIL by the cascade. Validators
    can return any Decision; the AMBIGUOUS handling lives in the run loop,
    not here. This test only checks that AMBIGUOUS is a returnable value."""

    class _V:
        name = "v"

        async def validate(self, result, *, task):
            return Decision.AMBIGUOUS

    v = _V()
    out = await v.validate(None, task="t")
    assert out == Decision.AMBIGUOUS


async def test_cascade_returns_sas_result_when_validator_passes(make_container, stub_llm):
    """SAS attempt PASSes the validator → return SAS result; no escalation."""
    from voussoir.agent.agent import Agent

    p = stub_llm(content="A faithful answer.", input_tokens=5, output_tokens=2)

    class _PassValidator:
        name = "pass"

        async def validate(self, result, *, task):
            return Decision.PASS

    smart_lead = Agent(
        name="smart_lead",
        cascade=RequestCascade(verifier=_PassValidator()),
        container=make_container(p),
    )
    result = await smart_lead.run("question")
    assert result.output == "A faithful answer."
    assert p.chat.await_count == 1  # only the SAS attempt


async def test_cascade_escalates_on_fail(make_container, stub_llm):
    """Validator FAIL → run cascade.escalation, return its result."""
    from voussoir.agent.agent import Agent

    sas_llm = stub_llm(content="weak SAS attempt", input_tokens=3, output_tokens=1)
    mas_llm = stub_llm(content="strong MAS answer", input_tokens=8, output_tokens=3)

    class _FailValidator:
        name = "fail"

        async def validate(self, result, *, task):
            return Decision.FAIL

    escalation = Agent(name="multi_lead", container=make_container(mas_llm))
    smart_lead = Agent(
        name="smart_lead",
        cascade=RequestCascade(
            verifier=_FailValidator(),
            escalation=escalation,
        ),
        container=make_container(sas_llm),
    )
    result = await smart_lead.run("question")
    assert result.output == "strong MAS answer"


async def test_cascade_returns_last_attempt_when_max_attempts_exhausted(make_container, stub_llm):
    """Both attempts FAIL; cascade returns the last attempt with
    finish_reason='error'."""
    from voussoir.agent.agent import Agent

    p = stub_llm(content="still bad")

    class _AlwaysFail:
        name = "always-fail"

        async def validate(self, result, *, task):
            return Decision.FAIL

    escalation = Agent(name="other_lead", container=make_container(p))
    smart_lead = Agent(
        name="smart_lead",
        cascade=RequestCascade(
            verifier=_AlwaysFail(),
            escalation=escalation,
            max_attempts=2,
        ),
        container=make_container(p),
    )
    result = await smart_lead.run("question")
    assert result.finish_reason == "error"


async def test_cascade_treats_ambiguous_as_fail_and_escalates(make_container, stub_llm):
    """AMBIGUOUS is treated identically to FAIL by the cascade run loop:
    no PASS short-circuit, escalation runs. Catches the mutation
    `decision == PASS` → `!= FAIL` (which would treat AMBIGUOUS as PASS).
    """
    from voussoir.agent.agent import Agent

    sas_llm = stub_llm(content="weak SAS")
    mas_llm = stub_llm(content="strong MAS answer")

    class _AmbiguousValidator:
        name = "ambiguous"

        async def validate(self, result, *, task):
            return Decision.AMBIGUOUS

    escalation = Agent(name="multi_lead", container=make_container(mas_llm))
    smart_lead = Agent(
        name="smart_lead",
        cascade=RequestCascade(
            verifier=_AmbiguousValidator(),
            escalation=escalation,
        ),
        container=make_container(sas_llm),
    )
    result = await smart_lead.run("question")
    assert result.output == "strong MAS answer"
    # And the SAS attempt is recorded with reason="ambiguous".
    assert len(result.cascade_history) == 2
    assert result.cascade_history[0].reason == "ambiguous"
    assert result.cascade_history[0].escalated is True


async def test_cascade_sas_attempt_suppresses_delegate_tools(make_container, stub_llm):
    """When `cascade` triggers an SAS-first attempt, the agent's own
    `delegates` must not be synthesized as tools — the whole point of SAS
    is to gate single-agent capability before deploying multi-agent. Catches
    the mutation `_force_sas=True` → `False` in `_run_with_cascade`."""
    from voussoir.agent.agent import Agent

    sas_llm = stub_llm(content="ok")

    class _PassV:
        name = "pass"

        async def validate(self, result, *, task):
            return Decision.PASS

    # `delegates` is set on smart_lead, but `cascade` should keep them
    # off the SAS attempt's tool list.
    sub = Agent(name="sub", container=make_container(stub_llm()))
    smart_lead = Agent(
        name="smart_lead",
        delegates=[sub],
        cascade=RequestCascade(verifier=_PassV()),
        container=make_container(sas_llm),
    )
    await smart_lead.run("question")

    fns = sas_llm.chat.call_args.kwargs.get("functions") or []
    assert not any(f["name"].startswith("delegate_to_") for f in fns)


async def test_cascade_skips_sas_when_sas_attempt_first_false(make_container, stub_llm):
    """With sas_attempt_first=False, attempt 0 runs escalation directly;
    self.run's normal path is never invoked. Catches the mutation
    `attempt == 0 and sas_attempt_first` → `or` (which would always
    take the SAS path on attempt 0)."""
    from voussoir.agent.agent import Agent

    sas_llm = stub_llm(content="should not be called")
    mas_llm = stub_llm(content="MAS first")

    class _PassV:
        name = "pass"

        async def validate(self, result, *, task):
            return Decision.PASS

    escalation = Agent(name="multi_lead", container=make_container(mas_llm))
    smart_lead = Agent(
        name="smart_lead",
        cascade=RequestCascade(
            verifier=_PassV(),
            escalation=escalation,
            sas_attempt_first=False,
        ),
        container=make_container(sas_llm),
    )
    result = await smart_lead.run("question")
    assert result.output == "MAS first"
    sas_llm.chat.assert_not_awaited()
    mas_llm.chat.assert_awaited_once()


def test_request_cascade_default_max_cascade_depth_is_three():
    class _V:
        name = "v"

        async def validate(self, result, *, task):
            return Decision.PASS

    c = RequestCascade(verifier=_V())
    assert c.max_cascade_depth == 3


def test_request_cascade_rejects_max_cascade_depth_below_one():
    class _V:
        name = "v"

        async def validate(self, result, *, task):
            return Decision.PASS

    with pytest.raises(ValidationError):
        RequestCascade(verifier=_V(), max_cascade_depth=0)


async def test_cascade_refuses_recursion_past_max_cascade_depth(make_container, stub_llm):
    """Self-referencing escalation chain hits the depth cap and raises
    RuntimeError rather than recursing until Python's stack runs out.
    """
    from voussoir.agent.agent import Agent

    sas_llm = stub_llm(content="weak")

    class _AlwaysFail:
        name = "fail"

        async def validate(self, result, *, task):
            return Decision.FAIL

    # Build the cascade-aware Agent, then mutate its cascade to point
    # the escalation back at itself. RequestCascade can't reference the
    # Agent at construction time without this two-step.
    agent = Agent(name="recursive", container=make_container(sas_llm))
    agent.cascade = RequestCascade(
        verifier=_AlwaysFail(),
        escalation=agent,
        max_cascade_depth=2,
        max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="max_cascade_depth"):
        await agent.run("question")
