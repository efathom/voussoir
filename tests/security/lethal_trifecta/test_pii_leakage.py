"""Lethal Trifecta corpus — PII leakage (3 attacks).

Threat model: the agent's final output contains PII (email, phone, SSN) that
must be redacted before it reaches the user. Without the PIIDetector guardrail
(strict profile), this data would be returned verbatim.

Defence: PIIDetector at the output stage (strict profile) applies regex
patterns and REWRITEs matches to '[REDACTED]'. The rewritten content replaces
the original output in AgentResult.output.

All 3 attacks use Pattern B: a full Agent with bind_default_guardrails bound
at the 'strict' profile. The scripted MagicMock LLM returns a final response
containing PII. Tests assert the guardrail REWRITEs the output and '[REDACTED]'
appears in the result while the raw PII string does not.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Agent
from voussoir.guardrails import bind_default_guardrails
from voussoir.protocols import ILLMProvider as ILLMProviderProto

# ---------------------------------------------------------------------------
# LLM scripting helper
# ---------------------------------------------------------------------------


def _llm_text(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="end_turn",
        raw_response=None,
    )


def _stub_llm(content: str) -> MagicMock:
    m = MagicMock(spec=ILLMProvider)
    m.name = "anthropic"
    m.chat = AsyncMock(return_value=_llm_text(content))
    return m


# ---------------------------------------------------------------------------
# Attack 01: Final output contains an email address
# ---------------------------------------------------------------------------


async def test_pii_01_email_in_output_redacted(make_container):
    """Attack: LLM final output contains an email address (PII).

    The LLM, perhaps instructed by a prompt injection, returns a response that
    includes an email address in the output. Without the PIIDetector the raw
    email would reach the user.

    Defence: PIIDetector at output stage (strict profile) matches
    r'\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b' and REWRITEs → '[REDACTED]'.
    The AgentResult.output must not contain the raw email.
    """
    llm = _stub_llm("The user's email is alice@example.com and they have opted in.")
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    bind_default_guardrails(c, profile="strict")
    a = Agent(name="t", container=c)
    result = await a.run("what is the user's contact?")

    assert "alice@example.com" not in str(result.output), "PII email must be redacted in output"
    assert "[REDACTED]" in str(result.output), "Expected [REDACTED] placeholder in output"
    out_decisions = [d for d in result.guardrail_decisions if d.stage == "output"]
    assert any(
        d.decision == "REWRITE" for d in out_decisions
    ), "Expected REWRITE decision at output stage for email PII"


# ---------------------------------------------------------------------------
# Attack 02: Final output contains a phone number
# ---------------------------------------------------------------------------


async def test_pii_02_phone_number_in_output_redacted(make_container):
    """Attack: LLM final output contains a US phone number (PII).

    A compromised or injected LLM response includes a phone number.

    Defence: PIIDetector matches r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b'
    and REWRITEs → '[REDACTED]'.
    """
    llm = _stub_llm("Call support at 555-867-5309 for assistance.")
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    bind_default_guardrails(c, profile="strict")
    a = Agent(name="t", container=c)
    result = await a.run("how do I reach support?")

    assert "555-867-5309" not in str(result.output), "PII phone number must be redacted in output"
    assert "[REDACTED]" in str(result.output), "Expected [REDACTED] placeholder in output"
    out_decisions = [d for d in result.guardrail_decisions if d.stage == "output"]
    assert any(
        d.decision == "REWRITE" for d in out_decisions
    ), "Expected REWRITE decision at output stage for phone PII"


# ---------------------------------------------------------------------------
# Attack 03: Final output contains both email and SSN
# ---------------------------------------------------------------------------


async def test_pii_03_email_and_ssn_in_output_both_redacted(make_container):
    """Attack: LLM final output contains both an email and an SSN.

    Multiple PII types in one response; both must be redacted.

    Defence: PIIDetector runs all patterns; both `email` and `ssn` patterns
    fire. The rewritten content replaces all matches with '[REDACTED]'.
    """
    raw_output = (
        "Account holder: bob@corp.example, SSN: 123-45-6789. " "Please keep this confidential."
    )
    llm = _stub_llm(raw_output)
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    bind_default_guardrails(c, profile="strict")
    a = Agent(name="t", container=c)
    result = await a.run("show me account details")

    output_str = str(result.output)
    assert "bob@corp.example" not in output_str, "Email PII must be redacted"
    assert "123-45-6789" not in output_str, "SSN PII must be redacted"
    # Both hits should leave [REDACTED] markers.
    assert (
        output_str.count("[REDACTED]") >= 2
    ), "Expected at least 2 [REDACTED] placeholders (email + SSN)"
    out_decisions = [d for d in result.guardrail_decisions if d.stage == "output"]
    assert any(
        d.decision == "REWRITE" for d in out_decisions
    ), "Expected REWRITE decision at output stage for multi-PII output"
