"""voussoir.guardrails — Soft-policy guardrail chain, Trust taint model, and LLM judge.

Public API:
  Guardrail              — Protocol every guardrail must satisfy
  GuardrailPayload       — what a guardrail screens (stage + content + metadata)
  GuardrailVerdict       — what a guardrail returns (ALLOW / BLOCK / REWRITE / AMBIGUOUS)
  IGuardrailChain        — Protocol for chain implementations (v1.1.0 F2)
  DefaultGuardrailChain  — stage-indexed chain; first non-ALLOW short-circuits
  LLMGuardrailJudge      — wraps any Guardrail to resolve AMBIGUOUS via an LLM
  Trust                  — StrEnum for content trust levels (SYSTEM / USER / INTERNAL / UNTRUSTED)
  bind_default_guardrails — helper to register built-in guardrails at a given profile
  DEFAULT_GUARDRAIL_JUDGE_PROMPT — default system prompt for the LLM judge
"""

from voussoir.guardrails.bootstrap import bind_default_guardrails
from voussoir.guardrails.chain import DefaultGuardrailChain
from voussoir.guardrails.judge import (
    DEFAULT_GUARDRAIL_JUDGE_PROMPT,
    LLMGuardrailJudge,
)
from voussoir.guardrails.protocol import (
    Guardrail,
    GuardrailPayload,
    GuardrailVerdict,
    IGuardrailChain,
)
from voussoir.guardrails.trust import Trust

__all__ = [
    "DEFAULT_GUARDRAIL_JUDGE_PROMPT",
    "DefaultGuardrailChain",
    "Guardrail",
    "GuardrailPayload",
    "GuardrailVerdict",
    "IGuardrailChain",
    "LLMGuardrailJudge",
    "Trust",
    "bind_default_guardrails",
]
