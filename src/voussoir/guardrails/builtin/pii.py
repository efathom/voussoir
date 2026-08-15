"""PII detector guardrail (regex-based, REWRITE with [REDACTED]).

Use this on output-stage screening when you want to scrub common PII (email,
phone, SSN, credit-card-shaped digits) before the result reaches the user.
REWRITEs rather than BLOCKs — the rewritten content replaces the original.
"""

from __future__ import annotations

import re
from typing import Literal

from voussoir.guardrails.protocol import GuardrailPayload, GuardrailVerdict

_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone_us": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


class PIIDetector:
    """Redact common PII patterns from output content.

    Use this on output-stage screening when you want to scrub emails, phone
    numbers, SSNs, and credit-card-shaped numbers before results reach users.
    Returns REWRITE with sanitized content when PII is found; ALLOW otherwise.
    """

    name = "pii_detector"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "output"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        out = payload.content
        hits: list[str] = []
        for kind, pat in _PATTERNS.items():
            if pat.search(out):
                out = pat.sub("[REDACTED]", out)
                hits.append(kind)
        if hits:
            return GuardrailVerdict(
                verdict="REWRITE",
                rewrite=out,
                reason=f"PII redacted: {','.join(hits)}",
            )
        return GuardrailVerdict(verdict="ALLOW")
