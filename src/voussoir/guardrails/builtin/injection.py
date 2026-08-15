"""Prompt-injection heuristic guardrail (regex-based).

Use this on input or tool_output stages when you want a cheap, fast
deterministic check against common prompt-injection patterns. Not exhaustive —
catches the most common cases ("ignore previous instructions",
"SYSTEM: you are now…", "reveal your prompt"). For stronger coverage, layer
an LLMGuardrailJudge.

`stage` is configurable so the same regex defence can guard user input AND
attacker-controlled tool output. The standard profile registers two
instances — one for `input`, one for `tool_output`.
"""

from __future__ import annotations

import re
from typing import Literal

from voussoir.guardrails.protocol import GuardrailPayload, GuardrailVerdict

_PATTERNS = [
    re.compile(r"ignore\s+(?:previous|prior|all|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(?:previous|prior|all|above)", re.I),
    re.compile(r"system\s*[:>]\s*you\s+(?:are|must|will)", re.I),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?prompt", re.I),
]


def find_injection_pattern(text: str) -> re.Pattern[str] | None:
    """Return the first injection pattern that matches *text*, or None.

    Use this when you need to screen text for prompt-injection signatures
    outside the guardrail-chain flow (e.g. MCP tool description sanitization
    at registration time). The PromptInjectionHeuristic guardrail uses the
    same patterns; this helper exposes them for non-chain call sites.
    """
    for pat in _PATTERNS:
        if pat.search(text):
            return pat
    return None


class PromptInjectionHeuristic:
    """Prompt-injection heuristic guardrail (regex-based).

    Use this on input or tool_output stages when you want a cheap, fast
    deterministic first layer of injection defence. The standard profile
    registers two instances — one for `input`, one for `tool_output`.
    Layer an LLMGuardrailJudge for deeper coverage.
    """

    name = "prompt_injection"

    def __init__(
        self,
        *,
        stage: Literal["input", "tool_call", "tool_output", "output"] = "input",
    ) -> None:
        self.stage: Literal["input", "tool_call", "tool_output", "output"] = stage

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        for pat in _PATTERNS:
            if pat.search(payload.content):
                return GuardrailVerdict(
                    verdict="BLOCK",
                    reason=f"matches prompt-injection pattern: {pat.pattern!r}",
                )
        return GuardrailVerdict(verdict="ALLOW")
