"""LLMGuardrailJudge — AMBIGUOUS fallback composer.

Use this when you want a deterministic guardrail's AMBIGUOUS verdicts routed
to an LLM-backed binary judge. Mirrors Phase 3.5's `AmbiguousFallback` +
`LLMJudge` validator pattern, ported to the guardrail surface.

The judge:
1. Calls the wrapped primary guardrail's `screen()`.
2. If the primary returned anything other than AMBIGUOUS, passes the verdict
   through unchanged (ALLOW / BLOCK / REWRITE all surface as-is).
3. If the primary returned AMBIGUOUS, calls an LLM via the container's
   `ILLMProvider` binding with a forced-binary prompt. The LLM's "PASS" or
   "FAIL" response maps to GuardrailVerdict ALLOW or BLOCK respectively.
   Malformed responses default to BLOCK (fail-closed).

Resolves `ILLMProvider` from the container at `screen()` call time, not at
construction — keeps the judge usable as a plain `Guardrail` without exposing
the container to callers of `screen()`.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from ctxforge.protocols.llm import ChatMessage, ILLMProvider

from voussoir.container import Container
from voussoir.guardrails.protocol import Guardrail, GuardrailPayload, GuardrailVerdict
from voussoir.observability.sink import ITelemetrySink

DEFAULT_GUARDRAIL_JUDGE_PROMPT = """\
You are a guardrail decision judge. A deterministic guardrail named {name!r}
flagged the following content at stage {stage!r} as AMBIGUOUS.

Primary reason: {reason}

Content:
---
{content}
---

Decide whether this content should be ALLOWED to proceed or BLOCKED.

Reply with exactly one word: PASS (for ALLOW) or FAIL (for BLOCK).
"""


class LLMGuardrailJudge:
    """Wraps a primary `Guardrail`; routes AMBIGUOUS verdicts to an LLM judge.

    Use this when a deterministic guardrail returns AMBIGUOUS (e.g., a
    heuristic that isn't sure whether content is prompt-injection or
    legitimate). The judge forces a binary verdict via an LLM call.
    Construct one wrapper per primary guardrail; the wrapper's `stage`
    mirrors the primary's so the chain still groups it correctly.
    """

    def __init__(
        self,
        primary: Guardrail,
        *,
        container: Container,
        model: str | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self._primary = primary
        self._container = container
        self._model = model
        self._prompt_template = prompt_template or DEFAULT_GUARDRAIL_JUDGE_PROMPT
        for placeholder in ("{name", "{stage", "{reason", "{content"):
            if placeholder not in self._prompt_template:
                raise ValueError(
                    f"prompt_template must include placeholder {placeholder + '}'!r}; "
                    f"got: {self._prompt_template!r}"
                )

    @property
    def name(self) -> str:
        return f"{self._primary.name}+llm_judge"

    @property
    def stage(self) -> Literal["input", "tool_call", "tool_output", "output"]:
        return self._primary.stage

    async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict:
        primary = await self._primary.screen(payload, ctx)
        if primary.verdict != "AMBIGUOUS":
            return primary
        # AMBIGUOUS → forced binary LLM judge.
        prompt = self._prompt_template.format(
            name=self._primary.name,
            stage=self._primary.stage,
            reason=primary.reason,
            content=payload.content[:4000],  # cap LLM-visible content to avoid budget blowups
        )
        llm = self._container.resolve(ILLMProvider)
        sink = self._container.resolve(
            ITelemetrySink,  # type: ignore[type-abstract]
            default=None,
        )
        t0 = time.monotonic()
        # audit M3: `model` was stored in __init__ and never read, so a judge
        # configured with a cheap adjudication model silently ran every
        # judgment on the provider default.
        response = await llm.chat([ChatMessage(role="user", content=prompt)], model=self._model)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if sink is not None:
            from voussoir.llm.pricing import compute_cost

            sink.record_llm_call(
                name=self.name,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                cost_usd=compute_cost(
                    response.model, response.input_tokens, response.output_tokens
                ),
                duration_ms=elapsed_ms,
                payload={"primary_name": self._primary.name, "primary_stage": self._primary.stage},
            )
        # Strip + uppercase; exact "PASS" → ALLOW, anything else → BLOCK
        # (LLMGuardrailJudge resolves AMBIGUOUS and must commit to a binary verdict;
        # malformed responses fail closed). Inlined v1.0.1 to remove the guardrails →
        # agent layering inversion that imported voussoir.agent.validators._parse_judge_verdict.
        is_pass = response.content.strip().upper() == "PASS"
        return GuardrailVerdict(
            verdict="ALLOW" if is_pass else "BLOCK",
            reason=f"llm_judge ({self._primary.name}): {response.content.strip()[:120]}",
        )
