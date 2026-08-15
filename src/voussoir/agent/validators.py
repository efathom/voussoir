"""Built-in cascade validators.

Phase 3 shipped ToolUseFaithfulness — a permissive heuristic detecting
regression-prone "I called the X tool" claims that don't map to recorded
tool_call Steps.

Phase 3.5 adds LLMJudge (a one-shot LLM-driven validator) and
AmbiguousFallback (a two-stage composer: primary first, judge only on
AMBIGUOUS). Validator-call cost rolls into AgentResult via the
cascade-scoped ITelemetrySink (see voussoir.observability.sink).
"""

from __future__ import annotations

import re
import time
from typing import Any

from ctxforge.protocols.llm import ChatMessage, ILLMProvider

from voussoir.agent.cascade import Decision, Validator
from voussoir.agent.result import AgentResult
from voussoir.container import Container
from voussoir.observability.sink import ITelemetrySink


class ToolUseFaithfulness:
    """Validates that every tool-use claim in result.output maps to an
    actual tool_call Step in result.steps.

    Detects two definite claim patterns:
      - "[tool: <name>]" — explicit marker (suggested convention).
      - "I (called|invoked|used) (the )?<name>(\\s+tool|\\s+function)?" —
        natural-language claim from the LLM's narration.

    Plus one hedged claim pattern that signals uncertainty:
      - "I (might|may|possibly|think I|believe I) (have )?(called|invoked|used)
        (the )?<name>" — the model isn't sure whether it called a tool.

    Possible verdicts:
      - PASS if no claims found OR every definite claim matches an actual call.
      - FAIL if at least one definite claim has no actual call.
      - AMBIGUOUS if any hedged claim appears (regardless of match) — defer
        the verdict to a downstream LLM judge via `AmbiguousFallback`.
    """

    name = "tool_use_faithfulness"

    _CLAIM_RE = re.compile(
        r"\[tool:\s*([a-zA-Z0-9_-]+)\]"
        r"|I (?:called|invoked|used) (?:the )?([a-zA-Z0-9_-]+)(?:\s+tool|\s+function)?"
    )

    # Hedged-claim pattern. Matches the model wavering about whether it
    # used a tool — phrasings like "I might have used X", "I think I called Y",
    # "I believe I invoked Z". Returns AMBIGUOUS regardless of whether the
    # named tool was actually called: hedging is the signal, not the match.
    _HEDGE_RE = re.compile(
        r"I (?:might(?:\s+have)?|may(?:\s+have)?|possibly|think\s+I|believe\s+I)"
        r"\s+(?:called|invoked|used)"
    )

    async def validate(self, result: AgentResult[Any], *, task: str) -> Decision:
        # Heuristic only inspects result.steps + result.output; the
        # `task` kwarg exists for protocol compatibility (Phase 3.5).
        del task
        output = str(result.output)

        # Hedge detection runs first — if the model is uncertain, defer
        # regardless of whether definite claims also match.
        if self._HEDGE_RE.search(output):
            return Decision.AMBIGUOUS

        actual: set[str] = {s.name for s in result.steps if s.kind == "tool_call"}

        claimed: set[str] = set()
        for match in self._CLAIM_RE.finditer(output):
            name = match.group(1) or match.group(2)
            if name:
                claimed.add(name)

        if not claimed:
            return Decision.PASS

        unmatched = claimed - actual
        if unmatched:
            return Decision.FAIL
        return Decision.PASS


DEFAULT_LLM_JUDGE_PROMPT = """\
You are evaluating whether an agent's output meets a specific criterion.

TASK: {task}

OUTPUT: {output}

CRITERION: {criterion}

Reply with exactly one word: PASS or FAIL.
"""


def _parse_judge_verdict(raw: str) -> Decision:
    """Strip + uppercase. Exact 'PASS' → PASS; anything else → FAIL.

    LLMJudge never returns AMBIGUOUS — it is the AMBIGUOUS resolver and
    must commit to a binary verdict. Malformed responses are FAIL.
    """
    if raw.strip().upper() == "PASS":
        return Decision.PASS
    return Decision.FAIL


class LLMJudge:
    """Validator that issues an LLM call to render a binary verdict.

    Resolves an ILLMProvider from the container at validate() time and
    emits cost via ITelemetrySink (also from the container). The cascade
    scopes a BufferedTelemetrySink around validate() calls so the emission
    rolls into AgentResult.

    Always returns PASS or FAIL — never AMBIGUOUS. Anything other than
    an exact "PASS" response (after strip+upper) is treated as FAIL.
    """

    name: str = "llm_judge"

    def __init__(
        self,
        criterion: str,
        *,
        container: Container,
        model: str | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self.criterion = criterion
        self._container = container
        self._model = model
        self._prompt_template = prompt_template or DEFAULT_LLM_JUDGE_PROMPT
        for placeholder in ("{task}", "{output}", "{criterion}"):
            if placeholder not in self._prompt_template:
                raise ValueError(
                    f"prompt_template must include placeholder {placeholder!r}; "
                    f"got: {self._prompt_template!r}"
                )

    async def validate(self, result: AgentResult[Any], *, task: str) -> Decision:
        prompt = self._prompt_template.format(
            task=task,
            output=str(result.output),
            criterion=self.criterion,
        )
        llm = self._container.resolve(ILLMProvider)
        sink = self._container.resolve(
            ITelemetrySink,  # type: ignore[type-abstract]
            default=None,
        )
        t0 = time.monotonic()
        response = await llm.chat([ChatMessage(role="user", content=prompt)])
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
                payload={"criterion": self.criterion},
            )
        return _parse_judge_verdict(response.content)


class AmbiguousFallback:
    """Two-stage validator: primary first, judge only on AMBIGUOUS.

    Telemetry emitted by inner validators flows through whatever sink
    is active at validate() time (typically the cascade-scoped buffer
    set by Agent._run_with_cascade). No special wiring needed at the
    wrapper layer.
    """

    def __init__(self, primary: Validator, judge: Validator) -> None:
        self._primary = primary
        self._judge = judge
        self.name = f"ambiguous_fallback({primary.name},{judge.name})"

    async def validate(self, result: AgentResult[Any], *, task: str) -> Decision:
        primary_decision = await self._primary.validate(result, task=task)
        if primary_decision != Decision.AMBIGUOUS:
            return primary_decision
        return await self._judge.validate(result, task=task)
