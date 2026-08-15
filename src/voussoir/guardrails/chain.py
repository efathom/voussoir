"""DefaultGuardrailChain — the soft-policy dispatcher.

Use this when wiring a custom set of guardrails into an Agent; the framework's
`bind_default_guardrails` helper (in voussoir.guardrails.bootstrap) constructs
a chain populated by profile and binds it under the DefaultGuardrailChain
protocol key on the container.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from voussoir.guardrails.protocol import Guardrail, GuardrailPayload, GuardrailVerdict


class DefaultGuardrailChain:
    """Runs guardrails in declared order at each stage; first non-ALLOW short-circuits.

    Construction takes a flat `list[Guardrail]`; the chain groups them by `stage`
    internally so `screen(payload, ctx)` only dispatches the relevant subset. Empty
    chain returns `GuardrailVerdict(verdict="ALLOW")` for every call (a no-op).
    """

    def __init__(self, guardrails: list[Guardrail]) -> None:
        self._by_stage: dict[str, list[Guardrail]] = defaultdict(list)
        for g in guardrails:
            self._by_stage[g.stage].append(g)

    async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict:
        for g in self._by_stage[payload.stage]:
            verdict = await g.screen(payload, ctx)
            if verdict.verdict != "ALLOW":
                return verdict
        return GuardrailVerdict(verdict="ALLOW")

    def count(self) -> int:
        """Use this to get the total number of guardrails registered across all
        stages. Replaces private `chain._by_stage` access from outside the class.
        """
        return sum(len(gs) for gs in self._by_stage.values())
