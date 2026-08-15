"""Agent-side bridge that merges buffered telemetry into an AgentResult.

Phase 4.5a Task 1: was `BufferedTelemetrySink.merge_into` in
`voussoir.observability.sink`. Moved here to restore one-way layering —
observability is the lower layer and has no agent imports; the merge
happens on the agent side which already knows about AgentResult/Step.
"""

from __future__ import annotations

from typing import Any

from voussoir.agent.result import AgentResult, Step
from voussoir.observability.sink import TelemetryRecord


def merge_buffered_telemetry_into_result(
    result: AgentResult[Any], records: list[TelemetryRecord]
) -> AgentResult[Any]:
    """Return a new AgentResult with `records` folded in.

    Aggregates `tokens_in / tokens_out / cost_usd` deltas from records and
    appends one Step per record. Pure: input result is not mutated.
    """
    delta_in = sum(r.tokens_in for r in records)
    delta_out = sum(r.tokens_out for r in records)
    delta_cost = sum(r.cost_usd for r in records)
    new_steps = list(result.steps) + [
        Step(
            kind=r.kind,
            name=r.name,
            duration_ms=r.duration_ms,
            payload=r.payload,
        )
        for r in records
    ]
    return result.model_copy(
        update={
            "tokens_in": result.tokens_in + delta_in,
            "tokens_out": result.tokens_out + delta_out,
            "cost_usd": result.cost_usd + delta_cost,
            "steps": new_steps,
        }
    )
