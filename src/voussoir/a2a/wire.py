"""Phase 4.5a P0 #3 — sanitized wire response shape for A2A peers.

The pre-4.5a wire format returned `AgentResult.model_dump()` which
includes `steps` (with tool args + 200-char output_preview), the full
`delegation_chain`, and `trace_id`. That data is internal trace and
may contain PII or secrets. Phase 4.5a redacts the JSON-RPC response
to the user-observable summary fields only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from voussoir.agent.result import AgentResult, FinishReason


class WireAgentResult(BaseModel):
    """Redacted projection of AgentResult for over-the-wire JSON-RPC.

    Excludes: steps, delegation_chain, trace_id, cascade_history,
    guardrail_decisions. Includes only user-observable summary.
    """

    model_config = ConfigDict(extra="forbid")

    output: str
    finish_reason: FinishReason
    tokens_in: int
    tokens_out: int
    duration_ms: float

    @classmethod
    def from_agent_result(cls, r: AgentResult[str]) -> WireAgentResult:
        return cls(
            output=r.output,
            finish_reason=r.finish_reason,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            duration_ms=r.duration_ms,
        )
