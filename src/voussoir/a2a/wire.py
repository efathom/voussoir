"""Phase 4.5a P0 #3 — sanitized wire response shape for A2A peers.

The pre-4.5a wire format returned `AgentResult.model_dump()` which
includes `steps` (with tool args + 200-char output_preview), the full
`delegation_chain`, and `trace_id`. That data is internal trace and
may contain PII or secrets. Phase 4.5a redacts the JSON-RPC response
to the user-observable summary fields only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from voussoir.agent.result import AgentResult, FinishReason
from voussoir.guardrails.trust import Trust


class WireAgentResult(BaseModel):
    """Redacted projection of AgentResult for over-the-wire JSON-RPC.

    Excludes: steps, delegation_chain, trace_id, cascade_history,
    guardrail_decisions. Includes the user-observable summary plus the two
    fields the CALLER needs to keep enforcing its own budgets:

    - `taint` — the run's trust labels. v1.0.2 D5 made local sub-agents
      propagate taint so a sub-agent that reads UNTRUSTED content can't hand a
      clean context back and let the parent call EXFILTRATION tools. Leaving it
      off the wire reopened exactly that bypass one network hop out: a remote
      peer that read a poisoned page returned clean taint (audit H3). Coarse
      trust labels are not lineage, so carrying them doesn't undercut the
      redaction rationale.
    - `cost_usd` — without it a delegating agent's `AgentPolicy.max_cost_usd`
      could not bound spend across an A2A hop at all.
    """

    model_config = ConfigDict(extra="forbid")

    output: str
    finish_reason: FinishReason
    tokens_in: int
    tokens_out: int
    duration_ms: float
    taint: set[Trust] = Field(default_factory=set)
    cost_usd: float = 0.0

    @classmethod
    def from_agent_result(cls, r: AgentResult[str]) -> WireAgentResult:
        return cls(
            output=r.output,
            finish_reason=r.finish_reason,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            duration_ms=r.duration_ms,
            taint=set(r.taint),
            cost_usd=r.cost_usd,
        )
