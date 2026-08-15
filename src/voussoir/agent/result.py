"""AgentResult and AgentEvent — public output types for Agent.run / Agent.stream."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, Field

from voussoir.auth.decision import AuthzDecision
from voussoir.guardrails.trust import Trust

if TYPE_CHECKING:
    # Imported lazily at runtime inside `to_wire` to avoid a module-import cycle
    # (voussoir.a2a.wire imports AgentResult). The @overload signatures below
    # need WireAgentResult resolvable at type-check time.
    from voussoir.a2a.wire import WireAgentResult

Out = TypeVar("Out")

# Canonical `finish_reason` literal. Lives here (the result module) since
# AgentResult.finish_reason is the field that ultimately carries this value;
# `voussoir.agent.agent` and `voussoir.a2a.wire` import this alias rather
# than redefining the 8 variants. PEP 695 `type` statement so the alias has
# a runtime __doc__ (passes the public-symbol docstring sweep).
type FinishReason = Literal[
    "completed",
    "max_steps",
    "max_duration",
    "max_input_tokens",
    "max_output_tokens",
    "max_cost",
    "blocked",
    "error",
]


class Step(BaseModel):
    """A single step in an agent's run — LLM call, tool call, delegation, guardrail, or validator call."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "llm_call",
        "tool_call",
        "delegation",
        "guardrail",
        "validator_call",
    ]
    name: str
    duration_ms: float
    payload: dict[str, Any] = {}


class AgentEvent(BaseModel):
    """Streaming event yielded by Agent.stream()."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "token",
        "tool_started",
        "tool_finished",
        "tool_progress",
        "delegation_started",
        "delegation_finished",
        "cascade_escalated",
        "cascade_passed",
        "cascade_failed",
        "guardrail_triggered",
        "checkpoint",
        "error",
        "done",
    ]
    payload: dict[str, Any]
    span_id: str
    timestamp: datetime


class CascadeOutcome(BaseModel):
    """Outcome of a Request Cascading attempt — populated by Phase 3."""

    model_config = ConfigDict(extra="forbid")

    attempted: str
    escalated: bool
    reason: str = ""


class GuardrailDecision(BaseModel):
    """Guardrail verdict recorded for the run — audit log of what fired.

    Populated when a guardrail returns non-ALLOW from screen(). Distinct from
    voussoir.guardrails.GuardrailVerdict (the ephemeral return value).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    stage: Literal["input", "tool_call", "tool_output", "output"]
    decision: Literal["ALLOW", "BLOCK", "REWRITE", "AMBIGUOUS"]
    reason: str = ""
    rewrite: str | None = None


class AgentResult(BaseModel, Generic[Out]):  # noqa: UP046
    """Final result returned by Agent.run() / .run_sync()."""

    model_config = ConfigDict(extra="forbid")

    output: Out
    trace_id: str
    steps: list[Step]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    duration_ms: float
    delegation_chain: list[str]
    cascade_history: list[CascadeOutcome]
    guardrail_decisions: list[GuardrailDecision]
    authz_decisions: list[AuthzDecision] = Field(default_factory=list)
    # v1.0.2 D5: carry the run's taint set so a sub-agent's UNTRUSTED bits
    # propagate back to the parent. Without this, a sub-agent that reads
    # UNTRUSTED content "launders" its taint — the parent could then call
    # EXFILTRATION tools unblocked (Lethal Trifecta delegation bypass).
    # accumulate_outcomes merges this into the parent's AgentContext.taint.
    taint: set[Trust] = Field(default_factory=set)
    finish_reason: FinishReason

    @overload
    def to_wire(self, profile: Literal["public"] = "public") -> WireAgentResult: ...
    @overload
    def to_wire(self, profile: Literal["trusted"]) -> AgentResult[Out]: ...

    def to_wire(
        self, profile: Literal["public", "trusted"] = "public"
    ) -> AgentResult[Out] | WireAgentResult:
        """Use this when serializing for A2A. Two profiles:

        - `"public"` (default): returns a `voussoir.a2a.WireAgentResult` —
          strips `steps`, `delegation_chain`, `cascade_history`,
          `guardrail_decisions`, `trace_id`, `cost_usd`. Default for
          `make_a2a_router` since peers should not see lineage / costs.
        - `"trusted"`: returns `self` unchanged — full dump for intra-org
          peering where the caller is trusted to see lineage. Opt-in only
          via `make_a2a_router(wire_profile="trusted")`.
        """
        if profile == "public":
            # Lazy import — voussoir.a2a.wire imports AgentResult, so a
            # top-level import here would form a cycle. TYPE_CHECKING import
            # at module top covers the @overload signatures.
            from voussoir.a2a.wire import WireAgentResult

            return WireAgentResult.from_agent_result(cast("AgentResult[str]", self))
        return self
