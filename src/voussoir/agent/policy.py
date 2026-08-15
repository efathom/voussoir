"""AgentPolicy — stop conditions for an agent run."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from voussoir.errors import VoussoirError


class PolicyViolation(StrEnum):
    """Reasons an Agent run can be cut short by its `AgentPolicy`.

    Budget-style violations (`MAX_STEPS`, `MAX_DURATION`, `MAX_INPUT_TOKENS`,
    `MAX_OUTPUT_TOKENS`, `MAX_COST`) are raised when a budget threshold is
    crossed mid-run. Non-budget violations (`STREAMING_NOT_SUPPORTED`,
    `DELEGATE_NOT_FOUND`) share the same error type so callers have one
    place to catch policy failures. Capability and taint violations
    (`CAPABILITY_DENIED`, `TAINT_EXFILTRATION`, `CAPABILITY_CLAMPED_EMPTY`)
    are raised by executor-side enforcement and sub-agent clamping.
    """

    MAX_STEPS = "max_steps"
    MAX_DURATION = "max_duration"
    MAX_INPUT_TOKENS = "max_input_tokens"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    MAX_COST = "max_cost"
    # Phase 4.5a — non-budget violations surfaced through the same error type
    # so callers have one place to catch.
    STREAMING_NOT_SUPPORTED = "streaming_not_supported"
    DELEGATE_NOT_FOUND = "delegate_not_found"
    # Phase 5 — capability + taint enforcement
    CAPABILITY_DENIED = "capability_denied"
    TAINT_EXFILTRATION = "taint_exfiltration"
    CAPABILITY_CLAMPED_EMPTY = "capability_clamped_empty"
    # Phase 6 — authz enforcement
    AUTHZ_DENIED = "authz_denied"


class PolicyViolationError(VoussoirError):
    """Raised by AgentPolicy budget enforcement and security gates.

    Use this when catching agent-side policy failures. The `.violation`
    attribute is a `PolicyViolation` StrEnum naming the specific violation:

    Budget (raised only when `AgentPolicy.on_violation == "error"`):
      - MAX_STEPS, MAX_DURATION, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_COST

    Security (always raised regardless of policy):
      - CAPABILITY_DENIED — tool capability not in agent.allowed_capabilities
      - TAINT_EXFILTRATION — EXFILTRATION tool called with UNTRUSTED taint
      - AUTHZ_DENIED — Authorizer returned DENY
      - CAPABILITY_CLAMPED_EMPTY — child-clamp left no tools

    Dispatch:
      - STREAMING_NOT_SUPPORTED — Agent.stream() called with max_cascade_depth > 1
      - DELEGATE_NOT_FOUND — string delegate has no registry entry
    """

    def __init__(self, violation: PolicyViolation, message: str) -> None:
        super().__init__(message)
        self.violation = violation


class AgentPolicy(BaseModel):
    """Per-run budgets and the response when a budget is breached.

    Use this when you want hard limits on a run's step count, duration,
    token usage, or estimated cost. `on_violation` chooses between raising
    `PolicyViolationError` ("error") or terminating the loop cleanly with
    a summary step ("summarize_and_stop"). Defaults are conservative; tune
    per-agent for production workloads.
    """

    max_steps: int = 25
    max_duration_s: float = 300.0
    max_input_tokens: int = 200_000
    max_output_tokens: int = 8_000
    max_cost_usd: float = 1.00
    on_violation: Literal["error", "summarize_and_stop"] = "summarize_and_stop"

    def check(
        self,
        *,
        steps: int,
        duration_s: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> PolicyViolation | None:
        """Check current run state against budgets.

        Returns the violation if any budget is exceeded; raises if
        ``on_violation == "error"``. Returns None if all budgets are within limits.
        """
        violation = self._violation(
            steps=steps,
            duration_s=duration_s,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        if violation is None:
            return None
        if self.on_violation == "error":
            raise PolicyViolationError(violation, f"Policy budget exceeded: {violation.value}")
        return violation

    def _violation(
        self,
        *,
        steps: int,
        duration_s: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> PolicyViolation | None:
        if steps >= self.max_steps:
            return PolicyViolation.MAX_STEPS
        if duration_s >= self.max_duration_s:
            return PolicyViolation.MAX_DURATION
        if tokens_in >= self.max_input_tokens:
            return PolicyViolation.MAX_INPUT_TOKENS
        if tokens_out >= self.max_output_tokens:
            return PolicyViolation.MAX_OUTPUT_TOKENS
        if cost_usd >= self.max_cost_usd:
            return PolicyViolation.MAX_COST
        return None
