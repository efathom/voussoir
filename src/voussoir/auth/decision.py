"""AuthzDecision — record returned by Authorizer.authorize().

Use this when implementing a custom Authorizer: return ALLOW (proceed),
DENY (raises PolicyViolationError in the executor), or MASK (proceed +
strip fields from the tool output). Distinct from
voussoir.agent.result.GuardrailDecision (guardrail-chain verdicts) and
voussoir.agent.cascade.Decision (cascade verifier verdicts).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthzDecision(BaseModel):
    """Authorizer verdict + audit-log record.

    Use this when implementing a custom Authorizer: return AuthzDecision
    with decision ALLOW (proceed), DENY (executor raises PolicyViolationError),
    or MASK (proceed, then strip `masked_fields` from the tool output).

    Decisions accumulate in AgentResult.authz_decisions for the run.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ALLOW", "DENY", "MASK"]
    reason: str = ""
    masked_fields: list[str] = Field(default_factory=list)
    authorizer_name: str = ""
