"""Decision / Validator / RequestCascade — Phase 3 cascade primitives.

Run sequence (see plan §3.7 + design §6 for details):
  1. Single-agent attempt (delegates suppressed).
  2. validator.validate(result) → PASS / FAIL / AMBIGUOUS.
  3. PASS → return single-agent result.
  4. FAIL or AMBIGUOUS within max_attempts → run escalation Agent.
  5. Exhausted → return last result with finish_reason="error".

In v1, AMBIGUOUS is treated identically to FAIL by the run loop; a
real LLM-judge fallback for AMBIGUOUS lands in Phase 3.5.

`escalation` uses a string forward-ref to `Agent`. There is no actual
import cycle (agent.py does not import cascade.py at module top), so
the deferred resolution is just to break the alphabetical-import
ordering — `_ensure_cascade_rebuilt` (at the bottom of this module) is
called lazily from `RequestCascade.__init__` and finalises the model
on first construction (before Pydantic validates the input).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from voussoir.agent.result import AgentResult

if TYPE_CHECKING:
    from voussoir.agent.agent import Agent


class Decision(StrEnum):
    """A cascade validator's verdict on a candidate `AgentResult`.

    `PASS` accepts the result and stops the cascade; `FAIL` triggers
    escalation to the next attempt or escalation Agent; `AMBIGUOUS` is
    handled by an `AmbiguousFallback` policy (default: treat as `FAIL`).
    """

    PASS = "pass"
    FAIL = "fail"
    AMBIGUOUS = "ambiguous"


@runtime_checkable
class Validator(Protocol):
    """Protocol every cascade verifier satisfies.

    Implementations decide PASS / FAIL / AMBIGUOUS based on an
    AgentResult plus the original `task` (input string the agent
    received). They are async to allow LLM-judge implementations
    without API breakage.
    """

    name: str

    async def validate(self, result: AgentResult[Any], *, task: str) -> Decision: ...


class RequestCascade(BaseModel):
    """Single-agent-first, escalate-to-multi-agent on validator failure."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    sas_attempt_first: bool = True
    verifier: Validator
    escalation: Agent | None = None
    max_attempts: int = Field(default=2, ge=1)
    # Defends against escalation chains that point back to a cascade-aware
    # Agent (directly or transitively). Counts cascade re-entries, not
    # SAS attempts; the run loop guards on `current_depth >= max_cascade_depth`.
    max_cascade_depth: int = Field(default=3, ge=1)

    def __init__(self, /, **data: Any) -> None:
        # Trigger lazy forward-ref resolution before Pydantic validates the
        # constructor arguments. model_post_init can't be used here: when
        # input is invalid (e.g. max_attempts=0) Pydantic raises during
        # validation, before model_post_init would fire — and an unrebuilt
        # model refuses to validate at all.
        _ensure_cascade_rebuilt()
        super().__init__(**data)


def _ensure_cascade_rebuilt() -> None:
    """Lazily rebuild RequestCascade's forward refs on first construction.

    Pydantic sets RequestCascade.__pydantic_complete__ = True after the
    rebuild finishes; checking it here avoids re-importing Agent on every
    cascade construct. Tranche A's voussoir/agent/__init__.py called
    model_rebuild at module-top, requiring strict import ordering; this
    localizes the dependency.
    """
    if RequestCascade.__pydantic_complete__:
        return
    # Import Agent here (NOT at module-top) to avoid the import cycle
    # that motivated the original __init__-level rebuild.
    from voussoir.agent.agent import Agent

    RequestCascade.model_rebuild(_types_namespace={"Agent": Agent})
