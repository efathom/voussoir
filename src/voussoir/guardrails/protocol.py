"""Guardrail Protocol — soft policies that screen content at four stages.

Use this when implementing a custom guardrail; the framework binds your
implementations into a DefaultGuardrailChain (see voussoir.guardrails.chain).

`GuardrailVerdict` is the EPHEMERAL return value from screen(). Distinct
from `voussoir.agent.result.GuardrailDecision`, which is the *audit-log
record* stored in AgentResult.guardrail_decisions.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from voussoir.guardrails.trust import Trust
from voussoir.tools.protocol import Capability


class GuardrailPayload(BaseModel):
    """What a guardrail screens.

    Use this when constructing a payload to pass into `Guardrail.screen()`,
    or when writing a custom guardrail that pattern-matches on `stage`.

    `stage` determines which optional fields are populated. `content` is always
    set. For `tool_call` and `tool_output`, `tool_name` is set (and `tool_args`
    is set for `tool_call`, `trust`/`capability` for `tool_output`). For
    `input` and `output`, only `content` is meaningful.
    """

    model_config = ConfigDict(extra="forbid")

    stage: Literal["input", "tool_call", "tool_output", "output"]
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    trust: Trust | None = None
    capability: Capability | None = None


class GuardrailVerdict(BaseModel):
    """Ephemeral return value from `Guardrail.screen()`.

    Use this when returning from your `screen()` implementation: return
    `verdict="ALLOW"` if content passes, `"BLOCK"` to halt dispatch (with a
    `reason`), `"REWRITE"` with `rewrite=<replacement>` to sanitize, or
    `"AMBIGUOUS"` to defer to an LLM judge (see `LLMGuardrailJudge`, B4).

    Distinct from voussoir.agent.result.GuardrailDecision — the latter is the
    *audit-log record* of a verdict stored on AgentResult.guardrail_decisions.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["ALLOW", "BLOCK", "REWRITE", "AMBIGUOUS"]
    reason: str = ""
    rewrite: str | None = None


@runtime_checkable
class Guardrail(Protocol):
    """A soft policy that screens content at one of four stages.

    Use this when implementing a custom guardrail; the framework binds your
    implementations under this Protocol key and the `DefaultGuardrailChain`
    (see `voussoir.guardrails.chain`, B2) dispatches them per-stage.
    """

    name: str
    stage: Literal["input", "tool_call", "tool_output", "output"]

    async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict: ...


@runtime_checkable
class IGuardrailChain(Protocol):
    """Use this when implementing a non-default chain — remote policy
    service, dynamic chain composition, profile loader, etc.

    DefaultGuardrailChain (voussoir.guardrails.chain) is the default impl
    bound on default_container() with an empty per-stage list. Custom impls
    bind via c.bind(IGuardrailChain, MyChain()) or pass per-run via
    agent.run(guardrail_chain=...).
    """

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict: ...

    def count(self) -> int: ...
