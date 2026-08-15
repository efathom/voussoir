"""Tool Protocol — what a callable capability looks like to the Agent."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntFlag
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from voussoir.auth.credentials import Credentials
from voussoir.auth.decision import AuthzDecision
from voussoir.auth.principal import Principal, default_principal
from voussoir.container import Container
from voussoir.guardrails.trust import Trust


class Capability(IntFlag):
    """Capability scopes a tool requires.

    Use this when declaring `@tool(capability=...)` or checking what a tool needs.
    Bit values are stable wire ids — never reuse a freed bit.

    `NONE` is the empty flag (value 0). Under IntFlag subset semantics every
    non-empty flag contains it (`Capability.NONE in Capability.READ_PUBLIC` is
    True), so don't use `NONE in allowed` as a "no capability allowed" guard —
    test `allowed == Capability.NONE` instead.
    """

    NONE = 0
    READ_PUBLIC = 1 << 0  # public web, public docs
    READ_PRIVATE = 1 << 1  # internal corpus, internal APIs
    WRITE_PRIVATE = 1 << 2  # write to internal stores
    EXFILTRATION = 1 << 3  # send email, post external, render image-with-URL


def parse_capability_list(names: list[str] | None) -> Capability | None:
    """Parse a list of capability names into a Capability flag.

    Use this when applying yaml or env-var capability overrides; returns
    None when input is None (meaning "leave the builder's current value
    unchanged" — distinct from `Capability.NONE` which is "no capability
    allowed"). Raises ValueError with a list of valid names when an entry
    doesn't match a known Capability member.

    Promoted from `voussoir.agent.agent_builder._parse_capability_list` in
    v1.0.1 so CLI / config / builder paths share one public helper instead
    of crossing the agent/cli boundary via a private name.
    """
    if names is None:
        return None
    flag = Capability.NONE
    for name in names:
        try:
            flag |= Capability[name]
        except KeyError:
            valid = ", ".join(c.name for c in Capability if c.name)
            raise ValueError(f"Unknown capability {name!r}; valid values: {valid}") from None
    return flag


class ToolContext(BaseModel):
    """Per-invocation context passed to the tool's invoke().

    Use this when constructing test fixtures for the executor; production
    callers receive a context built by Agent.run().
    """

    run_id: str
    span_id: str
    allowed_capabilities: Capability = Capability.NONE
    taint: set[Trust] = Field(default_factory=set)
    # Phase 6 A2: principal + credentials threading.
    principal: Principal = Field(default_factory=default_principal)
    credentials: Credentials | None = None
    # Phase 6 A4: container for Authorizer / CredentialBroker resolution.
    container: Container | None = None
    # Phase 6 A4: per-invocation authz decisions (merged back into AgentContext by dispatch.py).
    authz_decisions: list[AuthzDecision] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    async def interrupt(self, kind: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Pause the agent for external input (v1.2).

        Default impl raises ``InterruptRequest`` -- the exception propagates
        through ``Agent.run`` via the existing on_error middleware fanout.
        Runtime layers (e.g. impost) catch the exception in a custom
        ``Middleware.on_error`` and persist it; on resume, replay re-enters
        the tool with a cached return value so this call appears to return
        the operator-supplied payload.

        Without a runtime, ``InterruptRequest`` bubbles out of ``Agent.run``
        to the caller -- no state is preserved.
        """
        # Local import: voussoir.agent.interrupts depends transitively on
        # voussoir.tools at runtime (via voussoir.agent.__init__ re-export),
        # so a module-level import here would create a cycle.
        from voussoir.agent.interrupts import InterruptRequest

        raise InterruptRequest(kind=kind, payload=payload)


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: type[BaseModel]
    capability: Capability

    async def invoke(self, args: BaseModel, ctx: ToolContext) -> Any: ...
