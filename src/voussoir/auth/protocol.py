"""Auth Protocol definitions for voussoir.

`Authorizer` and `CredentialBroker` are the two extension points for the
two-axis auth model (Axis 1 = authn via CredentialBroker; Axis 2 = authz
via Authorizer). Concrete bindings live in `voussoir.auth.brokers.*` and
`voussoir.auth.authorizers.*`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from voussoir.auth.credentials import AuthRequirement, Credentials
from voussoir.auth.decision import AuthzDecision
from voussoir.auth.principal import Principal

if TYPE_CHECKING:
    from voussoir.tools.protocol import Tool, ToolContext


@runtime_checkable
class CredentialBroker(Protocol):
    """Resolve auth requirements into ready-to-use credentials.

    `resolve` is called by the runtime when a Tool declares an
    `AuthRequirement`; `refresh` is called when the broker (or caller) detects
    that previously-issued credentials have expired and need to be renewed
    in-place rather than re-resolved from scratch.
    """

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials: ...

    async def refresh(self, creds: Credentials) -> Credentials: ...


@runtime_checkable
class Authorizer(Protocol):
    """Decide whether a principal may invoke a tool with the given args.

    Use this when implementing a custom authz policy. Return an AuthzDecision
    with decision ALLOW/DENY/MASK. The StandardExecutor calls authorize()
    inline before capability + taint + guardrail-chain checks (per spec
    §12.1 ordering); DENY short-circuits with PolicyViolationError.
    """

    name: str

    async def authorize(
        self,
        principal: Principal,
        tool: Tool,
        args: BaseModel,
        ctx: ToolContext,
    ) -> AuthzDecision: ...
