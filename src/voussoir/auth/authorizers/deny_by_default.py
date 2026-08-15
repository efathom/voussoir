"""DenyByDefaultAuthorizer — the fail-closed default.

`default_container()` binds this authorizer so tool invocations deny by default
until an operator layers a concrete grant source (e.g. `ChainedAuthorizer` with
a `RoleAuthorizer` / `DomainAuthorizer`). For a permissive dev posture, bind
`AllowAllAuthorizer` explicitly (it still emits a one-time `authz_unenforced`
warning).
"""

from __future__ import annotations

from voussoir.auth.decision import AuthzDecision


class DenyByDefaultAuthorizer:
    """Fail-closed authorizer; DENYs every authorize() call.

    A concrete grant source must be layered on top (e.g. ChainedAuthorizer with
    a RoleAuthorizer / DomainAuthorizer) for any tool to be invocable.
    """

    name = "deny_by_default"

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        del principal, tool, args, ctx
        return AuthzDecision(
            decision="DENY",
            reason="no authorizer grant: DenyByDefaultAuthorizer fails closed",
            authorizer_name=self.name,
        )
