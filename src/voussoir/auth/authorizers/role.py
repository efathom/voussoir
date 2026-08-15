"""RoleAuthorizer -- RBAC. Principal must have at least one of the tool's declared roles.

Use this for role-based access control. Apply @tool(roles=["admin", "ops"]) and
ensure principals carry their roles via Principal.roles.
"""

from __future__ import annotations

from voussoir.auth.decision import AuthzDecision


class RoleAuthorizer:
    """Allow when principal.roles intersects tool.roles; deny otherwise.

    Use this when tools declare role requirements via @tool(roles=[...]).
    Tools without role declarations are always allowed (permissive default).
    """

    name = "role"

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        """Allow if principal holds at least one of the tool's required roles."""
        del args, ctx
        tool_roles = list(getattr(tool, "roles", []) or [])
        if not tool_roles:
            return AuthzDecision(
                decision="ALLOW",
                authorizer_name=self.name,
                reason="tool declares no role requirement",
            )
        principal_roles: list[str] = list(getattr(principal, "roles", []) or [])
        if set(principal_roles) & set(tool_roles):
            return AuthzDecision(
                decision="ALLOW",
                authorizer_name=self.name,
                reason=f"principal has matching role(s) for {tool_roles}",
            )
        return AuthzDecision(
            decision="DENY",
            authorizer_name=self.name,
            reason=(
                f"principal roles {principal_roles} do not intersect" f" tool roles {tool_roles}"
            ),
        )
