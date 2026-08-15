"""DomainAuthorizer -- Fathom-style domain gating. principal.domains ∩ tool.domains must be non-empty.

Use this when tools are scoped to functional areas (e.g., "sre", "platform",
"backend") and principals carry their domain memberships via Principal.domains.
Apply @tool(domains=["sre"]) to restrict a tool to SRE principals only.
"""

from __future__ import annotations

from voussoir.auth.decision import AuthzDecision


class DomainAuthorizer:
    """Allow when principal.domains intersects tool.domains; deny otherwise.

    Use this when tools declare domain requirements via @tool(domains=[...]).
    Tools without domain declarations are always allowed (permissive default).
    """

    name = "domain"

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        """Allow if principal belongs to at least one of the tool's required domains."""
        del args, ctx
        tool_domains = list(getattr(tool, "domains", []) or [])
        if not tool_domains:
            return AuthzDecision(
                decision="ALLOW",
                authorizer_name=self.name,
                reason="tool declares no domain requirement",
            )
        principal_domains: list[str] = list(getattr(principal, "domains", []) or [])
        if set(principal_domains) & set(tool_domains):
            return AuthzDecision(
                decision="ALLOW",
                authorizer_name=self.name,
                reason=f"principal has matching domain(s) for {tool_domains}",
            )
        return AuthzDecision(
            decision="DENY",
            authorizer_name=self.name,
            reason=(
                f"principal domains {principal_domains} do not intersect"
                f" tool domains {tool_domains}"
            ),
        )
