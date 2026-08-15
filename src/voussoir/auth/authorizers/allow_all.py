"""AllowAllAuthorizer -- default; logs 'no authz enforced' once per process.

Use this when you haven't yet configured a concrete authz policy. Voussoir
binds this on the default container so executor.invoke always has an
Authorizer available; bind RoleAuthorizer / DomainAuthorizer / ChainedAuthorizer
for production.
"""

from __future__ import annotations

import structlog

from voussoir.auth.decision import AuthzDecision

_log = structlog.get_logger(__name__)
_warned = False


class AllowAllAuthorizer:
    """Permissive authorizer; ALLOWs every authorize() call.

    Use this when no authz policy is configured. Emits one 'authz_unenforced'
    warning on the first call (per process) to remind operators to bind a real
    Authorizer for production.
    """

    name = "allow_all"

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        """Allow unconditionally; warn once per process that authz is unenforced."""
        del principal, tool, args, ctx
        global _warned
        if not _warned:
            _log.warning(
                "authz_unenforced",
                authorizer="AllowAllAuthorizer",
                hint=(
                    "bind a concrete Authorizer (RoleAuthorizer, DomainAuthorizer,"
                    " ChainedAuthorizer) for production"
                ),
            )
            _warned = True
        return AuthzDecision(decision="ALLOW", authorizer_name=self.name)
