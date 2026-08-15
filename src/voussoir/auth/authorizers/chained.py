"""ChainedAuthorizer -- composes multiple authorizers; first non-ALLOW wins.

Use this to layer authz policies (e.g., role + domain). DENY short-circuits
the chain. MASK decisions accumulate; the final ALLOW emits a MASK with the
union of masked_fields if any authorizer in the chain returned MASK.
"""

from __future__ import annotations

from voussoir.auth.decision import AuthzDecision
from voussoir.auth.protocol import Authorizer


class ChainedAuthorizer:
    """Compose multiple Authorizers into a single chain.

    Use this when you want to combine RBAC + domain checks (or any other
    authz combinations). The chain short-circuits on the first DENY; MASK
    decisions are accumulated and merged into the final ALLOW.
    """

    name = "chained"

    def __init__(self, authorizers: list[Authorizer]) -> None:
        if not authorizers:
            raise ValueError("ChainedAuthorizer requires at least one authorizer")
        self._authorizers = list(authorizers)

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        """Run each authorizer in order; DENY short-circuits; MASKs accumulate."""
        masked: list[str] = []
        for a in self._authorizers:
            d = await a.authorize(principal, tool, args, ctx)  # type: ignore[arg-type]
            if d.decision == "DENY":
                return AuthzDecision(
                    decision="DENY",
                    reason=d.reason,
                    authorizer_name=a.name,
                )
            if d.decision == "MASK":
                masked.extend(d.masked_fields)
        if masked:
            return AuthzDecision(
                decision="MASK",
                masked_fields=sorted(set(masked)),
                authorizer_name=self.name,
            )
        return AuthzDecision(decision="ALLOW", authorizer_name=self.name)
