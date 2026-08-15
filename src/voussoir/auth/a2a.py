"""A2A Principal flow — inbound JWT -> Principal, outbound Principal -> JWT (Phase 6 A9).

Use this module when extending the A2A inbound or outbound Principal mapping
for non-default JWT claim shapes or cross-org token minting. The defaults
satisfy the same-org passthrough case (sub -> user_id, email -> email,
groups -> roles), which is the typical deployment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from voussoir.auth.principal import Principal

if TYPE_CHECKING:
    from voussoir.a2a.keys import KeyProvider

# Standard JWT claims to exclude from Principal.attributes (these are RFC 7519 reserved
# plus the voussoir-specific A2A claims that map to first-class Principal fields):
_STANDARD_JWT_CLAIMS = frozenset(
    {"sub", "email", "groups", "iat", "exp", "iss", "aud", "nbf", "jti"}
)


@runtime_checkable
class JWTPrincipalMapper(Protocol):
    """Map a verified JWT's claims dict into a Principal.

    Use this when you need custom claim-to-field mapping (e.g., your IdP
    emits 'preferred_username' instead of 'sub', or 'memberOf' instead of
    'groups'). Implement `map(claims)` and bind on the container.
    """

    async def map(self, claims: dict[str, Any]) -> Principal: ...


class DefaultJWTPrincipalMapper:
    """Standard claims-to-Principal mapping.

    Use this when your IdP emits RFC-7519 standard claims:
    `sub` -> user_id, `email` -> email, `groups` -> roles. Custom claims
    (anything not in the JWT standard set) become Principal.attributes.
    """

    name = "default_jwt_mapper"

    async def map(self, claims: dict[str, Any]) -> Principal:
        attributes = {k: v for k, v in claims.items() if k not in _STANDARD_JWT_CLAIMS}
        exp_ts = claims.get("exp")
        iat_ts = claims.get("iat", datetime.now(UTC).timestamp())
        return Principal(
            user_id=str(claims.get("sub", "unknown")),
            email=claims.get("email"),
            roles=list(claims.get("groups", [])),
            issued_at=datetime.fromtimestamp(float(iat_ts), tz=UTC),
            expires_at=(
                datetime.fromtimestamp(float(exp_ts), tz=UTC) if exp_ts is not None else None
            ),
            attributes=attributes,
        )


@runtime_checkable
class PrincipalForwarder(Protocol):
    """Mint a downstream JWT carrying the caller's Principal (outbound A2A).

    Use this for cross-service A2A calls where the receiving service needs
    to know who the original caller is. Same-org deployments typically use
    SameOrgPassthroughForwarder (re-sign with the publisher's AgentCard key);
    cross-org deployments implement an exchange to a federation issuer.
    """

    async def forward(self, principal: Principal, target_url: str, *, audience: str) -> str: ...


class SameOrgPassthroughForwarder:
    """Mint a short-lived JWT signed with the publisher's HS256 shared secret.

    Use this for the typical same-org deployment where every peer trusts
    the same shared secret (VOUSSOIR_A2A_JWT_SECRET). The minted token carries
    the full claim set required by the receiver's JWT verification:
    ``iss``, ``sub``, ``aud``, ``exp``, ``iat``, ``nbf``.

    The ``audience`` kwarg on :meth:`forward` must match the receiving agent's
    ``agent.name`` (which is what ``make_a2a_router`` verifies against).  In
    practice this is ``self.card.name`` at the ``AgentRef.delegate`` call site.

    Constructor args:
      key_provider: must implement ``jwt_secret()`` and ``jwt_algorithm()``
                    (both present on ``voussoir.a2a.keys.KeyProvider``).
      issuer:       the ``iss`` claim value this forwarder stamps on every token.
                    Must appear in the receiver's ``expected_issuers()`` allow-list
                    (env var ``VOUSSOIR_A2A_ALLOWED_ISSUERS``).  Typically the
                    calling agent's name or a shared org-wide issuer identifier.
      ttl_s:        token lifetime in seconds (default 300 = 5 minutes).
      nbf_skew_s:   «not-before» clock-skew buffer in seconds (default 30).
    """

    name = "same_org_passthrough"

    def __init__(
        self,
        *,
        key_provider: KeyProvider,
        issuer: str,
        ttl_s: int = 300,
        nbf_skew_s: int = 30,
    ) -> None:
        self._key_provider = key_provider
        self._issuer = issuer
        self._ttl_s = ttl_s
        self._nbf_skew_s = nbf_skew_s

    async def forward(self, principal: Principal, target_url: str, *, audience: str) -> str:
        """Mint a JWT accepted by make_a2a_router's /a2a endpoint.

        Args:
            principal:  The caller's identity to forward.
            target_url: The target endpoint URL (used only for routing at the
                        call site; audience binding comes from ``audience``).
            audience:   The receiving agent's name — must match ``agent.name``
                        on the ``make_a2a_router`` side (audience=agent.name).
        """
        del target_url  # routing concern at the call site; aud comes from `audience`
        # Lazy import: PyJWT is in the 'a2a' optional extra, not the base install.
        import jwt as _jwt

        now = datetime.now(UTC)
        now_ts = int(now.timestamp())
        signing_key: bytes = self._key_provider.jwt_secret()
        algorithm: str = self._key_provider.jwt_algorithm()

        payload: dict[str, Any] = {
            "iss": self._issuer,
            "sub": principal.user_id,
            "aud": audience,
            "iat": now_ts,
            "nbf": now_ts - self._nbf_skew_s,
            "exp": now_ts + self._ttl_s,
        }
        if principal.email:
            payload["email"] = principal.email
        if principal.roles:
            payload["groups"] = list(principal.roles)

        return str(_jwt.encode(payload, signing_key, algorithm=algorithm))
