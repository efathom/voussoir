"""KeyProvider Protocol + EnvKeyProvider default impl.

Phase 4b uses HS256 for JWT (caller and callee share a secret out-of-band)
and RS256 for AgentCard JWS signing. v1.5 adds asymmetric JWT (RS256) for
cross-org peers.

The default EnvKeyProvider:
- Reads VOUSSOIR_A2A_JWT_SECRET (base64). Without it AND with
  allow_ephemeral=True (dev mode), generates a random 32-byte secret and
  emits a structured warning via the voussoir.a2a.keys logger. Default
  allow_ephemeral=False raises at .jwt_secret() time so production
  deployments fail fast rather than silently using a fresh secret per
  process that no peer can authenticate against (Phase 4.5a P0 #5).
- Reads VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH (PEM file) or auto-generates an
  ephemeral RSA-2048 keypair in memory (lost on restart).

Production deployments supply a custom KeyProvider (KMS, Vault, HSM) via
`default_container(key_provider=...)`, or bind one on a hand-built Container.
"""

from __future__ import annotations

import base64
import os
import secrets
import uuid
from typing import Any, Protocol, runtime_checkable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from voussoir.errors import VoussoirError
from voussoir.observability.logging_setup import get_logger


class NoCardSigningKeyError(VoussoirError):
    """Raised by custom KeyProviders that explicitly refuse card signing
    (e.g. read-only verifier deployments). EnvKeyProvider always provides
    a key (auto-generated if absent); this exception exists so the
    publisher's card route can fall back to unsigned dev mode.
    """


class JWTKeyNotConfiguredError(VoussoirError):
    """Raised when a KeyProvider can't supply a key for the requested algorithm.

    Use this when an inbound verification path needs a public key it
    doesn't have (e.g. RS256 verification without an RSA pubkey), or an
    outbound signing path needs a private key it doesn't have
    (e.g. RS256 signing without an RSA private key). Operator action:
    bind a custom KeyProvider that implements the missing surface.

    Concrete failure mode (inbound): operator sets VOUSSOIR_A2A_JWT_ALGORITHM=RS256
    while keeping the default EnvKeyProvider, which only has a symmetric
    secret. Pre-E2 this surfaced as a generic 401 (PyJWT raised InvalidKeyError
    which was caught by the catch-all `except jwt.InvalidTokenError` clause).
    Post-E2 the publisher catches this exception and returns HTTP 500
    ("server misconfigured") so operators see the real failure mode.

    Bind a custom KeyProvider that returns an RSA public key from
    `jwt_verification_key('RS256')` to enable RS256 inbound JWT auth.
    """


# Back-compat alias (v1.0.4 E2 → v1.1.0 F8). Removed in v1.2.0.
InboundJWTVerificationNotConfiguredError = JWTKeyNotConfiguredError


@runtime_checkable
class KeyProvider(Protocol):
    """Abstracts the source of cryptographic material for A2A.

    Two roles:
      - JWT signing/verification (invocation auth on POST /a2a).
        v1 = symmetric HS256; v1.5 = asymmetric RS256.
      - JWS signing/verification (AgentCard provenance). Always asymmetric.
    """

    def jwt_secret(self) -> bytes: ...
    def jwt_algorithm(self) -> str: ...
    def expected_issuers(self) -> set[str]: ...

    def jwt_verification_key(self, alg: str) -> Any:
        """Return the key suitable for VERIFYING inbound JWTs under `alg`.

        v1.0.4 E2 / C4: previously the publisher always used `jwt_secret()`
        regardless of algorithm, which silently broke RS256 (PyJWT raised
        InvalidKeyError, caught by the generic InvalidTokenError clause,
        becoming a 401 with no diagnostic). This method centralizes the
        dispatch:

        - Symmetric algorithms (HS256/HS384/HS512): return bytes (typically
          the same value as `jwt_secret()`).
        - Asymmetric algorithms (RS256): return an RSA public key object,
          OR raise `JWTKeyNotConfiguredError` if the provider cannot supply one.

        Use this when implementing a custom KeyProvider for production with
        per-peer RSA verification keys (e.g. fetched from a KMS or a peer
        registry). EnvKeyProvider raises for RS256 because it only knows
        about a shared symmetric secret.
        """
        ...

    def jwt_signing_key(self, alg: str) -> Any:
        """Return the key suitable for SIGNING outbound JWTs under `alg`.

        v1.1.0 F8: mirrors jwt_verification_key for the outbound path.

        - HS algorithms (HS256/HS384/HS512): return bytes — typically the
          same value as `jwt_secret()` (PyJWT accepts bytes).
        - RS/ES/PS algorithms: return the appropriate private key
          (e.g. RSAPrivateKey from cryptography.hazmat.primitives.asymmetric.rsa).

        Raise JWTKeyNotConfiguredError if no key is configured for `alg`.

        Use this when implementing a custom KeyProvider that signs outbound
        JWTs with an asymmetric key (e.g. fetched from a KMS or HSM).
        """
        ...

    def card_signing_jwk(self) -> dict[str, Any]: ...
    def card_verification_jwks(self) -> dict[str, Any]: ...


class EnvKeyProvider:
    """Reads from environment variables; auto-generates for dev mode.

    Env vars:
      VOUSSOIR_A2A_JWT_SECRET             base64-encoded shared secret
      VOUSSOIR_A2A_JWT_ALGORITHM          "HS256" (default), "HS384", "HS512"
      VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH  path to PEM-encoded RSA private key
    """

    def __init__(
        self,
        allow_ephemeral: bool = False,
        *,
        jwt_secret: bytes | None = None,
    ) -> None:
        self._allow_ephemeral = allow_ephemeral
        # JWT secret is loaded lazily: card-only consumers shouldn't pay the
        # raise/warn cost just for constructing the provider.
        # jwt_secret= kwarg seeds a deterministic secret (test use; v1.1.0 F10).
        self._jwt_secret: bytes | None = jwt_secret
        self._jwt_algo = os.environ.get("VOUSSOIR_A2A_JWT_ALGORITHM", "HS256")
        self._card_private_jwk, self._card_public_jwks = self._load_card_keys()

    @staticmethod
    def _load_card_keys() -> tuple[dict[str, Any], dict[str, Any]]:
        path = os.environ.get("VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH")
        if path:
            with open(path, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise ValueError(f"VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH={path!r} is not an RSA key")
        else:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        kid = uuid.uuid4().hex
        private_jwk = _rsa_private_to_jwk(private_key, kid=kid)
        public_jwks = {"keys": [_rsa_public_to_jwk(private_key.public_key(), kid=kid)]}
        return private_jwk, public_jwks

    def jwt_secret(self) -> bytes:
        """Phase 4.5a P0 #5: lazy load.
        1. VOUSSOIR_A2A_JWT_SECRET (base64) → use it.
        2. Otherwise + allow_ephemeral=True → generate 32 bytes, warn (via
           logger, NOT stderr — secrets must not land in log aggregation).
        3. Otherwise → raise so production fails fast.
        """
        if self._jwt_secret is not None:
            return self._jwt_secret
        encoded = os.environ.get("VOUSSOIR_A2A_JWT_SECRET")
        if encoded:
            self._jwt_secret = base64.b64decode(encoded)
            return self._jwt_secret
        if not self._allow_ephemeral:
            raise RuntimeError(
                "VOUSSOIR_A2A_JWT_SECRET is not set. Either set the env var, "
                "or construct EnvKeyProvider(allow_ephemeral=True) in dev mode."
            )
        self._jwt_secret = secrets.token_bytes(32)
        get_logger(__name__).warning(
            "ephemeral_jwt_secret_generated",
            note=(
                "VOUSSOIR_A2A_JWT_SECRET not set; using ephemeral 32-byte secret. "
                "Inter-process A2A auth will fail. Set the env var in prod."
            ),
        )
        return self._jwt_secret

    def jwt_algorithm(self) -> str:
        return self._jwt_algo

    def jwt_verification_key(self, alg: str) -> bytes:
        """v1.0.4 E2 / C4: dispatch by algorithm family.

        EnvKeyProvider only knows about a symmetric secret. Symmetric
        algorithms (HS*) return that secret; asymmetric algorithms (RS256)
        raise `JWTKeyNotConfiguredError` so operators see a clean 500 instead
        of a silent 401 when they misconfigure VOUSSOIR_A2A_JWT_ALGORITHM=RS256
        without binding a custom provider.
        """
        if alg.startswith("HS"):
            return self.jwt_secret()
        raise JWTKeyNotConfiguredError(
            f"EnvKeyProvider cannot verify inbound JWTs under alg={alg!r}; "
            "RS256 inbound JWT auth requires a custom KeyProvider that returns "
            "an RSA public key from jwt_verification_key('RS256'). Either bind "
            "such a provider on the agent's container, or downgrade "
            "VOUSSOIR_A2A_JWT_ALGORITHM to HS256/HS384/HS512."
        )

    def jwt_signing_key(self, alg: str) -> bytes:
        """v1.1.0 F8: dispatch outbound signing key by algorithm family.

        EnvKeyProvider only knows about a symmetric secret. Symmetric
        algorithms (HS*) return that secret; asymmetric algorithms raise
        `JWTKeyNotConfiguredError` with an actionable operator message.
        """
        if alg.startswith("HS"):
            return self.jwt_secret()
        raise JWTKeyNotConfiguredError(
            f"EnvKeyProvider has no asymmetric signing key for alg={alg!r}. "
            f"For RS256/ES256 outbound JWTs, bind a custom KeyProvider that "
            f"implements jwt_signing_key() with the appropriate private key."
        )

    def expected_issuers(self) -> set[str]:
        """Issuer allow-list for inbound JWT validation (Phase 4.5a P0 #2).
        Reads VOUSSOIR_A2A_ALLOWED_ISSUERS (comma-separated). Empty set
        means reject all incoming JWTs — default-deny in prod."""
        raw = os.environ.get("VOUSSOIR_A2A_ALLOWED_ISSUERS", "")
        return {s.strip() for s in raw.split(",") if s.strip()}

    def card_signing_jwk(self) -> dict[str, Any]:
        return self._card_private_jwk

    def card_verification_jwks(self) -> dict[str, Any]:
        return self._card_public_jwks


def _rsa_private_to_jwk(key: rsa.RSAPrivateKey, *, kid: str) -> dict[str, Any]:
    """Encode an RSA private key as a JWK (RFC 7517)."""
    numbers = key.private_numbers()
    pub = numbers.public_numbers
    return {
        "kty": "RSA",
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
        "n": _int_to_b64url(pub.n),
        "e": _int_to_b64url(pub.e),
        "d": _int_to_b64url(numbers.d),
        "p": _int_to_b64url(numbers.p),
        "q": _int_to_b64url(numbers.q),
        "dp": _int_to_b64url(numbers.dmp1),
        "dq": _int_to_b64url(numbers.dmq1),
        "qi": _int_to_b64url(numbers.iqmp),
    }


def _rsa_public_to_jwk(key: rsa.RSAPublicKey, *, kid: str) -> dict[str, Any]:
    """Encode an RSA public key as a JWK (public-only)."""
    pub = key.public_numbers()
    return {
        "kty": "RSA",
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
        "n": _int_to_b64url(pub.n),
        "e": _int_to_b64url(pub.e),
    }


def _int_to_b64url(n: int) -> str:
    """Convert a positive int to base64url with no padding (per RFC 7518)."""
    length = (n.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode("ascii")
