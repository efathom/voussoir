"""A2A discovery — discover_card() and its verification helpers.

discover_card(url) fetches, verifies, and parses an AgentCard from a peer.
The AgentRef IDelegate impl lives in voussoir.a2a.agent_ref.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

from voussoir.a2a.card import AgentCard
from voussoir.a2a.errors import CardVerificationError


async def discover_card(
    url: str,
    *,
    verify_signature: bool = True,
    http_client: httpx.AsyncClient | None = None,
    timeout_s: float = 5.0,
) -> AgentCard:
    """Fetch + parse + verify an AgentCard from a peer's well-known endpoint.

    `url` is the peer's base URL — `/.well-known/agent-card.json` is appended.
    `verify_signature=False` skips JWS verification (dev / tests only).
    `http_client` lets tests inject an ASGI-backed client; default is a
    fresh httpx.AsyncClient that this function creates and closes.
    `timeout_s` sets the httpx client timeout in seconds (default 5.0).

    v1.0.3 rename: this kwarg was previously named `timeout`; renamed to
    `timeout_s` (1) to silence ruff ASYNC109 (which discourages a `timeout`
    parameter on async functions in favor of `asyncio.timeout()`) and
    (2) to match voussoir's `_s` suffix convention for second-valued
    parameters (`refresh_buffer_s`, `ttl_s`). Callers passing `timeout=`
    will see a clear `TypeError: unexpected keyword argument`.

    Raises:
      httpx.HTTPError: transport failure (DNS, connection refused, timeout).
      CardVerificationError: signature verification failed or kid not found.
      pydantic.ValidationError: card schema doesn't match.
    """
    base = url.rstrip("/")
    card_url = f"{base}/.well-known/agent-card.json"

    if http_client is None:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            return await _fetch_and_verify(
                client, base, card_url, verify_signature=verify_signature
            )

    return await _fetch_and_verify(http_client, base, card_url, verify_signature=verify_signature)


async def _fetch_and_verify(
    client: httpx.AsyncClient,
    base: str,
    card_url: str,
    *,
    verify_signature: bool,
) -> AgentCard:
    accept = "application/jose, application/json" if verify_signature else "application/json"
    resp = await client.get(card_url, headers={"Accept": accept})
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")

    if not verify_signature:
        if "application/jose" in content_type:
            # JWS Compact — decode payload without verification.
            try:
                payload = jwt.decode(
                    resp.text,
                    options={
                        "verify_signature": False,
                        "verify_aud": False,
                        "verify_exp": False,
                    },
                )
            except jwt.InvalidTokenError as exc:
                raise CardVerificationError(f"malformed JWS: {exc}") from exc
            return AgentCard.model_validate(payload)
        body = resp.json()
        if "payload" in body and "signature" in body:
            # JWS JSON serialization — extract payload, base64-decode.
            payload_b64 = body["payload"]
            padding = "=" * (-len(payload_b64) % 4)
            decoded = base64.urlsafe_b64decode(payload_b64 + padding)
            return AgentCard.model_validate(json.loads(decoded))
        return AgentCard.model_validate(body)

    # Verify signature.
    if "application/jose" not in content_type:
        raise CardVerificationError(
            f"verify_signature=True but card returned content-type={content_type!r}; "
            f"set verify_signature=False to accept unsigned cards."
        )

    token = resp.text
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise CardVerificationError("card JWS header missing 'kid'")

    unverified_payload = jwt.decode(
        token,
        options={"verify_signature": False, "verify_aud": False, "verify_exp": False},
    )

    public_jwk = await _resolve_public_jwk(client, base, kid, unverified_payload)
    public_key = RSAAlgorithm.from_jwk(public_jwk)

    # v1.0.4 E2 / C3: enforce `exp` when present on the card. PyJWT's
    # `verify_exp` only fires when `exp` is in the payload, so absent `exp`
    # preserves back-compat (pre-E2 cards still verify cleanly).
    # ExpiredSignatureError is a subclass of InvalidTokenError; catch it
    # first so we surface a distinct, more helpful error message.
    try:
        verified_payload = jwt.decode(
            token,
            public_key,  # type: ignore[arg-type]
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_exp": True},
        )
    except jwt.ExpiredSignatureError as exc:
        # Re-parse the (unverified) payload to surface the exp value in the
        # error message; signature was already verified above so we trust it.
        exp_val = unverified_payload.get("exp")
        raise CardVerificationError(f"Agent card expired (exp={exp_val})") from exc
    except jwt.InvalidTokenError as exc:
        raise CardVerificationError(f"card signature invalid: {exc}") from exc

    card = AgentCard.model_validate(verified_payload)
    _validate_card_endpoint(str(card.endpoint))
    return card


# Hosts allowed to serve plaintext http. Compared against the parsed hostname,
# never against a URL prefix: `startswith("http://localhost")` also accepted
# `http://localhost.evil.com/`, and `startswith("http://127.0.0.1")` accepted
# `http://127.0.0.1.evil.com/` — turning HTTPS enforcement into a naming
# convention an attacker could satisfy by registering a subdomain (audit H5).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback(hostname: str | None) -> bool:
    """True for exactly the loopback hostnames, matched whole and case-folded."""
    return hostname is not None and hostname.lower() in _LOOPBACK_HOSTS


def _validate_card_endpoint(endpoint: str) -> None:
    """Phase 4.5a P1 #14: peers must publish HTTPS endpoints; an http:// peer
    means JWTs (carrying sub/trace_id) traverse cleartext. Loopback (127.0.0.1,
    ::1 or localhost) is allowed for local dev / smoke tests.

    Parses the URL rather than prefix-matching it — see `_LOOPBACK_HOSTS`.
    """
    parsed = urlsplit(endpoint)
    if parsed.scheme.lower() == "https":
        return
    if parsed.scheme.lower() == "http" and _is_loopback(parsed.hostname):
        return
    raise CardVerificationError(
        f"card endpoint must be HTTPS (loopback allowed for dev): {endpoint}"
    )


async def _resolve_public_jwk(
    client: httpx.AsyncClient,
    base: str,
    kid: str,
    unverified_payload: dict[str, Any],
) -> dict[str, Any]:
    """Locate the public JWK that matches `kid`.

    Phase 4.5a P0 #1: `inline_jwk` is no longer trusted. A peer that publishes
    inline_jwk inside a card payload could sign that payload with the matching
    private key and have discover_card() "verify" against a key of the peer's
    choosing. Cards must publish via `jwks_uri` (HTTPS) or fall back to the
    well-known JWKS endpoint on the same host.

    Phase 4.5a P1 #13: JWK `kty` / `alg` are validated here so a malformed
    JWK can't crash PyJWT's RSAAlgorithm.from_jwk in an attacker-controlled way.

    Resolution order:
      1. jwks_uri on the card (HTTPS, and same-origin with the card).
      2. {base}/.well-known/jwks.json fallback.
    """
    fallback_uri = f"{base}/.well-known/jwks.json"
    jwks_uri = unverified_payload.get("jwks_uri") or fallback_uri

    # The card body has NOT been verified yet, so `jwks_uri` is attacker-
    # controlled input. Letting it name any host meant a peer could sign its
    # own card and point verification at the matching public key — the exact
    # self-selected-key property that motivated dropping `inline_jwk` in P0 #1,
    # moved one HTTP hop away (audit H5). Requiring the key to come from the
    # card's own origin restores the binding: TLS then attests that the key
    # really belongs to the host the caller chose to talk to.
    base_host = urlsplit(base).hostname
    jwks_parsed = urlsplit(jwks_uri)
    if jwks_parsed.hostname != base_host:
        raise CardVerificationError(
            f"jwks_uri host {jwks_parsed.hostname!r} does not match the card's "
            f"origin {base_host!r}; a peer may not select its own verification "
            f"key source. Publish the JWKS on the card's own host, or bind a "
            f"custom KeyProvider that pins the peer's key out-of-band."
        )
    # Parsed-host checks, not prefix matches: `startswith("http://test")` let
    # any domain beginning with "test" (e.g. test.attacker.com) serve keys over
    # cleartext, and the loopback prefixes had the same subdomain hole.
    _allowed = jwks_parsed.scheme.lower() == "https" or (
        jwks_parsed.scheme.lower() == "http"
        # ASGITransport tests use http://test/... — no real network call
        # happens, and the origin check above already pins it to the card's own
        # host, so this can't widen the production surface.
        and (_is_loopback(jwks_parsed.hostname) or jwks_parsed.hostname == "test")
    )
    if not _allowed:
        raise CardVerificationError(f"jwks_uri must be HTTPS: {jwks_uri}")
    resp = await client.get(jwks_uri)
    resp.raise_for_status()
    jwks = resp.json()
    for jwk in jwks.get("keys", []):
        if not isinstance(jwk, dict) or jwk.get("kid") != kid:
            continue
        if jwk.get("kty") != "RSA":
            raise CardVerificationError(
                f"unsupported jwk.kty={jwk.get('kty')!r}; only RSA supported"
            )
        if jwk.get("alg", "RS256") != "RS256":
            raise CardVerificationError(
                f"unsupported jwk.alg={jwk.get('alg')!r}; only RS256 supported"
            )
        return dict(jwk)

    raise CardVerificationError(f"no JWK with kid={kid!r} found in JWKS at {jwks_uri}")
