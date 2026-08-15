"""Phase 4.5a P0 #1 + P1 #13 + P1 #14 — card signing hardening.

The discovery layer must:
  - Reject non-HTTPS jwks_uri (HTTPS or loopback only).
  - Reject non-RSA / non-RS256 JWKs (no key-confusion attacks).
  - Reject non-HTTPS / non-loopback card endpoints.

inline_jwk resolution is no longer honored at all — cards must publish
a jwks_uri. We don't exhaustively test the inline_jwk-rejection path
here because the path was simply removed; if a card sets inline_jwk
the resolver ignores it and falls back to jwks_uri (HTTPS required).
"""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from voussoir.a2a.discovery import (
    CardVerificationError,
    _resolve_public_jwk,
    _validate_card_endpoint,
)


def test_validate_card_endpoint_accepts_https() -> None:
    _validate_card_endpoint("https://peer.example/a2a")


def test_validate_card_endpoint_accepts_loopback() -> None:
    _validate_card_endpoint("http://127.0.0.1:8080/a2a")
    _validate_card_endpoint("http://localhost:8080/a2a")


def test_validate_card_endpoint_rejects_plain_http() -> None:
    with pytest.raises(CardVerificationError, match="HTTPS"):
        _validate_card_endpoint("http://insecure.example/a2a")


@pytest.mark.asyncio
async def test_resolve_public_jwk_rejects_non_rsa_kty() -> None:
    import httpx

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://peer") as mock:
            mock.get("/jwks.json").mock(
                return_value=Response(
                    200,
                    json={"keys": [{"kid": "k1", "kty": "oct", "k": "AAAA"}]},
                )
            )
            with pytest.raises(CardVerificationError, match="kty"):
                await _resolve_public_jwk(
                    client,
                    base="https://peer",
                    kid="k1",
                    unverified_payload={"jwks_uri": "https://peer/jwks.json"},
                )


@pytest.mark.asyncio
async def test_resolve_public_jwk_rejects_non_rs256_alg() -> None:
    import httpx

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://peer") as mock:
            mock.get("/jwks.json").mock(
                return_value=Response(
                    200,
                    json={
                        "keys": [{"kid": "k1", "kty": "RSA", "alg": "HS256", "n": "X", "e": "X"}]
                    },
                )
            )
            with pytest.raises(CardVerificationError, match="alg"):
                await _resolve_public_jwk(
                    client,
                    base="https://peer",
                    kid="k1",
                    unverified_payload={"jwks_uri": "https://peer/jwks.json"},
                )


@pytest.mark.asyncio
async def test_resolve_public_jwk_rejects_non_https_jwks_uri() -> None:
    import httpx

    async with httpx.AsyncClient() as client:
        with pytest.raises(CardVerificationError, match="HTTPS"):
            await _resolve_public_jwk(
                client,
                base="https://peer",
                kid="k1",
                unverified_payload={"jwks_uri": "http://insecure.example/jwks.json"},
            )


@pytest.mark.asyncio
async def test_inline_jwk_no_longer_honored() -> None:
    """Phase 4.5a P0 #1: inline_jwk in payload is ignored. With no jwks_uri
    set and no JWKS at the default location, resolution falls back to
    {base}/.well-known/jwks.json which here returns nothing — should raise."""
    import httpx

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://peer") as mock:
            mock.get("/.well-known/jwks.json").mock(
                return_value=Response(200, json={"keys": []}),
            )
            with pytest.raises(CardVerificationError, match="no JWK with kid"):
                await _resolve_public_jwk(
                    client,
                    base="https://peer",
                    kid="k1",
                    unverified_payload={
                        # inline_jwk is set but Phase 4.5a no longer trusts it.
                        "inline_jwk": {
                            "kid": "k1",
                            "kty": "RSA",
                            "alg": "RS256",
                            "n": "X",
                            "e": "X",
                        },
                    },
                )
