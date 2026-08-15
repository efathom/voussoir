"""Locks the JWTPrincipalMapper + PrincipalForwarder defaults (Phase 6 A9)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from voussoir.auth import Principal
from voussoir.auth.a2a import (
    DefaultJWTPrincipalMapper,
    JWTPrincipalMapper,
    PrincipalForwarder,
    SameOrgPassthroughForwarder,
)

# ------- DefaultJWTPrincipalMapper -------


@pytest.mark.asyncio
async def test_mapper_minimal_claims() -> None:
    m = DefaultJWTPrincipalMapper()
    p = await m.map({"sub": "alice"})
    assert p.user_id == "alice"


@pytest.mark.asyncio
async def test_mapper_full_claims() -> None:
    m = DefaultJWTPrincipalMapper()
    iat = int(datetime.now(UTC).timestamp())
    exp = iat + 3600
    p = await m.map(
        {
            "sub": "alice",
            "email": "a@b.c",
            "groups": ["admin", "ops"],
            "iat": iat,
            "exp": exp,
            "iss": "fathom.example.com",
            "aud": "voussoir",
            "custom_claim": "value",
        }
    )
    assert p.user_id == "alice"
    assert p.email == "a@b.c"
    assert p.roles == ["admin", "ops"]
    # Standard claims (iss/aud) should NOT be in attributes; custom claims should.
    assert p.attributes.get("custom_claim") == "value"
    assert "iss" not in p.attributes


@pytest.mark.asyncio
async def test_mapper_no_sub_uses_unknown() -> None:
    m = DefaultJWTPrincipalMapper()
    p = await m.map({"email": "a@b.c"})
    assert p.user_id == "unknown"


@pytest.mark.asyncio
async def test_mapper_no_groups_gives_empty_roles() -> None:
    m = DefaultJWTPrincipalMapper()
    p = await m.map({"sub": "bob"})
    assert p.roles == []


@pytest.mark.asyncio
async def test_mapper_issued_at_and_expires_at_populated() -> None:
    m = DefaultJWTPrincipalMapper()
    iat = int(datetime.now(UTC).timestamp())
    exp = iat + 300
    p = await m.map({"sub": "alice", "iat": iat, "exp": exp})
    assert p.issued_at is not None
    assert p.expires_at is not None
    assert int(p.issued_at.timestamp()) == iat
    assert int(p.expires_at.timestamp()) == exp


@pytest.mark.asyncio
async def test_mapper_implements_protocol() -> None:
    assert isinstance(DefaultJWTPrincipalMapper(), JWTPrincipalMapper)


@pytest.mark.asyncio
async def test_mapper_standard_claims_excluded_from_attributes() -> None:
    """All RFC-7519 reserved claims must NOT appear in attributes."""
    m = DefaultJWTPrincipalMapper()
    iat = int(datetime.now(UTC).timestamp())
    p = await m.map(
        {
            "sub": "u",
            "iat": iat,
            "exp": iat + 60,
            "nbf": iat,
            "jti": "some-id",
            "iss": "issuer",
            "aud": "audience",
            "email": "x@y.z",
            "groups": [],
        }
    )
    for reserved in ("sub", "iat", "exp", "nbf", "jti", "iss", "aud", "email", "groups"):
        assert reserved not in p.attributes, f"{reserved!r} should not be in attributes"


# ------- SameOrgPassthroughForwarder -------


@pytest.mark.asyncio
async def test_forwarder_mints_jwt_with_principal_claims() -> None:
    """Forwarder produces a Bearer JWT that round-trips back to the same Principal
    via DefaultJWTPrincipalMapper."""
    from voussoir.a2a.keys import EnvKeyProvider

    key_provider = EnvKeyProvider(allow_ephemeral=True)
    forwarder = SameOrgPassthroughForwarder(
        key_provider=key_provider, issuer="caller-agent", ttl_s=300
    )

    p = Principal(
        user_id="alice",
        email="a@b.c",
        roles=["admin"],
        issued_at=datetime.now(UTC),
    )
    token = await forwarder.forward(p, "https://peer.example.com/a2a/", audience="peer-agent")
    assert isinstance(token, str)
    assert len(token) > 50  # JWT shape (base64-ish)
    assert token.count(".") == 2  # three-part JWT


@pytest.mark.asyncio
async def test_forwarder_token_contains_correct_claims() -> None:
    """Decode the minted JWT and verify all required claims are present."""
    import jwt as _jwt

    from voussoir.a2a.keys import EnvKeyProvider

    key_provider = EnvKeyProvider(allow_ephemeral=True)
    forwarder = SameOrgPassthroughForwarder(
        key_provider=key_provider, issuer="caller-agent", ttl_s=60
    )

    p = Principal(
        user_id="bob",
        email="bob@example.com",
        roles=["reader", "writer"],
        issued_at=datetime.now(UTC),
    )
    token = await forwarder.forward(p, "https://peer.example.com/a2a/", audience="peer-agent")

    # Decode without verification to inspect claims (this is a unit test of payload shape)
    claims = _jwt.decode(token, options={"verify_signature": False})
    assert claims["sub"] == "bob"
    assert claims["email"] == "bob@example.com"
    assert claims["groups"] == ["reader", "writer"]
    assert claims["iss"] == "caller-agent"
    assert claims["aud"] == "peer-agent"
    assert "exp" in claims
    assert "iat" in claims
    assert "nbf" in claims


@pytest.mark.asyncio
async def test_forwarder_token_no_email_when_absent() -> None:
    """If Principal has no email, the minted JWT should not include 'email'."""
    import jwt as _jwt

    from voussoir.a2a.keys import EnvKeyProvider

    key_provider = EnvKeyProvider(allow_ephemeral=True)
    forwarder = SameOrgPassthroughForwarder(
        key_provider=key_provider, issuer="caller-agent", ttl_s=60
    )

    p = Principal(user_id="svc-account", issued_at=datetime.now(UTC))
    token = await forwarder.forward(p, "https://peer.example.com/a2a/", audience="peer-agent")

    claims = _jwt.decode(token, options={"verify_signature": False})
    assert "email" not in claims
    assert "groups" not in claims  # empty roles → no groups claim


@pytest.mark.asyncio
async def test_forwarder_implements_protocol() -> None:
    from voussoir.a2a.keys import EnvKeyProvider

    f = SameOrgPassthroughForwarder(
        key_provider=EnvKeyProvider(allow_ephemeral=True), issuer="caller-agent"
    )
    assert isinstance(f, PrincipalForwarder)


# ------- Round-trip integration: forwarder token accepted by make_a2a_router -------


@pytest.mark.asyncio
async def test_forwarder_token_accepted_by_make_a2a_router(
    make_container, stub_llm, fresh_env
) -> None:
    """Prove that a SameOrgPassthroughForwarder-minted token passes the full
    JWT verification in make_a2a_router's /a2a endpoint.

    This is the regression test for the bug where the forwarder omitted iss/aud/nbf
    and every forwarder-minted token was rejected with 'Token is missing the "iss" claim'.
    """
    import json

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from voussoir.a2a.keys import EnvKeyProvider
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    # The issuer name must appear in VOUSSOIR_A2A_ALLOWED_ISSUERS.
    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller-agent")

    # Build a shared KeyProvider (same secret used for minting + verification).
    key_provider = EnvKeyProvider(allow_ephemeral=True)

    # Build the receiving agent + router.
    server_c = make_container(stub_llm(content="forwarded ok"))
    # Inject the shared key provider so the publisher verifies with the same secret.
    from voussoir.a2a.keys import KeyProvider

    server_c.bind(KeyProvider, key_provider)  # type: ignore[type-abstract]
    server_agent = Agent("researcher", description="r", container=server_c)

    app = FastAPI()
    app.include_router(make_a2a_router(server_agent, endpoint="https://test/a2a"))

    # Build the forwarder bound to the same shared secret + issuer.
    forwarder = SameOrgPassthroughForwarder(
        key_provider=key_provider,
        issuer="caller-agent",
        ttl_s=300,
    )

    # Mint a token for a sample Principal, targeting the receiving agent.
    principal = Principal(
        user_id="alice",
        email="alice@example.com",
        roles=["reader"],
        issued_at=datetime.now(UTC),
    )
    token = await forwarder.forward(
        principal,
        "https://test/a2a",
        audience=server_agent.name,  # "researcher"
    )

    # POST to /a2a with the forwarder-minted token and assert it is ACCEPTED (200,
    # not 401).  Prior to the fix, this returned 401 "invalid bearer token" because
    # PyJWT raised MissingRequiredClaimError("iss").
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        resp = await client.post(
            "/a2a",
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "voussoir.delegate",
                    "params": {"task": "hello"},
                    "id": "rt-1",
                }
            ),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    assert (
        resp.status_code == 200
    ), f"Expected 200 (token accepted), got {resp.status_code}: {resp.text}"
    body = resp.json()
    # Must have a result, not an error — authentication passed.
    assert (
        "error" not in body or body.get("error") is None
    ), f"Unexpected JSON-RPC error: {body.get('error')}"
    assert body.get("result", {}).get("output") == "forwarded ok"
