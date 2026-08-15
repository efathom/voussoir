"""v1.0.4 E2 — AgentCard exp + inbound JWT alg/key dispatch + jwks_uri (Round-3 critical cluster).

Three bundled fixes (touch overlapping files):
  C3 — AgentCard has no `exp`; JWS verified with verify_exp=False → cards valid
       forever. Add `exp: int | None` field + `ttl_s` kwarg to card_from_agent;
       enforce exp in discover_card.
  C4 — Inbound JWT decoded with `provider.jwt_secret()` regardless of `alg`;
       RS256 silently 401s. Branch on alg in publisher; KeyProvider grows a
       `jwt_verification_key(alg)` method with a clear error for unsupported
       configurations.
  C5 — make_a2a_router passes jwks_uri=None to card_from_agent; consumer
       fallback breaks under non-root mounts. Derive jwks_uri from the endpoint
       and pass it explicitly.
"""

from __future__ import annotations

import json
import time
import uuid

import jwt
import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mint_jwt(provider, *, aud: str, iss: str = "test-caller", exp_delta: int = 300) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": iss,
            "aud": aud,
            "iat": now,
            "nbf": now,
            "exp": now + exp_delta,
            "sub": "test-user",
            "trace_id": uuid.uuid4().hex,
        },
        provider.jwt_secret(),
        algorithm=provider.jwt_algorithm(),
    )


# ---------------------------------------------------------------------------
# C3 — AgentCard.exp + discover_card expiry enforcement
# ---------------------------------------------------------------------------


def test_card_from_agent_default_has_no_exp(make_container, stub_llm) -> None:
    """Back-compat: card_from_agent without ttl_s produces exp=None."""
    from voussoir.a2a.card import card_from_agent
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    card = card_from_agent(agent, endpoint="https://h/a2a")  # type: ignore[arg-type]
    assert card.exp is None


def test_card_from_agent_stamps_exp_when_ttl_set(make_container, stub_llm) -> None:
    """ttl_s=3600 → exp ≈ now + 3600."""
    from voussoir.a2a.card import card_from_agent
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    before = int(time.time())
    card = card_from_agent(
        agent,
        endpoint="https://h/a2a",  # type: ignore[arg-type]
        ttl_s=3600,
    )
    after = int(time.time())
    assert card.exp is not None
    assert before + 3600 <= card.exp <= after + 3600


def test_agent_card_exp_field_validates() -> None:
    """AgentCard.exp accepts int and None."""
    from voussoir.a2a.card import AgentCard

    card_a = AgentCard(name="r", endpoint="https://h/a2a", exp=1_700_000_000)
    assert card_a.exp == 1_700_000_000
    card_b = AgentCard(name="r", endpoint="https://h/a2a")
    assert card_b.exp is None


def _sign_card_payload(payload: dict, private_jwk: dict) -> str:
    """Sign an arbitrary AgentCard JSON payload as a JWS Compact token."""
    signing_key = jwt.algorithms.RSAAlgorithm.from_jwk(private_jwk)
    return jwt.encode(
        payload,
        signing_key,  # type: ignore[arg-type]
        algorithm="RS256",
        headers={"typ": "agent-card+jwt", "kid": private_jwk["kid"]},
    )


@pytest.fixture
async def publisher_app_client(make_container, stub_llm, fresh_env):
    """Reuse the pattern from tests/a2a/test_discovery.py."""
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("researcher", description="r-desc", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        yield client, c, agent


@pytest.mark.asyncio
async def test_card_with_past_exp_rejected(publisher_app_client) -> None:
    """A card whose exp is in the past raises CardVerificationError mentioning 'expired'."""
    from voussoir.a2a.discovery import discover_card
    from voussoir.a2a.errors import CardVerificationError
    from voussoir.a2a.keys import KeyProvider

    client, c, _agent = publisher_app_client
    provider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    private_jwk = provider.card_signing_jwk()
    jwks_json = (await client.get("/.well-known/jwks.json")).json()

    payload = {
        "name": "researcher",
        "description": "",
        "version": "1.0.0",
        "endpoint": "https://test/a2a",
        "capabilities": [],
        "input_schema": {},
        "output_schema": {},
        "auth": [],
        "supports": ["json-rpc/2.0"],
        "jwks_uri": None,
        "exp": int(time.time()) - 100,
    }
    token = _sign_card_payload(payload, private_jwk)

    with respx.mock(base_url="https://expired-peer") as mock:
        mock.get("/.well-known/agent-card.json").mock(
            return_value=Response(
                200,
                content=token.encode(),
                headers={"content-type": "application/jose"},
            )
        )
        mock.get("/.well-known/jwks.json").mock(return_value=Response(200, json=jwks_json))
        with pytest.raises(CardVerificationError, match="expired"):
            await discover_card("https://expired-peer")


@pytest.mark.asyncio
async def test_card_without_exp_still_accepted(publisher_app_client) -> None:
    """Back-compat: cards with no exp (most current cards) verify cleanly."""
    from voussoir.a2a.discovery import discover_card

    client, _c, _agent = publisher_app_client
    card = await discover_card("https://test", http_client=client)
    assert card.name == "researcher"
    assert card.exp is None


@pytest.mark.asyncio
async def test_card_with_future_exp_accepted(publisher_app_client) -> None:
    """A card with exp far in the future verifies cleanly."""
    from voussoir.a2a.discovery import discover_card
    from voussoir.a2a.keys import KeyProvider

    client, c, _agent = publisher_app_client
    provider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    private_jwk = provider.card_signing_jwk()
    jwks_json = (await client.get("/.well-known/jwks.json")).json()

    payload = {
        "name": "researcher",
        "description": "",
        "version": "1.0.0",
        "endpoint": "https://test/a2a",
        "capabilities": [],
        "input_schema": {},
        "output_schema": {},
        "auth": [],
        "supports": ["json-rpc/2.0"],
        "jwks_uri": None,
        "exp": int(time.time()) + 3600,
    }
    token = _sign_card_payload(payload, private_jwk)

    with respx.mock(base_url="https://future-peer") as mock:
        mock.get("/.well-known/agent-card.json").mock(
            return_value=Response(
                200,
                content=token.encode(),
                headers={"content-type": "application/jose"},
            )
        )
        mock.get("/.well-known/jwks.json").mock(return_value=Response(200, json=jwks_json))
        card = await discover_card("https://future-peer")
        assert card.name == "researcher"
        assert card.exp is not None


# ---------------------------------------------------------------------------
# C4 — JWT alg/key dispatch
# ---------------------------------------------------------------------------


def test_envkey_jwt_verification_key_returns_secret_for_hs256(fresh_env) -> None:
    """HS256 path: provider.jwt_verification_key('HS256') == provider.jwt_secret()."""
    from voussoir.a2a.keys import EnvKeyProvider

    p = EnvKeyProvider(allow_ephemeral=True)
    secret = p.jwt_secret()
    assert p.jwt_verification_key("HS256") == secret
    assert p.jwt_verification_key("HS384") == secret
    assert p.jwt_verification_key("HS512") == secret


def test_envkey_jwt_verification_key_raises_for_rs256(fresh_env) -> None:
    """RS256 inbound JWT auth needs a separate verification key; EnvKeyProvider
    doesn't yet support it. Must raise a clear error (not silently fall back
    to the symmetric secret).
    """
    from voussoir.a2a.keys import EnvKeyProvider, InboundJWTVerificationNotConfiguredError

    p = EnvKeyProvider(allow_ephemeral=True)
    with pytest.raises(InboundJWTVerificationNotConfiguredError, match="RS256"):
        p.jwt_verification_key("RS256")


@pytest.mark.asyncio
async def test_rs256_inbound_jwt_misconfig_returns_500(make_container, stub_llm, fresh_env) -> None:
    """When VOUSSOIR_A2A_JWT_ALGORITHM=RS256 + EnvKeyProvider is bound, the
    publisher returns 500 (server misconfiguration), NOT a generic 401.
    """
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "test-caller")
    fresh_env.setenv("VOUSSOIR_A2A_JWT_ALGORITHM", "RS256")
    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    # Mint an HS256 token (which the misconfigured server can't verify); the
    # important assertion is the response code, not the token contents.
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "test-caller",
            "aud": "r",
            "iat": now,
            "nbf": now,
            "exp": now + 60,
            "sub": "u",
        },
        b"unused",
        algorithm="HS256",
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        resp = await client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "voussoir.delegate",
                "params": {"task": "x"},
                "id": "1",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 500
    body = json.loads(resp.content)
    assert "misconfigured" in body.get("error", "").lower()


@pytest.mark.asyncio
async def test_hs256_inbound_jwt_still_works(make_container, stub_llm, fresh_env) -> None:
    """The default HS256 happy path is unchanged after the dispatch refactor."""
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "test-caller")
    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name)  # type: ignore[type-abstract]
        resp = await client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "voussoir.delegate",
                "params": {"task": "x"},
                "id": "1",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert "result" in resp.json()


# ---------------------------------------------------------------------------
# C5 — jwks_uri propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_a2a_router_publishes_card_with_jwks_uri(publisher_app_client) -> None:
    """The card payload's jwks_uri must point at the well-known JWKS URL
    derived from the endpoint's origin, so consumers don't have to fall back
    to deriving it from the endpoint (which breaks under non-root mounts).
    """
    from voussoir.a2a.keys import KeyProvider

    client, c, _agent = publisher_app_client
    resp = await client.get("/.well-known/agent-card.json")
    token = resp.text
    provider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    jwks = provider.card_verification_jwks()
    public_jwk = jwks["keys"][0]
    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(public_jwk)
    payload = jwt.decode(
        token,
        public_key,  # type: ignore[arg-type]
        algorithms=["RS256"],
        options={"verify_aud": False, "verify_exp": False},
    )
    assert payload["jwks_uri"] is not None
    # Origin must match the endpoint's origin.
    # endpoint is "https://test/a2a", so jwks_uri should be
    # "https://test/.well-known/jwks.json".
    assert payload["jwks_uri"] == "https://test/.well-known/jwks.json"
