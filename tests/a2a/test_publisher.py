"""Publisher: card route + JWKS route + JSON-RPC /a2a endpoint."""

from __future__ import annotations

import time
import uuid

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def publisher_client(make_container, stub_llm, fresh_env):
    """Build a FastAPI app exposing a stub Agent over A2A."""
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    # Phase 4.5a P0 #2: publisher requires iss allow-list non-empty.
    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "test-caller")
    c = make_container(stub_llm())
    agent = Agent("researcher", description="r-desc", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        yield client, c, agent


@pytest.mark.asyncio
async def test_card_route_returns_jws(publisher_client) -> None:
    client, _c, _agent = publisher_client
    resp = await client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/jose")
    body = resp.text
    assert body.count(".") == 2  # JWS Compact: header.payload.signature


@pytest.mark.asyncio
async def test_card_route_json_serialization(publisher_client) -> None:
    """Accept: application/json switches to JSON serialization."""
    client, _c, _agent = publisher_client
    resp = await client.get(
        "/.well-known/agent-card.json",
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert "payload" in body
    assert "protected" in body
    assert "signature" in body


@pytest.mark.asyncio
async def test_card_route_cache_control(publisher_client) -> None:
    client, _c, _agent = publisher_client
    resp = await client.get("/.well-known/agent-card.json")
    assert "max-age=60" in resp.headers.get("cache-control", "")


@pytest.mark.asyncio
async def test_card_payload_matches_agent(publisher_client) -> None:
    """Decode the JWS, verify against the JWKS, check the AgentCard contents."""
    from voussoir.a2a.keys import KeyProvider

    client, c, _agent = publisher_client
    resp = await client.get("/.well-known/agent-card.json")
    token = resp.text
    provider = c.resolve(KeyProvider)
    jwks = provider.card_verification_jwks()
    public_jwk = jwks["keys"][0]
    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(public_jwk)
    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        options={"verify_aud": False, "verify_exp": False},
    )
    assert payload["name"] == "researcher"
    assert payload["description"] == "r-desc"
    assert payload["endpoint"] == "https://test/a2a"


@pytest.mark.asyncio
async def test_jwks_route_returns_public_keys(publisher_client) -> None:
    client, _c, _agent = publisher_client
    resp = await client.get("/.well-known/jwks.json")
    assert resp.status_code == 200
    body = resp.json()
    assert "keys" in body
    assert len(body["keys"]) == 1
    public = body["keys"][0]
    assert public["kty"] == "RSA"
    assert "n" in public and "e" in public
    for field in ("d", "p", "q"):
        assert field not in public


@pytest.mark.asyncio
async def test_jwks_route_cache_control(publisher_client) -> None:
    client, _c, _agent = publisher_client
    resp = await client.get("/.well-known/jwks.json")
    assert "max-age=300" in resp.headers.get("cache-control", "")


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


@pytest.mark.asyncio
async def test_a2a_post_missing_auth_returns_401(publisher_client) -> None:
    client, _c, _agent = publisher_client
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "method": "voussoir.delegate", "params": {"task": "x"}, "id": "1"},
    )
    assert resp.status_code == 401
    assert "Bearer" in resp.headers.get("www-authenticate", "")


@pytest.mark.asyncio
async def test_a2a_post_malformed_bearer_returns_401(publisher_client) -> None:
    client, _c, _agent = publisher_client
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "method": "voussoir.delegate", "params": {"task": "x"}, "id": "1"},
        headers={"Authorization": "Bearer not-a-jwt"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_a2a_post_expired_jwt_returns_401(publisher_client) -> None:
    from voussoir.a2a.keys import KeyProvider

    client, c, agent = publisher_client
    token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name, exp_delta=-10)
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "method": "voussoir.delegate", "params": {"task": "x"}, "id": "1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_a2a_post_wrong_audience_returns_401(publisher_client) -> None:
    from voussoir.a2a.keys import KeyProvider

    client, c, _agent = publisher_client
    token = _mint_jwt(c.resolve(KeyProvider), aud="wrong-agent")
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "method": "voussoir.delegate", "params": {"task": "x"}, "id": "1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_a2a_post_malformed_json_returns_parse_error(publisher_client) -> None:
    from voussoir.a2a.errors import A2AErrorCode
    from voussoir.a2a.keys import KeyProvider

    client, c, agent = publisher_client
    token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name)
    resp = await client.post(
        "/a2a",
        content=b"{not json",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == A2AErrorCode.PARSE_ERROR


@pytest.mark.asyncio
async def test_a2a_post_wrong_method_returns_method_not_found(publisher_client) -> None:
    from voussoir.a2a.errors import A2AErrorCode
    from voussoir.a2a.keys import KeyProvider

    client, c, agent = publisher_client
    token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name)
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "method": "voussoir.cancel", "params": {}, "id": "1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == A2AErrorCode.METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_a2a_post_missing_task_returns_invalid_params(publisher_client) -> None:
    from voussoir.a2a.errors import A2AErrorCode
    from voussoir.a2a.keys import KeyProvider

    client, c, agent = publisher_client
    token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name)
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "method": "voussoir.delegate", "params": {}, "id": "1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == A2AErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_a2a_post_happy_path_returns_agent_result(publisher_client) -> None:
    from voussoir.a2a.keys import KeyProvider

    client, c, agent = publisher_client
    token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name)
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "voussoir.delegate",
            "params": {"task": "research X"},
            "id": "r1",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "r1"
    assert "result" in body
    assert body["result"]["output"] == "ok"


@pytest.mark.asyncio
async def test_a2a_post_internal_error_sanitized(make_container, fresh_env) -> None:
    """An unhandled exception inside agent.run surfaces as -32603 with a
    sanitized message (no internal traceback on the wire)."""
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider

    from voussoir.a2a.errors import A2AErrorCode
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent
    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "test-caller")
    broken_llm = MagicMock(spec=ILLMProvider)
    broken_llm.name = "anthropic"
    broken_llm.chat = AsyncMock(side_effect=RuntimeError("internal-detail-do-not-leak"))
    c = make_container(broken_llm)
    c.bind(ILLMProviderProto, broken_llm)
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        token = _mint_jwt(c.resolve(KeyProvider), aud=agent.name)
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
    body = resp.json()
    assert body["error"]["code"] == A2AErrorCode.INTERNAL_ERROR
    assert "internal-detail-do-not-leak" not in str(body)
