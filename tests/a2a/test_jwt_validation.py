"""Phase 4.5a P0 #2 — JWT validation: iss/exp/aud/nbf/iat/sub required; iss
allow-listed against KeyProvider.expected_issuers(); algorithm allow-listed."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from voussoir.a2a.keys import KeyProvider
from voussoir.a2a.publisher import make_a2a_router
from voussoir.agent.agent import Agent


def _issue(payload: dict, secret: bytes, alg: str = "HS256") -> str:
    return jwt.encode(payload, secret, algorithm=alg)


@pytest.mark.asyncio
async def test_jwt_without_iss_rejected(make_container, stub_llm, fresh_env, monkeypatch) -> None:
    """A JWT missing the iss claim is rejected even if everything else is valid."""
    monkeypatch.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    provider: KeyProvider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    secret = provider.jwt_secret()
    now = int(time.time())
    token = _issue(
        {"aud": "r", "iat": now, "exp": now + 60, "nbf": now, "sub": "u"},  # no iss
        secret,
        provider.jwt_algorithm(),
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
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_with_disallowed_iss_rejected(
    make_container, stub_llm, fresh_env, monkeypatch
) -> None:
    """iss not in KeyProvider.expected_issuers() is rejected."""
    monkeypatch.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "only_this_peer")
    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    provider: KeyProvider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    secret = provider.jwt_secret()
    now = int(time.time())
    token = _issue(
        {"iss": "attacker", "aud": "r", "iat": now, "exp": now + 60, "nbf": now, "sub": "u"},
        secret,
        provider.jwt_algorithm(),
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
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_alg_none_rejected(make_container, stub_llm, fresh_env, monkeypatch) -> None:
    """alg=none in JWT header is rejected (algorithm allow-list)."""
    monkeypatch.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    now = int(time.time())
    bad_token = jwt.encode(
        {"iss": "caller", "aud": "r", "iat": now, "exp": now + 60, "nbf": now, "sub": "u"},
        "",
        algorithm="none",
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
            headers={"Authorization": f"Bearer {bad_token}"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_no_expected_issuers_rejects_all(make_container, stub_llm, fresh_env) -> None:
    """Phase 4.5a P0 #2: empty issuer allow-list (default in prod) → 401."""
    # fresh_env clears VOUSSOIR_A2A_ALLOWED_ISSUERS — empty allow-list.
    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    provider: KeyProvider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    secret = provider.jwt_secret()
    now = int(time.time())
    token = _issue(
        {"iss": "caller", "aud": "r", "iat": now, "exp": now + 60, "nbf": now, "sub": "u"},
        secret,
        provider.jwt_algorithm(),
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
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_happy_path_with_iss_works(
    make_container, stub_llm, fresh_env, monkeypatch
) -> None:
    """Sanity: a fully-claimed JWT with allowed iss succeeds, and the response
    body uses the redacted WireAgentResult shape (no steps/delegation_chain)."""
    monkeypatch.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    c = make_container(stub_llm(content="ok"))
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    provider: KeyProvider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    secret = provider.jwt_secret()
    now = int(time.time())
    token = _issue(
        {"iss": "caller", "aud": "r", "iat": now, "exp": now + 60, "nbf": now, "sub": "u"},
        secret,
        provider.jwt_algorithm(),
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
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["output"] == "ok"
    # Phase 4.5a P0 #3: wire model redacts these.
    assert "steps" not in body["result"]
    assert "delegation_chain" not in body["result"]
    assert "trace_id" not in body["result"]
