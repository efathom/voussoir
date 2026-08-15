"""Phase 4b exit criteria — gates the v0.4.0b-phase4b tag."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


# Exit 1 — AgentCard schema validates required fields; extra='forbid'.
def test_exit_1_agent_card_schema() -> None:
    from pydantic import ValidationError

    from voussoir.a2a import AgentCard

    card = AgentCard(name="x", endpoint="https://h/a2a")
    assert card.name == "x"
    with pytest.raises(ValidationError):
        AgentCard(name="x", endpoint="https://h/a2a", unknown="nope")  # type: ignore[call-arg]


# Exit 2 — publisher card route → JWS → verify → AgentCard.
@pytest.mark.asyncio
async def test_exit_2_publisher_card_round_trip(make_container, stub_llm, fresh_env) -> None:
    from voussoir.a2a import AgentCard, KeyProvider, make_a2a_router
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("researcher", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        resp = await client.get("/.well-known/agent-card.json")
        token = resp.text
        jwks = (await client.get("/.well-known/jwks.json")).json()
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwks["keys"][0])
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_exp": False},
        )
        card = AgentCard.model_validate(payload)
    assert card.name == "researcher"
    _ = KeyProvider  # imported for side-effects (re-export check)


# Exit 3 — JSON-RPC happy path.
@pytest.mark.asyncio
async def test_exit_3_jsonrpc_happy_path(make_container, stub_llm, fresh_env) -> None:
    from voussoir.a2a import KeyProvider, make_a2a_router
    from voussoir.agent.agent import Agent

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    c = make_container(stub_llm(content="answered"))
    agent = Agent("researcher", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    provider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "caller",
            "aud": "researcher",
            "iat": now,
            "nbf": now,
            "exp": now + 60,
            "sub": "u",
        },
        provider.jwt_secret(),
        algorithm=provider.jwt_algorithm(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        resp = await client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "voussoir.delegate",
                "params": {"task": "do"},
                "id": "1",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["output"] == "answered"


# Exit 4 — JSON-RPC auth failures.
@pytest.mark.asyncio
async def test_exit_4_jsonrpc_auth_failures(make_container, stub_llm, fresh_env) -> None:
    from voussoir.a2a import KeyProvider, make_a2a_router
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        body = {
            "jsonrpc": "2.0",
            "method": "voussoir.delegate",
            "params": {"task": "x"},
            "id": "1",
        }
        # Missing auth.
        assert (await client.post("/a2a", json=body)).status_code == 401
        # Bad token.
        assert (
            await client.post("/a2a", json=body, headers={"Authorization": "Bearer xxx"})
        ).status_code == 401
        # Expired token.
        provider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
        now = int(time.time())
        expired = jwt.encode(
            {
                "iss": "c",
                "aud": "r",
                "iat": now - 1000,
                "nbf": now - 1000,
                "exp": now - 500,
                "sub": "u",
            },
            provider.jwt_secret(),
            algorithm=provider.jwt_algorithm(),
        )
        assert (
            await client.post("/a2a", json=body, headers={"Authorization": f"Bearer {expired}"})
        ).status_code == 401


# Exit 5 — Internal-error path produces a sanitized -32603.
@pytest.mark.asyncio
async def test_exit_5_internal_error_sanitized(make_container, fresh_env) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider

    from voussoir.a2a import A2AErrorCode, KeyProvider, make_a2a_router
    from voussoir.agent.agent import Agent
    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "c")
    broken_llm = MagicMock(spec=ILLMProvider)
    broken_llm.name = "anthropic"
    broken_llm.chat = AsyncMock(side_effect=RuntimeError("internal-detail-do-not-leak"))
    c = make_container(broken_llm)
    c.bind(ILLMProviderProto, broken_llm)
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    provider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "c",
            "aud": "r",
            "iat": now,
            "nbf": now,
            "exp": now + 60,
            "sub": "u",
        },
        provider.jwt_secret(),
        algorithm=provider.jwt_algorithm(),
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
    body = resp.json()
    assert body["error"]["code"] == A2AErrorCode.INTERNAL_ERROR
    assert "internal-detail-do-not-leak" not in str(body)


# Exit 6 — AgentRef satisfies IDelegate; discovery round-trip.
@pytest.mark.asyncio
async def test_exit_6_agent_ref_round_trip(make_container, stub_llm, fresh_env) -> None:
    from voussoir.a2a import AgentRef, make_a2a_router
    from voussoir.agent.agent import Agent
    from voussoir.agent.delegate import IDelegate

    c = make_container(stub_llm())
    agent = Agent("researcher", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)
    assert isinstance(ref, IDelegate)
    assert ref.name == "researcher"


# Exit 7 — Mixed delegates: AgentRef + local Agent in lead's delegates list.
@pytest.mark.asyncio
async def test_exit_7_mixed_delegates(make_container, stub_llm, fresh_env) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    from voussoir.a2a import AgentRef, KeyProvider, make_a2a_router
    from voussoir.agent.agent import Agent
    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "lead")
    server_c = make_container(stub_llm(content="from-remote"))
    server_agent = Agent("remote_helper", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(server_agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)

        lead_llm = MagicMock(spec=ILLMProvider)
        lead_llm.name = "anthropic"
        lead_llm.chat = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    model="stub",
                    input_tokens=1,
                    output_tokens=1,
                    finish_reason="tool_use",
                    raw_response={
                        "tool_calls": [
                            {
                                "id": "t1",
                                "name": "delegate_to_remote_helper",
                                "arguments": {"task": "x"},
                            }
                        ]
                    },
                ),
                LLMResponse(
                    content="lead saw remote.",
                    model="stub",
                    input_tokens=1,
                    output_tokens=1,
                    finish_reason="end_turn",
                    raw_response=None,
                ),
            ]
        )
        caller_c = make_container(lead_llm)
        caller_c.bind(ILLMProviderProto, lead_llm)
        caller_c.bind(KeyProvider, server_c.resolve(KeyProvider))  # type: ignore[type-abstract]

        lead = Agent("lead", instructions="", delegates=[ref], container=caller_c)
        result = await lead.run("test")
    assert "lead saw remote" in result.output
    assert "remote_helper" in result.delegation_chain


# Exit 8 — Tampered AgentCard signature → CardVerificationError.
@pytest.mark.asyncio
async def test_exit_8_tampered_card_rejected(make_container, stub_llm, fresh_env) -> None:
    import respx
    from httpx import Response

    from voussoir.a2a import CardVerificationError, discover_card, make_a2a_router
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        raw = (await client.get("/.well-known/agent-card.json")).text
        jwks_json = (await client.get("/.well-known/jwks.json")).json()

    h, p, sig = raw.split(".")
    tampered = f"{h}.{p}.{sig[:-2]}XX"

    # v1.0.4 E2 / C5: card now carries an explicit jwks_uri derived from
    # the publisher's endpoint origin (https://test), so the resolver
    # fetches the JWKS from https://test/.well-known/jwks.json (not
    # https://tampered/...). Mock both.
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://tampered/.well-known/agent-card.json").mock(
            return_value=Response(
                200, content=tampered.encode(), headers={"content-type": "application/jose"}
            )
        )
        mock.get("https://test/.well-known/jwks.json").mock(
            return_value=Response(200, json=jwks_json)
        )
        with pytest.raises(CardVerificationError):
            await discover_card("https://tampered")


# Exit 9 — Phase 3.5 + 4a regressions still importable.
def test_exit_9_prior_phase_regressions() -> None:
    import tests.test_phase4a_exit as p4a
    import tests.test_phase35_exit as p35

    for name in (
        "test_exit_1_validator_protocol_passes_task",
        "test_exit_6_three_agent_yaml_round_trip",
    ):
        assert hasattr(p35, name), f"Phase 3.5 exit test {name!r} missing"
    for name in (
        "test_exit_1_agent_satisfies_idelegate",
        "test_exit_3_local_agent_delegation_e2e",
    ):
        assert hasattr(p4a, name), f"Phase 4a exit test {name!r} missing"
