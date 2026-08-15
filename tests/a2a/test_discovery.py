"""discover_card() — fetch + verify + parse AgentCard."""

from __future__ import annotations

import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from voussoir.a2a.discovery import discover_card
from voussoir.a2a.errors import CardVerificationError


@pytest.fixture
async def publisher_app_client(make_container, stub_llm, fresh_env):
    """A publisher app + httpx client bound to it via ASGITransport."""
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("researcher", description="r-desc", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        yield client, c, agent


@pytest.mark.asyncio
async def test_discover_signed_card_round_trip(publisher_app_client) -> None:
    client, _c, _agent = publisher_app_client
    card = await discover_card("https://test", http_client=client)
    assert card.name == "researcher"
    assert str(card.endpoint) == "https://test/a2a"


@pytest.mark.asyncio
async def test_discover_rejects_tampered_signature(publisher_app_client) -> None:
    """Fetch the card, flip a byte in the signature, re-feed via respx.

    v1.0.4 E2 / C5: the card payload now carries an explicit `jwks_uri`
    (https://test/.well-known/jwks.json — derived from the publisher's
    endpoint origin) instead of relying on the consumer fallback. We mock
    BOTH the tampered card endpoint AND the publisher's JWKS endpoint so
    the resolver can fetch the public key before failing signature
    verification.
    """
    client, _c, _agent = publisher_app_client
    raw = (await client.get("/.well-known/agent-card.json")).text
    jwks_json = (await client.get("/.well-known/jwks.json")).json()

    header_b64, payload_b64, sig_b64 = raw.split(".")
    tampered = f"{header_b64}.{payload_b64}.{sig_b64[:-2]}XX"

    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://tampered/.well-known/agent-card.json").mock(
            return_value=Response(
                200,
                content=tampered.encode(),
                headers={"content-type": "application/jose"},
            )
        )
        mock.get("https://test/.well-known/jwks.json").mock(
            return_value=Response(200, json=jwks_json)
        )
        with pytest.raises(CardVerificationError):
            await discover_card("https://tampered")


@pytest.mark.asyncio
async def test_discover_unsigned_dev_mode(publisher_app_client) -> None:
    """verify_signature=False accepts JWS-Compact cards without verifying."""
    client, _c, _agent = publisher_app_client
    card = await discover_card(
        "https://test",
        verify_signature=False,
        http_client=client,
    )
    assert card.name == "researcher"


@pytest.mark.asyncio
async def test_discover_invalid_card_schema_raises() -> None:
    """A response that doesn't match AgentCard schema raises ValidationError."""
    from pydantic import ValidationError

    with respx.mock(base_url="http://bad") as mock:
        mock.get("/.well-known/agent-card.json").mock(
            return_value=Response(200, json={"unknown_field": "nope"})
        )
        with pytest.raises(ValidationError):
            await discover_card("http://bad", verify_signature=False)


@pytest.mark.asyncio
async def test_discover_unreachable_url() -> None:
    """Connection errors propagate."""
    import httpx

    with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
        await discover_card("http://localhost:1", timeout_s=0.5)
