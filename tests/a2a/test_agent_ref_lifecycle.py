"""Phase 4.5a P0 #8 — AgentRef requires async-with or explicit http_client."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from voussoir.a2a.agent_ref import AgentRef
from voussoir.a2a.publisher import make_a2a_router
from voussoir.agent.agent import Agent


@pytest.mark.asyncio
async def test_bare_construct_then_delegate_raises(make_container, stub_llm, fresh_env) -> None:
    """AgentRef(card) without async-with and without http_client= raises on
    .delegate() with a clear hint at the two supported patterns."""
    server_c = make_container(stub_llm(content="from remote"))
    agent = Agent("r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref_external = await AgentRef.discover("https://test", http_client=client)

    # Now construct a bare AgentRef using only the card; no http_client passed,
    # no async-with. .delegate() must raise.
    bare = AgentRef(ref_external.card)
    with pytest.raises(RuntimeError, match="async with"):
        await bare.delegate("hello", parent_ctx=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_with_constructs_and_closes_client(make_container, stub_llm, fresh_env) -> None:
    """`async with AgentRef(card) as ref:` lazily constructs the client on
    __aenter__ and clears it on __aexit__."""
    server_c = make_container(stub_llm(content="from remote"))
    agent = Agent("r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)

    bare = AgentRef(ref.card)
    # Before async-with: no client.
    assert bare._http is None  # noqa: SLF001
    assert bare._owns_client is False  # noqa: SLF001

    async with bare as bare_entered:
        # Inside async-with: client is constructed, owned.
        assert bare_entered._http is not None  # noqa: SLF001
        assert bare_entered._owns_client is True  # noqa: SLF001

    # After async-with: client cleared.
    assert bare._http is None  # noqa: SLF001
    assert bare._owns_client is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_external_client_no_async_with_needed(make_container, stub_llm, fresh_env) -> None:
    """AgentRef(card, http_client=external_client) works without async-with;
    `.delegate()` does not raise the lifecycle guard."""
    server_c = make_container(stub_llm(content="from remote"))
    agent = Agent("r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)

        # Build a fresh ref against the same card, explicitly passing the client.
        explicit = AgentRef(ref.card, http_client=client)
        # External client: _http is set immediately; AgentRef does NOT own it.
        assert explicit._http is client  # noqa: SLF001
        assert explicit._owns_client is False  # noqa: SLF001
