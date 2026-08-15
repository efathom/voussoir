"""Phase 4.5b B2 — full DelegationError subclass coverage for AgentRef.

One test per documented failure mode, driven by respx. This file is the
single canonical home for AgentRef.delegate() failure-mode coverage; it
consolidates the 4 tests Phase 4.5a originally placed in
test_delegation_errors.py (deleted in Tranche B) plus 3 net-new cases
(connection refused, 403 forbidden, non-JSON response body).

Coverage map (transport failure -> DelegationError subclass):
- httpx.ConnectError / 5xx              -> RemoteUnreachable
- 401 / 403                             -> RemoteAuthFailed
- JSON-RPC `error` envelope             -> RemoteProtocolError
- JSON parse failure / valid JSON-RPC missing `result` -> RemoteMalformed

Add a test here when a new DelegationError subclass lands or a new
failure mode is discovered.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response

from voussoir.a2a.agent_ref import AgentRef
from voussoir.a2a.card import AgentCard, AuthMethod
from voussoir.a2a.errors import (
    RemoteAuthFailed,
    RemoteMalformed,
    RemoteProtocolError,
    RemoteUnreachable,
)
from voussoir.agent.context import AgentContext
from voussoir.container import Container


def _make_card() -> AgentCard:
    """Build a minimal AgentCard pointing at the respx-mocked peer.

    These tests skip the discover step (which verifies the card's JWS) and
    construct AgentRef directly from a synthesized card; the signature /
    JWKS plumbing is exercised in test_card_signing.py.
    """
    return AgentCard(
        name="remote",
        description="r",
        endpoint="https://peer/a2a",  # type: ignore[arg-type]
        capabilities=[],
        input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
        output_schema={"type": "string"},
        auth=[AuthMethod(type="bearer")],
        supports=["json-rpc/2.0"],
    )


async def _open_ctx(c: Container) -> AgentContext:
    """Open an AgentContext bound to `c`. The caller is responsible for
    setting ctx.agent_name (which becomes the JWT `iss`) before invoking
    ref.delegate().
    """
    return await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u")


@pytest.fixture
async def delegate_caller(make_container, stub_llm, fresh_env):  # type: ignore[no-untyped-def]
    """Yields (client, mock, ctx, ref) wired up to peer 'https://peer/a2a'.

    Each test mocks specific responses on `mock` and asserts the resulting
    DelegationError subclass. The boilerplate (env var, container, ctx,
    httpx client, respx mock, AgentRef construction) is centralized here.
    """
    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    c = make_container(stub_llm())
    async with httpx.AsyncClient() as client, respx.mock(base_url="https://peer") as mock:
        ref = AgentRef(_make_card(), http_client=client)
        async with await _open_ctx(c) as ctx:
            ctx.agent_name = "caller"
            yield client, mock, ctx, ref


# --- RemoteUnreachable -------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_refused_raises_remote_unreachable(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """httpx.ConnectError -> RemoteUnreachable (covers the httpx.HTTPError
    branch around the POST in AgentRef.delegate)."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(side_effect=httpx.ConnectError("connection refused"))
    with pytest.raises(RemoteUnreachable):
        await ref.delegate("task", parent_ctx=ctx)


@pytest.mark.asyncio
async def test_500_raises_remote_unreachable(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """HTTP 500 from peer -> RemoteUnreachable. The peer is reachable but
    overloaded; the spec maps 5xx into "unreachable" rather than "protocol
    error" because the caller's correct response is to retry/back-off."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(return_value=Response(500, text="upstream error"))
    with pytest.raises(RemoteUnreachable):
        await ref.delegate("task", parent_ctx=ctx)


# --- RemoteAuthFailed --------------------------------------------------------


@pytest.mark.asyncio
async def test_401_raises_remote_auth_failed(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """HTTP 401 from peer -> RemoteAuthFailed (caller's JWT rejected)."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(return_value=Response(401, json={"error": "unauthorized"}))
    with pytest.raises(RemoteAuthFailed):
        await ref.delegate("task", parent_ctx=ctx)


@pytest.mark.asyncio
async def test_403_raises_remote_auth_failed(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """HTTP 403 from peer -> RemoteAuthFailed (caller authenticated but
    not authorized for voussoir.delegate on this peer)."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(return_value=Response(403, json={"error": "forbidden"}))
    with pytest.raises(RemoteAuthFailed):
        await ref.delegate("task", parent_ctx=ctx)


# --- RemoteProtocolError -----------------------------------------------------


@pytest.mark.asyncio
async def test_jsonrpc_error_envelope_raises_remote_protocol_error(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """JSON-RPC body with `error` block -> RemoteProtocolError. The peer
    spoke valid JSON-RPC but returned a protocol-level error, which is
    distinct from a transport failure (the peer IS reachable) and from a
    malformed response (the envelope IS well-formed)."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(
        return_value=Response(
            200,
            json={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "internal error"},
                "id": "1",
            },
        )
    )
    with pytest.raises(RemoteProtocolError):
        await ref.delegate("task", parent_ctx=ctx)


# --- RemoteMalformed ---------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_result_field_raises_remote_malformed(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """JSON-RPC body missing both `result` and `error` -> RemoteMalformed.
    The envelope passes JsonRpcResponse validation (id is present, no extra
    keys) but the "exactly one of result/error" invariant is violated."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(return_value=Response(200, json={"jsonrpc": "2.0", "id": "1"}))
    with pytest.raises(RemoteMalformed):
        await ref.delegate("task", parent_ctx=ctx)


@pytest.mark.asyncio
async def test_non_json_body_raises_remote_malformed(delegate_caller) -> None:  # type: ignore[no-untyped-def]
    """200 OK with a non-JSON body -> RemoteMalformed. AgentRef.delegate
    catches ValueError / TypeError from resp.json() + JsonRpcResponse
    validation and surfaces them as RemoteMalformed."""
    _client, mock, ctx, ref = delegate_caller
    mock.post("/a2a").mock(
        return_value=Response(
            200,
            text="not json",
            headers={"content-type": "text/plain"},
        )
    )
    with pytest.raises(RemoteMalformed):
        await ref.delegate("task", parent_ctx=ctx)
