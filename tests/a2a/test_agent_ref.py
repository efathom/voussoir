"""AgentRef — IDelegate impl for remote agents."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from voussoir.a2a.agent_ref import AgentRef
from voussoir.agent.delegate import IDelegate


@pytest.fixture
async def publisher_app_client(make_container, stub_llm, fresh_env):
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    # Phase 4.5a P0 #2: publisher requires iss allow-list non-empty. The
    # caller_agent_name below mints JWTs with iss="caller_agent", so allow it.
    fresh_env.setenv(
        "VOUSSOIR_A2A_ALLOWED_ISSUERS",
        "caller-lead,caller,caller_agent,test-caller,lead",
    )
    c = make_container(stub_llm(content="from researcher"))
    agent = Agent("researcher", description="r-desc", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        yield client, c, agent


@pytest.mark.asyncio
async def test_agent_ref_satisfies_idelegate(publisher_app_client) -> None:
    client, _c, _agent = publisher_app_client
    ref = await AgentRef.discover("https://test", http_client=client)
    assert isinstance(ref, IDelegate)


@pytest.mark.asyncio
async def test_agent_ref_attrs_from_card(publisher_app_client) -> None:
    client, _c, _agent = publisher_app_client
    ref = await AgentRef.discover("https://test", http_client=client)
    assert ref.name == "researcher"
    assert ref.description == "r-desc"


@pytest.mark.asyncio
async def test_agent_ref_delegate_round_trip(publisher_app_client, make_container) -> None:
    """AgentRef.delegate makes the JSON-RPC call and parses the result."""
    from voussoir.a2a.keys import KeyProvider
    from voussoir.agent.context import AgentContext

    client, server_container, _server_agent = publisher_app_client
    ref = await AgentRef.discover("https://test", http_client=client)

    # Caller-side container shares the server's KeyProvider (HS256 shared
    # secret) so JWT verification succeeds.
    caller_c = make_container()
    caller_c.bind(KeyProvider, server_container.resolve(KeyProvider))  # type: ignore[type-abstract]

    async with await AgentContext.open(
        container=caller_c, run_id="caller-r", session_id="cs", user_id="alice"
    ) as parent_ctx:
        parent_ctx.agent_name = "caller-lead"
        result = await ref.delegate("research X", parent_ctx=parent_ctx)
    assert "from researcher" in result.output


@pytest.mark.asyncio
async def test_agent_ref_handles_remote_401(publisher_app_client, make_container) -> None:
    """A caller with the wrong JWT secret hits 401; AgentRef raises
    RemoteAuthFailed (Phase 4.5a P1 #25 — was PolicyViolationError pre-4.5a)."""
    from voussoir.a2a.errors import RemoteAuthFailed
    from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
    from voussoir.agent.context import AgentContext

    client, _server_c, _agent = publisher_app_client
    ref = await AgentRef.discover("https://test", http_client=client)

    # Caller has a DIFFERENT JWT secret → server-side decode fails.
    caller_c = make_container()
    bad_provider = EnvKeyProvider(allow_ephemeral=True)
    bad_provider._jwt_secret = b"wrong-secret-32bytes-padded!!!!!!"  # noqa: SLF001
    caller_c.bind(KeyProvider, bad_provider)  # type: ignore[type-abstract]

    async with await AgentContext.open(
        container=caller_c, run_id="r", session_id="s", user_id="u"
    ) as parent_ctx:
        parent_ctx.agent_name = "caller"
        with pytest.raises(RemoteAuthFailed):
            await ref.delegate("X", parent_ctx=parent_ctx)


@pytest.mark.asyncio
async def test_agent_ref_in_local_agent_delegates(publisher_app_client, make_container) -> None:
    """An AgentRef can sit in Agent(delegates=[...]) alongside a local
    Agent. The synthetic-tool layer doesn't distinguish kinds (Phase 4a)."""
    from unittest.mock import AsyncMock, MagicMock

    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    from voussoir.a2a.keys import KeyProvider
    from voussoir.agent.agent import Agent
    from voussoir.protocols import ILLMProvider as ILLMProviderProto

    client, server_container, _server_agent = publisher_app_client
    ref = await AgentRef.discover("https://test", http_client=client)

    # Lead LLM: emit tool_use to delegate_to_researcher, then final text.
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
                        {"id": "t1", "name": "delegate_to_researcher", "arguments": {"task": "go"}}
                    ]
                },
            ),
            LLMResponse(
                content="Lead saw researcher.",
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
    caller_c.bind(KeyProvider, server_container.resolve(KeyProvider))  # type: ignore[type-abstract]

    lead = Agent("lead", instructions="", delegates=[ref], container=caller_c)
    result = await lead.run("test")
    assert "Lead saw researcher" in result.output
    assert "researcher" in result.delegation_chain
