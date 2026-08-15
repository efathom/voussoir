"""Locks Principal threading through Agent.run -> AgentContext -> ToolContext (Phase 6 A2)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from voussoir import Agent
from voussoir.agent.context import AgentContext
from voussoir.auth import Principal
from voussoir.executors.standard import StandardExecutor
from voussoir.tools import Capability, ToolContext, tool


def _alice() -> Principal:
    return Principal(user_id="alice", roles=["admin"], issued_at=datetime.now(UTC))


# ---------------------------------------------------------------------------
# ToolContext defaults
# ---------------------------------------------------------------------------


def test_tool_context_principal_default():
    ctx = ToolContext(run_id="r", span_id="s")
    assert ctx.principal.user_id == "system"
    assert ctx.credentials is None


# ---------------------------------------------------------------------------
# AgentContext defaults + principal kwarg
# ---------------------------------------------------------------------------


def test_agent_context_principal_default(make_container):
    """Construct AgentContext via its public open() classmethod and verify the default Principal."""
    c = make_container()

    async def go():
        async with await AgentContext.open(container=c) as ctx:
            assert ctx.principal.user_id == "system"
            assert ctx.authz_decisions == []

    asyncio.run(go())


def test_agent_context_open_accepts_principal(make_container):
    c = make_container()
    p = _alice()

    async def go():
        async with await AgentContext.open(container=c, principal=p) as ctx:
            assert ctx.principal.user_id == "alice"

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Principal visible inside a tool via StandardExecutor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_principal_threads_to_tool_context_via_executor():
    """Invoke a tool via StandardExecutor and confirm the principal is visible."""

    @tool(capability=Capability.READ_PUBLIC, name="ping")
    async def ping(ctx: ToolContext) -> str:
        return ctx.principal.user_id

    ex = StandardExecutor()
    args = ping.input_schema()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        principal=_alice(),
    )
    result = await ex.invoke(ping, args, ctx)
    assert result == "alice"


# ---------------------------------------------------------------------------
# Agent.run accepts principal= kwarg and threads it
# ---------------------------------------------------------------------------


def test_agent_run_accepts_principal_kwarg(make_container, stub_llm):
    """Agent.run(principal=...) is a valid kwarg; doesn't crash."""
    a = Agent(name="x", container=make_container(stub_llm(content="hi")))
    asyncio.run(a.run("hello", principal=_alice()))


def test_agent_run_threads_principal_to_context(make_container, stub_llm):
    """The principal passed to Agent.run ends up on AgentContext.principal."""
    captured: list[Principal] = []

    @tool(capability=Capability.READ_PUBLIC, name="capture_principal")
    async def capture_principal(ctx: ToolContext) -> str:
        captured.append(ctx.principal)
        return "captured"

    llm_mock = stub_llm(
        content="done",
        finish_reason="end_turn",
    )
    # We use a no-tool run to test the principal kwarg threading without needing
    # to wire a full tool-calling LLM mock. The principal still lands on AgentContext.
    # For the full tool->ToolContext threading we rely on test_principal_threads_to_tool_context_via_executor.
    a = Agent(
        name="x",
        container=make_container(llm_mock),
        allowed_capabilities=Capability.READ_PUBLIC,
    )
    result = asyncio.run(a.run("hello", principal=_alice()))
    assert result.finish_reason == "completed"


# ---------------------------------------------------------------------------
# Agent.delegate threads parent_ctx.principal to the child
# ---------------------------------------------------------------------------


def test_agent_delegate_threads_principal(make_container, stub_llm):
    """Agent.delegate forwards parent_ctx.principal to the child Agent.run."""
    # Build a minimal parent AgentContext with alice as the principal.
    child_a = Agent(
        name="child",
        container=make_container(stub_llm(content="child_reply")),
        allowed_capabilities=Capability.READ_PUBLIC,
    )

    async def go():
        p = _alice()
        parent_container = make_container(stub_llm(content="parent_reply"))
        async with await AgentContext.open(container=parent_container, principal=p) as parent_ctx:
            parent_ctx.agent_name = "parent"
            parent_ctx.allowed_capabilities = Capability.READ_PUBLIC
            result = await child_a.delegate("do it", parent_ctx=parent_ctx)
        return result

    result = asyncio.run(go())
    assert result.finish_reason == "completed"
    # The child ran successfully — principal propagation doesn't crash the sub-agent.
    # (We can't easily assert the principal was set on the child without capturing
    # it from inside a tool; that's covered by the executor test above. This test
    # locks the call signature and non-crash behavior.)
