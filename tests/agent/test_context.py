import pytest

from voussoir.agent.context import AgentContext
from voussoir.container import Container
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.protocols import IMemoryStore, ISessionStore


@pytest.fixture
def container() -> Container:
    c = Container()
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    return c


async def test_agent_context_opens_run_scope(container):
    async with await AgentContext.open(
        container=container,
        run_id="r1",
        session_id="s1",
        user_id="u1",
    ) as ctx:
        assert ctx.run_id == "r1"
        assert ctx.session_id == "s1"
        assert ctx.user_id == "u1"
        assert ctx.engine is not None  # ctxforge.CtxForge


async def test_agent_context_creates_session(container):
    async with await AgentContext.open(
        container=container,
        run_id="r1",
        session_id="s1",
        user_id="u1",
    ) as ctx:
        session = await ctx.engine.get_session("s1", "u1")
        assert session is not None
        assert session.session_id == "s1"
        assert session.user_id == "u1"


async def test_agent_context_record_user_message(container):
    async with await AgentContext.open(
        container=container,
        run_id="r1",
        session_id="s1",
        user_id="u1",
    ) as ctx:
        await ctx.record_user_message("hello")
        # Engine recorded an event; verify via the session.
        session = await ctx.engine.get_session("s1", "u1")
        assert session is not None


async def test_agent_context_default_delegation_fields(container):
    """Phase 3: AgentContext exposes delegation lineage with safe defaults."""
    async with await AgentContext.open(
        container=container,
        run_id="r1",
        session_id="s1",
        user_id="u1",
    ) as ctx:
        assert ctx.agent_name == ""
        assert ctx.parent_run_id is None
        assert ctx.delegation_depth == 0
        assert ctx.delegation_chain == []
        assert ctx.max_delegation_depth == 3
