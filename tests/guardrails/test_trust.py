from contextlib import AsyncExitStack
from unittest.mock import MagicMock

from voussoir.agent.context import AgentContext
from voussoir.guardrails import Trust


def test_trust_values():
    assert Trust.SYSTEM == "system"
    assert Trust.USER == "user"
    assert Trust.INTERNAL == "internal"
    assert Trust.UNTRUSTED == "untrusted"


def test_trust_is_str_enum():
    assert isinstance(Trust.UNTRUSTED, str)


def _make_ctx(make_container) -> AgentContext:
    """Build a minimal AgentContext for taint-field tests."""
    c = make_container()
    return AgentContext(
        container=c,
        run_id="r",
        trace_id="t",
        session_id="s",
        user_id="test-user",
        engine=MagicMock(),
        _stack=AsyncExitStack(),
    )


def test_agent_context_taint_starts_empty(make_container):
    ctx = _make_ctx(make_container)
    assert ctx.taint == set()


def test_agent_context_taint_accumulates(make_container):
    ctx = _make_ctx(make_container)
    ctx.taint.add(Trust.UNTRUSTED)
    ctx.taint.add(Trust.INTERNAL)
    assert Trust.UNTRUSTED in ctx.taint
    assert Trust.INTERNAL in ctx.taint
    assert Trust.SYSTEM not in ctx.taint
