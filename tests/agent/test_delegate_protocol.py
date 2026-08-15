"""IDelegate Protocol shape + auto-wrap + NamedDelegate behavior."""

from __future__ import annotations

import inspect

import pytest

from voussoir.agent.delegate import IDelegate


def test_idelegate_is_protocol() -> None:
    """IDelegate is a runtime-checkable Protocol with name, description, delegate."""
    # delegate.py uses `from __future__ import annotations`, so class-level
    # annotations are stringified (PEP 563). Compare strings, not types.
    annotations = IDelegate.__annotations__
    assert annotations.get("name") == "str"
    assert annotations.get("description") == "str"

    delegate_method = IDelegate.delegate
    assert inspect.iscoroutinefunction(delegate_method)


def test_idelegate_delegate_signature() -> None:
    """`delegate` takes task: str and keyword-only parent_ctx: AgentContext."""
    sig = inspect.signature(IDelegate.delegate)
    params = sig.parameters
    assert "task" in params
    assert "parent_ctx" in params
    assert params["parent_ctx"].kind is inspect.Parameter.KEYWORD_ONLY


def test_runtime_checkable_accepts_minimal_impl() -> None:
    """Any object with name, description, and async delegate() satisfies IDelegate."""

    class _Minimal:
        name = "x"
        description = ""

        async def delegate(self, task, *, parent_ctx):  # type: ignore[no-untyped-def]
            return None

    assert isinstance(_Minimal(), IDelegate)


def test_named_delegate_attrs() -> None:
    from voussoir.agent.delegate import NamedDelegate

    nd = NamedDelegate("researcher")
    assert nd.name == "researcher"
    assert nd.description == ""


def test_named_delegate_satisfies_protocol() -> None:
    from voussoir.agent.delegate import NamedDelegate

    assert isinstance(NamedDelegate("x"), IDelegate)


@pytest.mark.asyncio
async def test_named_delegate_no_registry_raises(make_container, stub_llm) -> None:
    """Container with no AgentRegistry → PolicyViolationError."""
    from voussoir.agent.context import AgentContext
    from voussoir.agent.delegate import NamedDelegate
    from voussoir.agent.policy import PolicyViolationError

    c = make_container(stub_llm())
    nd = NamedDelegate("missing")
    async with await AgentContext.open(
        container=c, run_id="r1", session_id="s1", user_id="u1"
    ) as ctx:
        with pytest.raises(PolicyViolationError, match="AgentRegistry not bound"):
            await nd.delegate("X", parent_ctx=ctx)


@pytest.mark.asyncio
async def test_named_delegate_resolves_via_registry(make_container, stub_llm) -> None:
    """When the registry has the name, NamedDelegate.delegate dispatches to
    the resolved Agent's .delegate() method."""
    from voussoir.agent.agent import Agent
    from voussoir.agent.context import AgentContext
    from voussoir.agent.delegate import NamedDelegate
    from voussoir.agent.registry import register_agent

    c = make_container(stub_llm(content="researched"))
    register_agent(c, Agent("researcher", instructions="r", container=c))
    nd = NamedDelegate("researcher")
    async with await AgentContext.open(
        container=c, run_id="r1", session_id="s1", user_id="u1"
    ) as ctx:
        result = await nd.delegate("research X", parent_ctx=ctx)
    assert "researched" in result.output


@pytest.mark.asyncio
async def test_agent_satisfies_idelegate(make_container, stub_llm) -> None:
    """An Agent instance satisfies IDelegate (runtime-checkable)."""
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("x", instructions="", container=c)
    assert isinstance(agent, IDelegate)


@pytest.mark.asyncio
async def test_agent_delegate_runs_as_subagent(make_container, stub_llm) -> None:
    """Agent.delegate(task, parent_ctx=ctx) runs the agent as a sub-agent
    and returns its AgentResult."""
    from voussoir.agent.agent import Agent
    from voussoir.agent.context import AgentContext

    c = make_container(stub_llm(content="sub-agent ran"))
    agent = Agent("worker", instructions="", container=c)
    async with await AgentContext.open(
        container=c, run_id="r1", session_id="s1", user_id="u1"
    ) as ctx:
        result = await agent.delegate("do the task", parent_ctx=ctx)
    assert "sub-agent ran" in result.output


@pytest.mark.asyncio
async def test_named_delegate_unknown_name_raises(make_container, stub_llm) -> None:
    """Registry exists but name missing → PolicyViolationError with the name."""
    from voussoir.agent.agent import Agent
    from voussoir.agent.context import AgentContext
    from voussoir.agent.delegate import NamedDelegate
    from voussoir.agent.policy import PolicyViolationError
    from voussoir.agent.registry import register_agent

    c = make_container(stub_llm())
    register_agent(c, Agent("researcher", instructions="", container=c))
    nd = NamedDelegate("typo")
    async with await AgentContext.open(
        container=c, run_id="r1", session_id="s1", user_id="u1"
    ) as ctx:
        with pytest.raises(PolicyViolationError, match="typo"):
            await nd.delegate("X", parent_ctx=ctx)


def test_string_delegate_auto_wrapped(make_container, stub_llm) -> None:
    """Agent.__init__(delegates=['x']) wraps 'x' into NamedDelegate internally.
    Phase 4.5a P1 #23 requires AgentRegistry to be bound at construction; the
    wrap behavior is unchanged."""
    from voussoir.agent.agent import Agent
    from voussoir.agent.delegate import NamedDelegate
    from voussoir.agent.registry import AgentRegistry

    c = make_container(stub_llm())
    c.bind(AgentRegistry, AgentRegistry())
    lead = Agent("lead", instructions="", delegates=["x"], container=c)
    assert len(lead.delegates) == 1
    assert isinstance(lead.delegates[0], NamedDelegate)
    assert lead.delegates[0].name == "x"


def test_agent_delegate_passes_through(make_container, stub_llm) -> None:
    """An Agent instance in the delegates list passes through unchanged."""
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper], container=c)
    assert lead.delegates[0] is helper


def test_mixed_delegates_after_wrap(make_container, stub_llm) -> None:
    """Mixed list: Agent passes through, str wrapped to NamedDelegate.
    Phase 4.5a P1 #23 requires AgentRegistry binding when strings are present."""
    from voussoir.agent.agent import Agent
    from voussoir.agent.delegate import NamedDelegate
    from voussoir.agent.registry import AgentRegistry

    c = make_container(stub_llm())
    c.bind(AgentRegistry, AgentRegistry())
    helper = Agent("helper", instructions="", container=c)
    lead = Agent("lead", instructions="", delegates=[helper, "x"], container=c)
    assert lead.delegates[0] is helper
    assert isinstance(lead.delegates[1], NamedDelegate)
    assert lead.delegates[1].name == "x"


def test_custom_idelegate_accepted(make_container, stub_llm) -> None:
    """A user-defined class satisfying IDelegate is acceptable in delegates."""
    from voussoir.agent.agent import Agent
    from voussoir.agent.result import AgentResult

    class CustomDelegate:
        name = "custom"
        description = ""

        async def delegate(self, task, *, parent_ctx):  # type: ignore[no-untyped-def]
            return AgentResult[str](
                output=f"custom: {task}",
                trace_id="t",
                steps=[],
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                duration_ms=0.0,
                delegation_chain=[],
                cascade_history=[],
                guardrail_decisions=[],
                finish_reason="completed",
            )

    c = make_container(stub_llm())
    custom = CustomDelegate()
    lead = Agent("lead", instructions="", delegates=[custom], container=c)
    assert lead.delegates[0] is custom
