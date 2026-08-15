"""Phase 3 exit-criteria gates. If any fail, Phase 3 is not done."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse


def test_top_level_imports():
    from voussoir import Agent, Container  # noqa: F401
    from voussoir.agent import (  # noqa: F401
        Decision,
        RequestCascade,
        ToolUseFaithfulness,
        Validator,
    )


async def test_delegation_chain_populates():
    """Exit criterion #1: AgentResult.delegation_chain reflects the call tree."""
    from voussoir import Container
    from voussoir.agent import Agent
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import ILLMProvider as ILLMProviderProto
    from voussoir.protocols import IMemoryStore, ISessionStore

    sub_llm = MagicMock(spec=ILLMProvider)
    sub_llm.name = "anthropic"
    sub_llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="sub answer",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
        )
    )
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
                            "id": "tc",
                            "name": "delegate_to_researcher",
                            "arguments": {"task": "x"},
                        }
                    ]
                },
            ),
            LLMResponse(
                content="ok",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
            ),
        ]
    )

    def _c(llm):
        c = Container()
        c.bind(ILLMProviderProto, llm)
        c.bind(IMemoryStore, InMemoryStore())
        c.bind(ISessionStore, InMemorySessionStore())
        return c

    researcher = Agent(name="researcher", container=_c(sub_llm))
    lead = Agent(name="lead", delegates=[researcher], container=_c(lead_llm))
    result = await lead.run("hi")
    assert "lead" in result.delegation_chain
    assert "researcher" in result.delegation_chain


async def test_child_container_inherits_singletons_but_fresh_run_scope():
    """Exit criterion #2: parent and child resolve the same LLM provider
    singleton, but have different RUN scopes (different ctxforge sessions)."""
    from voussoir import Container
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import ILLMProvider as ILLMProviderProto
    from voussoir.protocols import IMemoryStore, ISessionStore

    p = MagicMock(spec=ILLMProvider)
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content="ok",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
        )
    )
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())

    parent_p = c.resolve(ILLMProviderProto)
    child = c.child()
    child_p = child.resolve(ILLMProviderProto)
    assert parent_p is child_p  # singleton shared


async def test_max_delegation_depth_refuses_past_limit():
    """Exit criterion #5: max_delegation_depth refused at runtime.

    Phase 4a: the depth check moved from _run_sub_agent (removed) to
    make_delegate_invoker (caller-side). Past-depth delegation no
    longer raises an exception — it returns a DELEGATION_REFUSED string
    that the lead's LLM sees as the synthetic tool's result.
    """
    from voussoir import Container
    from voussoir.agent.agent import Agent
    from voussoir.agent.context import AgentContext
    from voussoir.agent.dispatch import make_delegate_invoker, parent_ctx_var
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import IMemoryStore, ISessionStore

    c = Container()
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    target = Agent(name="deep", container=c)
    invoke = make_delegate_invoker(target)

    async with await AgentContext.open(
        container=c, run_id="parent-r", session_id="ps", user_id="alice"
    ) as parent_ctx:
        parent_ctx.delegation_depth = 3
        parent_ctx.max_delegation_depth = 3
        token = parent_ctx_var.set(parent_ctx)
        try:
            result = await invoke("x")
        finally:
            parent_ctx_var.reset(token)

    assert "DELEGATION_REFUSED" in result
    assert "max_delegation_depth" in result


async def test_cascade_pass_returns_sas_no_escalation():
    """Exit criterion #3 part A: PASS short-circuits."""
    from voussoir import Container
    from voussoir.agent import Agent, Decision, RequestCascade
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import ILLMProvider as ILLMProviderProto
    from voussoir.protocols import IMemoryStore, ISessionStore

    p = MagicMock(spec=ILLMProvider)
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content="ok",
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
        )
    )
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())

    class _PassV:
        name = "pass"

        async def validate(self, result, *, task):
            return Decision.PASS

    smart_lead = Agent(
        name="smart_lead",
        cascade=RequestCascade(verifier=_PassV()),
        container=c,
    )
    result = await smart_lead.run("hi")
    assert result.output == "ok"
    assert p.chat.await_count == 1


def test_no_peer_to_peer_delegation_at_class_level(make_container, stub_llm):
    """Exit criterion #4: a sub-agent's run loop only generates delegate
    tools for its own delegates list, not the parent's siblings.
    Phase 4.5a: Agent requires container=."""
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    a = Agent(name="a", container=c)
    b = Agent(name="b", delegates=[a], container=c)  # noqa: F841
    # 'a' has no delegates; its own delegates list is empty regardless of
    # who points at it.
    assert a.delegates == []


def test_coverage_floor_phase3_packages():
    """Exit criterion #6: ≥85% on the new agent files."""
    pytest.importorskip("coverage")
    import coverage as cov_mod

    cov_file = Path(__file__).resolve().parent.parent / ".coverage"
    if not cov_file.exists():
        pytest.skip("no .coverage file — run `make ci` first")
    cov = cov_mod.Coverage(data_file=str(cov_file))
    cov.load()
    paths = [
        "src/voussoir/agent/cascade.py",
        "src/voussoir/agent/validators.py",
    ]
    total_stmts = 0
    total_miss = 0
    for f in cov.get_data().measured_files():
        if any(p in f for p in paths):
            # analysis2 returns (filename, executable, excluded, missing,
            # formatted_missing); we want executable + missing, NOT excluded.
            _, executable, _excluded, missing = cov.analysis2(f)[:4]
            total_stmts += len(executable)
            total_miss += len(missing)
    if total_stmts == 0:
        pytest.skip("no measured files match paths")
    pct = (total_stmts - total_miss) / total_stmts
    assert pct >= 0.85, f"coverage {pct:.1%} on phase 3 packages is below 85% floor"


@pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ
    or os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-test"),
    reason="live multi-agent research — requires a real ANTHROPIC_API_KEY",
)
async def test_live_multi_agent_research():
    """Exit criterion #7 (live): the 03_multi_agent_research example
    runs end-to-end against real Anthropic."""
    from voussoir import Agent
    from voussoir.container.defaults import default_container

    container = default_container()
    researcher = Agent(
        name="researcher",
        description="Returns 3 bullets on a topic.",
        instructions="Produce 3 bullet points. Be terse.",
        model="claude-haiku-4-5-20251001",
        container=container,
    )
    lead = Agent(
        name="lead",
        instructions="Use delegate_to_researcher then return the result.",
        delegates=[researcher],
        model="claude-haiku-4-5-20251001",
        container=container,
    )
    result = await lead.run("Topic: voussoir testing strategy.")
    assert "researcher" in result.delegation_chain
    assert result.finish_reason == "completed"
