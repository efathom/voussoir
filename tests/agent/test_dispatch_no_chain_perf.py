"""v1.0.4 E5 — _dispatch_one skips Pydantic audit when guardrail chain is empty.

Round-3 perf finding I5: previously _dispatch_one always constructed
GuardrailVerdict + GuardrailDecision per tool call, even with no chain.
With the fix, both the tool_call and tool_output screen+audit blocks are
skipped entirely when chain.count() == 0.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import _dispatch_one
from voussoir.executors.standard import StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain, GuardrailPayload, GuardrailVerdict
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability
from voussoir.tools.registry import ToolRegistry


@tool(capability=Capability.READ_PUBLIC, name="ping_e5", description="ping for E5 perf test")
async def _ping() -> str:
    return "pong"


@pytest.mark.asyncio
async def test_dispatch_no_chain_skips_audit_records(make_container) -> None:
    """When ctx.guardrail_chain is empty, _dispatch_one creates ZERO GuardrailDecision records."""
    container = make_container()
    # Don't bind a chain — ctx.guardrail_chain defaults to DefaultGuardrailChain([]).
    registry = ToolRegistry()
    registry.register(_ping)

    async with await AgentContext.open(
        container=container,
        run_id="r",
        session_id="s",
        user_id="u",
        allowed_capabilities=Capability.READ_PUBLIC,
    ) as ctx:
        assert ctx.guardrail_chain.count() == 0  # sanity: empty chain

        outcome = await _dispatch_one(
            tc={"name": "ping_e5", "arguments": {}, "id": "tc1"},
            registry=registry,
            executor=StandardExecutor(),
            ctx=ctx,
        )

        assert outcome.output_str == "pong"
        # NO audit records were created since no chain was screening
        assert ctx.guardrail_decisions == []


@pytest.mark.asyncio
async def test_dispatch_with_chain_still_writes_audit(make_container) -> None:
    """Sanity: when a chain IS bound and screens ALLOW, audit records still land."""

    class _AllowToolCall:
        name = "allow_tc"
        stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_call"

        async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict:
            return GuardrailVerdict(verdict="ALLOW")

    class _AllowToolOutput:
        name = "allow_to"
        stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_output"

        async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict:
            return GuardrailVerdict(verdict="ALLOW")

    chain = DefaultGuardrailChain([_AllowToolCall(), _AllowToolOutput()])  # type: ignore[list-item]
    container = make_container()

    registry = ToolRegistry()
    registry.register(_ping)

    async with await AgentContext.open(
        container=container,
        run_id="r",
        session_id="s",
        user_id="u",
        allowed_capabilities=Capability.READ_PUBLIC,
        guardrail_chain=chain,
    ) as ctx:
        assert ctx.guardrail_chain.count() == 2

        outcome = await _dispatch_one(
            tc={"name": "ping_e5", "arguments": {}, "id": "tc1"},
            registry=registry,
            executor=StandardExecutor(),
            ctx=ctx,
        )

        assert outcome.output_str == "pong"
        # Both tool_call and tool_output audit records were created
        stages = {d.stage for d in ctx.guardrail_decisions}
        assert "tool_call" in stages
        assert "tool_output" in stages
