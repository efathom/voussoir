"""v1.0.2 D10 — branch coverage for tool_output REWRITE → re-screen-still-fails.

Targets the branch in ``src/voussoir/agent/dispatch.py``::

    elif out_verdict.verdict == "REWRITE":
        output_str = out_verdict.rewrite or output_str
        rescreen_payload = GuardrailPayload(...)
        rescreen_verdict = await chain.screen(rescreen_payload, ctx)
        if rescreen_verdict.verdict != "ALLOW":
            output_str = f"[blocked: tool_output rewrite still flagged ({rescreen_verdict.verdict})]"
            out_record.decision = "BLOCK"

The branch fires when the first tool_output screen returns REWRITE *and*
the re-screen of the rewritten content returns non-ALLOW (BLOCK or another
REWRITE). It is the chain's safety hatch against guardrails that keep
proposing rewrites that themselves get flagged.

These tests exercise the branch via ``_dispatch_one`` directly so we can
assert on both invariants — the audit-record mutation to BLOCK *and* the
output marker string — without depending on agent.run + LLM scaffolding.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import _dispatch_one
from voussoir.executors.standard import StandardExecutor
from voussoir.guardrails import (
    DefaultGuardrailChain,
    GuardrailPayload,
    GuardrailVerdict,
)
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability
from voussoir.tools.registry import ToolRegistry


@tool(capability=Capability.READ_PUBLIC, name="echo_branch", description="echo for branch test")
async def _echo_branch(text: str) -> str:
    return f"echo:{text}"


class _RewriteThenBlock:
    """Guardrail that REWRITEs on first screen, then BLOCKs on re-screen."""

    name = "rewrite_then_block"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_output"

    def __init__(self) -> None:
        self.call_count = 0

    async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict:
        del ctx
        self.call_count += 1
        if self.call_count == 1:
            return GuardrailVerdict(
                verdict="REWRITE", rewrite="rewritten_output", reason="first pass"
            )
        return GuardrailVerdict(verdict="BLOCK", reason="still bad on re-screen")


class _RewriteThenRewrite:
    """Guardrail that REWRITEs on every call — re-screen returns REWRITE (non-ALLOW)."""

    name = "rewrite_then_rewrite"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_output"

    def __init__(self) -> None:
        self.call_count = 0

    async def screen(self, payload: GuardrailPayload, ctx: Any) -> GuardrailVerdict:
        del ctx
        self.call_count += 1
        return GuardrailVerdict(verdict="REWRITE", rewrite="rewritten_again", reason="always")


@pytest.mark.asyncio
async def test_tool_output_rewrite_then_rescreen_block_replaces_output(
    make_container, stub_llm
) -> None:
    """REWRITE → re-screen BLOCK: output is replaced with the [blocked:...] marker
    AND the audit record's decision is mutated to BLOCK."""
    guardrail = _RewriteThenBlock()
    chain = DefaultGuardrailChain([guardrail])  # type: ignore[list-item]
    registry = ToolRegistry()
    registry.register_many([_echo_branch])
    c = make_container(stub_llm())

    async with await AgentContext.open(
        container=c,
        run_id="r",
        session_id="s",
        user_id="u",
        allowed_capabilities=Capability.READ_PUBLIC,
        guardrail_chain=chain,
    ) as ctx:
        outcome = await _dispatch_one(
            {"id": "tc1", "name": "echo_branch", "arguments": {"text": "hi"}},
            registry=registry,
            executor=StandardExecutor(),
            ctx=ctx,
        )

        # Chain was called twice: once for the REWRITE, once for the re-screen.
        assert guardrail.call_count == 2

        # Invariant 1: output_str is the [blocked: ...] marker carrying the
        # re-screen verdict (BLOCK in this case).
        assert outcome.output_str == "[blocked: tool_output rewrite still flagged (BLOCK)]"

        # Invariant 2: the tool_output audit record has been mutated to BLOCK
        # (the REWRITE decision is "downgraded" because the re-screen failed).
        out_decisions = [d for d in ctx.guardrail_decisions if d.stage == "tool_output"]
        assert len(out_decisions) == 1
        assert out_decisions[0].decision == "BLOCK"
        assert out_decisions[0].name == "chain.tool_output.echo_branch"

        # No execution error — the tool itself succeeded; the chain rewrote
        # then blocked.
        assert outcome.error is None


@pytest.mark.asyncio
async def test_tool_output_rewrite_then_rescreen_rewrite_also_blocks(
    make_container, stub_llm
) -> None:
    """REWRITE → re-screen REWRITE (also non-ALLOW): same blocked-marker
    treatment, marker carries 'REWRITE' as the verdict. Confirms the
    branch fires on any non-ALLOW re-screen verdict, not just BLOCK."""
    guardrail = _RewriteThenRewrite()
    chain = DefaultGuardrailChain([guardrail])  # type: ignore[list-item]
    registry = ToolRegistry()
    registry.register_many([_echo_branch])
    c = make_container(stub_llm())

    async with await AgentContext.open(
        container=c,
        run_id="r",
        session_id="s",
        user_id="u",
        allowed_capabilities=Capability.READ_PUBLIC,
        guardrail_chain=chain,
    ) as ctx:
        outcome = await _dispatch_one(
            {"id": "tc2", "name": "echo_branch", "arguments": {"text": "hi"}},
            registry=registry,
            executor=StandardExecutor(),
            ctx=ctx,
        )

        # Chain called twice — first REWRITE, then re-screen (also REWRITE).
        assert guardrail.call_count == 2

        # Marker carries REWRITE as the re-screen verdict.
        assert outcome.output_str == "[blocked: tool_output rewrite still flagged (REWRITE)]"

        out_decisions = [d for d in ctx.guardrail_decisions if d.stage == "tool_output"]
        assert len(out_decisions) == 1
        assert out_decisions[0].decision == "BLOCK"
