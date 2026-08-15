"""Unit tests for apply_guardrail_verdict (T10: guardrail-verdict helper).

Verifies that the helper:
- records exactly one GuardrailDecision in ctx.guardrail_decisions
- returns (content, False) on PASS/ALLOW
- returns (rewritten_content, False) on REWRITE
- returns (blocked_content, True) on BLOCK
"""

from __future__ import annotations

import pytest

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import apply_guardrail_verdict
from voussoir.guardrails.protocol import GuardrailVerdict


@pytest.fixture()
def make_container():
    """Minimal container for AgentContext.open."""
    from voussoir.container import Container
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import IMemoryStore, ISessionStore

    c = Container()
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    return c


async def test_pass_verdict_returns_content_unchanged(make_container):
    """ALLOW verdict: content unchanged, is_blocked=False, one decision recorded."""
    async with await AgentContext.open(
        container=make_container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        verdict = GuardrailVerdict(verdict="ALLOW")
        new_content, is_blocked = apply_guardrail_verdict(
            verdict, "hello", stage="input", ctx=ctx, blocked_content="[blocked]"
        )
    assert new_content == "hello"
    assert is_blocked is False
    assert len(ctx.guardrail_decisions) == 1
    assert ctx.guardrail_decisions[0].decision == "ALLOW"
    assert ctx.guardrail_decisions[0].stage == "input"


async def test_rewrite_verdict_returns_rewritten_content(make_container):
    """REWRITE verdict: content replaced by verdict.rewrite, is_blocked=False."""
    async with await AgentContext.open(
        container=make_container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        verdict = GuardrailVerdict(verdict="REWRITE", rewrite="sanitized", reason="pii")
        new_content, is_blocked = apply_guardrail_verdict(
            verdict, "dirty input", stage="output", ctx=ctx, blocked_content="[blocked]"
        )
    assert new_content == "sanitized"
    assert is_blocked is False
    assert len(ctx.guardrail_decisions) == 1
    rec = ctx.guardrail_decisions[0]
    assert rec.decision == "REWRITE"
    assert rec.rewrite == "sanitized"
    assert rec.stage == "output"


async def test_rewrite_verdict_falls_back_to_content_when_rewrite_is_none(make_container):
    """REWRITE with rewrite=None: falls back to original content, is_blocked=False."""
    async with await AgentContext.open(
        container=make_container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        verdict = GuardrailVerdict(verdict="REWRITE", rewrite=None)
        new_content, is_blocked = apply_guardrail_verdict(
            verdict, "original", stage="input", ctx=ctx, blocked_content="[blocked]"
        )
    assert new_content == "original"
    assert is_blocked is False


async def test_block_verdict_returns_blocked_content(make_container):
    """BLOCK verdict: new_content is blocked_content, is_blocked=True."""
    async with await AgentContext.open(
        container=make_container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        verdict = GuardrailVerdict(verdict="BLOCK", reason="dangerous content")
        new_content, is_blocked = apply_guardrail_verdict(
            verdict,
            "bad message",
            stage="input",
            ctx=ctx,
            blocked_content="[blocked by input guardrail: dangerous content]",
        )
    assert new_content == "[blocked by input guardrail: dangerous content]"
    assert is_blocked is True
    assert len(ctx.guardrail_decisions) == 1
    rec = ctx.guardrail_decisions[0]
    assert rec.decision == "BLOCK"
    assert rec.reason == "dangerous content"
    assert rec.stage == "input"


async def test_multiple_calls_accumulate_decisions(make_container):
    """Each call appends exactly one record; multiple calls accumulate correctly."""
    async with await AgentContext.open(
        container=make_container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        apply_guardrail_verdict(
            GuardrailVerdict(verdict="ALLOW"),
            "hello",
            stage="input",
            ctx=ctx,
            blocked_content="[blocked]",
        )
        apply_guardrail_verdict(
            GuardrailVerdict(verdict="BLOCK", reason="dangerous"),
            "final",
            stage="output",
            ctx=ctx,
            blocked_content="[output blocked]",
        )
    assert len(ctx.guardrail_decisions) == 2
    assert ctx.guardrail_decisions[0].stage == "input"
    assert ctx.guardrail_decisions[0].decision == "ALLOW"
    assert ctx.guardrail_decisions[1].stage == "output"
    assert ctx.guardrail_decisions[1].decision == "BLOCK"
