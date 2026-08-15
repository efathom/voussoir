"""Locks DefaultGuardrailChain dispatch + bind_default_guardrails profile wiring (B2).

The chain runs guardrails per-stage in declared order; first non-ALLOW short-circuits.
Profile binding shape is locked here; profile *content* (the actual built-ins) lands
in Task B3 and the tests below tolerate that gap via importorskip.
"""

from __future__ import annotations

from typing import Literal

import pytest

from voussoir.container import Container
from voussoir.guardrails import (
    DefaultGuardrailChain,
    GuardrailPayload,
    GuardrailVerdict,
    IGuardrailChain,
    bind_default_guardrails,
)


class _AlwaysAllow:
    name = "always-allow"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "input"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del payload, ctx
        return GuardrailVerdict(verdict="ALLOW")


class _AlwaysBlock:
    name = "always-block"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "input"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del payload, ctx
        return GuardrailVerdict(verdict="BLOCK", reason="nope")


class _AlwaysRewrite:
    name = "always-rewrite"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_output"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del payload, ctx
        return GuardrailVerdict(verdict="REWRITE", rewrite="cleaned", reason="x")


# ---- chain dispatch ----


async def test_chain_returns_allow_when_no_guardrails():
    chain = DefaultGuardrailChain([])
    v = await chain.screen(GuardrailPayload(stage="input", content="hi"), ctx=None)
    assert v.verdict == "ALLOW"


async def test_chain_returns_allow_when_all_allow():
    chain = DefaultGuardrailChain([_AlwaysAllow()])
    v = await chain.screen(GuardrailPayload(stage="input", content="hi"), ctx=None)
    assert v.verdict == "ALLOW"


async def test_chain_short_circuits_on_first_block():
    chain = DefaultGuardrailChain([_AlwaysAllow(), _AlwaysBlock(), _AlwaysAllow()])
    v = await chain.screen(GuardrailPayload(stage="input", content="hi"), ctx=None)
    assert v.verdict == "BLOCK"
    assert v.reason == "nope"


async def test_chain_dispatches_per_stage():
    """A tool_output guardrail does not fire on input-stage payloads."""
    chain = DefaultGuardrailChain([_AlwaysRewrite()])  # stage = tool_output
    v = await chain.screen(GuardrailPayload(stage="input", content="hi"), ctx=None)
    assert v.verdict == "ALLOW"
    # Same chain fires on tool_output:
    v2 = await chain.screen(
        GuardrailPayload(stage="tool_output", content="x", tool_name="t"), ctx=None
    )
    assert v2.verdict == "REWRITE"


async def test_chain_groups_guardrails_by_stage():
    """Multiple guardrails on different stages don't cross-contaminate."""
    chain = DefaultGuardrailChain([_AlwaysAllow(), _AlwaysRewrite()])
    v_in = await chain.screen(GuardrailPayload(stage="input", content="hi"), ctx=None)
    assert v_in.verdict == "ALLOW"  # only _AlwaysAllow ran
    v_out = await chain.screen(
        GuardrailPayload(stage="tool_output", content="x", tool_name="t"), ctx=None
    )
    assert v_out.verdict == "REWRITE"  # only _AlwaysRewrite ran


# ---- bind_default_guardrails (binding shape only — profile content is B3) ----


def test_bind_default_guardrails_off_profile_binds_a_chain():
    """`profile='off'` binds a DefaultGuardrailChain singleton."""
    pytest.importorskip("voussoir.guardrails.builtin.schema")
    c = Container()
    bind_default_guardrails(c, profile="off")
    chain = c.resolve(IGuardrailChain)
    assert isinstance(chain, DefaultGuardrailChain)


def test_bind_default_guardrails_standard_profile_binds_a_chain():
    pytest.importorskip("voussoir.guardrails.builtin.schema")
    c = Container()
    bind_default_guardrails(c, profile="standard")
    chain = c.resolve(IGuardrailChain)
    assert isinstance(chain, DefaultGuardrailChain)


def test_bind_default_guardrails_strict_warns_on_empty_allowlist(caplog):
    """strict profile without url_allowlist logs a warning but still binds."""
    pytest.importorskip("voussoir.guardrails.builtin.urls")
    import structlog

    c = Container()
    with structlog.testing.capture_logs() as captured:
        bind_default_guardrails(c, profile="strict")
    chain = c.resolve(IGuardrailChain)
    assert isinstance(chain, DefaultGuardrailChain)
    assert any("url_allowlist" in str(r.get("event", "")).lower() for r in captured)


def test_bind_default_guardrails_strict_with_allowlist_no_warning(caplog):
    pytest.importorskip("voussoir.guardrails.builtin.urls")
    import structlog

    c = Container()
    with structlog.testing.capture_logs() as captured:
        bind_default_guardrails(c, profile="strict", url_allowlist=["docs.example.com"])
    chain = c.resolve(IGuardrailChain)
    assert isinstance(chain, DefaultGuardrailChain)
    assert not any("url_allowlist requires" in str(r.get("event", "")).lower() for r in captured)
