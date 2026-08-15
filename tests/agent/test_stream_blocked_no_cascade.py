"""v1.0.4 E4 — blocked output guardrail must not trigger the cascade gate.

Regression tests for the bug where stream()'s two output-guardrail sites
used `full, _ = apply_guardrail_verdict(...)`, discarding `is_blocked` so
`finish_reason` stayed `"completed"` and the cascade gate fired on the
blocked placeholder output.

Fix: capture `is_blocked` and set `finish_reason = "blocked"` in both:
  - simple-path  (no tools)
  - tool-path    (agent with tools, multi-turn)
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Agent
from voussoir.agent.cascade import Decision, RequestCascade
from voussoir.guardrails import (
    DefaultGuardrailChain,
    GuardrailPayload,
    GuardrailVerdict,
    IGuardrailChain,
)
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.tools import Capability, tool

# ---------------------------------------------------------------------------
# Shared guardrail + cascade helpers
# ---------------------------------------------------------------------------


class _BlockOutputGuardrail:
    """Output-stage guardrail that always returns BLOCK."""

    name = "block-output"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "output"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        return GuardrailVerdict(verdict="BLOCK", reason="test block")


class _NeverPassVerifier:
    """Cascade verifier that always returns FAIL — used to assert cascade did NOT fire."""

    name = "never-pass"

    async def validate(self, result: object, *, task: str) -> Decision:
        return Decision.FAIL


# ---------------------------------------------------------------------------
# Simple-path (no tools)
# ---------------------------------------------------------------------------


def _simple_streaming_llm(content: str = "would be blocked") -> MagicMock:
    """Mock ILLMProvider whose .stream yields a single chunk (simple-path)."""

    async def _gen(*args: object, **kwargs: object) -> object:
        yield content

    m = MagicMock(spec=ILLMProvider)
    m.name = "stub"
    m.stream = MagicMock(return_value=_gen())
    return m


async def test_simple_stream_blocked_output_no_cascade_event(make_container) -> None:
    """Simple-path: output BLOCK must set finish_reason='blocked', suppressing the cascade gate."""
    llm = _simple_streaming_llm("secret data")
    c = make_container(llm)
    c.bind(IGuardrailChain, DefaultGuardrailChain([_BlockOutputGuardrail()]))  # type: ignore[type-abstract]

    cascade = RequestCascade(verifier=_NeverPassVerifier(), max_cascade_depth=1)
    a = Agent(name="x", container=c, cascade=cascade)

    events = []
    async for ev in a.stream("input"):
        events.append(ev)

    kinds = [e.kind for e in events]
    # The cascade gate must NOT fire for a blocked run.
    cascade_evs = [e for e in events if e.kind.startswith("cascade")]
    assert len(cascade_evs) == 0, (
        f"cascade gate fired on blocked simple-path output: {cascade_evs}\n"
        f"all event kinds: {kinds}"
    )


async def test_simple_stream_blocked_output_done_event_emitted(make_container) -> None:
    """Simple-path: a done event IS still emitted (carrying the placeholder), just no cascade."""
    llm = _simple_streaming_llm("blocked content")
    c = make_container(llm)
    c.bind(IGuardrailChain, DefaultGuardrailChain([_BlockOutputGuardrail()]))  # type: ignore[type-abstract]

    a = Agent(name="x", container=c)

    events = []
    async for ev in a.stream("input"):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert "done" in kinds, f"expected 'done' event even when output is blocked; got: {kinds}"

    # done event payload should carry the blocked placeholder, not the original content
    done_ev = next(e for e in events if e.kind == "done")
    assert (
        "blocked" in str(done_ev.payload).lower()
    ), f"done event payload should contain blocked marker; got: {done_ev.payload}"


async def test_simple_stream_unblocked_output_cascade_still_fires(make_container) -> None:
    """Sanity: when no guardrail blocks, the cascade gate still fires normally."""
    llm = _simple_streaming_llm("normal output")
    c = make_container(llm)
    # No guardrail chain bound — no blocking

    cascade = RequestCascade(verifier=_NeverPassVerifier(), max_cascade_depth=1)
    a = Agent(name="x", container=c, cascade=cascade)

    events = []
    async for ev in a.stream("input"):
        events.append(ev)

    kinds = [e.kind for e in events]
    # cascade_failed should fire (because _NeverPassVerifier always fails)
    assert (
        "cascade_failed" in kinds
    ), f"expected 'cascade_failed' for unblocked output with fail-verifier; got: {kinds}"


# ---------------------------------------------------------------------------
# Tool-path (agent has tools, multi-turn chat)
# ---------------------------------------------------------------------------


@tool(capability=Capability.READ_PUBLIC, name="e4_echo")
async def _echo_tool(text: str) -> str:
    """Trivial echo tool for tool-path stream tests."""
    return f"echo:{text}"


def _tool_path_llm() -> MagicMock:
    """LLM that emits one tool_use turn then end_turn with final content."""
    m = MagicMock(spec=ILLMProvider)
    m.name = "anthropic"
    m.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="tool_use",
                raw_response={
                    "tool_calls": [{"id": "tc1", "name": "e4_echo", "arguments": {"text": "hello"}}]
                },
            ),
            LLMResponse(
                content="final answer",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
        ]
    )
    return m


async def test_tool_stream_blocked_output_no_cascade_event(make_container) -> None:
    """Tool-path: output BLOCK must set finish_reason='blocked', suppressing the cascade gate."""
    llm = _tool_path_llm()
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, DefaultGuardrailChain([_BlockOutputGuardrail()]))  # type: ignore[type-abstract]

    cascade = RequestCascade(verifier=_NeverPassVerifier(), max_cascade_depth=1)
    a = Agent(name="x", container=c, tools=[_echo_tool], cascade=cascade)

    events = []
    async for ev in a.stream("input"):
        events.append(ev)

    kinds = [e.kind for e in events]
    cascade_evs = [e for e in events if e.kind.startswith("cascade")]
    assert len(cascade_evs) == 0, (
        f"cascade gate fired on blocked tool-path output: {cascade_evs}\n"
        f"all event kinds: {kinds}"
    )


async def test_tool_stream_blocked_output_done_event_emitted(make_container) -> None:
    """Tool-path: done event is still emitted (with blocked placeholder) even when output blocked."""
    llm = _tool_path_llm()
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, DefaultGuardrailChain([_BlockOutputGuardrail()]))  # type: ignore[type-abstract]

    a = Agent(name="x", container=c, tools=[_echo_tool])

    events = []
    async for ev in a.stream("input"):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert "done" in kinds, f"expected 'done' event on tool-path blocked output; got: {kinds}"

    done_ev = next(e for e in events if e.kind == "done")
    assert (
        "blocked" in str(done_ev.payload).lower()
    ), f"tool-path done payload should contain blocked marker; got: {done_ev.payload}"


async def test_tool_stream_unblocked_output_cascade_still_fires(make_container) -> None:
    """Sanity: tool-path without blocking still fires the cascade gate."""
    llm = _tool_path_llm()
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    # No guardrail chain

    cascade = RequestCascade(verifier=_NeverPassVerifier(), max_cascade_depth=1)
    a = Agent(name="x", container=c, tools=[_echo_tool], cascade=cascade)

    events = []
    async for ev in a.stream("input"):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert (
        "cascade_failed" in kinds
    ), f"expected 'cascade_failed' for unblocked tool-path output with fail-verifier; got: {kinds}"
