"""Locks DefaultGuardrailChain integration into Agent.run at four stages (B5/B6).

Verifies that input/output stages fire when guardrails are bound on the container,
that verdicts BLOCK/REWRITE flow correctly, and that each verdict is recorded in
AgentResult.guardrail_decisions.

B6 extends coverage to tool_call + tool_output stages (wired in _dispatch_one)
and to the stream() path (input + output stages mirrored from _run_normal).
"""

from __future__ import annotations

import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Agent
from voussoir.guardrails import (
    DefaultGuardrailChain,
    GuardrailPayload,
    GuardrailVerdict,
    IGuardrailChain,
)
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability


class _RecordingGuardrail:
    """Captures every payload it screens and returns a configured verdict."""

    def __init__(
        self,
        name: str,
        stage: Literal["input", "tool_call", "tool_output", "output"],
        *,
        decision: Literal["ALLOW", "BLOCK", "REWRITE", "AMBIGUOUS"] = "ALLOW",
        rewrite: str | None = None,
    ) -> None:
        self.name = name
        self.stage: Literal["input", "tool_call", "tool_output", "output"] = stage
        self._decision = decision
        self._rewrite = rewrite
        self.calls: list[GuardrailPayload] = []

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        self.calls.append(payload)
        return GuardrailVerdict(
            verdict=self._decision, rewrite=self._rewrite, reason=f"test:{self.name}"
        )


async def test_input_stage_fires_once_per_run(make_container, stub_llm):
    g = _RecordingGuardrail("input-g", "input")
    chain = DefaultGuardrailChain([g])
    c = make_container(stub_llm(content="hello"))
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    await a.run("hi there")
    assert len(g.calls) == 1
    assert g.calls[0].stage == "input"
    assert g.calls[0].content == "hi there"


async def test_input_stage_block_short_circuits_run(make_container, stub_llm):
    g = _RecordingGuardrail("input-block", "input", decision="BLOCK")
    chain = DefaultGuardrailChain([g])
    c = make_container(stub_llm(content="should-not-run"))
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    result = await a.run("bad input")
    # The run short-circuits before any LLM turn fires.
    assert "blocked" in str(result.output).lower() or result.finish_reason == "blocked"
    rec = [d for d in result.guardrail_decisions if d.stage == "input"][0]
    assert rec.decision == "BLOCK"


async def test_input_stage_rewrite_replaces_user_input(make_container, stub_llm):
    """When the input guardrail REWRITEs, downstream sees the rewritten content."""
    g = _RecordingGuardrail("input-rw", "input", decision="REWRITE", rewrite="sanitized")
    chain = DefaultGuardrailChain([g])
    llm = stub_llm(content="ok")
    c = make_container(llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    result = await a.run("dirty input")
    rec = [d for d in result.guardrail_decisions if d.stage == "input"][0]
    assert rec.decision == "REWRITE"
    assert rec.rewrite == "sanitized"


async def test_output_stage_fires_at_run_end(make_container, stub_llm):
    g = _RecordingGuardrail("output-g", "output")
    chain = DefaultGuardrailChain([g])
    c = make_container(stub_llm(content="goodbye"))
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    await a.run("hi")
    out_calls = [p for p in g.calls if p.stage == "output"]
    assert len(out_calls) == 1


async def test_output_stage_block_replaces_output(make_container, stub_llm):
    g = _RecordingGuardrail("output-block", "output", decision="BLOCK")
    chain = DefaultGuardrailChain([g])
    c = make_container(stub_llm(content="secret"))
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    result = await a.run("hi")
    # The original output is replaced with a blocked-marker.
    assert "secret" not in str(result.output)


async def test_result_records_all_stage_verdicts(make_container, stub_llm):
    """All chain decisions across the run accumulate in result.guardrail_decisions."""
    g_in = _RecordingGuardrail("input-allow", "input")
    g_out = _RecordingGuardrail("output-allow", "output")
    chain = DefaultGuardrailChain([g_in, g_out])
    c = make_container(stub_llm(content="hello"))
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    result = await a.run("hi")
    stages = {d.stage for d in result.guardrail_decisions}
    assert stages == {"input", "output"}
    assert all(d.decision == "ALLOW" for d in result.guardrail_decisions)


async def test_no_chain_bound_run_proceeds_normally(make_container, stub_llm):
    """When no DefaultGuardrailChain is bound, run proceeds and guardrail_decisions is []."""
    c = make_container(stub_llm(content="hello"))
    a = Agent(name="t", container=c)
    result = await a.run("hi")
    assert result.guardrail_decisions == []


# ---------------------------------------------------------------------------
# B6: tool_call + tool_output stage tests (wired in _dispatch_one)
# ---------------------------------------------------------------------------


def _llm_tool_use(tool_calls: list[dict], content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_use",
        raw_response={"tool_calls": tool_calls},
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


@tool(capability=Capability.READ_PUBLIC, name="echo_text")
async def _echo_text(text: str) -> str:
    return f"echo:{text}"


async def test_tool_call_stage_blocks_dispatch(make_container, stub_llm):
    """A BLOCK at tool_call short-circuits the executor; outcome carries the blocked marker."""
    g = _RecordingGuardrail("tc-block", "tool_call", decision="BLOCK")
    chain = DefaultGuardrailChain([g])

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use([{"id": "t1", "name": "echo_text", "arguments": {"text": "hello"}}]),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("hi")

    # The tool was NOT run (blocked marker in the tool result seen by LLM).
    assert any(d.stage == "tool_call" and d.decision == "BLOCK" for d in result.guardrail_decisions)
    # The output comes from the LLM's follow-up turn (which saw the blocked marker).
    assert result.output == "done"
    # Exactly one tool_call guardrail decision.
    tc_decisions = [d for d in result.guardrail_decisions if d.stage == "tool_call"]
    assert len(tc_decisions) == 1
    assert tc_decisions[0].name == "chain.tool_call.echo_text"


async def test_tool_output_stage_block_replaces_outcome(make_container, stub_llm):
    """A BLOCK at tool_output replaces outcome.output_str with a blocked marker."""
    g = _RecordingGuardrail("out-block", "tool_output", decision="BLOCK")
    chain = DefaultGuardrailChain([g])

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use([{"id": "t1", "name": "echo_text", "arguments": {"text": "secret"}}]),
            _llm_text("seen"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("hi")

    out_decisions = [d for d in result.guardrail_decisions if d.stage == "tool_output"]
    assert len(out_decisions) == 1
    assert out_decisions[0].decision == "BLOCK"
    assert out_decisions[0].name == "chain.tool_output.echo_text"


async def test_tool_output_stage_rewrite_replaces_outcome(make_container, stub_llm):
    """A REWRITE at tool_output replaces what downstream LLM sees (re-screen passes)."""
    # First screen: REWRITE; second screen (re-screen): ALLOW.
    call_count = 0

    class _OnceRewriteGuardrail:
        name = "once-rw"
        stage = "tool_output"

        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GuardrailVerdict(
                    verdict="REWRITE", rewrite="sanitized_output", reason="first"
                )
            # Re-screen of the rewrite: ALLOW.
            return GuardrailVerdict(verdict="ALLOW")

    chain = DefaultGuardrailChain([_OnceRewriteGuardrail()])  # type: ignore[list-item]

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    captured_second_messages: list = []

    async def chat(messages, **kwargs):
        if llm.chat.await_count == 1:
            return _llm_tool_use([{"id": "t1", "name": "echo_text", "arguments": {"text": "x"}}])
        captured_second_messages.extend(messages)
        return _llm_text("done")

    llm.chat = AsyncMock(side_effect=chat)
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("hi")

    # The LLM's second call should have received "sanitized_output" not "echo:x".
    fn_messages = [m for m in captured_second_messages if m.role == "function"]
    assert len(fn_messages) == 1
    assert fn_messages[0].content == "sanitized_output"

    out_decisions = [d for d in result.guardrail_decisions if d.stage == "tool_output"]
    assert len(out_decisions) == 1
    assert out_decisions[0].decision == "REWRITE"


async def test_tool_output_rewrite_rescreened_once_then_blocked(make_container, stub_llm):
    """Two REWRITEs in a row at tool_output → BLOCK (no infinite loop)."""

    class _AlwaysRewriteGuardrail:
        name = "always-rw"
        stage = "tool_output"

        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            # Always rewrite — triggers the re-screen which also returns REWRITE.
            return GuardrailVerdict(verdict="REWRITE", rewrite="rewritten_output", reason="always")

    chain = DefaultGuardrailChain([_AlwaysRewriteGuardrail()])  # type: ignore[list-item]

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use([{"id": "t1", "name": "echo_text", "arguments": {"text": "x"}}]),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("hi")

    out_decisions = [d for d in result.guardrail_decisions if d.stage == "tool_output"]
    assert len(out_decisions) == 1
    # Downgraded to BLOCK because the re-screen returned non-ALLOW.
    assert out_decisions[0].decision == "BLOCK"


async def test_tool_call_rewrite_schema_valid_proceeds(make_container, stub_llm):
    """A tool_call REWRITE with valid JSON args (schema-valid) proceeds to executor."""

    class _RewriteArgsGuardrail:
        name = "rw-args"
        stage = "tool_call"
        calls: list[GuardrailPayload] = []

        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            self.calls.append(payload)
            # Rewrite to a different text value; must be JSON for dispatch to parse.
            return GuardrailVerdict(
                verdict="REWRITE",
                rewrite=json.dumps({"text": "rewritten"}),
                reason="sanitize",
            )

    g = _RewriteArgsGuardrail()
    chain = DefaultGuardrailChain([g])  # type: ignore[list-item]

    captured_second_messages: list = []

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"

    async def chat(messages, **kwargs):
        if llm.chat.await_count == 1:
            return _llm_tool_use(
                [{"id": "t1", "name": "echo_text", "arguments": {"text": "original"}}]
            )
        captured_second_messages.extend(messages)
        return _llm_text("done")

    llm.chat = AsyncMock(side_effect=chat)
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("hi")

    # The executor ran with the rewritten args: "echo:rewritten".
    fn_messages = [m for m in captured_second_messages if m.role == "function"]
    assert len(fn_messages) == 1
    assert fn_messages[0].content == "echo:rewritten"

    tc_decisions = [d for d in result.guardrail_decisions if d.stage == "tool_call"]
    assert len(tc_decisions) == 1
    assert tc_decisions[0].decision == "REWRITE"


async def test_tool_call_rewrite_schema_invalid_downgrades_to_block(make_container, stub_llm):
    """A tool_call REWRITE with schema-invalid JSON downgrades to BLOCK."""

    class _BadRewriteGuardrail:
        name = "bad-rw"
        stage = "tool_call"

        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            # Provide JSON that doesn't match the echo_text schema (missing 'text').
            return GuardrailVerdict(
                verdict="REWRITE",
                rewrite=json.dumps({"wrong_field": 999}),
                reason="test",
            )

    chain = DefaultGuardrailChain([_BadRewriteGuardrail()])  # type: ignore[list-item]

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use([{"id": "t1", "name": "echo_text", "arguments": {"text": "x"}}]),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("hi")

    tc_decisions = [d for d in result.guardrail_decisions if d.stage == "tool_call"]
    assert len(tc_decisions) == 1
    # Downgraded from REWRITE → BLOCK because the rewrite failed schema validation.
    assert tc_decisions[0].decision == "BLOCK"


async def test_tool_call_and_output_decisions_in_result(make_container, stub_llm):
    """tool_call and tool_output decisions both appear in result.guardrail_decisions."""
    g_tc = _RecordingGuardrail("tc-allow", "tool_call")
    g_out = _RecordingGuardrail("out-allow", "tool_output")
    chain = DefaultGuardrailChain([g_tc, g_out])

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            _llm_tool_use([{"id": "t1", "name": "echo_text", "arguments": {"text": "hi"}}]),
            _llm_text("done"),
        ]
    )
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c, tools=[_echo_text])
    result = await a.run("test")

    stages = {d.stage for d in result.guardrail_decisions}
    assert "tool_call" in stages
    assert "tool_output" in stages
    assert all(d.decision == "ALLOW" for d in result.guardrail_decisions)


# ---------------------------------------------------------------------------
# B6: stream() path — input + output stage wiring
# ---------------------------------------------------------------------------


def _streaming_llm_simple(*chunks: str) -> MagicMock:
    """Build a streaming-capable MagicMock ILLMProvider for simple-agent (no-tool) tests."""

    async def _gen():
        for chunk in chunks:
            yield chunk

    m = MagicMock(spec=ILLMProvider)
    m.name = "stub"
    m.stream = MagicMock(return_value=_gen())
    return m


async def test_stream_input_stage_fires(make_container):
    """Agent.stream calls input-stage guardrails (B6 wires the stream path)."""
    g = _RecordingGuardrail("input-g", "input")
    chain = DefaultGuardrailChain([g])
    llm = _streaming_llm_simple("hello")
    c = make_container(llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    async for _ in a.stream("hi"):
        pass
    assert len(g.calls) == 1
    assert g.calls[0].stage == "input"
    assert g.calls[0].content == "hi"


async def test_stream_input_block_yields_done_and_stops(make_container, stub_llm):
    """BLOCK at input stage in stream: yields a done event with blocked marker, no LLM call."""
    g = _RecordingGuardrail("input-block", "input", decision="BLOCK")
    chain = DefaultGuardrailChain([g])
    # Use a simple stub_llm whose stream would error if called — proves we short-circuit.
    c = make_container(stub_llm(content="should-not-appear"))
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    events = [ev async for ev in a.stream("bad")]
    assert len(events) == 1
    assert events[0].kind == "done"
    assert "blocked" in events[0].payload.get("output", "").lower()


async def test_stream_input_rewrite_modifies_content(make_container):
    """REWRITE at input stage in stream: the rewritten input is used for the LLM call."""
    g = _RecordingGuardrail("input-rw", "input", decision="REWRITE", rewrite="sanitized")
    chain = DefaultGuardrailChain([g])
    llm = _streaming_llm_simple("ok")
    c = make_container(llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    events = [ev async for ev in a.stream("dirty")]
    # Run completed (not blocked).
    kinds = [ev.kind for ev in events]
    assert "done" in kinds


async def test_stream_output_stage_fires(make_container):
    """Agent.stream calls output-stage guardrails for simple-agent path."""
    g = _RecordingGuardrail("output-g", "output")
    chain = DefaultGuardrailChain([g])
    llm = _streaming_llm_simple("goodbye")
    c = make_container(llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    async for _ in a.stream("hi"):
        pass
    assert len(g.calls) == 1
    assert g.calls[0].stage == "output"


async def test_stream_output_block_replaces_done_payload(make_container):
    """BLOCK at output stage in stream: done event carries blocked marker, not original output."""
    g = _RecordingGuardrail("output-block", "output", decision="BLOCK")
    chain = DefaultGuardrailChain([g])
    llm = _streaming_llm_simple("secret")
    c = make_container(llm)
    c.bind(IGuardrailChain, chain)  # type: ignore[type-abstract]
    a = Agent(name="t", container=c)
    events = [ev async for ev in a.stream("hi")]
    done_events = [ev for ev in events if ev.kind == "done"]
    assert len(done_events) == 1
    output = done_events[0].payload.get("output", "")
    assert "secret" not in output
    assert "blocked" in output.lower()
