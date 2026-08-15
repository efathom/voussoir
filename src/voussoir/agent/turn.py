"""Two-phase single-turn helper for Agent's tool loop.

One assistant turn splits into two phases:

  1. ``tool_turn_prepare`` — call ``llm.chat`` and parse the response.
     Returns a ``PreparedTurn`` carrying the finish_reason, assistant
     text/content, the tool-call declarations (which tools the model
     wants, with arguments), token deltas from the chat itself, and the
     assistant ``ChatMessage`` to append before dispatch.

  2. ``tool_turn_dispatch`` — execute the declared tool calls
     concurrently and accumulate their outcomes. Returns a
     ``TurnResult`` with outcomes, the cost delta from any sub-agent
     work, and the function-result ``ChatMessage``s to append after
     dispatch.

The two-phase split exists so ``Agent.stream`` can emit
``tool_started`` / ``delegation_started`` events BETWEEN phases — the
events fire BEFORE the tool body executes, restoring real-time
semantics. ``Agent._run_normal`` emits no events but uses the same
two-phase shape so both call paths share one body.

Both helpers take only the pieces they need (`llm`, `messages`,
`registry`, `executor`, `ctx`, plus the LLM kwargs) — never an `Agent`
instance — so this module has zero dependency on
``voussoir.agent.agent``. The arrow is one-way: ``agent.py → turn.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ctxforge.protocols.llm import ChatMessage, ILLMProvider, LLMResponse

from voussoir.agent.context import AgentContext
from voussoir.agent.dispatch import (
    ToolCallOutcome,
    accumulate_outcomes,
    dispatch_tool_calls,
)
from voussoir.agent.result import Step
from voussoir.agent.turn_adapter import ToolCallAdapter, adapter_for
from voussoir.executors import IToolExecutor
from voussoir.observability import span as otel_span
from voussoir.tools.registry import ToolRegistry


@dataclass(frozen=True)
class PreparedTurn:
    """Output of phase 1 (``tool_turn_prepare``).

    The assistant has just responded — we know the finish_reason, the
    assistant text/content, and the tool-call declarations (which
    tools the model wants to invoke + their arguments) — but we have
    NOT yet executed any tool. Callers may inspect
    ``tool_call_declarations`` to emit ``tool_started`` /
    ``delegation_started`` events before calling
    ``tool_turn_dispatch``.

    ``assistant_message`` is the ``ChatMessage`` the caller must append
    to ``messages`` BEFORE dispatch fires (so tool-result messages line
    up with their owning assistant turn). On the ``stop`` path it is
    still produced for callers that want to record the final assistant
    text into the message history.

    ``tokens_in_delta`` / ``tokens_out_delta`` carry ONLY the
    chat-itself token counts. Sub-agent token accumulation happens in
    phase 2 via ``TurnResult.cost_delta`` and the steps that
    ``tool_turn_dispatch`` appends.

    ``adapter`` is the per-provider tool-calling adapter selected by
    ``adapter_for(llm)`` at the start of the turn. Set to ``None`` on the
    ``stop`` path (no tools were used this turn), so ``adapter_for`` is not
    called for tool-less agents running against non-Anthropic providers.
    Carried on ``PreparedTurn`` so ``tool_turn_dispatch`` can pass it through
    to ``accumulate_outcomes`` without needing a separate ``llm`` reference.
    """

    finish_reason: Literal["tool_use", "stop"]
    response: LLMResponse
    tool_call_declarations: list[dict[str, Any]]
    tokens_in_delta: int
    tokens_out_delta: int
    assistant_message: ChatMessage
    adapter: ToolCallAdapter | None


@dataclass(frozen=True)
class TurnResult:
    """Output of phase 2 (``tool_turn_dispatch``).

    Carries the outcomes of executing the tools that ``PreparedTurn``
    declared, plus the function-result ``ChatMessage``s the caller
    must append to ``messages`` AFTER dispatch. Token deltas from
    sub-agent runs are folded into ``tokens_in_delta`` /
    ``tokens_out_delta``; ``cost_delta`` reports the aggregated cost
    of those sub-agent runs.

    On the ``stop`` path (no tool calls declared)
    ``tool_turn_dispatch`` returns a zero-state result immediately —
    empty outcomes, zero deltas, empty ``tool_result_messages``.
    """

    tool_call_outcomes: list[ToolCallOutcome]
    tokens_in_delta: int
    tokens_out_delta: int
    cost_delta: float
    tool_result_messages: list[ChatMessage]


async def tool_turn_prepare(
    *,
    llm: ILLMProvider,
    messages: list[ChatMessage],
    registry: ToolRegistry,
    ctx: AgentContext,
    model: str | None,
    temperature: float | None,
    has_tools: bool,
    steps: list[Step],
) -> PreparedTurn:
    """Phase 1: call ``llm.chat`` and parse the response.

    Appends an ``llm_call`` Step to ``steps``. Does NOT execute any
    tool calls. Does NOT mutate ``messages`` — the caller appends
    ``PreparedTurn.assistant_message`` itself, then (in the streaming
    path) emits pre-dispatch events, then calls
    ``tool_turn_dispatch``.

    Returns a ``PreparedTurn`` with ``finish_reason="stop"`` when the
    response has no ``tool_calls`` (final answer in
    ``response.content``); otherwise ``finish_reason="tool_use"`` and
    ``tool_call_declarations`` is non-empty.
    """
    del ctx  # signature stable for future per-step ctx access
    # adapter_for is only called when tools are present — toolless agents must
    # be able to run against any ILLMProvider (e.g. OpenAI) without hitting the
    # NotImplementedError that guards unimplemented adapters. When has_tools is
    # False, functions is None and adapter stays None (used only on the stop
    # branch where tool dispatch never fires).
    adapter: ToolCallAdapter | None = None
    functions: list[dict[str, Any]] | None = None
    if has_tools:
        adapter = adapter_for(llm)
        functions = [adapter.serialize_tool(registry.resolve(n)) for n in registry.names()]
    chat_kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "functions": functions,
    }
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    # TODO(C3): replace "anthropic" with llm.provider_name() when ILLMProvider
    # exposes a system-name property.
    with otel_span(
        "llm.complete",
        **{
            "gen_ai.system": getattr(llm, "name", "anthropic"),
            "gen_ai.request.model": model or "",
        },
    ) as llm_span:
        response: LLMResponse = await llm.chat(**chat_kwargs)
        llm_span.set_attribute("gen_ai.usage.input_tokens", response.input_tokens)
        llm_span.set_attribute("gen_ai.usage.output_tokens", response.output_tokens)
        llm_span.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
        from voussoir.llm.pricing import compute_cost

        llm_span.set_attribute(
            "cost_usd",
            compute_cost(model, response.input_tokens, response.output_tokens),
        )
    steps.append(
        Step(
            kind="llm_call",
            name="chat",
            duration_ms=response.latency_ms,
            payload={"finish_reason": response.finish_reason},
        )
    )

    tool_calls = adapter.extract_tool_calls(response) if adapter is not None else []
    if not tool_calls:
        # No tool calls: build a plain assistant message for callers
        # that want to record final assistant text into the history.
        # _run_normal doesn't append it (the run-tail handles that via
        # ctx.record_assistant_message); stream uses the response
        # content directly. Carrying it on PreparedTurn keeps the type
        # uniform across both branches.
        return PreparedTurn(
            finish_reason="stop",
            response=response,
            tool_call_declarations=[],
            tokens_in_delta=response.input_tokens,
            tokens_out_delta=response.output_tokens,
            assistant_message=ChatMessage(
                role="assistant",
                content=response.content or "",
            ),
            adapter=None,
        )

    # Tool calls present: build the assistant message carrying them.
    # Caller appends this BEFORE dispatch (so tool-result messages
    # line up with the assistant turn that requested them).
    assistant_msg = ChatMessage(
        role="assistant",
        content=response.content or "",
        function_call={"tool_calls": tool_calls},
    )
    return PreparedTurn(
        finish_reason="tool_use",
        response=response,
        tool_call_declarations=tool_calls,
        tokens_in_delta=response.input_tokens,
        tokens_out_delta=response.output_tokens,
        assistant_message=assistant_msg,
        adapter=adapter,
    )


async def tool_turn_dispatch(
    *,
    prepared: PreparedTurn,
    registry: ToolRegistry,
    executor: IToolExecutor,
    ctx: AgentContext,
    steps: list[Step],
) -> TurnResult:
    """Phase 2: execute tool calls concurrently and accumulate outcomes.

    Pass in the ``PreparedTurn`` from phase 1. Concurrent dispatch
    happens via ``dispatch_tool_calls`` + ``accumulate_outcomes``.
    Per-outcome ``ctx.record_tool_use`` runs here (both call paths
    share it). Sub-agent tokens/cost are folded into the returned
    deltas; ``tool_result_messages`` carries the function-result
    ``ChatMessage``s for the caller to ``extend`` onto its messages
    list AFTER dispatch.

    Fast path: when ``prepared.tool_call_declarations`` is empty (the
    ``stop`` branch from phase 1) returns a zero-state ``TurnResult``
    without touching the executor.
    """
    if not prepared.tool_call_declarations:
        return TurnResult(
            tool_call_outcomes=[],
            tokens_in_delta=0,
            tokens_out_delta=0,
            cost_delta=0.0,
            tool_result_messages=[],
        )

    outcomes = await dispatch_tool_calls(
        prepared.tool_call_declarations,
        registry=registry,
        executor=executor,
        ctx=ctx,
    )

    for outcome in outcomes:
        tc = outcome.tc
        await ctx.record_tool_use(
            tc["name"],
            tc["arguments"],
            outcome.output_str,
            tool_call_id=tc.get("id"),
        )

    # accumulate_outcomes mutates the messages list it's given. Pass a
    # fresh list and surface it as tool_result_messages so the caller
    # decides when to extend its own messages (matters for stream,
    # which emits pre-dispatch events between the assistant-message
    # append and the tool-result extend).
    # prepared.adapter is always non-None here: we only reach this branch
    # when prepared.tool_call_declarations is non-empty, which only happens
    # when adapter_for(llm) succeeded (i.e. has_tools=True with a supported
    # provider). The assert narrows the type for mypy.
    assert prepared.adapter is not None, "adapter must be set when tool calls are present"
    tool_result_messages: list[ChatMessage] = []
    d_in, d_out, d_cost = accumulate_outcomes(
        outcomes,
        ctx=ctx,
        steps=steps,
        messages=tool_result_messages,
        adapter=prepared.adapter,
    )
    return TurnResult(
        tool_call_outcomes=outcomes,
        tokens_in_delta=d_in,
        tokens_out_delta=d_out,
        cost_delta=d_cost,
        tool_result_messages=tool_result_messages,
    )
