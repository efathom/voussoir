"""Concurrent tool-call dispatch + delegate invocation primitives.

Phase 4.5a: moved out of agent.py (which had grown to 855 LOC) to restore
file-level focus. Behavior is unchanged from Phase 4d.

Exports:
  - ToolCallOutcome (dataclass)
  - _dispatch_one — execute one tool_use, capture outcome
  - dispatch_tool_calls — run a batch concurrently, return outcomes
  - accumulate_outcomes — sync bookkeeping (Steps + function messages)
  - make_delegate_invoker — closure that wraps an IDelegate as the invoke
    callback for delegation.make_delegate_tool
  - parent_ctx_var, last_sub_result_var — ContextVars used by the above
  - verdict_to_record — convert GuardrailVerdict → audit-log GuardrailDecision
  - apply_guardrail_verdict — record + apply a verdict, return (content, is_blocked)

The ContextVars MUST live with the helpers that read them; both are
task-local-snapshot via asyncio.create_task semantics, so co-locating
them with the create_task callsite keeps the contract obvious.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import pydantic
from ctxforge.protocols.llm import ChatMessage

from voussoir.a2a.errors import DelegationError
from voussoir.agent import delegation
from voussoir.agent.context import AgentContext
from voussoir.agent.delegate import IDelegate
from voussoir.agent.policy import PolicyViolationError
from voussoir.agent.result import AgentResult, GuardrailDecision, Step
from voussoir.agent.turn_adapter import ToolCallAdapter
from voussoir.executors import IToolExecutor
from voussoir.guardrails import GuardrailPayload, GuardrailVerdict
from voussoir.observability import span as otel_span
from voussoir.observability.metrics import GUARDRAIL_DECISIONS, TOOL_CALLS
from voussoir.tools.protocol import ToolContext
from voussoir.tools.registry import ToolRegistry

parent_ctx_var: contextvars.ContextVar[AgentContext | None] = contextvars.ContextVar(
    "voussoir_parent_ctx", default=None
)
last_sub_result_var: contextvars.ContextVar[AgentResult[str] | None] = contextvars.ContextVar(
    "voussoir_last_sub_result", default=None
)

# Bound on concurrent tool execution process-wide. Without this, a single turn
# that asks for N tools launches N unbounded coroutines (each potentially doing
# blocking I/O), which can saturate the event loop and exhaust downstream
# connections. Tune with VOUSSOIR_MAX_CONCURRENT_TOOLS.
_tool_semaphore = asyncio.Semaphore(int(os.environ.get("VOUSSOIR_MAX_CONCURRENT_TOOLS", "8")))


def verdict_to_record(
    stage: Literal["input", "tool_call", "tool_output", "output"],
    verdict: GuardrailVerdict,
    *,
    tool_name: str | None = None,
) -> GuardrailDecision:
    """Convert an ephemeral GuardrailVerdict to an audit-log GuardrailDecision.

    Public-named because this is the canonical helper used across agent.py
    (input/output stages) and dispatch.py (per-tool tool_call/tool_output
    stages). The optional `tool_name` lands on the record's `name` field
    as `chain.<stage>.<tool>` for per-tool audit-log fan-out.
    """
    name = f"chain.{stage}"
    if tool_name is not None:
        name = f"{name}.{tool_name}"
    GUARDRAIL_DECISIONS.add(
        1,
        {
            "stage": stage,
            "verdict": verdict.verdict,
            "guardrail_name": name,
        },
    )
    return GuardrailDecision(
        name=name,
        stage=stage,
        decision=verdict.verdict,
        reason=verdict.reason,
        rewrite=verdict.rewrite,
    )


def apply_guardrail_verdict(
    verdict: GuardrailVerdict,
    content: str,
    *,
    stage: Literal["input", "tool_call", "tool_output", "output"],
    ctx: AgentContext,
    blocked_content: str,
) -> tuple[str, bool]:
    """Record the guardrail decision and apply the verdict to `content`.

    Records one GuardrailDecision in ``ctx.guardrail_decisions`` (via
    :func:`verdict_to_record`), then returns ``(new_content, is_blocked)``:

    - ``BLOCK``: ``new_content`` is ``blocked_content``; ``is_blocked=True``.
    - ``REWRITE``: ``new_content`` is ``verdict.rewrite or content``; ``is_blocked=False``.
    - ``ALLOW``/``AMBIGUOUS``: ``new_content`` is ``content`` unchanged; ``is_blocked=False``.

    The caller decides what to do when ``is_blocked`` is ``True`` — for
    input-stage checks this typically means an early-return; for output-stage
    checks it means substituting the final content string.  The two-value
    return keeps the helper free of any side-effects beyond the audit log.
    """
    ctx.guardrail_decisions.append(verdict_to_record(stage, verdict))
    if verdict.verdict == "BLOCK":
        return blocked_content, True
    if verdict.verdict == "REWRITE":
        return verdict.rewrite or content, False
    return content, False  # ALLOW / AMBIGUOUS


@dataclass
class ToolCallOutcome:
    """Result of one tool-call dispatch (concurrent batch member)."""

    tc: dict[str, Any]
    output_str: str
    sub_result: AgentResult[str] | None
    duration_ms: float
    error: BaseException | None


async def _dispatch_one(
    tc: dict[str, Any],
    *,
    registry: ToolRegistry,
    executor: IToolExecutor,
    ctx: AgentContext,
) -> ToolCallOutcome:
    """Execute one tool_use block and capture its outcome.

    Each invocation runs in its own asyncio task context when called from
    asyncio.gather, so last_sub_result_var.get() reads the sub-result
    written by this task's make_delegate_invoker._invoke() (if any),
    independent of sibling tasks.
    """
    tool_obj = registry.resolve(tc["name"])
    # v1.0.2 D8: read the cached chain from ctx — set at run-start by Agent
    # from self._guardrail_chain — so tool_call / tool_output screening uses
    # the SAME chain instance as input/output screening in agent.py. Previously
    # this re-resolved per tool call from ctx.container, which would diverge
    # from the agent's cached chain if anything rebound the chain mid-run.
    chain = ctx.guardrail_chain

    # tool_call stage — fire BEFORE executor.invoke.
    # I5 (v1.0.4 E5): skip entire screen+audit block when no guardrails are
    # configured — avoids ~2 Pydantic allocs + 1 OTel counter call per tool call
    # in the common no-guardrails case (mirrors agent.py lines 497 and 657).
    if chain.count():
        tc_payload = GuardrailPayload(
            stage="tool_call",
            content=str(tc["arguments"]),
            tool_name=tc["name"],
            tool_args=tc["arguments"],
            capability=tool_obj.capability,
        )
        with otel_span("guardrail.tool_call") as gtc_span:
            tc_verdict = await chain.screen(tc_payload, ctx)
            gtc_span.set_attribute("verdict", tc_verdict.verdict)
            gtc_span.set_attribute("reason", tc_verdict.reason or "")
        tc_record = verdict_to_record("tool_call", tc_verdict, tool_name=tc["name"])
        ctx.guardrail_decisions.append(tc_record)

        if tc_verdict.verdict == "BLOCK":
            return ToolCallOutcome(
                tc=tc,
                output_str=f"[blocked by tool_call guardrail: {tc_verdict.reason}]",
                sub_result=None,
                duration_ms=0.0,
                error=None,
            )
        if tc_verdict.verdict == "REWRITE":
            # REWRITE on tool_call: rewrite is a JSON-string of args.
            # Validate against the tool's pydantic input_schema. On fail → BLOCK.
            rewrite_str = tc_verdict.rewrite or ""
            try:
                rewritten_args = json.loads(rewrite_str)
                tool_obj.input_schema.model_validate(rewritten_args)
                tc = {**tc, "arguments": rewritten_args}
            except (json.JSONDecodeError, pydantic.ValidationError) as e:
                tc_record.decision = "BLOCK"
                tc_record.reason = f"tool_call REWRITE failed schema validation: {e}"
                return ToolCallOutcome(
                    tc=tc,
                    output_str="[blocked: tool_call rewrite failed schema validation]",
                    sub_result=None,
                    duration_ms=0.0,
                    error=None,
                )

    # Now invoke the tool (with potentially rewritten args).
    # Wrap args instantiation so schema validation failures become a chain-level
    # BLOCK rather than a hard ValidationError propagating to the caller.
    try:
        args = tool_obj.input_schema(**tc["arguments"])
    except pydantic.ValidationError as e:
        bad_args_record = GuardrailDecision(
            name=f"chain.tool_call.{tc['name']}",
            stage="tool_call",
            decision="BLOCK",
            reason=f"args schema validation failed: {e.errors()[0]['msg']}",
            rewrite=None,
        )
        ctx.guardrail_decisions.append(bad_args_record)
        return ToolCallOutcome(
            tc=tc,
            output_str=f"[blocked: args failed schema validation: {e.errors()[0]['msg']}]",
            sub_result=None,
            duration_ms=0.0,
            error=None,
        )

    tool_ctx = ToolContext(
        run_id=ctx.run_id,
        span_id=ctx.trace_id,
        allowed_capabilities=ctx.allowed_capabilities,
        taint=ctx.taint,
        principal=ctx.principal,
        container=ctx.container,  # Phase 6 A4: thread container for Authorizer/CredentialBroker
    )
    t_start = time.monotonic()
    error: BaseException | None = None
    output_str = ""
    try:
        async with _tool_semaphore:
            output = await executor.invoke(tool_obj, args, tool_ctx)
        output_str = str(output)
    except asyncio.CancelledError:
        raise  # never swallow cancellation
    except PolicyViolationError:
        # Hard security denials must propagate to the caller — they must NOT
        # be silently surfaced as TOOL_ERROR text to the LLM.
        raise
    except Exception as exc:
        # Phase 4.5a P1 #10: narrowed from BaseException so KeyboardInterrupt
        # and SystemExit propagate. CancelledError is still caught explicitly
        # above so concurrent cancellation cleanup works.
        output_str = f"TOOL_ERROR: {exc}"
        error = exc
    finally:
        # Pydantic deep-copies ToolContext.taint at construction; any trust
        # tag added by the executor (e.g. UNTRUSTED for READ_PUBLIC output)
        # lives only on tool_ctx.taint. Merge back so the next dispatch call
        # sees the accumulated taint set. Runs on success AND on error so
        # partial taint from aborted calls is not silently discarded.
        ctx.taint |= tool_ctx.taint
        ctx.authz_decisions.extend(
            tool_ctx.authz_decisions
        )  # Phase 6 A4: merge-back authz audit log
    duration_ms = (time.monotonic() - t_start) * 1000

    # tool_output stage — fire AFTER executor.invoke ONLY on success
    # (no useful output to screen on error).
    # I5 (v1.0.4 E5): same chain.count() guard as the tool_call stage above —
    # skips screen+audit entirely when no guardrails are configured.
    if error is None and chain.count():
        out_payload = GuardrailPayload(
            stage="tool_output",
            content=output_str,
            tool_name=tc["name"],
            capability=tool_obj.capability,
        )
        with otel_span("guardrail.tool_output") as gto_span:
            out_verdict = await chain.screen(out_payload, ctx)
            gto_span.set_attribute("verdict", out_verdict.verdict)
            gto_span.set_attribute("reason", out_verdict.reason or "")
        out_record = verdict_to_record("tool_output", out_verdict, tool_name=tc["name"])
        ctx.guardrail_decisions.append(out_record)
        if out_verdict.verdict == "BLOCK":
            output_str = f"[blocked by tool_output guardrail: {out_verdict.reason}]"
        elif out_verdict.verdict == "REWRITE":
            # Apply rewrite, then re-screen ONCE.
            output_str = out_verdict.rewrite or output_str
            rescreen_payload = GuardrailPayload(
                stage="tool_output",
                content=output_str,
                tool_name=tc["name"],
                capability=tool_obj.capability,
            )
            rescreen_verdict = await chain.screen(rescreen_payload, ctx)
            if rescreen_verdict.verdict != "ALLOW":
                output_str = (
                    f"[blocked: tool_output rewrite still flagged " f"({rescreen_verdict.verdict})]"
                )
                out_record.decision = "BLOCK"

    TOOL_CALLS.add(
        1,
        {
            "tool_name": tc["name"],
            "capability": str(tool_obj.capability),
            "success": str(error is None),
        },
    )
    sub_result = last_sub_result_var.get()
    last_sub_result_var.set(None)
    return ToolCallOutcome(
        tc=tc,
        output_str=output_str,
        sub_result=sub_result,
        duration_ms=duration_ms,
        error=error,
    )


async def dispatch_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    registry: ToolRegistry,
    executor: IToolExecutor,
    ctx: AgentContext,
) -> list[ToolCallOutcome]:
    """Run every tool_use block concurrently; return outcomes in declared order.

    Phase 4.5a P0 #6: uses asyncio.gather inside try/finally so unfinished
    tasks are cancelled and drained when the caller is cancelled mid-flight.
    Pre-4.5a used as_completed which leaked the in-flight _dispatch_one
    tasks (they kept holding ctx/registry/HTTP refs).

    Order: gather preserves submission order, so outcomes line up with
    tool_calls indexes naturally — the previous outcomes_by_id re-ordering
    map is unnecessary.

    Propagation: _dispatch_one captures regular Exception as outcome.error
    internally, but DELIBERATELY re-raises both CancelledError (cooperative
    cancellation) AND PolicyViolationError (hard security denials —
    CAPABILITY_DENIED / TAINT_EXFILTRATION / AUTHZ_DENIED must never be
    silently converted to TOOL_ERROR strings the LLM could see). When
    PolicyViolationError fires in a concurrent batch, gather raises it
    immediately, the finally block cancels sibling tasks, and the exception
    propagates up to tool_turn_dispatch / Agent._run_normal — which is the
    intended fail-loud behavior.
    """
    if not tool_calls:
        return []
    tasks = [
        asyncio.create_task(_dispatch_one(tc, registry=registry, executor=executor, ctx=ctx))
        for tc in tool_calls
    ]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if any(not t.done() for t in tasks):
            # Drain so cancellation actually propagates before we return.
            await asyncio.gather(*tasks, return_exceptions=True)


def accumulate_outcomes(
    outcomes: list[ToolCallOutcome],
    *,
    ctx: AgentContext,
    steps: list[Step],
    messages: list[ChatMessage],
    adapter: ToolCallAdapter,
) -> tuple[int, int, float]:
    """Append tool_call (+ delegation, when applicable) Steps and a
    function-result ChatMessage per outcome. Aggregate sub-agent
    tokens/cost. Returns (delta_tokens_in, delta_tokens_out, delta_cost).
    """
    # v1.0.2 D5: merge sub-agent taint back into parent context. Without this,
    # a sub-agent that reads UNTRUSTED content "launders" its taint — the
    # parent gets a clean ctx.taint back and could then call EXFILTRATION
    # tools unblocked (Lethal Trifecta delegation bypass). Direction is
    # strictly sub → parent (union), never the reverse.
    for outcome in outcomes:
        if outcome.sub_result is not None:
            ctx.taint |= outcome.sub_result.taint

    delta_in = delta_out = 0
    delta_cost = 0.0
    for outcome in outcomes:
        tc = outcome.tc
        steps.append(
            Step(
                kind="tool_call",
                name=tc["name"],
                duration_ms=outcome.duration_ms,
                payload={
                    "args": tc["arguments"],
                    "output_preview": outcome.output_str[:200],
                },
            )
        )
        if outcome.sub_result is not None:
            sr = outcome.sub_result
            delta_in += sr.tokens_in
            delta_out += sr.tokens_out
            delta_cost += sr.cost_usd
            steps.append(
                Step(
                    kind="delegation",
                    name=tc["name"].removeprefix(delegation.DELEGATE_TOOL_PREFIX),
                    duration_ms=sr.duration_ms,
                    payload={
                        "task_preview": str(tc["arguments"].get("task", ""))[:200],
                        "delegation_chain": sr.delegation_chain,
                        "cost_usd": sr.cost_usd,
                        "finish_reason": sr.finish_reason,
                    },
                )
            )
        messages.append(
            adapter.build_tool_result_message(
                tc.get("id", ""),
                outcome.output_str,
                tc["name"],
            )
        )
    return delta_in, delta_out, delta_cost


def make_delegate_invoker(
    delegate: IDelegate,
) -> Callable[[str], Awaitable[str]]:
    """Build the invoke callback that delegation.make_delegate_tool wraps."""

    async def _invoke(task: str) -> str:
        parent_ctx = parent_ctx_var.get()
        if parent_ctx is None:
            raise RuntimeError("delegate_to_* tool invoked outside an Agent.run context")

        if parent_ctx.delegation_depth + 1 > parent_ctx.max_delegation_depth:
            return delegation.wrap_delegate_output(
                delegate.name,
                f"DELEGATION_REFUSED: max_delegation_depth "
                f"({parent_ctx.max_delegation_depth}) exceeded.",
            )

        with otel_span(
            "delegation.dispatch." + delegate.name,
            child_agent=delegate.name,
            delegate_kind=type(delegate).__name__,
        ):
            try:
                sub_result = await delegate.delegate(task, parent_ctx=parent_ctx)
            except PolicyViolationError as exc:
                return delegation.wrap_delegate_output(delegate.name, f"DELEGATION_REFUSED: {exc}")
            except DelegationError as exc:
                # Phase 4.5a P1 #25: AgentRef.delegate raises typed
                # DelegationError subclasses; we wrap them as DELEGATION_REFUSED
                # the same way (lead's LLM sees a string, not an exception type).
                return delegation.wrap_delegate_output(delegate.name, f"DELEGATION_REFUSED: {exc}")

        last_sub_result_var.set(sub_result)
        return delegation.wrap_delegate_output(delegate.name, str(sub_result.output))

    return _invoke
