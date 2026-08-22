"""AgentEvent construction helpers for `Agent.stream`.

Stream is an async generator: it can `yield` events but cannot delegate
`yield` into a helper. These functions return `AgentEvent` instances
(or lists thereof) that `stream` then yields.

Lives alongside `voussoir.agent.turn` (the per-turn body shared with
`_run_normal`): turn does the work, this module shapes the events.
Both are leaf modules — neither imports `Agent`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from voussoir.agent import delegation
from voussoir.agent.dispatch import ToolCallOutcome
from voussoir.agent.result import AgentEvent


def token_event(text: str, span_id: str) -> AgentEvent:
    return AgentEvent(
        kind="token",
        payload={"text": text},
        span_id=span_id,
        timestamp=datetime.now(UTC),
    )


def done_event(output: str, span_id: str, finish_reason: str = "completed") -> AgentEvent:
    """Terminal stream event.

    Carries `finish_reason` so a consumer can tell a clean finish from a
    blocked or budget-cut one. Without it `voussoir run` had no way to detect a
    failed stream and always exited 0, contradicting its own "exit 1 on
    failures" docstring (audit, minor).
    """
    return AgentEvent(
        kind="done",
        payload={"output": output, "finish_reason": finish_reason},
        span_id=span_id,
        timestamp=datetime.now(UTC),
    )


def cascade_passed_event(validator_name: str, *, span_id: str) -> AgentEvent:
    """Emit when a streaming cascade gate returns PASS after `done`."""
    return AgentEvent(
        kind="cascade_passed",
        payload={"validator_name": validator_name, "verdict": "PASS"},
        span_id=span_id,
        timestamp=datetime.now(UTC),
    )


def cascade_failed_event(
    validator_name: str,
    verdict: str,
    reason: str,
    *,
    span_id: str,
) -> AgentEvent:
    """Emit when a streaming cascade gate returns non-PASS after `done`."""
    return AgentEvent(
        kind="cascade_failed",
        payload={
            "validator_name": validator_name,
            "verdict": verdict,
            "reason": reason,
        },
        span_id=span_id,
        timestamp=datetime.now(UTC),
    )


def pre_dispatch_events(tool_calls: list[dict[str, Any]], span_id: str) -> list[AgentEvent]:
    """Build tool_started + delegation_started events for one turn's
    tool_calls, preserving declared order (tool_started before
    delegation_started for the same tc).
    """
    out: list[AgentEvent] = []
    for tc in tool_calls:
        out.append(
            AgentEvent(
                kind="tool_started",
                payload={
                    "tool_call_id": tc.get("id", ""),
                    "name": tc["name"],
                    "args": tc["arguments"],
                },
                span_id=span_id,
                timestamp=datetime.now(UTC),
            )
        )
        if tc["name"].startswith(delegation.DELEGATE_TOOL_PREFIX):
            out.append(
                AgentEvent(
                    kind="delegation_started",
                    payload={
                        "tool_call_id": tc.get("id", ""),
                        "delegate_name": tc["name"].removeprefix(delegation.DELEGATE_TOOL_PREFIX),
                        "task_preview": str(tc["arguments"].get("task", ""))[:200],
                    },
                    span_id=span_id,
                    timestamp=datetime.now(UTC),
                )
            )
    return out


def post_dispatch_events(outcomes: list[ToolCallOutcome], span_id: str) -> list[AgentEvent]:
    """Build tool_finished + delegation_finished events for one turn's
    outcomes. delegation_finished is omitted when outcome.sub_result is
    None (delegate refusal / depth-cap / typed errors take this path).
    """
    out: list[AgentEvent] = []
    for outcome in outcomes:
        tc = outcome.tc
        out.append(
            AgentEvent(
                kind="tool_finished",
                payload={
                    "tool_call_id": tc.get("id", ""),
                    "name": tc["name"],
                    "output_preview": outcome.output_str[:200],
                    "duration_ms": outcome.duration_ms,
                    "error": str(outcome.error) if outcome.error else None,
                },
                span_id=span_id,
                timestamp=datetime.now(UTC),
            )
        )
        if outcome.sub_result is not None:
            sr = outcome.sub_result
            out.append(
                AgentEvent(
                    kind="delegation_finished",
                    payload={
                        "tool_call_id": tc.get("id", ""),
                        "delegate_name": tc["name"].removeprefix(delegation.DELEGATE_TOOL_PREFIX),
                        "output_preview": str(sr.output)[:200],
                        "cost_usd": sr.cost_usd,
                        "duration_ms": sr.duration_ms,
                        "finish_reason": sr.finish_reason,
                    },
                    span_id=span_id,
                    timestamp=datetime.now(UTC),
                )
            )
    return out
