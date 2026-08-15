"""ITelemetrySink protocol + 3 reference implementations.

Distinct from `get_tracer()`: tracer carries spans for distributed
tracing; sink carries usage records (tokens / cost / duration / Step)
that aggregate into `AgentResult.tokens_in/out`, `cost_usd`, and `steps`.

Phase 3.5 introduces the protocol and uses it exclusively for the new
LLMJudge / cascade-validator path. Existing LLM/tool/sub-agent
accumulation paths are not migrated in this phase.

Scoping is implemented via an internal ContextVar inside `_BaseSink`.
The `scoped(child)` context manager pushes `child` onto the var; every
sink's `record_*` call routes through `_emit_record`, which redirects
to the active scoped child if one is set. Async-safe: ContextVar gives
each task its own snapshot, so concurrent cascades don't collide.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# StepKind enumerates the kinds of work an Agent records via the telemetry
# sink. Used as the discriminator on `TelemetryRecord.kind` and on
# `ITelemetrySink.record_step(kind=...)`. Kept as a Literal alias rather
# than a StrEnum so it's a zero-cost type hint at call sites.
type StepKind = Literal[
    "llm_call",
    "tool_call",
    "delegation",
    "guardrail",
    "validator_call",
]


@dataclass
class TelemetryRecord:
    """One emission. Distinguishes pure-Step records (no cost) from llm-call
    records (tokens + cost + Step in one)."""

    kind: StepKind
    name: str
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ITelemetrySink(Protocol):
    """Per-run sink for usage records (tokens / cost / step kind / duration).

    Use this when you need to capture or aggregate an agent's emitted usage
    independent of distributed tracing — e.g. summing tokens across a run
    for `AgentResult.tokens_in/out`, or buffering validator-call costs in a
    scoped child sink. Three reference implementations ship in this module:
    `NullTelemetrySink` (default), `BufferedTelemetrySink` (cascade), and
    `InMemoryTelemetrySink` (tests).
    """

    def record_llm_call(
        self,
        *,
        name: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an LLM call: tokens in / out, cost, and duration."""
        ...

    def record_step(
        self,
        *,
        kind: StepKind,
        name: str,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a non-LLM step (tool call, delegation, guardrail, etc.)."""
        ...

    @contextmanager
    def scoped(self, child: ITelemetrySink) -> Iterator[None]:
        """Redirect emissions on this sink to `child` for the block's
        duration. Async-safe via ContextVar — concurrent scopes don't leak.
        """
        ...


_active_sink_var: contextvars.ContextVar[ITelemetrySink | None] = contextvars.ContextVar(
    "voussoir_telemetry_active_sink", default=None
)


class _BaseSink:
    """Shared scoping logic.

    Every concrete sink's `record_*` calls `_emit_record(rec)` first.
    `_emit_record` checks the active-sink ContextVar — if a child is
    active and isn't `self`, the call is redirected to the child.
    Otherwise the subclass's `_store_record` hook handles persistence.
    """

    def _store_record(self, rec: TelemetryRecord) -> None:
        """Subclass hook. Default: drop on the floor (NullTelemetrySink)."""
        del rec

    def _emit_record(self, rec: TelemetryRecord) -> None:
        active = _active_sink_var.get()
        if active is not None and active is not self:
            if rec.tokens_in or rec.tokens_out or rec.cost_usd:
                active.record_llm_call(
                    name=rec.name,
                    tokens_in=rec.tokens_in,
                    tokens_out=rec.tokens_out,
                    cost_usd=rec.cost_usd,
                    duration_ms=rec.duration_ms,
                    payload=rec.payload,
                )
            else:
                active.record_step(
                    kind=rec.kind,
                    name=rec.name,
                    duration_ms=rec.duration_ms,
                    payload=rec.payload,
                )
            return
        self._store_record(rec)

    @contextmanager
    def scoped(self, child: ITelemetrySink) -> Iterator[None]:
        token = _active_sink_var.set(child)
        try:
            yield
        finally:
            _active_sink_var.reset(token)

    def record_llm_call(
        self,
        *,
        name: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        # llm-call records use kind="llm_call" by default; BufferedTelemetrySink
        # overrides to mark them as validator_call (its only consumer).
        self._emit_record(
            TelemetryRecord(
                kind="llm_call",
                name=name,
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                payload=payload or {},
            )
        )

    def record_step(
        self,
        *,
        kind: StepKind,
        name: str,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._emit_record(
            TelemetryRecord(
                kind=kind,
                name=name,
                duration_ms=duration_ms,
                payload=payload or {},
            )
        )


class NullTelemetrySink(_BaseSink):
    """Default container binding. Records dropped; `scoped(child)` redirects."""

    # _store_record inherited (no-op).


class BufferedTelemetrySink(_BaseSink):
    """Buffers records in memory; `merge_into(result)` folds them into an AgentResult.

    Used by the cascade as the scoped child during validator.validate() —
    every emission inside the `with sink.scoped(buffer):` block lands here.
    LLM-call records are tagged kind="validator_call" because the cascade's
    only emitter today is LLMJudge (a validator-call by definition).
    """

    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []

    @property
    def records(self) -> list[TelemetryRecord]:
        return list(self._records)

    def record_llm_call(
        self,
        *,
        name: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._records.append(
            TelemetryRecord(
                kind="validator_call",
                name=name,
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                payload=payload or {},
            )
        )

    def record_step(
        self,
        *,
        kind: StepKind,
        name: str,
        duration_ms: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._records.append(
            TelemetryRecord(
                kind=kind,
                name=name,
                duration_ms=duration_ms,
                payload=payload or {},
            )
        )


class InMemoryTelemetrySink(_BaseSink):
    """Public test fixture. Exposes `records` directly for assertion."""

    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []

    @property
    def records(self) -> list[TelemetryRecord]:
        return list(self._records)

    def _store_record(self, rec: TelemetryRecord) -> None:
        self._records.append(rec)
