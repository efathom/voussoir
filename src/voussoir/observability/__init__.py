"""voussoir.observability — OpenTelemetry spans, metrics, and telemetry sinks.

Public API:
  configure_otel         — set up default TracerProvider + MeterProvider (called lazily by Agent)
  get_tracer / get_meter — access the framework's tracer and meter instances
  span                   — context-manager helper for creating child spans
  metrics                — module exposing the 10 named metric handles
  ITelemetrySink         — Protocol for test doubles and custom backends
  InMemoryTelemetrySink  — in-memory sink for unit tests
  NullTelemetrySink      — no-op sink for tests that don't need telemetry
  BufferedTelemetrySink  — batching sink for integration tests
  TelemetryRecord        — structured record emitted by the sink
  StepKind               — step-kind literals used in TelemetryRecord
"""

from voussoir.observability import metrics
from voussoir.observability.sink import (
    BufferedTelemetrySink,
    InMemoryTelemetrySink,
    ITelemetrySink,
    NullTelemetrySink,
    StepKind,
    TelemetryRecord,
)
from voussoir.observability.tracer import (
    configure_otel,
    get_meter,
    get_tracer,
    span,
)

__all__ = [
    "BufferedTelemetrySink",
    "ITelemetrySink",
    "InMemoryTelemetrySink",
    "NullTelemetrySink",
    "StepKind",
    "TelemetryRecord",
    "configure_otel",
    "get_meter",
    "get_tracer",
    "metrics",
    "span",
]
