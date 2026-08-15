"""OpenTelemetry tracer + meter helpers.

Use this when instrumenting voussoir-internal code. Real OTel SDK under;
respects OTEL_SDK_DISABLED and VOUSSOIR_OTEL_DISABLED.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Span

try:
    PKG_VERSION = importlib.metadata.version("voussoir")
except importlib.metadata.PackageNotFoundError:
    PKG_VERSION = "0.0.0+local"  # editable / not-installed dev path


def _is_disabled() -> bool:
    """True when OTel should no-op for this process.

    Honors the official OTEL_SDK_DISABLED env var and voussoir's
    convenience VOUSSOIR_OTEL_DISABLED equivalent. Either set to a
    truthy value disables span emission (the SDK still resolves but
    yields NonRecordingSpan instances).
    """
    return os.environ.get("VOUSSOIR_OTEL_DISABLED", "").strip() in (
        "1",
        "true",
        "TRUE",
    ) or os.environ.get("OTEL_SDK_DISABLED", "").strip() in (
        "1",
        "true",
        "TRUE",
    )


class _VoussoirTracer:
    """Thin kwarg-ergonomic wrapper around OTel's Tracer.

    Use this when you want voussoir's existing `start_as_current_span(name, **attrs)`
    kwarg style — the wrapper translates kwargs to OTel's `attributes=` dict.
    Real instrumentation sites can also use the module-level `span()` helper
    (which uses this wrapper internally), or drop down to `trace.get_tracer(...)`
    directly for full OTel API access.
    """

    def __init__(self, otel_tracer: trace.Tracer) -> None:
        self._tracer = otel_tracer

    @contextmanager
    def start_as_current_span(self, name: str, **attrs: Any) -> Iterator[Span]:
        with self._tracer.start_as_current_span(name, attributes=attrs) as s:
            yield s


def get_tracer(name: str) -> _VoussoirTracer:
    """Use this when instrumenting voussoir-internal code; returns a tracer wrapper.

    The wrapper preserves voussoir's pre-Phase-5 kwarg-style API. The underlying
    OTel Tracer is created via `opentelemetry.trace.get_tracer(name, voussoir_version)`;
    callers needing the standard OTel API can call `trace.get_tracer(...)` directly.
    """
    return _VoussoirTracer(trace.get_tracer(name, PKG_VERSION))


def get_meter(name: str) -> metrics.Meter:
    """Use this when emitting voussoir-internal metrics; returns a real OTel Meter."""
    return metrics.get_meter(name, PKG_VERSION)


_VOUSSOIR_TRACER = trace.get_tracer("voussoir", PKG_VERSION)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Use this as the canonical span context manager across voussoir instrumentation.

    Equivalent to `with OTel_Tracer.start_as_current_span(name, attributes=attrs)`
    but with kwarg-style attribute ergonomics. Reuses a module-scoped tracer
    named `"voussoir"` (the library name) — observability backends group all
    voussoir spans under a single instrumentation library, with the span name
    (e.g., `agent.run`, `tool.call.<name>`, `guardrail.input`) encoding the
    site of origin.
    """
    with _VOUSSOIR_TRACER.start_as_current_span(name, attributes=attrs) as s:
        yield s


def configure_otel(c: Any = None) -> None:
    """Idempotent default OTel setup.

    Use this when you want voussoir to install a TracerProvider + MeterProvider
    driven by the standard OTel env vars. Called from `default_container()`.
    Skipped when:
      - VOUSSOIR_OTEL_DISABLED / OTEL_SDK_DISABLED is set, OR
      - a TracerProvider is already configured (user override via container,
        or e.g., a pytest fixture already installed one).

    Exporter selection (traces and metrics alike):
      - OTEL_EXPORTER_OTLP_ENDPOINT set → OTLP HTTP/protobuf exporter.
      - OTEL_TRACES_EXPORTER=console / OTEL_METRICS_EXPORTER=console → console.
      - otherwise → no exporter is registered (spans/metrics are dropped
        silently) so a library never spams stdout in production.
    """
    del c  # reserved for a future container-driven config; keep the signature stable.
    if _is_disabled():
        return
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return  # user (or pytest fixture) already installed one — respect it.

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "voussoir"),
            "service.version": PKG_VERSION,
        }
    )

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    traces_exporter = os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower()

    provider = SDKTracerProvider(resource=resource)
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif traces_exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # Metrics: install a MeterProvider too. Pre-v1.3.0 only a TracerProvider
    # was installed, so the metric handles in voussoir.observability.metrics
    # were created but never exported.
    from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    metrics_exporter = os.environ.get("OTEL_METRICS_EXPORTER", "").strip().lower()
    metric_readers: list[Any] = []
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    elif metrics_exporter == "console":
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    metrics.set_meter_provider(SDKMeterProvider(resource=resource, metric_readers=metric_readers))
