"""Locks that OpenTelemetry SDK is importable as a base dependency (Phase 5 Task C1).

Phase 5 §6.3 commits to the OTel SDK being a base dep (not an extra) so
`opentelemetry.trace.get_tracer(...)` and the OTLP exporter work without
`pip install voussoir[observability]`. The base wheel grows by ~10 transitive
deps; users opt out via OTEL_SDK_DISABLED=true or VOUSSOIR_OTEL_DISABLED=1
(landing in C5).
"""

from __future__ import annotations


def test_otel_api_importable() -> None:
    """opentelemetry-api carries Tracer/Meter Protocol shells."""
    import opentelemetry.metrics
    import opentelemetry.trace

    assert opentelemetry.trace.get_tracer("voussoir.test")
    assert opentelemetry.metrics.get_meter("voussoir.test")


def test_otel_sdk_importable() -> None:
    """opentelemetry-sdk provides TracerProvider, BatchSpanProcessor, ConsoleSpanExporter."""
    import opentelemetry.sdk.metrics
    import opentelemetry.sdk.trace
    import opentelemetry.sdk.trace.export

    assert opentelemetry.sdk.trace.TracerProvider
    assert opentelemetry.sdk.trace.export.BatchSpanProcessor
    assert opentelemetry.sdk.trace.export.ConsoleSpanExporter


def test_otel_otlp_http_exporter_importable() -> None:
    """opentelemetry-exporter-otlp-proto-http is the HTTP exporter base deps need."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    assert OTLPSpanExporter
    assert OTLPMetricExporter
