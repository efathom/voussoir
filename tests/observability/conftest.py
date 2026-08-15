"""Shared OTel fixtures for the observability test suite.

Both test_span_hierarchy.py and test_authz_span.py need a session-scoped
InMemorySpanExporter. OTel's set_tracer_provider is guarded by an internal
Once — only the first call takes effect; subsequent calls are silently ignored.
Therefore the provider + exporter must be installed exactly once per session,
and all tests that need span data must share the same exporter instance.

Placing these fixtures in conftest.py (rather than individual test files)
ensures pytest resolves them as a single session-scoped singleton even when
multiple test modules request them.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session")
def _span_provider() -> InMemorySpanExporter:
    """Install a single real TracerProvider + InMemorySpanExporter for the session.

    OTel allows the ProxyTracerProvider to be replaced exactly once; all
    span tests share this provider so module-level tracers are wired up
    correctly regardless of import order.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def span_exporter(
    monkeypatch: pytest.MonkeyPatch, _span_provider: InMemorySpanExporter
) -> InMemorySpanExporter:
    """Per-test exporter view: unset the OTel-disable env var, clear previous
    spans, yield the shared exporter.

    The autouse ``_disable_otel_in_tests`` fixture (conftest.py) sets
    ``VOUSSOIR_OTEL_DISABLED=1``; we unset it here so voussoir's span
    calls are live for these tests.
    """
    monkeypatch.delenv("VOUSSOIR_OTEL_DISABLED", raising=False)
    _span_provider.clear()
    return _span_provider
