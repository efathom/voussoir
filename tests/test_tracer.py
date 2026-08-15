"""Locks voussoir.observability.tracer API (Phase 5 Task C2).

Replaces the Phase 0 no-op tests. The tracer now wraps real OTel; we verify
get_tracer/get_meter return objects with the expected shape, span() ergonomic
wrapper works, and the disabled flag honors env vars.
"""

from __future__ import annotations

from voussoir.observability import configure_otel, get_meter, get_tracer, span


def test_get_tracer_returns_kwarg_compatible_wrapper():
    """get_tracer wraps OTel's Tracer with voussoir's kwarg-style API."""
    tracer = get_tracer("voussoir.test")
    # Existing pattern: kwargs as attributes
    with tracer.start_as_current_span("my-op", attr1="v1", attr2=42) as s:
        assert s is not None


def test_get_meter_returns_otel_meter():
    """get_meter returns a real OTel Meter."""
    from opentelemetry.metrics import Meter

    meter = get_meter("voussoir.test")
    assert isinstance(meter, Meter)


def test_span_helper_kwarg_attributes():
    """span() is the module-level ergonomic kwarg helper."""
    with span("test.op", attr="value") as s:
        s.set_attribute("more", "stuff")


def test_is_disabled_honors_voussoir_env(monkeypatch):
    """VOUSSOIR_OTEL_DISABLED=1 toggles the disable check."""
    from voussoir.observability.tracer import _is_disabled

    monkeypatch.setenv("VOUSSOIR_OTEL_DISABLED", "1")
    assert _is_disabled() is True
    monkeypatch.setenv("VOUSSOIR_OTEL_DISABLED", "0")
    assert _is_disabled() is False


def test_is_disabled_honors_otel_sdk_env(monkeypatch):
    """OTEL_SDK_DISABLED=true toggles the disable check (official OTel env var)."""
    from voussoir.observability.tracer import _is_disabled

    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("VOUSSOIR_OTEL_DISABLED", raising=False)
    assert _is_disabled() is True


def test_configure_otel_is_idempotent_and_respects_disabled(monkeypatch):
    """configure_otel skips when disabled; can be called multiple times safely."""
    monkeypatch.setenv("VOUSSOIR_OTEL_DISABLED", "1")
    configure_otel()  # must not raise
    configure_otel()  # second call also fine
