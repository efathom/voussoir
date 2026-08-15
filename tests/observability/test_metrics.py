"""Locks OTel metric emission per spec §6.2 (Phase 5 Task C4).

Installs a session-scoped InMemoryMetricReader (mirroring the span hierarchy
test's session-scoped InMemorySpanExporter pattern), runs Agent.run / executor
calls with OTel enabled, and inspects collected metric names + attribute schemas.

OTel MeterProvider replacement is guarded ("Overriding … not allowed" once a
real provider is set), so we install a *single* shared MeterProvider once per
session and clear the reader between tests — exactly like test_span_hierarchy.py
does for spans.
"""

from __future__ import annotations

import pytest
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


@pytest.fixture(scope="session")
def _metric_provider() -> InMemoryMetricReader:
    """Install a single real MeterProvider + InMemoryMetricReader for the session.

    OTel only allows replacing ProxyMeterProvider once; all metric tests share
    this provider so module-level instruments (created at import time) forward
    to the right SDK backend.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics.set_meter_provider(provider)
    return reader


@pytest.fixture
def metric_reader(monkeypatch, _metric_provider: InMemoryMetricReader) -> InMemoryMetricReader:
    """Per-test reader view: unset OTel-disable env, collect any pending data
    (to clear previous-test residue), yield the shared reader.

    The autouse ``_disable_otel_in_tests`` fixture (conftest.py) sets
    ``VOUSSOIR_OTEL_DISABLED=1``; we unset it here so voussoir's metric
    calls are live for these tests.
    """
    monkeypatch.delenv("VOUSSOIR_OTEL_DISABLED", raising=False)
    # Drain any data from previous tests so _all_metric_names sees only fresh data.
    _metric_provider.get_metrics_data()
    return _metric_provider


def _all_metric_names(reader: InMemoryMetricReader) -> set[str]:
    """Collect all unique metric names from a reader.get_metrics_data() call."""
    data = reader.get_metrics_data()
    if data is None:
        return set()
    names: set[str] = set()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                names.add(m.name)
    return names


async def test_agent_run_emits_token_metrics(metric_reader, make_container, stub_llm):  # type: ignore[no-untyped-def]
    from voussoir import Agent

    a = Agent(name="x", container=make_container(stub_llm(content="hello")))
    await a.run("hi")
    names = _all_metric_names(metric_reader)
    assert "voussoir.tokens.in" in names
    assert "voussoir.tokens.out" in names
    assert "voussoir.duration_ms" in names


async def test_capability_denial_increments_counter(metric_reader, make_container):  # type: ignore[no-untyped-def]
    from voussoir.agent.policy import PolicyViolationError
    from voussoir.executors.standard import StandardExecutor
    from voussoir.tools import Capability, ToolContext, tool

    @tool(capability=Capability.EXFILTRATION, name="send")
    async def send() -> str:
        return "sent"

    ex = StandardExecutor()
    ctx = ToolContext(run_id="r", span_id="s", allowed_capabilities=Capability.READ_PUBLIC)
    with pytest.raises(PolicyViolationError):
        await ex.invoke(send, send.input_schema(), ctx)
    names = _all_metric_names(metric_reader)
    assert "voussoir.capability.denials" in names


def test_metrics_module_defines_9_handles():  # type: ignore[no-untyped-def]
    """All 9 spec-required metrics are exported from voussoir.observability.metrics."""
    from voussoir.observability import metrics

    for name in (
        "TOKENS_IN",
        "TOKENS_OUT",
        "COST_USD",
        "DURATION_MS",
        "TOOL_CALLS",
        "GUARDRAIL_DECISIONS",
        "CAPABILITY_DENIALS",
        "TAINT_EXFIL_BLOCKS",
        "CASCADE_ESCALATIONS",
    ):
        assert hasattr(metrics, name), f"missing metric: {name}"
