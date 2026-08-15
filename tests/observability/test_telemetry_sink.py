"""ITelemetrySink protocol + reference implementations."""

from __future__ import annotations

from voussoir.agent.result import AgentResult
from voussoir.observability.sink import (
    BufferedTelemetrySink,
    InMemoryTelemetrySink,
    ITelemetrySink,
    NullTelemetrySink,
)


def _make_result() -> AgentResult[str]:
    return AgentResult(
        output="hello",
        trace_id="t",
        steps=[],
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.01,
        duration_ms=200.0,
        delegation_chain=[],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )


def test_null_sink_record_llm_call_is_noop() -> None:
    sink: ITelemetrySink = NullTelemetrySink()
    sink.record_llm_call(
        name="llm_judge",
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.001,
        duration_ms=100.0,
    )
    # No exceptions, no observable state.


def test_null_sink_scoped_redirects_to_child() -> None:
    parent = NullTelemetrySink()
    child = InMemoryTelemetrySink()
    with parent.scoped(child):
        parent.record_llm_call(
            name="llm_judge", tokens_in=1, tokens_out=1, cost_usd=0.001, duration_ms=1.0
        )
    assert len(child.records) == 1
    # After exit, parent is back to no-op
    parent.record_llm_call(name="x", tokens_in=1, tokens_out=1, cost_usd=0.001, duration_ms=1.0)
    assert len(child.records) == 1


def test_null_sink_scoped_restores_on_exception() -> None:
    parent = NullTelemetrySink()
    child = InMemoryTelemetrySink()
    try:
        with parent.scoped(child):
            parent.record_llm_call(
                name="boom",
                tokens_in=1,
                tokens_out=1,
                cost_usd=0.001,
                duration_ms=1.0,
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert len(child.records) == 1


def test_buffered_sink_merge_into_result() -> None:
    buf = BufferedTelemetrySink()
    buf.record_llm_call(
        name="llm_judge",
        tokens_in=12,
        tokens_out=7,
        cost_usd=0.0023,
        duration_ms=42.0,
    )
    from voussoir.agent.telemetry import merge_buffered_telemetry_into_result

    merged = merge_buffered_telemetry_into_result(_make_result(), buf.records)
    assert merged.tokens_in == 112
    assert merged.tokens_out == 57
    assert abs(merged.cost_usd - 0.0123) < 1e-9
    assert len(merged.steps) == 1
    assert merged.steps[0].kind == "validator_call"
    assert merged.steps[0].name == "llm_judge"


def test_buffered_sink_merge_does_not_mutate_original() -> None:
    buf = BufferedTelemetrySink()
    buf.record_llm_call(
        name="llm_judge",
        tokens_in=5,
        tokens_out=5,
        cost_usd=0.001,
        duration_ms=1.0,
    )
    from voussoir.agent.telemetry import merge_buffered_telemetry_into_result

    original = _make_result()
    merge_buffered_telemetry_into_result(original, buf.records)
    assert original.tokens_in == 100
    assert original.cost_usd == 0.01
    assert original.steps == []


def test_in_memory_sink_records_step() -> None:
    sink = InMemoryTelemetrySink()
    sink.record_step(kind="tool_call", name="search", duration_ms=15.0)
    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec.kind == "tool_call"
    assert rec.name == "search"
    assert rec.duration_ms == 15.0
    assert rec.tokens_in == 0
    assert rec.tokens_out == 0
    assert rec.cost_usd == 0.0


def test_default_container_binds_null_sink(monkeypatch) -> None:
    """default_container() binds NullTelemetrySink as the production default."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from voussoir.container.defaults import default_container

    c = default_container()
    sink = c.resolve(ITelemetrySink)
    assert isinstance(sink, NullTelemetrySink)
