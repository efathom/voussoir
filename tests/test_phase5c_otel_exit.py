"""Phase 5 Tranche C exit gate — invariants from spec §6 + §7.

These tests lock the OTel + carry-overs claim for Tranche C. Future changes
that break any of these will fail at the exit gate before they reach CI.
"""

from __future__ import annotations

import os


def test_exit_1_otel_sdk_is_base_dep() -> None:
    """Spec §6.3: OTel SDK is a base dependency, not an extra."""
    import opentelemetry.sdk.metrics
    import opentelemetry.sdk.trace

    assert opentelemetry.sdk.trace.TracerProvider
    assert opentelemetry.sdk.metrics.MeterProvider


def test_exit_2_get_tracer_returns_real_otel() -> None:
    """Spec §6.4: voussoir.observability.span returns a real OTel span context."""
    from voussoir.observability import span

    with span("voussoir.exit-test", attr="value"):
        pass  # must not raise


def test_exit_3_voussoir_otel_disabled_var_honored(monkeypatch) -> None:
    """Spec §6.4: VOUSSOIR_OTEL_DISABLED + OTEL_SDK_DISABLED both honored."""
    from voussoir.observability.tracer import _is_disabled

    monkeypatch.setenv("VOUSSOIR_OTEL_DISABLED", "1")
    assert _is_disabled() is True
    monkeypatch.setenv("VOUSSOIR_OTEL_DISABLED", "0")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert _is_disabled() is True


def test_exit_4_a2a_wire_result_public_default() -> None:
    """Spec §7.1: WireAgentResult is publicly re-exported."""
    from voussoir import WireAgentResult
    from voussoir.a2a import WireAgentResult as A2AWireAgentResult

    assert WireAgentResult is A2AWireAgentResult


def test_exit_5_agent_result_has_to_wire() -> None:
    """Spec §7.1: AgentResult.to_wire(profile=) method exists."""
    from voussoir.agent import AgentResult

    assert "to_wire" in dir(AgentResult)


def test_exit_6_stream_cascade_events_present() -> None:
    """Spec §7.2: AgentEvent.kind Literal contains cascade_passed/failed."""
    import typing

    from voussoir.agent.result import AgentEvent

    kind_field = AgentEvent.model_fields["kind"]
    kind_args = typing.get_args(kind_field.annotation)
    assert "cascade_passed" in kind_args
    assert "cascade_failed" in kind_args


def test_exit_7_env_var_overrides_apply(monkeypatch, make_container) -> None:
    """Spec §7.3: bind_agent_registry applies VOUSSOIR_AGENT_<NAME>_<FIELD> env overrides."""
    from voussoir import Agent
    from voussoir.agent import bind_agent_registry, register_agent
    from voussoir.agent.registry import AgentRegistry

    c = make_container()
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    register_agent(c, Agent(name="exit7-agent", container=c, model="claude-opus-4-7"))
    monkeypatch.setenv("VOUSSOIR_AGENT_EXIT7_AGENT_MODEL", "claude-haiku-4-5")
    bind_agent_registry(c, load_plugins=False)
    assert c.resolve(AgentRegistry).get("exit7-agent").model == "claude-haiku-4-5"


def test_exit_8_examples_dir_present() -> None:
    """examples/05_observability/ ships three demos + README."""
    examples = "examples/05_observability"
    assert os.path.isdir(examples)
    expected = {
        "README.md",
        "console_exporter_demo.py",
        "otlp_phoenix_demo.py",
        "streaming_cascade_demo.py",
    }
    actual = set(os.listdir(examples))
    assert expected.issubset(actual), f"missing demos: {expected - actual}"
