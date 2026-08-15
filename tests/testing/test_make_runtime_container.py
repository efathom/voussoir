"""Tests for voussoir.testing.make_runtime_container (v1.2 G3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voussoir.executors import IToolExecutor, StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain, IGuardrailChain
from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
from voussoir.protocols import ILLMProvider, IMemoryStore, ISessionStore
from voussoir.testing import make_runtime_container


def test_runtime_container_has_default_memory_and_session_stores() -> None:
    container = make_runtime_container()
    assert container.resolve(IMemoryStore) is not None
    assert container.resolve(ISessionStore) is not None


def test_runtime_container_binds_default_telemetry_sink() -> None:
    container = make_runtime_container()
    sink = container.resolve(ITelemetrySink)
    assert isinstance(sink, NullTelemetrySink)


def test_runtime_container_binds_default_executor_and_guardrail_chain() -> None:
    container = make_runtime_container()
    assert isinstance(container.resolve(IToolExecutor), StandardExecutor)
    assert isinstance(container.resolve(IGuardrailChain), DefaultGuardrailChain)


def test_runtime_container_accepts_override_kwargs() -> None:
    fake_llm = MagicMock(spec=ILLMProvider)
    fake_sink = MagicMock(spec=ITelemetrySink)
    fake_executor = MagicMock(spec=IToolExecutor)
    fake_chain = MagicMock(spec=IGuardrailChain)

    container = make_runtime_container(
        llm=fake_llm,
        telemetry_sink=fake_sink,
        tool_executor=fake_executor,
        guardrail_chain=fake_chain,
    )
    assert container.resolve(ILLMProvider) is fake_llm
    assert container.resolve(ITelemetrySink) is fake_sink
    assert container.resolve(IToolExecutor) is fake_executor
    assert container.resolve(IGuardrailChain) is fake_chain


def test_runtime_container_does_not_freeze_anything() -> None:
    """Unlike default_container, runtime layers must be able to rebind anything."""
    container = make_runtime_container()
    replacement = MagicMock(spec=ITelemetrySink)
    container.bind(ITelemetrySink, replacement)
    assert container.resolve(ITelemetrySink) is replacement


def test_runtime_container_omits_llm_when_none() -> None:
    """When llm=None, ILLMProvider is unbound; resolve raises LookupError."""
    container = make_runtime_container()
    with pytest.raises(LookupError):
        container.resolve(ILLMProvider)


def test_runtime_container_accepts_extra_bindings() -> None:
    class OpaqueProtocol:
        pass

    impl = OpaqueProtocol()
    container = make_runtime_container(extra_bindings={OpaqueProtocol: impl})
    assert container.resolve(OpaqueProtocol) is impl
