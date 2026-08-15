"""Shared pytest fixtures.

`make_container` and `stub_llm` were originally agent-suite-only fixtures
(tests/agent/conftest.py); Phase 3.5 lifts them to the top-level conftest
so cross-suite tests (e.g. tests/test_phase35_exit.py) can use them too.
The fixture bodies remain in tests/agent/conftest.py; we just re-export
the factory callables here.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voussoir.container import Container


def pytest_configure(config: pytest.Config) -> None:
    """Clean stale `.coverage` when running without `--cov` (v1.0.3 housekeeping).

    `tests/test_phase3_exit.py::test_coverage_floor_phase3_packages` reads the
    repo-root `.coverage` file and asserts a >=85% floor on Phase-3 packages.
    The test already skips when the file is missing, but a partial `.coverage`
    left over from a previous `make ci` run would otherwise be read by this
    test under a plain `pytest --no-cov` and trip the floor assertion against
    stale/partial data.

    Solution: when pytest-cov is NOT actively collecting coverage this session,
    remove the stale file at session-start so the existing `pytest.skip` guard
    fires. When `--cov=...` IS passed (the `make ci` path), `cov_source` is
    populated and we leave the file alone — pytest-cov needs to write to it.
    """
    cov_active = bool(getattr(config.option, "cov_source", None))
    if cov_active:
        return
    cov_file = Path(__file__).resolve().parent.parent / ".coverage"
    if cov_file.exists():
        # Best-effort cleanup; if removal fails, the coverage test will
        # fail loudly downstream rather than corrupting unrelated runs.
        with contextlib.suppress(OSError):
            cov_file.unlink()


@pytest.fixture(autouse=True)
def _disable_otel_in_tests(monkeypatch):  # type: ignore[no-untyped-def]
    """Default-off OTel in the test suite (Phase 5 Task C5).

    `voussoir.observability.tracer.configure_otel(c)` would otherwise install
    a ConsoleSpanExporter that prints every span at run-end. With ~15 spans
    per Agent.run after C3 lands, a 700+ test suite floods stdout with span
    noise. Tests that need real spans/metrics (e.g. span-hierarchy assertion
    tests in C3) override via their own fixtures that unset this var and
    install an InMemorySpanExporter.
    """
    monkeypatch.setenv("VOUSSOIR_OTEL_DISABLED", "1")


# Sentinel dummy used by _ensure_llm_key. Never a live credential; the live
# integration tests skip when the key matches this prefix.
_DUMMY_API_KEY = "sk-ant-test-only"


@pytest.fixture(autouse=True)
def _ensure_llm_key(monkeypatch):  # type: ignore[no-untyped-def]
    """Provide a dummy ANTHROPIC_API_KEY when neither provider key is set.

    `default_container()` raises RuntimeError when no LLM key is present, so any
    test that constructs a default container needs a key. Setting a dummy here
    (only when the environment is otherwise clean) keeps the suite runnable
    without secrets. Tests that assert "raises when no key" delete both keys
    themselves, which wins because this fixture runs first.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", _DUMMY_API_KEY)


@pytest.fixture(autouse=True)
def reset_dispatch_contextvars():  # type: ignore[no-untyped-def]
    """Reset dispatch ContextVars before and after every test.

    Tranche A's gather-based dispatch always restores them on the happy
    path, but exception paths and out-of-band ContextVar.set() calls in
    tests can leak state into the next test.
    """
    from voussoir.agent.dispatch import last_sub_result_var, parent_ctx_var

    parent_ctx_var.set(None)
    last_sub_result_var.set(None)
    try:
        yield
    finally:
        parent_ctx_var.set(None)
        last_sub_result_var.set(None)


@pytest.fixture(autouse=True)
def reset_warn_once_flags():  # type: ignore[no-untyped-def]
    """Reset module-level warn-once flags between tests (v1.0.4 E8).

    `voussoir.auth.authorizers.allow_all._warned` and
    `voussoir.executors.standard._fallback_warned` are process-scoped
    "log on first invocation" sentinels. Without a per-test reset, the
    first test that touches either path latches the flag to True for
    the rest of the session — any later test that asserts the warning
    fires becomes execution-order-dependent. Reset before AND after so
    unrelated tests in subsequent files also see a clean slate.

    We also re-bind each module's structlog logger to a fresh lazy
    proxy. `voussoir.observability.logging_setup.configure_logging`
    installs structlog with `cache_logger_on_first_use=True`. Once any
    test exercises a code path that lands a `.warning()` call after that
    configuration, the lazy proxy caches its wrapped stdlib logger;
    `structlog.testing.capture_logs()` cannot intercept the cached
    instance. Re-issuing `structlog.get_logger(__name__)` produces a
    fresh proxy with `_logger=None`, so the next `.warning()` resolves
    through whatever processor chain is currently active.
    """
    import structlog

    import voussoir.auth.authorizers.allow_all as _allow_all
    import voussoir.executors.standard as _standard

    _allow_all._warned = False
    _standard._fallback_warned = False
    _allow_all._log = structlog.get_logger(_allow_all.__name__)
    _standard._fallback_log = structlog.get_logger(_standard.__name__)
    try:
        yield
    finally:
        _allow_all._warned = False
        _standard._fallback_warned = False
        _allow_all._log = structlog.get_logger(_allow_all.__name__)
        _standard._fallback_log = structlog.get_logger(_standard.__name__)


@pytest.fixture
def make_mini_delegate():  # type: ignore[no-untyped-def]
    """Factory for tiny IDelegate-conforming instances. Call as `make_mini_delegate(name, output="ok")`.

    Structural (duck-typed) satisfaction of IDelegate — matches the
    convention used in tests/agent/test_delegate_protocol.py and
    tests/agent/test_plugins.py. Avoids ~5-10 lines of inline class
    definition per test.
    """
    from voussoir.agent.context import AgentContext
    from voussoir.agent.result import AgentResult

    class _MiniDelegate:
        def __init__(self, name: str, output: str = "ok") -> None:
            self.name = name
            self.description = f"mini delegate {name}"
            self._output = output

        async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
            return AgentResult[str](
                output=self._output,
                trace_id=f"trace-{self.name}",
                steps=[],
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                duration_ms=0.0,
                delegation_chain=[self.name],
                cascade_history=[],
                guardrail_decisions=[],
                finish_reason="completed",
            )

    return _MiniDelegate


@pytest.fixture
def freeze_time(monkeypatch):  # type: ignore[no-untyped-def]
    """Freeze time.time() to a known value so expiry-related tests are deterministic."""
    import time

    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    yield 1_700_000_000.0


@pytest.fixture
def fresh_env(monkeypatch):  # type: ignore[no-untyped-def]
    """Strip VOUSSOIR_A2A_* env vars so each test starts clean."""
    import os

    for key in list(os.environ):
        if key.startswith("VOUSSOIR_A2A_"):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


@pytest.fixture
def make_container() -> Callable[..., Container]:
    """Thin wrapper around voussoir.testing.make_container (v1.1.0 F11).

    Returns the public helper callable so existing tests calling
    ``make_container(llm)`` continue to work unchanged.
    """
    from voussoir.testing import make_container as helper

    return helper  # type: ignore[return-value]


@pytest.fixture
def stub_llm() -> Callable[..., MagicMock]:
    """Thin wrapper around voussoir.testing.stub_llm (v1.1.0 F11).

    Returns the public helper callable so existing tests calling
    ``stub_llm(content=...)``, ``stub_llm(side_effect=BoomError)``,
    or ``stub_llm(raw_response=..., name="openai")`` continue to work.
    """
    from voussoir.testing import stub_llm as helper

    return helper
