"""Phase 1 exit-criteria gates. If any fail, Phase 1 is not done."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse


def test_top_level_imports():
    from voussoir import Agent, AgentResult, Container, Scope  # noqa: F401


def test_tool_decorator_importable():
    from voussoir.tools import Capability, tool

    @tool(capability=Capability.READ_PUBLIC)
    async def x(q: str) -> str:
        return q

    assert x.name == "x"


async def test_run_against_stub_provider():
    """Exit criterion #1 (mocked): Agent.run() returns AgentResult against any ILLMProvider."""
    from voussoir import Agent, Container
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import ILLMProvider as ILLMProviderProto
    from voussoir.protocols import IMemoryStore, ISessionStore

    p = MagicMock(spec=ILLMProvider)
    p.chat = AsyncMock(
        return_value=LLMResponse(
            content="phase1",
            model="stub",
            input_tokens=3,
            output_tokens=1,
            finish_reason="end_turn",
        )
    )
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())

    agent = Agent(name="phase1", container=c)
    result = await agent.run("ping")
    assert result.output == "phase1"
    assert result.delegation_chain == []
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    assert result.duration_ms >= 0


def test_default_middleware_classes_importable():
    """Exit criterion #4: built-in middleware classes are present."""
    from voussoir.agent import (
        BudgetMiddleware,
        LoggingMiddleware,
        RetryMiddleware,
    )

    assert all([BudgetMiddleware, LoggingMiddleware, RetryMiddleware])


def test_streaming_method_exists():
    """Exit criterion #3: Agent.stream is defined."""
    import inspect

    from voussoir import Agent

    assert hasattr(Agent, "stream")
    assert inspect.isasyncgenfunction(Agent.stream) or inspect.iscoroutinefunction(Agent.stream)


@pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ
    or os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-test"),
    reason="exit criterion #1 (live) — requires a real ANTHROPIC_API_KEY",
)
async def test_run_against_real_anthropic():
    """Exit criterion #1 (live): the 5-line demo against the real provider.

    Pinned to claude-haiku-4-5-20251001: the default opus-4-7 routinely returns
    HTTP 529 (Overloaded) under load and would make this gate flaky.
    """
    from voussoir import Agent
    from voussoir.container.defaults import default_container

    agent = Agent(
        name="live",
        instructions="Be terse.",
        model="claude-haiku-4-5-20251001",
        container=default_container(),
    )
    result = await agent.run("Say 'pong'.")
    assert "pong" in result.output.lower()
    assert result.finish_reason == "completed"


def test_coverage_floor_voussoir_agent():
    """Exit criterion #6: ≥85% on voussoir/agent/, voussoir/executors/standard.py, voussoir/tools/.

    Coverage is enforced by `make ci` running pytest --cov; this test reads the
    most-recent coverage data and asserts the floor. Skipped if no .coverage file.
    """
    pytest.importorskip("coverage")
    from pathlib import Path

    import coverage as cov_mod

    cov_file = Path(__file__).resolve().parent.parent / ".coverage"
    if not cov_file.exists():
        pytest.skip("no .coverage file — run `make ci` first")
    cov = cov_mod.Coverage(data_file=str(cov_file))
    cov.load()
    paths = [
        "src/voussoir/agent",
        "src/voussoir/executors/standard.py",
        "src/voussoir/tools",
    ]
    # Aggregate coverage across the listed paths; ≥85% required.
    total_stmts = 0
    total_miss = 0
    for f in cov.get_data().measured_files():
        if any(p in f for p in paths):
            _, stmts, miss, _ = cov.analysis2(f)[:4]
            total_stmts += len(stmts)
            total_miss += len(miss)
    if total_stmts == 0:
        pytest.skip("no measured files match paths")
    pct = (total_stmts - total_miss) / total_stmts
    assert pct >= 0.85, f"coverage {pct:.1%} on agent/executors/tools is below 85% floor"
