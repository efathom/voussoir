"""Locks executor + guardrail_chain override precedence (v1.1.0 F4).

Precedence ladder: run-kwarg > init-kwarg > container resolve > fallback.
"""

from __future__ import annotations

from typing import Any

from voussoir import Agent
from voussoir.executors import StandardExecutor
from voussoir.guardrails import DefaultGuardrailChain


class _MarkerExecutor:
    """A StandardExecutor delegate that records which instance ran (.name)."""

    name = "marker"

    def __init__(self, label: str) -> None:
        self._label = label
        self._inner = StandardExecutor()

    async def invoke(self, tool: Any, args: Any, ctx: Any) -> Any:
        # tag the call on the ctx so the test can read it
        ctx.executor_label = self._label  # type: ignore[attr-defined]
        return await self._inner.invoke(tool, args, ctx)


# --- executor precedence tests ------------------------------------------


async def test_run_kwarg_executor_takes_precedence_over_init(
    make_container: Any, stub_llm: Any
) -> None:
    """agent.run(executor=X) wins over Agent(executor=Y)."""
    a = Agent(
        name="a",
        container=make_container(stub_llm(content="ok")),
        executor=_MarkerExecutor("init"),
    )
    run_exec = _MarkerExecutor("run")
    # Direct precedence-helper assertion (no tool call required):
    resolved = a._resolve_executor(run_kwarg=run_exec)
    assert resolved is run_exec


async def test_init_kwarg_executor_wins_over_container(make_container: Any, stub_llm: Any) -> None:
    init_exec = _MarkerExecutor("init")
    a = Agent(
        name="a",
        container=make_container(stub_llm(content="ok")),
        executor=init_exec,
    )
    resolved = a._resolve_executor(run_kwarg=None)
    assert resolved is init_exec


async def test_container_executor_used_when_no_override(make_container: Any, stub_llm: Any) -> None:
    """Default container binds StandardExecutor via IToolExecutor; that's resolved."""
    a = Agent(name="a", container=make_container(stub_llm(content="ok")))
    resolved = a._resolve_executor(run_kwarg=None)
    assert isinstance(resolved, StandardExecutor)


# --- guardrail_chain precedence tests ----------------------------------


async def test_run_kwarg_chain_takes_precedence_over_init(
    make_container: Any, stub_llm: Any
) -> None:
    init_chain = DefaultGuardrailChain([])
    run_chain = DefaultGuardrailChain([])
    assert init_chain is not run_chain
    a = Agent(
        name="a",
        container=make_container(stub_llm(content="ok")),
        guardrail_chain=init_chain,
    )
    resolved = a._resolve_guardrail_chain(run_kwarg=run_chain)
    assert resolved is run_chain


async def test_init_kwarg_chain_wins_over_container(make_container: Any, stub_llm: Any) -> None:
    init_chain = DefaultGuardrailChain([])
    a = Agent(
        name="a",
        container=make_container(stub_llm(content="ok")),
        guardrail_chain=init_chain,
    )
    resolved = a._resolve_guardrail_chain(run_kwarg=None)
    assert resolved is init_chain


async def test_container_chain_used_when_no_override(make_container: Any, stub_llm: Any) -> None:
    """Default container binds an empty chain via IGuardrailChain (F2 default)."""
    a = Agent(name="a", container=make_container(stub_llm(content="ok")))
    resolved = a._resolve_guardrail_chain(run_kwarg=None)
    assert isinstance(resolved, DefaultGuardrailChain)
    assert resolved.count() == 0
