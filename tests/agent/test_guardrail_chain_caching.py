"""Locks D8 fixes: cached guardrail chain through ctx + public count() helper.

Before D8, _dispatch_one re-resolved DefaultGuardrailChain from the
container per tool call. If something rebinds the chain mid-run, the
agent's input/output screening kept using its cached chain while tool-
call screening picked up the new one. Asymmetric enforcement.

Also: agent.py reached into chain._by_stage (a private attr) to count
guardrails — the public count() helper closes that boundary violation.
"""

from __future__ import annotations

import inspect

from voussoir.guardrails import DefaultGuardrailChain, GuardrailVerdict, IGuardrailChain


class _FakeGuardrail:
    """Minimal duck-typed Guardrail used for chain population tests."""

    name = "fake"
    stage = "input"

    async def screen(self, payload: object, ctx: object) -> GuardrailVerdict:
        return GuardrailVerdict(verdict="ALLOW")


def test_chain_count_is_public_helper() -> None:
    """DefaultGuardrailChain.count() exists and returns int."""
    empty = DefaultGuardrailChain([])
    assert empty.count() == 0

    populated = DefaultGuardrailChain([_FakeGuardrail()])  # type: ignore[list-item]
    assert populated.count() == 1


def test_agent_uses_count_not_private_by_stage() -> None:
    """Agent code must call chain.count() — not reach into chain._by_stage."""
    from voussoir.agent import agent as agent_mod

    source = inspect.getsource(agent_mod)
    assert "_by_stage" not in source, (
        "agent.py reaches into DefaultGuardrailChain._by_stage — use the "
        "public count() helper instead."
    )


def test_agent_context_has_guardrail_chain_field() -> None:
    """AgentContext exposes guardrail_chain so dispatch can use the cached
    chain instead of re-resolving from the container per tool call."""
    from dataclasses import fields

    from voussoir.agent.context import AgentContext

    field_names = {f.name for f in fields(AgentContext)}
    assert "guardrail_chain" in field_names


def test_dispatch_reads_chain_from_ctx_not_container() -> None:
    """_dispatch_one reads ctx.guardrail_chain — never re-resolves from
    ctx.container.resolve(DefaultGuardrailChain).
    """
    from voussoir.agent import dispatch as dispatch_mod

    source = inspect.getsource(dispatch_mod._dispatch_one)
    assert "container.resolve(DefaultGuardrailChain" not in source, (
        "_dispatch_one still re-resolves DefaultGuardrailChain per tool call; "
        "read ctx.guardrail_chain instead."
    )
    assert (
        "ctx.guardrail_chain" in source
    ), "_dispatch_one must read the cached chain from ctx.guardrail_chain."


async def test_rebind_mid_run_does_not_change_dispatch_chain(
    make_container,  # type: ignore[no-untyped-def]
) -> None:
    """A mid-run container.bind on DefaultGuardrailChain doesn't affect the
    chain used by _dispatch_one (it uses ctx.guardrail_chain, not container
    lookup).

    Constructs AgentContext via the .open() async-context-manager (the
    canonical entry path), threads in a specific guardrail chain, then
    rebinds the container's chain to a different instance after open and
    asserts ctx.guardrail_chain is still the original cached one.
    """
    from voussoir.agent.context import AgentContext

    original_chain = DefaultGuardrailChain([])
    container = make_container()
    container.bind(IGuardrailChain, original_chain)  # type: ignore[type-abstract]

    async with await AgentContext.open(
        container=container,
        guardrail_chain=original_chain,
    ) as ctx:
        # Rebind the container's chain to a different instance:
        new_chain = DefaultGuardrailChain([_FakeGuardrail()])  # type: ignore[list-item]
        container.bind(IGuardrailChain, new_chain)  # type: ignore[type-abstract]

        # ctx.guardrail_chain must still be `original_chain`, not new_chain.
        assert ctx.guardrail_chain is original_chain
        assert ctx.guardrail_chain is not new_chain
