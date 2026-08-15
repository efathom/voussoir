"""Locks IGuardrailChain Protocol + container binding (v1.1.0 F2)."""

from __future__ import annotations

from voussoir.guardrails import (
    DefaultGuardrailChain,
    GuardrailPayload,
    GuardrailVerdict,
    IGuardrailChain,
)


def test_igc_is_runtime_checkable():
    assert isinstance(DefaultGuardrailChain([]), IGuardrailChain)


def test_custom_chain_satisfies_protocol():
    class _RemoteChain:
        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            return GuardrailVerdict(verdict="ALLOW")

        def count(self) -> int:
            return 0

    assert isinstance(_RemoteChain(), IGuardrailChain)


def test_default_container_binds_chain_via_protocol():
    from voussoir.container.defaults import default_container

    c = default_container()
    resolved = c.resolve(IGuardrailChain)
    assert isinstance(resolved, DefaultGuardrailChain)


def test_default_container_chain_binds_standard_profile_by_default():
    """The default chain is no longer empty — default_container() wires the
    'standard' profile so the framework is safe by default (length caps,
    injection heuristic, exfil scan)."""
    from voussoir.container.defaults import default_container

    c = default_container()
    chain = c.resolve(IGuardrailChain)
    assert chain.count() > 0
