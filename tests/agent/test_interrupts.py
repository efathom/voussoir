"""Tests for InterruptRequest + IInterruptable Protocol (v1.2 G1)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from voussoir import IInterruptable, InterruptRequest
from voussoir.errors import VoussoirError


def test_interrupt_request_is_voussoir_error() -> None:
    exc = InterruptRequest(kind="approve", payload={"amount": 100})
    assert isinstance(exc, VoussoirError)
    assert isinstance(exc, Exception)


def test_interrupt_request_carries_kind_and_payload() -> None:
    exc = InterruptRequest(
        kind="manager_approval_required",
        payload={"amount": 10_000, "vendor": "ACME"},
    )
    assert exc.kind == "manager_approval_required"
    assert exc.payload == {"amount": 10_000, "vendor": "ACME"}


def test_interrupt_request_message_includes_kind() -> None:
    exc = InterruptRequest(kind="needs_human", payload={})
    assert "needs_human" in str(exc)


def test_interrupt_request_defensive_copies_payload() -> None:
    src: dict[str, Any] = {"amount": 5}
    exc = InterruptRequest(kind="x", payload=src)
    src["amount"] = 999
    assert exc.payload == {"amount": 5}


def test_interrupt_request_accepts_mapping_proxy() -> None:
    payload: Mapping[str, Any] = MappingProxyType({"k": "v"})
    exc = InterruptRequest(kind="x", payload=payload)
    assert exc.payload == {"k": "v"}


class _ConformingImpl:
    async def interrupt(self, kind: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"approved": True}


def test_iinterruptable_is_runtime_checkable() -> None:
    assert isinstance(_ConformingImpl(), IInterruptable)


def test_iinterruptable_runtime_check_rejects_non_conforming() -> None:
    class _Missing:
        pass

    assert not isinstance(_Missing(), IInterruptable)


@pytest.mark.asyncio
async def test_iinterruptable_runtime_impl_returns_mapping() -> None:
    sentinel = _ConformingImpl()
    result = await sentinel.interrupt("x", {})
    assert result == {"approved": True}
