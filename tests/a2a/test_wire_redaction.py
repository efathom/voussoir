"""Locks A2A wire-result redaction + AgentResult.to_wire() (Phase 5 Task C6).

Phase 4.5a's `WireAgentResult.from_agent_result(...)` already redacted the wire
shape inside publisher.py. Phase 5 C6 promotes the redaction to a first-class
`AgentResult.to_wire(profile=...)` method, adds a `wire_profile=` kwarg to
`make_a2a_router`, and re-exports `WireAgentResult` at `voussoir.a2a` AND
`voussoir` top level.
"""

from __future__ import annotations

import inspect

from voussoir import AgentResult, WireAgentResult
from voussoir.a2a import WireAgentResult as A2AWireAgentResult
from voussoir.a2a import make_a2a_router


def _make_result() -> AgentResult[str]:
    return AgentResult(
        output="hello",
        trace_id="trace-1",
        steps=[],
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.0042,
        duration_ms=12.5,
        delegation_chain=["sub-1", "sub-2"],
        cascade_history=[],
        guardrail_decisions=[],
        finish_reason="completed",
    )


def test_wire_agent_result_public_re_exports():
    """WireAgentResult is accessible from both voussoir.a2a and top-level voussoir."""
    assert WireAgentResult is A2AWireAgentResult


def test_to_wire_public_strips_lineage_and_costs():
    """profile='public' returns a WireAgentResult — no steps/delegation_chain/cost."""
    result = _make_result()
    wire = result.to_wire(profile="public")
    assert isinstance(wire, WireAgentResult)
    dumped = wire.model_dump()
    assert "delegation_chain" not in dumped
    assert "steps" not in dumped
    assert "cost_usd" not in dumped
    assert "trace_id" not in dumped
    assert dumped["output"] == "hello"
    assert dumped["finish_reason"] == "completed"
    assert dumped["tokens_in"] == 10
    assert dumped["tokens_out"] == 20
    assert dumped["duration_ms"] == 12.5


def test_to_wire_trusted_returns_full_self():
    """profile='trusted' returns the AgentResult unchanged — lineage + costs visible."""
    result = _make_result()
    wire = result.to_wire(profile="trusted")
    assert wire is result
    dumped = wire.model_dump()
    assert dumped["delegation_chain"] == ["sub-1", "sub-2"]
    assert dumped["cost_usd"] == 0.0042
    assert dumped["trace_id"] == "trace-1"


def test_to_wire_defaults_to_public():
    """No-arg call uses the public profile (fail-closed default)."""
    result = _make_result()
    wire = result.to_wire()
    assert isinstance(wire, WireAgentResult)


def test_make_a2a_router_default_wire_profile_is_public():
    """make_a2a_router defaults wire_profile='public' (fail-closed)."""
    sig = inspect.signature(make_a2a_router)
    assert "wire_profile" in sig.parameters
    assert sig.parameters["wire_profile"].default == "public"
