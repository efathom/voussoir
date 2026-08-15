"""Locks WireAgentResult.authz_decisions profile rules (Phase 6 A5).

Profile rules per spec §4.6:
  public  — authz_decisions is EXCLUDED (not present in wire dump)
  trusted — authz_decisions is INCLUDED
"""

from __future__ import annotations

from voussoir.a2a.wire import WireAgentResult
from voussoir.agent.result import AgentResult
from voussoir.auth import AuthzDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_agent_result(*, with_authz: bool = True) -> AgentResult[str]:
    decisions: list[AuthzDecision] = (
        [AuthzDecision(decision="ALLOW", authorizer_name="role-a5")] if with_authz else []
    )
    return AgentResult(
        output="x",
        trace_id="trace-a5",
        steps=[],
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
        duration_ms=1.0,
        delegation_chain=[],
        cascade_history=[],
        guardrail_decisions=[],
        authz_decisions=decisions,
        finish_reason="completed",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wire_public_strips_authz_decisions() -> None:
    """profile='public' → WireAgentResult dump must NOT expose authz_decisions."""
    r = _stub_agent_result(with_authz=True)
    wire_public = r.to_wire(profile="public")
    assert isinstance(wire_public, WireAgentResult)
    dump = wire_public.model_dump()
    # authz_decisions must be absent from the public wire shape.
    assert "authz_decisions" not in dump


def test_wire_trusted_includes_authz_decisions() -> None:
    """profile='trusted' → full AgentResult returned, authz_decisions is populated."""
    r = _stub_agent_result(with_authz=True)
    wire_trusted = r.to_wire(profile="trusted")
    # trusted profile returns self unchanged
    assert wire_trusted is r
    dump = wire_trusted.model_dump()
    assert "authz_decisions" in dump
    assert len(dump["authz_decisions"]) == 1
    assert dump["authz_decisions"][0]["decision"] == "ALLOW"
    assert dump["authz_decisions"][0]["authorizer_name"] == "role-a5"


def test_wire_public_with_no_authz_decisions() -> None:
    """Even when authz_decisions is empty, public profile keeps it out of the dump."""
    r = _stub_agent_result(with_authz=False)
    wire_public = r.to_wire(profile="public")
    dump = wire_public.model_dump()
    assert "authz_decisions" not in dump


def test_wire_model_fields_do_not_include_authz() -> None:
    """WireAgentResult model itself must not define an authz_decisions field."""
    assert "authz_decisions" not in WireAgentResult.model_fields
