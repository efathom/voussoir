"""Locks the AuthzDecision audit-log record + extra-forbid (Phase 6 Task A1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voussoir.auth import AuthzDecision


def test_decision_allow_minimal():
    d = AuthzDecision(decision="ALLOW")
    assert d.decision == "ALLOW"
    assert d.reason == ""
    assert d.masked_fields == []
    assert d.authorizer_name == ""


def test_decision_deny_with_reason():
    d = AuthzDecision(decision="DENY", reason="forbidden", authorizer_name="role")
    assert d.decision == "DENY"
    assert d.reason == "forbidden"


def test_decision_mask_with_fields():
    d = AuthzDecision(decision="MASK", masked_fields=["email", "ssn"])
    assert d.masked_fields == ["email", "ssn"]


def test_decision_extra_forbidden():
    with pytest.raises(ValidationError):
        AuthzDecision(decision="ALLOW", bogus="x")
