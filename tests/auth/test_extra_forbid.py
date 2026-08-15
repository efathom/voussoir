"""Locks extra='forbid' on Principal / AuthRequirement / Credentials (v1.0.2 D2).

These three models are user-facing; typos like Principal(roels=[...]) must
raise ValidationError, not silently drop the unknown field.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from voussoir.auth import AuthRequirement, AuthType, Credentials, Principal

_NOW = datetime(2025, 1, 1, tzinfo=UTC)


def test_principal_rejects_extra_field():
    with pytest.raises(ValidationError):
        Principal(user_id="alice", issued_at=_NOW, roles=["admin"], bogus="x")  # type: ignore[call-arg]


def test_principal_typo_caught():
    """Common typo: 'roels' instead of 'roles'."""
    with pytest.raises(ValidationError):
        Principal(user_id="alice", issued_at=_NOW, roels=["admin"])  # type: ignore[call-arg]


def test_auth_requirement_rejects_extra_field():
    with pytest.raises(ValidationError):
        AuthRequirement(auth_type=AuthType.BEARER, service="atlassian", bogus="x")  # type: ignore[call-arg]


def test_credentials_rejects_extra_field():
    with pytest.raises(ValidationError):
        Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer x"}, bogus="x")  # type: ignore[call-arg]
