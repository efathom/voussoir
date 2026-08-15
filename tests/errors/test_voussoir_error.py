"""v1.0.4 E6 — VoussoirError base hierarchy + top-level re-exports."""

from __future__ import annotations

import pytest


def test_voussoir_error_is_exception_subclass() -> None:
    from voussoir.errors import VoussoirError

    assert issubclass(VoussoirError, Exception)


def test_auth_error_is_voussoir_error() -> None:
    from voussoir.auth import AuthError
    from voussoir.errors import VoussoirError

    assert issubclass(AuthError, VoussoirError)


def test_delegation_error_is_voussoir_error() -> None:
    from voussoir.a2a.errors import DelegationError
    from voussoir.errors import VoussoirError

    assert issubclass(DelegationError, VoussoirError)


def test_card_verification_error_is_voussoir_error() -> None:
    from voussoir.a2a.errors import CardVerificationError
    from voussoir.errors import VoussoirError

    assert issubclass(CardVerificationError, VoussoirError)


def test_policy_violation_error_is_voussoir_error() -> None:
    from voussoir.agent.policy import PolicyViolationError
    from voussoir.errors import VoussoirError

    assert issubclass(PolicyViolationError, VoussoirError)


def test_no_card_signing_key_error_is_voussoir_error() -> None:
    from voussoir.a2a.keys import NoCardSigningKeyError
    from voussoir.errors import VoussoirError

    assert issubclass(NoCardSigningKeyError, VoussoirError)


def test_inbound_jwt_misconfig_is_voussoir_error() -> None:
    from voussoir.a2a.keys import InboundJWTVerificationNotConfiguredError
    from voussoir.errors import VoussoirError

    assert issubclass(InboundJWTVerificationNotConfiguredError, VoussoirError)


def test_voussoir_error_reexported_at_top_level() -> None:
    from voussoir import AuthError, PolicyViolation, PolicyViolationError, VoussoirError

    assert VoussoirError is not None
    assert PolicyViolationError is not None
    assert PolicyViolation is not None
    assert AuthError is not None


def test_catch_all_voussoir_errors_at_one_clause() -> None:
    """Callers can catch all voussoir errors with one except clause."""
    from voussoir.auth import MissingCredentialError
    from voussoir.errors import VoussoirError

    with pytest.raises(VoussoirError):
        raise MissingCredentialError("test")
