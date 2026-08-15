from datetime import UTC, datetime, timedelta

from voussoir.auth.principal import Principal


def test_principal_minimal():
    p = Principal(user_id="alice", issued_at=datetime.now(UTC))
    assert p.user_id == "alice"
    assert p.classification == "internal"
    assert p.roles == []


def test_principal_with_roles_teams_domains():
    p = Principal(
        user_id="alice",
        roles=["incident-responder"],
        teams=["sre"],
        domains=["sre"],
        issued_at=datetime.now(UTC),
    )
    assert "incident-responder" in p.roles


def test_principal_expiry_check():
    now = datetime.now(UTC)
    p = Principal(
        user_id="alice",
        issued_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    assert p.is_expired() is True
