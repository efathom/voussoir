from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials


def test_auth_requirement_default_refresh_true():
    r = AuthRequirement(auth_type=AuthType.OAUTH2, service="github")
    assert r.refresh is True


def test_credentials_expiry_with_buffer(freeze_time):
    c = Credentials(auth_type=AuthType.BEARER, expires_at=1_700_000_000.0 + 30)
    # 30s before expiry, with default 60s buffer → expired
    assert c.is_expired() is True
    c2 = Credentials(auth_type=AuthType.BEARER, expires_at=1_700_000_000.0 + 120)
    assert c2.is_expired() is False
