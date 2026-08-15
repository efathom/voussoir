"""KeyProvider Protocol + EnvKeyProvider behavior."""

from __future__ import annotations

from voussoir.a2a.keys import EnvKeyProvider, KeyProvider, NoCardSigningKeyError


def test_envkey_provider_satisfies_protocol(fresh_env) -> None:
    p = EnvKeyProvider()
    assert isinstance(p, KeyProvider)


def test_envkey_jwt_secret_from_env(fresh_env) -> None:
    fresh_env.setenv("VOUSSOIR_A2A_JWT_SECRET", "dGVzdC1zZWNyZXQ=")  # base64 "test-secret"
    p = EnvKeyProvider()
    assert p.jwt_secret() == b"test-secret"


def test_envkey_jwt_secret_autogen_when_unset(fresh_env) -> None:
    """Phase 4.5a P0 #5: no env var + allow_ephemeral=True → generate a
    random secret and emit a STRUCTURED LOG warning (no longer stderr-prints
    the secret, which would leak it into log aggregation).
    """
    from structlog.testing import capture_logs

    p = EnvKeyProvider(allow_ephemeral=True)
    with capture_logs() as cap_logs:
        secret = p.jwt_secret()
    assert isinstance(secret, bytes)
    assert len(secret) >= 32
    assert any(log.get("event") == "ephemeral_jwt_secret_generated" for log in cap_logs)


def test_envkey_jwt_algorithm_default(fresh_env) -> None:
    assert EnvKeyProvider().jwt_algorithm() == "HS256"


def test_envkey_jwt_algorithm_override(fresh_env) -> None:
    fresh_env.setenv("VOUSSOIR_A2A_JWT_ALGORITHM", "HS512")
    assert EnvKeyProvider().jwt_algorithm() == "HS512"


def test_envkey_card_signing_jwk_autogen(fresh_env) -> None:
    """No key-path env var → generate an ephemeral RSA-2048 keypair."""
    p = EnvKeyProvider()
    jwk = p.card_signing_jwk()
    assert jwk["kty"] == "RSA"
    assert "d" in jwk  # private exponent — private JWK
    assert "n" in jwk and "e" in jwk
    assert "kid" in jwk and jwk["kid"]


def test_envkey_card_signing_jwk_same_instance_consistent(fresh_env) -> None:
    """Same EnvKeyProvider instance returns the same keypair across calls."""
    p = EnvKeyProvider()
    jwk1 = p.card_signing_jwk()
    jwk2 = p.card_signing_jwk()
    assert jwk1["kid"] == jwk2["kid"]
    assert jwk1["n"] == jwk2["n"]


def test_envkey_card_verification_jwks_strips_private(fresh_env) -> None:
    """JWKS endpoint returns public-only keys (no `d`, `p`, `q`, etc.)."""
    p = EnvKeyProvider()
    p.card_signing_jwk()  # force keygen
    jwks = p.card_verification_jwks()
    assert "keys" in jwks
    assert len(jwks["keys"]) == 1
    public = jwks["keys"][0]
    for private_field in ("d", "p", "q", "dp", "dq", "qi"):
        assert private_field not in public
    assert public["kty"] == "RSA"
    assert "n" in public and "e" in public
    assert "kid" in public


def test_envkey_card_signing_from_pem_file(fresh_env, tmp_path) -> None:
    """VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH points at a PEM file."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(pem)
    fresh_env.setenv("VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH", str(key_path))

    p = EnvKeyProvider()
    jwk = p.card_signing_jwk()
    assert jwk["kty"] == "RSA"


def test_no_card_signing_key_error_is_exception() -> None:
    """NoCardSigningKeyError is a real exception class for custom providers."""
    assert issubclass(NoCardSigningKeyError, Exception)


def test_default_container_binds_key_provider(monkeypatch) -> None:
    """default_container() binds an EnvKeyProvider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from voussoir.container.defaults import default_container

    c = default_container()
    p = c.resolve(KeyProvider)
    assert isinstance(p, EnvKeyProvider)
