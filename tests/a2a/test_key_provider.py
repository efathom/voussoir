"""Phase 4.5a — EnvKeyProvider safety against accidental ephemeral secrets."""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from voussoir.a2a.keys import EnvKeyProvider


def test_default_allow_ephemeral_is_false() -> None:
    """Phase 4.5a default: ephemeral generation disabled."""
    provider = EnvKeyProvider()
    assert provider._allow_ephemeral is False  # noqa: SLF001 — internal-state check


def test_no_secret_and_no_ephemeral_raises(fresh_env) -> None:
    """Without VOUSSOIR_A2A_JWT_SECRET and default allow_ephemeral=False,
    jwt_secret() raises with an actionable message."""
    provider = EnvKeyProvider()
    with pytest.raises(RuntimeError, match="VOUSSOIR_A2A_JWT_SECRET"):
        provider.jwt_secret()


def test_no_secret_with_allow_ephemeral_warns_via_logger(fresh_env) -> None:
    """With allow_ephemeral=True and no env secret, jwt_secret() returns
    a generated secret AND emits a structured warning. Crucially: does
    NOT print to stderr (the pre-Phase-4.5a behavior leaked secrets to
    docker/k8s log aggregation)."""
    provider = EnvKeyProvider(allow_ephemeral=True)
    with capture_logs() as cap_logs:
        secret = provider.jwt_secret()
    assert len(secret) == 32  # bytes
    assert any(log.get("event") == "ephemeral_jwt_secret_generated" for log in cap_logs)


def test_env_var_secret_used_when_set(monkeypatch, fresh_env) -> None:
    """With env var set, allow_ephemeral default still works because we
    never fall through to the ephemeral path. The env var is base64-encoded
    per the existing contract."""
    import base64

    raw = b"my-test-secret-32bytes-padded-ok!"
    monkeypatch.setenv("VOUSSOIR_A2A_JWT_SECRET", base64.b64encode(raw).decode())
    provider = EnvKeyProvider()  # default allow_ephemeral=False is fine
    secret = provider.jwt_secret()
    assert secret == raw
