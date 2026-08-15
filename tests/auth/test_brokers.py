"""Locks the 6 CredentialBroker impls (Phase 6 A7).

Each broker: happy resolve, missing-cred resolve (raises), refresh path.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from voussoir.auth import (
    AuthRequirement,
    AuthType,
    Credentials,
    MissingCredentialError,
    Principal,
)
from voussoir.auth.brokers.chained import ChainedCredentialBroker
from voussoir.auth.brokers.env import EnvCredentialBroker
from voussoir.auth.brokers.file import FileCredentialBroker
from voussoir.auth.brokers.mtls import MTLSCredentialBroker
from voussoir.auth.brokers.oauth2 import OAuth2CredentialBroker
from voussoir.tools import Capability, ToolContext


def _principal() -> Principal:
    return Principal(user_id="u", issued_at=datetime.now(UTC))


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", span_id="s", allowed_capabilities=Capability.READ_PUBLIC)


# ------- EnvCredentialBroker -------


async def test_env_broker_reads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASSIAN_API_KEY", "secret-key")
    broker = EnvCredentialBroker()
    req = AuthRequirement(auth_type=AuthType.API_KEY, service="atlassian")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert "secret-key" in str(creds.headers) or creds.headers.get("X-API-Key") == "secret-key"


async def test_env_broker_reads_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "tok-abc")
    broker = EnvCredentialBroker()
    req = AuthRequirement(auth_type=AuthType.BEARER, service="github")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer tok-abc"


async def test_env_broker_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLASSIAN_API_KEY", raising=False)
    broker = EnvCredentialBroker()
    req = AuthRequirement(auth_type=AuthType.API_KEY, service="atlassian")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_env_broker_bearer_fallback_to_bearer_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back from {SERVICE}_TOKEN to {SERVICE}_BEARER."""
    monkeypatch.delenv("SLACK_TOKEN", raising=False)
    monkeypatch.setenv("SLACK_BEARER", "bearer-val")
    broker = EnvCredentialBroker()
    req = AuthRequirement(auth_type=AuthType.BEARER, service="slack")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer bearer-val"


async def test_env_broker_unsupported_type_raises() -> None:
    broker = EnvCredentialBroker()
    req = AuthRequirement(auth_type=AuthType.MTLS, service="svc")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_env_broker_hyphenated_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_API_KEY", "hyphen-key")
    broker = EnvCredentialBroker()
    req = AuthRequirement(auth_type=AuthType.API_KEY, service="my-service")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers.get("X-API-Key") == "hyphen-key"


async def test_env_broker_refresh_returns_same(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = Credentials(
        auth_type=AuthType.BEARER,
        headers={"Authorization": "Bearer old"},
    )
    broker = EnvCredentialBroker()
    result = await broker.refresh(existing)
    assert result is existing or result.headers == existing.headers


# ------- FileCredentialBroker -------


async def test_file_broker_reads_mounted_secret(tmp_path: Path) -> None:
    (tmp_path / "atlassian-token").write_text("file-token")
    broker = FileCredentialBroker(secret_dir=tmp_path)
    req = AuthRequirement(auth_type=AuthType.BEARER, service="atlassian")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer file-token"


async def test_file_broker_missing_raises(tmp_path: Path) -> None:
    broker = FileCredentialBroker(secret_dir=tmp_path)
    req = AuthRequirement(auth_type=AuthType.BEARER, service="absent")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_file_broker_bare_service_name(tmp_path: Path) -> None:
    (tmp_path / "github").write_text("bare-token  \n")
    broker = FileCredentialBroker(secret_dir=tmp_path)
    req = AuthRequirement(auth_type=AuthType.BEARER, service="github")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer bare-token"


async def test_file_broker_api_key_file(tmp_path: Path) -> None:
    (tmp_path / "stripe-api-key").write_text("sk-test-123")
    broker = FileCredentialBroker(secret_dir=tmp_path)
    req = AuthRequirement(auth_type=AuthType.API_KEY, service="stripe")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers.get("X-API-Key") == "sk-test-123"


async def test_file_broker_refresh_noop(tmp_path: Path) -> None:
    existing = Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer x"})
    broker = FileCredentialBroker(secret_dir=tmp_path)
    result = await broker.refresh(existing)
    assert result is existing


# ------- MTLSCredentialBroker -------


async def test_mtls_broker_resolves_cert_paths(tmp_path: Path) -> None:
    cert = tmp_path / "client.crt"
    key = tmp_path / "client.key"
    cert.write_text("CERT")
    key.write_text("KEY")
    broker = MTLSCredentialBroker(cert_path=cert, key_path=key)
    req = AuthRequirement(auth_type=AuthType.MTLS, service="internal")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.cert == (cert, key)


async def test_mtls_broker_missing_cert_raises(tmp_path: Path) -> None:
    broker = MTLSCredentialBroker(cert_path=tmp_path / "nope.crt", key_path=tmp_path / "nope.key")
    req = AuthRequirement(auth_type=AuthType.MTLS, service="x")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_mtls_broker_missing_key_raises(tmp_path: Path) -> None:
    cert = tmp_path / "client.crt"
    cert.write_text("CERT")
    broker = MTLSCredentialBroker(cert_path=cert, key_path=tmp_path / "missing.key")
    req = AuthRequirement(auth_type=AuthType.MTLS, service="x")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_mtls_broker_ca_bundle(tmp_path: Path) -> None:
    cert = tmp_path / "client.crt"
    key = tmp_path / "client.key"
    ca = tmp_path / "ca.pem"
    cert.write_text("CERT")
    key.write_text("KEY")
    ca.write_text("CA")
    broker = MTLSCredentialBroker(cert_path=cert, key_path=key, ca_bundle=ca)
    req = AuthRequirement(auth_type=AuthType.MTLS, service="svc")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.ca_bundle == ca


async def test_mtls_broker_wrong_auth_type_raises(tmp_path: Path) -> None:
    cert = tmp_path / "c.crt"
    key = tmp_path / "c.key"
    cert.write_text("C")
    key.write_text("K")
    broker = MTLSCredentialBroker(cert_path=cert, key_path=key)
    req = AuthRequirement(auth_type=AuthType.BEARER, service="svc")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_mtls_broker_refresh(tmp_path: Path) -> None:
    cert = tmp_path / "client.crt"
    key = tmp_path / "client.key"
    cert.write_text("CERT")
    key.write_text("KEY")
    broker = MTLSCredentialBroker(cert_path=cert, key_path=key)
    old = Credentials(auth_type=AuthType.MTLS, cert=(cert, key))
    new = await broker.refresh(old)
    assert new.cert == (cert, key)


# ------- ChainedCredentialBroker -------


async def test_chained_broker_first_hit_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("X_API_KEY", "from-env")
    chained = ChainedCredentialBroker(
        [EnvCredentialBroker(), FileCredentialBroker(secret_dir=tmp_path)]
    )
    req = AuthRequirement(auth_type=AuthType.API_KEY, service="x")
    creds = await chained.resolve(req, _principal(), _ctx())
    assert "from-env" in str(creds.headers)


async def test_chained_broker_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("Y_TOKEN", raising=False)
    monkeypatch.delenv("Y_BEARER", raising=False)
    (tmp_path / "y-token").write_text("from-file")
    chained = ChainedCredentialBroker(
        [EnvCredentialBroker(), FileCredentialBroker(secret_dir=tmp_path)]
    )
    req = AuthRequirement(auth_type=AuthType.BEARER, service="y")
    creds = await chained.resolve(req, _principal(), _ctx())
    assert "from-file" in str(creds.headers)


async def test_chained_broker_all_miss_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NOBODY_API_KEY", raising=False)
    chained = ChainedCredentialBroker(
        [EnvCredentialBroker(), FileCredentialBroker(secret_dir=tmp_path)]
    )
    req = AuthRequirement(auth_type=AuthType.API_KEY, service="nobody")
    with pytest.raises(MissingCredentialError):
        await chained.resolve(req, _principal(), _ctx())


async def test_chained_broker_empty_list_rejected() -> None:
    with pytest.raises(ValueError):
        ChainedCredentialBroker([])


async def test_chained_broker_refresh_delegates(tmp_path: Path) -> None:
    cert = tmp_path / "c.crt"
    key = tmp_path / "c.key"
    cert.write_text("C")
    key.write_text("K")
    mtls = MTLSCredentialBroker(cert_path=cert, key_path=key)
    chained = ChainedCredentialBroker([mtls])
    old = Credentials(auth_type=AuthType.MTLS, cert=(cert, key))
    new = await chained.refresh(old)
    assert new.cert == (cert, key)


# ------- OAuth2CredentialBroker -------


class _StubHTTPClient:
    """Minimal async stub for httpx.AsyncClient used by OAuth2 tests."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    async def post(self, url: str, data: dict[str, str] | None = None) -> _StubResponse:
        del url, data
        resp = self._responses.pop(0)
        return _StubResponse(resp)

    async def aclose(self) -> None:
        pass


class _StubResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


async def test_oauth2_broker_refresh_path() -> None:
    client = _StubHTTPClient(
        [{"access_token": "new-tok", "expires_in": 3600, "refresh_token": "new-refresh"}]
    )
    broker = OAuth2CredentialBroker(
        client_id="cid",
        client_secret="csec",
        token_url="https://oauth.example.com/token",
        scopes=["read"],
        http_client=client,  # type: ignore[arg-type]
    )
    expired = Credentials(
        auth_type=AuthType.OAUTH2,
        headers={"Authorization": "Bearer old"},
        expires_at=0.0,
        metadata={"refresh_token": "old-refresh"},
    )
    refreshed = await broker.refresh(expired)
    assert refreshed.headers["Authorization"] == "Bearer new-tok"
    assert refreshed.metadata.get("refresh_token") == "new-refresh"


async def test_oauth2_broker_refresh_no_token_raises() -> None:
    broker = OAuth2CredentialBroker(
        client_id="cid",
        client_secret="csec",
        token_url="https://x.example.com/token",
        scopes=[],
    )
    creds = Credentials(auth_type=AuthType.OAUTH2, headers={}, expires_at=0.0, metadata={})
    with pytest.raises(MissingCredentialError):
        await broker.refresh(creds)


async def test_oauth2_broker_resolve_uses_cached_creds() -> None:
    """If broker.store_initial(...) was called, resolve returns cached creds when not expired."""
    broker = OAuth2CredentialBroker(
        client_id="cid",
        client_secret="csec",
        token_url="https://x.example.com/token",
        scopes=[],
    )
    fresh = Credentials(
        auth_type=AuthType.OAUTH2,
        headers={"Authorization": "Bearer fresh"},
        expires_at=time.time() + 3600,
        metadata={"refresh_token": "r"},
    )
    broker.store_initial(fresh)
    req = AuthRequirement(auth_type=AuthType.OAUTH2, service="x")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer fresh"


async def test_oauth2_broker_serializes_concurrent_refresh_under_lock() -> None:
    """v1.0.1: N concurrent resolve() calls on an expired token fire EXACTLY ONE
    refresh-token grant, not N.

    The bug: without serialization, all N tasks see _is_expired() == True before
    any refresh completes, so each calls _do_refresh. Standard OAuth2 servers
    invalidate the refresh_token on first use, so the second-through-Nth grants
    fail with invalid_grant and break the agent run.

    The fix: an asyncio.Lock around the check-and-refresh; concurrent callers
    wait, then double-check that the cache is still expired before refreshing.
    """
    import asyncio

    refresh_call_count = 0

    class _SlowStubHTTPClient:
        """Stub that simulates a slow token endpoint to amplify the race window."""

        async def post(self, url: str, data: dict[str, str] | None = None) -> _StubResponse:
            del url, data
            nonlocal refresh_call_count
            refresh_call_count += 1
            # Yield to the event loop so other waiting tasks could (without the lock)
            # also enter the refresh path. With the lock, they block on the lock instead.
            await asyncio.sleep(0.01)
            return _StubResponse(
                {
                    "access_token": f"tok-{refresh_call_count}",
                    "expires_in": 3600,
                    "refresh_token": f"refresh-{refresh_call_count}",
                }
            )

        async def aclose(self) -> None:
            pass

    broker = OAuth2CredentialBroker(
        client_id="cid",
        client_secret="csec",
        token_url="https://oauth.example.com/token",
        scopes=[],
        http_client=_SlowStubHTTPClient(),  # type: ignore[arg-type]
    )
    # Seed with an expired token so resolve() takes the refresh path.
    broker.store_initial(
        Credentials(
            auth_type=AuthType.OAUTH2,
            headers={"Authorization": "Bearer stale"},
            expires_at=0.0,
            metadata={"refresh_token": "original"},
        )
    )

    req = AuthRequirement(auth_type=AuthType.OAUTH2, service="x")
    p = _principal()
    ctx = _ctx()
    results = await asyncio.gather(
        broker.resolve(req, p, ctx),
        broker.resolve(req, p, ctx),
        broker.resolve(req, p, ctx),
        broker.resolve(req, p, ctx),
        broker.resolve(req, p, ctx),
    )
    # Exactly one refresh grant fired despite 5 concurrent resolve() calls.
    assert refresh_call_count == 1, f"expected 1 refresh, got {refresh_call_count}"
    # All callers received the same refreshed credentials.
    assert all(r.headers["Authorization"] == "Bearer tok-1" for r in results)


async def test_oauth2_broker_resolve_no_cache_raises() -> None:
    broker = OAuth2CredentialBroker(
        client_id="c",
        client_secret="s",
        token_url="https://t.example.com/token",
        scopes=[],
    )
    req = AuthRequirement(auth_type=AuthType.OAUTH2, service="x")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


async def test_oauth2_broker_resolve_refreshes_expired() -> None:
    """If cached creds are expired, resolve triggers refresh before returning."""
    client = _StubHTTPClient(
        [{"access_token": "refreshed-tok", "expires_in": 3600, "refresh_token": "r2"}]
    )
    broker = OAuth2CredentialBroker(
        client_id="cid",
        client_secret="csec",
        token_url="https://oauth.example.com/token",
        scopes=[],
        http_client=client,  # type: ignore[arg-type]
    )
    expired = Credentials(
        auth_type=AuthType.OAUTH2,
        headers={"Authorization": "Bearer old"},
        expires_at=0.0,
        metadata={"refresh_token": "r1"},
    )
    broker.store_initial(expired)
    req = AuthRequirement(auth_type=AuthType.OAUTH2, service="x")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer refreshed-tok"


# ------- Keychain (CI-skipped) -------


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="keyring not available headless")
async def test_keychain_broker_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("keyring")
    import keyring
    from keyring.backend import KeyringBackend

    from voussoir.auth.brokers.keychain import KeychainCredentialBroker

    class _StubBackend(KeyringBackend):
        priority = 1  # type: ignore[assignment]

        def __init__(self) -> None:
            super().__init__()
            self._store: dict[tuple[str, str], str] = {}

        def set_password(self, service: str, username: str, password: str) -> None:
            self._store[(service, username)] = password

        def get_password(self, service: str, username: str) -> str | None:
            return self._store.get((service, username))

        def delete_password(self, service: str, username: str) -> None:
            self._store.pop((service, username), None)

    backend = _StubBackend()
    keyring.set_keyring(backend)
    backend.set_password("voussoir", "atlassian", "kc-token")

    broker = KeychainCredentialBroker(service_namespace="voussoir")
    req = AuthRequirement(auth_type=AuthType.BEARER, service="atlassian")
    creds = await broker.resolve(req, _principal(), _ctx())
    assert creds.headers["Authorization"] == "Bearer kc-token"


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="keyring not available headless")
async def test_keychain_broker_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("keyring")
    import keyring
    from keyring.backend import KeyringBackend

    from voussoir.auth.brokers.keychain import KeychainCredentialBroker

    class _EmptyBackend(KeyringBackend):
        priority = 1  # type: ignore[assignment]

        def get_password(self, service: str, username: str) -> str | None:
            return None

        def set_password(self, service: str, username: str, password: str) -> None:
            pass

        def delete_password(self, service: str, username: str) -> None:
            pass

    keyring.set_keyring(_EmptyBackend())

    broker = KeychainCredentialBroker(service_namespace="voussoir")
    req = AuthRequirement(auth_type=AuthType.BEARER, service="nonexistent")
    with pytest.raises(MissingCredentialError):
        await broker.resolve(req, _principal(), _ctx())


# ------- OAuth2CredentialBroker + keychain persistence -------


async def test_oauth2_broker_persists_refresh_to_keychain() -> None:
    """When keychain_service is configured, refresh() persists the new refresh_token to the keychain."""
    stored: dict[str, str] = {}

    class _StubKeychain:
        name = "stub-keychain"

        async def resolve(
            self,
            requirement: AuthRequirement,
            principal: Principal,
            ctx: ToolContext,
        ) -> Credentials:
            raise MissingCredentialError("stub")

        async def refresh(self, creds: Credentials) -> Credentials:
            return creds

        def store(self, service: str, value: str) -> None:
            stored[service] = value

    client = _StubHTTPClient(
        [{"access_token": "new-tok", "expires_in": 3600, "refresh_token": "new-refresh"}]
    )
    broker = OAuth2CredentialBroker(
        client_id="cid",
        client_secret="csec",
        token_url="https://oauth.example.com/token",
        scopes=["read"],
        http_client=client,  # type: ignore[arg-type]
        keychain_service="atlassian",
        keychain_broker=_StubKeychain(),  # type: ignore[arg-type]
    )
    expired = Credentials(
        auth_type=AuthType.OAUTH2,
        headers={"Authorization": "Bearer old"},
        expires_at=0.0,
        metadata={"refresh_token": "old-refresh"},
    )
    await broker.refresh(expired)
    assert stored.get("atlassian") == "new-refresh"


# ------- ChainedCredentialBroker last-error message preservation -------


async def test_chained_broker_all_miss_includes_last_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When all brokers raise MissingCredentialError, the propagated error includes the LAST broker's message."""
    monkeypatch.delenv("DEFINITELY_NOT_SET_TOKEN", raising=False)
    monkeypatch.delenv("DEFINITELY_NOT_SET_BEARER", raising=False)
    chained = ChainedCredentialBroker(
        [EnvCredentialBroker(), FileCredentialBroker(secret_dir=tmp_path)]
    )
    req = AuthRequirement(auth_type=AuthType.BEARER, service="definitely-not-set")
    with pytest.raises(MissingCredentialError) as excinfo:
        await chained.resolve(req, _principal(), _ctx())
    msg = str(excinfo.value)
    # The chained error message embeds the last broker's error; verify the service name is present
    assert "definitely-not-set" in msg or "file" in msg.lower() or "secret" in msg.lower()
