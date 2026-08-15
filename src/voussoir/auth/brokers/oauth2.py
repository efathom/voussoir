"""OAuth2CredentialBroker — manages OAuth2 access tokens with refresh-token loop.

Use this when the upstream service requires OAuth2 bearer tokens.  The broker
caches the current access token in memory (and optionally in a
``KeychainCredentialBroker``) and handles the refresh-token grant automatically
when the token nears expiry.  It does NOT bootstrap a brand-new grant (that
requires an interactive flow); instead, seed it with an initial ``Credentials``
via ``store_initial`` after obtaining the first token out-of-band.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx

from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials
from voussoir.auth.errors import MissingCredentialError
from voussoir.auth.principal import Principal
from voussoir.tools.protocol import ToolContext

if TYPE_CHECKING:
    from voussoir.auth.brokers.keychain import KeychainCredentialBroker

_DEFAULT_REFRESH_BUFFER_S = 60


class OAuth2CredentialBroker:
    """Manage OAuth2 access tokens with automatic refresh-token renewal.

    Use this when the upstream service uses OAuth2 bearer tokens and you
    have a refresh token available.  Seed the broker with an initial
    ``Credentials`` object (obtained via an interactive or device-code flow
    out-of-band) using ``store_initial``; thereafter ``resolve`` returns the
    cached token, refreshing it transparently when it approaches expiry.

    Constructor args:
        client_id: OAuth2 client identifier.
        client_secret: OAuth2 client secret.
        token_url: The token endpoint URL (e.g. ``https://auth.example.com/oauth/token``).
        scopes: List of OAuth2 scope strings to request on refresh.
        refresh_buffer_s: Seconds before nominal expiry at which the token is
            treated as expired (default 60 s).
        http_client: Optional pre-constructed ``httpx.AsyncClient``; injected in
            tests to avoid real network calls.
        keychain_service: When set, the broker persists the new refresh token to the
            OS keychain under this service name after every successful refresh.  Pass
            this when you want refresh tokens to survive process restarts on developer
            workstations.  Has no effect in headless / CI environments where
            ``keyring`` is unavailable.
        keychain_broker: Optional pre-constructed ``KeychainCredentialBroker`` to use
            for keychain persistence.  If ``None`` and ``keychain_service`` is set, a
            default ``KeychainCredentialBroker()`` is constructed lazily on first use.
            Inject a stub here in tests to avoid touching the real OS keychain.
    """

    name = "oauth2"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scopes: list[str],
        refresh_buffer_s: int = _DEFAULT_REFRESH_BUFFER_S,
        http_client: httpx.AsyncClient | None = None,
        keychain_service: str | None = None,
        keychain_broker: KeychainCredentialBroker | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._scopes = scopes
        self._buffer = refresh_buffer_s
        self._http_client = http_client
        self._keychain_service = keychain_service
        self._keychain_broker = keychain_broker
        self._cached: Credentials | None = None
        # Serialize concurrent refresh attempts. Multiple concurrent tool calls
        # hitting an expired token would otherwise each fire a refresh-token grant
        # in parallel; standard OAuth2 servers invalidate the refresh_token on
        # first use, so the second-through-Nth grants would fail with
        # invalid_grant and break the whole agent run. Lock is constructed lazily
        # on first acquire so the broker is safe to instantiate outside an event
        # loop.
        self._refresh_lock: asyncio.Lock | None = None

    def store_initial(self, creds: Credentials) -> None:
        """Seed the broker with an initial ``Credentials`` obtained out-of-band.

        Call this once after completing an interactive / device-code / client-
        credentials grant to allow subsequent ``resolve`` calls to serve from
        cache and auto-refresh.
        """
        self._cached = creds

    def _is_expired(self, creds: Credentials) -> bool:
        if creds.expires_at is None:
            return False
        return time.time() > (creds.expires_at - self._buffer)

    async def _do_refresh(self, refresh_token: str) -> Credentials:
        """Execute the OAuth2 refresh-token grant and return new Credentials."""
        data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scopes:
            data["scope"] = " ".join(self._scopes)

        if self._http_client is not None:
            resp = await self._http_client.post(self._token_url, data=data)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        else:
            async with httpx.AsyncClient() as client:
                r = await client.post(self._token_url, data=data)
                r.raise_for_status()
                payload = r.json()

        access_token: str = payload["access_token"]
        new_refresh: str = payload.get("refresh_token", refresh_token)
        expires_in: int = int(payload.get("expires_in", 3600))
        expires_at = time.time() + expires_in

        return Credentials(
            auth_type=AuthType.OAUTH2,
            headers={"Authorization": f"Bearer {access_token}"},
            expires_at=expires_at,
            metadata={"refresh_token": new_refresh},
        )

    def _get_lock(self) -> asyncio.Lock:
        """Lazily construct the refresh lock on first acquire."""
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        if self._cached is None:
            raise MissingCredentialError(
                "OAuth2CredentialBroker has no cached credentials; "
                "call store_initial() with an initial token first."
            )

        # Fast path: cache is fresh, no lock needed.
        if not self._is_expired(self._cached):
            return self._cached

        # Slow path: acquire the lock and re-check (another task may have
        # refreshed while we were waiting). This is the canonical async
        # double-checked-locking pattern.
        async with self._get_lock():
            if self._cached is None or self._is_expired(self._cached):
                self._cached = await self._refresh_locked(self._cached)
            return self._cached

    async def refresh(self, creds: Credentials) -> Credentials:
        """Perform the OAuth2 refresh-token grant and update cached credentials.

        Concurrent calls are serialized so only one refresh-token grant fires
        per expiry window. Raises ``MissingCredentialError`` if *creds* does
        not contain a ``refresh_token`` in its metadata dict.
        """
        async with self._get_lock():
            # If another concurrent caller already refreshed past `creds`, return
            # the newer cached value rather than firing a second grant that would
            # invalidate the first one's refresh_token.
            if (
                self._cached is not None
                and self._cached is not creds
                and not self._is_expired(self._cached)
            ):
                return self._cached
            self._cached = await self._refresh_locked(creds)
            return self._cached

    async def _refresh_locked(self, creds: Credentials | None) -> Credentials:
        """Execute the refresh-token grant. MUST be called with self._refresh_lock held."""
        if creds is None:
            raise MissingCredentialError(
                "Cannot refresh OAuth2 credentials: no cached credentials to refresh from"
            )
        refresh_token: str | None = creds.metadata.get("refresh_token")
        if not refresh_token:
            raise MissingCredentialError(
                "Cannot refresh OAuth2 credentials: no refresh_token in metadata"
            )

        new_creds = await self._do_refresh(refresh_token)

        if self._keychain_service:
            new_refresh_token: str | None = new_creds.metadata.get("refresh_token")
            if new_refresh_token:
                from voussoir.auth.brokers.keychain import KeychainCredentialBroker  # noqa: PLC0415

                broker = self._keychain_broker or KeychainCredentialBroker()
                broker.store(self._keychain_service, new_refresh_token)

        return new_creds
