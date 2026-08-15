"""MTLSCredentialBroker — resolves mTLS credentials from cert/key paths.

Use this when the upstream service requires mutual TLS authentication (common
in service-mesh environments like Istio/Linkerd and in zero-trust network
architectures).  The broker validates that the cert and key files exist at
resolve-time, then returns a ``Credentials`` object with the ``cert`` tuple
populated for direct use with ``httpx`` (``cert=(cert_path, key_path)``).
"""

from __future__ import annotations

from pathlib import Path

from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials
from voussoir.auth.errors import MissingCredentialError
from voussoir.auth.principal import Principal
from voussoir.tools.protocol import ToolContext


class MTLSCredentialBroker:
    """Resolve mTLS credentials from certificate and key file paths.

    Use this when the downstream service requires client-certificate
    authentication.  Pass ``cert_path`` and ``key_path`` (PEM files) and
    an optional ``ca_bundle`` path for server verification.  The broker
    does NOT perform any network handshake — it merely validates that the
    files exist and returns a ``Credentials`` with the ``cert`` and
    ``ca_bundle`` fields set so the caller's HTTP client can use them
    directly (e.g. ``httpx.AsyncClient(cert=creds.cert, verify=creds.ca_bundle)``).
    """

    name = "mtls"

    def __init__(
        self,
        cert_path: Path,
        key_path: Path,
        ca_bundle: Path | None = None,
    ) -> None:
        self._cert_path = cert_path
        self._key_path = key_path
        self._ca_bundle = ca_bundle

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        if requirement.auth_type != AuthType.MTLS:
            raise MissingCredentialError(
                f"MTLSCredentialBroker only supports auth_type=MTLS, "
                f"got {requirement.auth_type!r}"
            )

        if not self._cert_path.is_file():
            raise MissingCredentialError(f"mTLS cert file not found: {self._cert_path}")
        if not self._key_path.is_file():
            raise MissingCredentialError(f"mTLS key file not found: {self._key_path}")

        return Credentials(
            auth_type=AuthType.MTLS,
            cert=(self._cert_path, self._key_path),
            ca_bundle=self._ca_bundle,
        )

    async def refresh(self, creds: Credentials) -> Credentials:
        """Re-verify cert/key files exist and return fresh Credentials.

        Useful when certificate rotation replaces files in-place (short-lived
        cert patterns in service meshes).
        """
        if not self._cert_path.is_file():
            raise MissingCredentialError(f"mTLS cert file not found on refresh: {self._cert_path}")
        if not self._key_path.is_file():
            raise MissingCredentialError(f"mTLS key file not found on refresh: {self._key_path}")
        return Credentials(
            auth_type=AuthType.MTLS,
            cert=(self._cert_path, self._key_path),
            ca_bundle=self._ca_bundle,
        )
