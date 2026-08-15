"""KeychainCredentialBroker — resolves credentials from the OS keychain.

Use this on developer workstations or long-running processes where secrets
are stored in the system keychain (macOS Keychain, GNOME Keyring, Windows
Credential Manager) via the ``keyring`` library.  Not appropriate for
headless CI environments — skip those tests with
``pytest.mark.skipif(os.environ.get("CI") is not None, ...)``.

``keyring`` is imported lazily inside each method so importing this module
does not force all consumers to install ``keyring``.
"""

from __future__ import annotations

from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials
from voussoir.auth.errors import MissingCredentialError
from voussoir.auth.principal import Principal
from voussoir.tools.protocol import ToolContext


class KeychainCredentialBroker:
    """Resolve credentials from the OS keychain via the ``keyring`` library.

    Use this when running on a developer workstation or managed compute
    where secrets are persisted in the OS secret store.  The ``service_namespace``
    constructor argument scopes all keychain entries
    (``keyring.get_password(service_namespace, service_name)``), preventing
    collisions with other applications.

    The ``store`` helper lets other brokers (e.g. ``OAuth2CredentialBroker``)
    persist refresh tokens into the keychain for later retrieval.
    """

    name = "keychain"

    def __init__(self, service_namespace: str = "voussoir") -> None:
        self._namespace = service_namespace

    def store(self, service: str, value: str) -> None:
        """Persist *value* under *service* in the OS keychain.

        Use this to pre-populate entries (e.g. from an OAuth2 initial grant)
        before calling ``resolve``.
        """
        import keyring  # lazy import — keyring is optional at import-time

        keyring.set_password(self._namespace, service, value)

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        import keyring  # lazy import — keyring is optional at import-time

        service = requirement.service or ""
        if not service:
            raise MissingCredentialError(
                "KeychainCredentialBroker requires a non-empty requirement.service"
            )

        value: str | None = keyring.get_password(self._namespace, service)
        if not value:
            raise MissingCredentialError(
                f"No keychain entry found for namespace={self._namespace!r}, "
                f"service={service!r}"
            )

        if requirement.auth_type in (AuthType.BEARER, AuthType.SERVICE_TOKEN):
            return Credentials(
                auth_type=requirement.auth_type,
                headers={"Authorization": f"Bearer {value}"},
            )

        if requirement.auth_type == AuthType.API_KEY:
            return Credentials(
                auth_type=AuthType.API_KEY,
                headers={"X-API-Key": value},
            )

        raise MissingCredentialError(
            f"KeychainCredentialBroker does not support auth_type={requirement.auth_type!r}"
        )

    async def refresh(self, creds: Credentials) -> Credentials:
        """Re-read the keychain entry (value may have rotated since last resolve).

        Returns a fresh ``Credentials`` object with updated header values.
        If the entry has been deleted, raises ``MissingCredentialError``.
        """
        import keyring  # lazy import

        # Derive service from existing metadata if available
        service = creds.metadata.get("keychain_service", "")
        if not service:
            # Cannot re-look up without knowing the service name; return unchanged
            return creds

        value: str | None = keyring.get_password(self._namespace, service)
        if not value:
            raise MissingCredentialError(
                f"Keychain entry deleted during refresh: "
                f"namespace={self._namespace!r}, service={service!r}"
            )

        if creds.auth_type in (AuthType.BEARER, AuthType.SERVICE_TOKEN):
            return Credentials(
                auth_type=creds.auth_type,
                headers={"Authorization": f"Bearer {value}"},
                metadata={"keychain_service": service},
            )

        if creds.auth_type == AuthType.API_KEY:
            return Credentials(
                auth_type=creds.auth_type,
                headers={"X-API-Key": value},
                metadata={"keychain_service": service},
            )

        return creds
