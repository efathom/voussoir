"""FileCredentialBroker — resolves credentials from mounted secret files.

Use this when secrets are mounted as files in the container filesystem,
which is the standard pattern for Kubernetes Secrets, Docker secrets, and
many CI/CD vault integrations (e.g. ``/var/run/secrets/<service>-token``).
It reads the file content at resolve-time so secret rotation is picked up
on the next invocation without restarting the process.
"""

from __future__ import annotations

from pathlib import Path

from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials
from voussoir.auth.errors import MissingCredentialError
from voussoir.auth.principal import Principal
from voussoir.tools.protocol import ToolContext

_CANDIDATE_SUFFIXES = ("", "-token", "-api-key")


class FileCredentialBroker:
    """Resolve credentials by reading mounted secret files.

    Use this when secrets are projected into a directory as plain-text files
    (Kubernetes ``secretKeyRef`` volume mounts, Docker secrets at
    ``/run/secrets/…``, etc.).  The constructor takes the root directory;
    at resolve-time the broker looks for files named:

        ``<service>``, ``<service>-token``, ``<service>-api-key``

    The first match (lowercased service name, hyphens normalised) wins.
    File content is stripped of surrounding whitespace before use.
    """

    name = "file"

    def __init__(self, secret_dir: Path) -> None:
        self._dir = secret_dir

    def _find_file(self, service: str) -> Path | None:
        base = service.lower().replace("_", "-")
        for suffix in _CANDIDATE_SUFFIXES:
            p = self._dir / f"{base}{suffix}"
            if p.is_file():
                return p
        return None

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        service = requirement.service or ""
        if not service:
            raise MissingCredentialError(
                "FileCredentialBroker requires a non-empty requirement.service"
            )

        path = self._find_file(service)
        if path is None:
            raise MissingCredentialError(
                f"No secret file found for service={service!r} under {self._dir}"
            )

        value = path.read_text().strip()
        if not value:
            raise MissingCredentialError(f"Secret file {path} is empty")

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

        # For unknown types, expose the raw value in a generic header so it
        # can still be forwarded; callers may post-process the Credentials.
        return Credentials(
            auth_type=requirement.auth_type,
            headers={"X-Secret": value},
        )

    async def refresh(self, creds: Credentials) -> Credentials:
        """File brokers do not hold state — refresh is a no-op.

        The next call to ``resolve`` will re-read the file, picking up any
        rotation automatically.
        """
        return creds
