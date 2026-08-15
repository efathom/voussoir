"""EnvCredentialBroker — resolves credentials from environment variables.

Use this when running in any containerised environment where secrets are
injected as env vars (e.g. Kubernetes with external-secrets, GitHub Actions,
Heroku config vars). It requires zero external dependencies and is typically
the first broker in a ChainedCredentialBroker stack.
"""

from __future__ import annotations

import os

from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials
from voussoir.auth.errors import MissingCredentialError
from voussoir.auth.principal import Principal
from voussoir.tools.protocol import ToolContext


class EnvCredentialBroker:
    """Resolve credentials from environment variables.

    Use this when secrets are available as env vars.  The env var name is
    derived from ``requirement.service`` by uppercasing and replacing hyphens
    with underscores:

    - ``BEARER``  → ``{SERVICE}_TOKEN`` (also falls back to ``{SERVICE}_BEARER``)
    - ``API_KEY`` → ``{SERVICE}_API_KEY``

    All other ``AuthType`` values raise ``MissingCredentialError``.
    """

    name = "env"

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        service = (requirement.service or "").upper().replace("-", "_")
        if not service:
            raise MissingCredentialError(
                "EnvCredentialBroker requires a non-empty requirement.service"
            )

        if requirement.auth_type == AuthType.BEARER:
            token = os.environ.get(f"{service}_TOKEN") or os.environ.get(f"{service}_BEARER")
            if not token:
                raise MissingCredentialError(
                    f"No env var {service}_TOKEN or {service}_BEARER found"
                )
            return Credentials(
                auth_type=AuthType.BEARER,
                headers={"Authorization": f"Bearer {token}"},
            )

        if requirement.auth_type == AuthType.API_KEY:
            key = os.environ.get(f"{service}_API_KEY")
            if not key:
                raise MissingCredentialError(f"No env var {service}_API_KEY found")
            return Credentials(
                auth_type=AuthType.API_KEY,
                headers={"X-API-Key": key},
            )

        raise MissingCredentialError(
            f"EnvCredentialBroker does not support auth_type={requirement.auth_type!r}"
        )

    async def refresh(self, creds: Credentials) -> Credentials:
        """Return credentials unchanged.

        Env-var credentials are read once at process start; there is no
        rotation mechanism. To pick up new values, restart the process or
        use a different broker (FileCredentialBroker for K8s secret rotation,
        OAuth2CredentialBroker for token refresh).
        """
        return creds
