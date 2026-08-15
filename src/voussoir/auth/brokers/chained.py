"""ChainedCredentialBroker — tries a list of brokers in order; first hit wins.

Use this as the primary composition mechanism for production credential
resolution.  A typical production chain might be:

    ChainedCredentialBroker([
        EnvCredentialBroker(),          # fastest; CI/CD env vars
        MTLSCredentialBroker(...),      # service-mesh identity
        FileCredentialBroker(...),      # K8s secret volume mounts
        KeychainCredentialBroker(),     # developer workstation fallback
    ])

``MissingCredentialError`` from a broker causes transparent fall-through to
the next broker.  Any other exception (network error, permission error, etc.)
propagates immediately.
"""

from __future__ import annotations

from voussoir.auth.credentials import AuthRequirement, Credentials
from voussoir.auth.errors import MissingCredentialError
from voussoir.auth.principal import Principal
from voussoir.auth.protocol import CredentialBroker
from voussoir.tools.protocol import ToolContext


class ChainedCredentialBroker:
    """Try a list of CredentialBrokers in order; return the first success.

    Use this as the top-level broker in production setups to compose multiple
    resolution strategies.  Brokers are tried in insertion order;
    ``MissingCredentialError`` triggers fall-through to the next broker.
    Non-missing exceptions (e.g. network errors, permission errors) short-
    circuit the chain immediately.

    Raises ``ValueError`` if constructed with an empty list.
    """

    name = "chained"

    def __init__(self, brokers: list[CredentialBroker]) -> None:
        if not brokers:
            raise ValueError("ChainedCredentialBroker requires at least one broker")
        self._brokers = brokers

    async def resolve(
        self,
        requirement: AuthRequirement,
        principal: Principal,
        ctx: ToolContext,
    ) -> Credentials:
        last_err: MissingCredentialError | None = None
        for broker in self._brokers:
            try:
                return await broker.resolve(requirement, principal, ctx)
            except MissingCredentialError as exc:
                last_err = exc
                continue

        raise MissingCredentialError(
            f"All brokers in chain exhausted for requirement={requirement!r}. "
            f"Last error: {last_err}"
        )

    async def refresh(self, creds: Credentials) -> Credentials:
        """Try each broker's refresh in order; return the first success.

        If all brokers raise ``MissingCredentialError``, re-raise the last one.
        Non-missing exceptions propagate immediately.
        """
        last_err: MissingCredentialError | None = None
        for broker in self._brokers:
            try:
                return await broker.refresh(creds)
            except MissingCredentialError as exc:
                last_err = exc
                continue

        if last_err is not None:
            raise last_err

        # Unreachable (brokers is non-empty), but satisfies type checker
        raise MissingCredentialError("No brokers available for refresh")
