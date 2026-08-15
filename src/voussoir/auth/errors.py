"""Auth-related exception types.

`AuthError` is the base for all voussoir auth exceptions, mirroring the
`DelegationError` pattern in voussoir.a2a.errors. Catch this for any
auth-domain failure.

`MissingCredentialError` fires from CredentialBroker when no broker is
bound or the broker can't produce credentials. `AuthenticationFailedError`
is the tool-author contract for 401/403 from an upstream service — the
executor catches it once, calls broker.refresh(), and retries tool.invoke.
"""

from __future__ import annotations

from voussoir.errors import VoussoirError


class AuthError(VoussoirError):
    """Base for all voussoir auth exceptions."""


class MissingCredentialError(AuthError):
    """Raised when a tool with auth_requirement runs but no CredentialBroker
    is bound, or the broker can't produce credentials for the declared
    requirement."""


class AuthenticationFailedError(AuthError):
    """Raised by a tool on 401/403 from its upstream service. The executor
    catches this once, calls broker.refresh(creds), and retries tool.invoke.
    A second failure propagates."""
