"""A2A error codes + voussoir-specific exceptions.

Error codes follow JSON-RPC 2.0 (reserved -32700..-32603) plus voussoir
additions in -32001..-32099 (per spec §5.3).

Phase 4.5a DelegationError subclasses (RemoteUnreachable, RemoteAuthFailed,
RemoteMalformed) intentionally omit the "Error" suffix — they read more
naturally as `except RemoteUnreachable:` than `except RemoteUnreachableError:`.
Suppressing pep8-naming N818 for this module only.
"""

# ruff: noqa: N818

from __future__ import annotations

from voussoir.errors import VoussoirError


class A2AErrorCode:
    """JSON-RPC error codes used by voussoir's A2A transport.

    JSON-RPC reserves -32700..-32603. Implementations may define application
    errors in -32000..-32099; voussoir uses -32001..-32004.
    """

    # JSON-RPC reserved
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # voussoir-defined
    AUTHENTICATION_FAILED = -32001
    AUTHORIZATION_FAILED = -32002
    POLICY_VIOLATION = -32003
    DELEGATION_REFUSED = -32004


class CardVerificationError(VoussoirError):
    """Raised when an AgentCard's JWS signature fails verification or the
    resolved JWKS doesn't contain the kid referenced by the card.
    """


# Phase 4.5a P1 #25: typed DelegationError hierarchy. Pre-4.5a, AgentRef.delegate
# raised PolicyViolationError(MAX_STEPS) for every failure mode (unreachable,
# auth, 5xx, malformed, missing-result) so callers and telemetry could not
# disambiguate. Each subclass below maps 1:1 to a distinct failure mode.


class DelegationError(VoussoirError):
    """Base class for AgentRef.delegate failures."""


class RemoteUnreachable(DelegationError):
    """HTTP timeout, connection refused, or 5xx response from the peer."""


class RemoteAuthFailed(DelegationError):
    """401 or 403 from the peer — caller's JWT was rejected."""


class RemoteProtocolError(DelegationError):
    """JSON-RPC envelope returned an `error` block, or content-type mismatch."""


class RemoteMalformed(DelegationError):
    """Response parsed as JSON-RPC OK but missing required fields, or was
    not valid JSON-RPC at all."""
