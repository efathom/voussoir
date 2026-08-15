"""voussoir.a2a — Agent-to-Agent remote delegation (Phase 4b).

Public API:
  AgentCard, AuthMethod        — schema for /.well-known/agent-card.json
  AgentRef                     — IDelegate impl wrapping a remote agent
  discover_card                — fetch + verify + parse AgentCard
  make_a2a_router, serve_a2a   — publisher entry points (lazy; require [a2a] extra)
  WireAgentResult              — redacted-for-the-wire AgentResult projection
  KeyProvider, EnvKeyProvider  — JWT + JWS key management
  A2AErrorCode                 — JSON-RPC error codes
  CardVerificationError, NoCardSigningKeyError,
  JWTKeyNotConfiguredError — exceptions
  InboundJWTVerificationNotConfiguredError — back-compat alias (removed v1.2.0)
"""

from __future__ import annotations

from voussoir.a2a.agent_ref import AgentRef
from voussoir.a2a.card import AgentCard, AuthMethod, card_from_agent
from voussoir.a2a.discovery import discover_card
from voussoir.a2a.errors import A2AErrorCode, CardVerificationError
from voussoir.a2a.keys import (
    EnvKeyProvider,
    InboundJWTVerificationNotConfiguredError,
    JWTKeyNotConfiguredError,
    KeyProvider,
    NoCardSigningKeyError,
)
from voussoir.a2a.wire import WireAgentResult

# make_a2a_router and serve_a2a require fastapi/uvicorn ([a2a] extra).
# Lazily resolved via __getattr__ so that `import voussoir.a2a` (or any
# submodule that triggers this __init__) does NOT eagerly pull those heavy
# deps for code that only uses AgentRef, KeyProvider, etc.
_LAZY_PUBLISHER_ATTRS = frozenset({"make_a2a_router", "serve_a2a"})


def __getattr__(name: str) -> object:
    if name in _LAZY_PUBLISHER_ATTRS:
        from voussoir.a2a.publisher import make_a2a_router, serve_a2a  # noqa: PLC0415

        globals()["make_a2a_router"] = make_a2a_router
        globals()["serve_a2a"] = serve_a2a
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A2AErrorCode",
    "AgentCard",
    "AgentRef",
    "AuthMethod",
    "CardVerificationError",
    "EnvKeyProvider",
    "InboundJWTVerificationNotConfiguredError",
    "JWTKeyNotConfiguredError",
    "KeyProvider",
    "NoCardSigningKeyError",
    "WireAgentResult",
    "card_from_agent",
    "discover_card",
    "make_a2a_router",
    "serve_a2a",
]
