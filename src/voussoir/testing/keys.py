"""make_key_provider() — KeyProvider builder for A2A tests."""

from __future__ import annotations

from voussoir.a2a.keys import EnvKeyProvider, KeyProvider


def make_key_provider(*, jwt_secret: bytes | None = None) -> KeyProvider:
    """Use this when testing A2A code (peer discovery, JWT minting).

    Builds an EnvKeyProvider(allow_ephemeral=True) with an optional
    deterministic secret for cross-test verifiability. If jwt_secret is
    None, the provider generates a fresh ephemeral secret per call.

    jwt_secret: seed a deterministic bytes secret. Useful for asserting
    JWT payloads round-trip correctly across provider instances.
    """
    return EnvKeyProvider(allow_ephemeral=True, jwt_secret=jwt_secret)
