"""Lock the public surface of voussoir.a2a."""

from __future__ import annotations


def test_package_exports() -> None:
    from voussoir.a2a import (
        A2AErrorCode,
        AgentCard,
        AgentRef,
        AuthMethod,
        CardVerificationError,
        EnvKeyProvider,
        KeyProvider,
        NoCardSigningKeyError,
        discover_card,
        make_a2a_router,
        serve_a2a,
    )

    assert AgentCard is not None
    assert AgentRef is not None
    assert make_a2a_router is not None
    assert serve_a2a is not None
    assert discover_card is not None
    assert KeyProvider is not None
    assert EnvKeyProvider is not None
    assert AuthMethod is not None
    assert A2AErrorCode is not None
    assert CardVerificationError is not None
    assert NoCardSigningKeyError is not None
