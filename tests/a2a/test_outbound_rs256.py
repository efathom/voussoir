"""Locks KeyProvider.jwt_signing_key dispatch + exception rename (v1.1.0 F8).

F9 tests (end-to-end AgentRef._issue_jwt path) are appended below.
"""

from __future__ import annotations

import pytest

from voussoir.a2a.keys import (
    EnvKeyProvider,
    JWTKeyNotConfiguredError,
)


def test_signing_key_returns_secret_for_hs256(monkeypatch: pytest.MonkeyPatch) -> None:
    """HS256 outbound JWTs use the symmetric secret."""
    # VOUSSOIR_A2A_JWT_SECRET is base64-encoded; "dGVzdC1zZWNyZXQ=" decodes to b"test-secret"
    monkeypatch.setenv("VOUSSOIR_A2A_JWT_SECRET", "dGVzdC1zZWNyZXQ=")
    provider = EnvKeyProvider(allow_ephemeral=False)
    key = provider.jwt_signing_key("HS256")
    assert isinstance(key, bytes)
    assert key == b"test-secret"


def test_signing_key_returns_secret_for_hs384(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOUSSOIR_A2A_JWT_SECRET", "dGVzdC1zZWNyZXQ=")
    provider = EnvKeyProvider(allow_ephemeral=False)
    key = provider.jwt_signing_key("HS384")
    assert key == b"test-secret"


def test_signing_key_raises_for_rs256(monkeypatch: pytest.MonkeyPatch) -> None:
    """RS256 outbound on EnvKeyProvider → clean error with operator guidance."""
    monkeypatch.setenv("VOUSSOIR_A2A_JWT_SECRET", "dGVzdC1zZWNyZXQ=")
    provider = EnvKeyProvider(allow_ephemeral=False)
    with pytest.raises(JWTKeyNotConfiguredError) as excinfo:
        provider.jwt_signing_key("RS256")
    msg = str(excinfo.value).lower()
    assert "rs256" in msg
    assert "envkeyprovider" in msg or "asymmetric" in msg
    assert "private" in msg  # error should explicitly say "private key" needed


def test_back_compat_alias_for_inbound_exception() -> None:
    """Old InboundJWTVerificationNotConfiguredError still importable + IS the new class."""
    from voussoir.a2a.keys import (
        InboundJWTVerificationNotConfiguredError,
        JWTKeyNotConfiguredError,
    )

    assert InboundJWTVerificationNotConfiguredError is JWTKeyNotConfiguredError


def test_back_compat_alias_via_a2a_package() -> None:
    """Old name still importable from voussoir.a2a top-level."""
    from voussoir.a2a import (
        InboundJWTVerificationNotConfiguredError,
        JWTKeyNotConfiguredError,
    )

    assert InboundJWTVerificationNotConfiguredError is JWTKeyNotConfiguredError


# ---------------------------------------------------------------------------
# F9 — AgentRef._issue_jwt end-to-end path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_ref_issue_jwt_rs256_misconfig_raises_clean_error(
    make_container: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: AgentRef._issue_jwt with EnvKeyProvider configured for RS256
    raises JWTKeyNotConfiguredError with operator-actionable guidance.

    Calls _issue_jwt directly (no HTTP round-trip required).
    """
    import base64

    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.card import AgentCard, AuthMethod
    from voussoir.a2a.keys import KeyProvider
    from voussoir.agent.context import AgentContext

    monkeypatch.setenv(
        "VOUSSOIR_A2A_JWT_SECRET",
        base64.b64encode(b"test-secret").decode(),
    )
    monkeypatch.setenv("VOUSSOIR_A2A_JWT_ALGORITHM", "RS256")

    # make_container provides ISessionStore + IMemoryStore + ITelemetrySink.
    # Override its KeyProvider with one that sees the RS256 env var above.
    container = (make_container)()  # type: ignore[call-arg,operator]
    provider = EnvKeyProvider(allow_ephemeral=False)
    container.bind(KeyProvider, provider)  # type: ignore[union-attr,type-abstract]

    card = AgentCard(
        name="remote",
        description="r",
        endpoint="https://peer/a2a",  # type: ignore[arg-type]
        capabilities=[],
        input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
        output_schema={"type": "string"},
        auth=[AuthMethod(type="bearer")],
        supports=["json-rpc/2.0"],
    )
    ref = AgentRef(card)

    async with await AgentContext.open(
        container=container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        ctx.agent_name = "caller"
        with pytest.raises(JWTKeyNotConfiguredError) as excinfo:
            ref._issue_jwt(ctx)  # noqa: SLF001

    msg = str(excinfo.value).lower()
    assert "rs256" in msg
    assert "private" in msg


@pytest.mark.asyncio
async def test_agent_ref_issue_jwt_hs256_still_works(
    make_container: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HS256 path (default): AgentRef._issue_jwt returns a valid JWT string
    without raising. Ensures the F9 change doesn't break the common case.
    """
    import base64

    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.card import AgentCard, AuthMethod
    from voussoir.a2a.keys import KeyProvider
    from voussoir.agent.context import AgentContext

    monkeypatch.setenv(
        "VOUSSOIR_A2A_JWT_SECRET",
        base64.b64encode(b"test-secret-32bytes-long-padded!").decode(),
    )
    monkeypatch.delenv("VOUSSOIR_A2A_JWT_ALGORITHM", raising=False)

    container = (make_container)()  # type: ignore[call-arg,operator]
    provider = EnvKeyProvider(allow_ephemeral=False)
    container.bind(KeyProvider, provider)  # type: ignore[union-attr,type-abstract]

    card = AgentCard(
        name="remote",
        description="r",
        endpoint="https://peer/a2a",  # type: ignore[arg-type]
        capabilities=[],
        input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
        output_schema={"type": "string"},
        auth=[AuthMethod(type="bearer")],
        supports=["json-rpc/2.0"],
    )
    ref = AgentRef(card)

    async with await AgentContext.open(
        container=container, run_id="r", session_id="s", user_id="u"
    ) as ctx:
        ctx.agent_name = "caller"
        token = ref._issue_jwt(ctx)  # noqa: SLF001

    assert isinstance(token, str)
    assert token.count(".") == 2  # three-part JWT
