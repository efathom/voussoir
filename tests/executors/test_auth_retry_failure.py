"""v1.0.4 E6 — Auth retry path surfaces MissingCredentialError + 2nd AuthError clearly."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from voussoir.auth import (
    AuthenticationFailedError,
    AuthRequirement,
    AuthType,
    CredentialBroker,
    Credentials,
    MissingCredentialError,
    Principal,
)
from voussoir.container import Container
from voussoir.executors.standard import StandardExecutor
from voussoir.tools import Capability, ToolContext, tool


@tool(
    capability=Capability.READ_PUBLIC,
    name="needs_auth_e6",
    auth_requirement=AuthRequirement(auth_type=AuthType.BEARER, service="x"),
)
async def _needs_auth(ctx: ToolContext) -> str:
    raise AuthenticationFailedError("401 from upstream")


class _RefreshFails:
    name = "refresh-fails"

    async def resolve(
        self, requirement: AuthRequirement, principal: Principal, ctx: ToolContext
    ) -> Credentials:
        return Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer stale"})

    async def refresh(self, creds: Credentials) -> Credentials:
        raise MissingCredentialError("refresh token revoked")


class _RefreshSucceedsButRetryFails:
    name = "permanent-401"

    async def resolve(
        self, requirement: AuthRequirement, principal: Principal, ctx: ToolContext
    ) -> Credentials:
        return Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer stale"})

    async def refresh(self, creds: Credentials) -> Credentials:
        return Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer fresh"})


async def test_refresh_missing_credential_re_raised_as_clear_auth_error(
    make_container: Callable[..., Container],
) -> None:
    """MissingCredentialError from broker.refresh() is re-raised as AuthenticationFailedError with a clear message."""
    c = make_container()
    c.bind(CredentialBroker, _RefreshFails())  # type: ignore[type-abstract]

    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    with pytest.raises(AuthenticationFailedError) as excinfo:
        await ex.invoke(_needs_auth, _needs_auth.input_schema(), ctx)
    # Error message mentions the permanent-failure nature
    msg = str(excinfo.value).lower()
    assert "refresh" in msg or "permanent" in msg
    # Causation preserved
    assert isinstance(excinfo.value.__cause__, MissingCredentialError)


async def test_second_auth_error_after_refresh_re_raised_clearly(
    make_container: Callable[..., Container],
) -> None:
    """When refresh succeeds but the second tool.invoke still fails 401, re-raise with a clear 'permanent' message."""
    c = make_container()
    c.bind(CredentialBroker, _RefreshSucceedsButRetryFails())  # type: ignore[type-abstract]

    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    with pytest.raises(AuthenticationFailedError) as excinfo:
        await ex.invoke(_needs_auth, _needs_auth.input_schema(), ctx)
    msg = str(excinfo.value).lower()
    assert "after refresh" in msg or "permanent" in msg or "still failed" in msg
    # Causation preserved: chain to the original 2nd-call AuthenticationFailedError
    assert isinstance(excinfo.value.__cause__, AuthenticationFailedError)
