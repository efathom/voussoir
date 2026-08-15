"""Locks the 4 Authorizer impls + default container binding (Phase 6 A8)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from voussoir.auth import AuthzDecision, Principal
from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
from voussoir.auth.authorizers.chained import ChainedAuthorizer
from voussoir.auth.authorizers.domain import DomainAuthorizer
from voussoir.auth.authorizers.role import RoleAuthorizer


def _alice(roles: list[str] | None = None, domains: list[str] | None = None) -> Principal:
    return Principal(
        user_id="alice",
        roles=roles or [],
        domains=domains or [],
        issued_at=datetime.now(UTC),
    )


class _FakeTool:
    """Stub Tool object — only the attributes A8 authorizers read."""

    def __init__(
        self, name: str, roles: list[str] | None = None, domains: list[str] | None = None
    ) -> None:
        self.name = name
        self.roles = roles or []
        self.domains = domains or []


# ------- AllowAllAuthorizer -------


async def test_allow_all_returns_allow():
    a = AllowAllAuthorizer()
    decision = await a.authorize(_alice(), _FakeTool("t"), None, None)  # type: ignore[arg-type]
    assert decision.decision == "ALLOW"
    assert decision.authorizer_name == "allow_all"


async def test_allow_all_logs_once_per_process(caplog: Any) -> None:
    """The 'no authz enforced' warning fires at most once per process."""
    # Reset the warned state for the test (the module-level flag).
    from voussoir.auth.authorizers import allow_all

    allow_all._warned = False  # type: ignore[attr-defined]

    a = AllowAllAuthorizer()
    # First call: warns
    import structlog

    with structlog.testing.capture_logs() as captured:
        await a.authorize(_alice(), _FakeTool("t"), None, None)  # type: ignore[arg-type]
        await a.authorize(_alice(), _FakeTool("t2"), None, None)  # type: ignore[arg-type]

    warns = [r for r in captured if "authz_unenforced" in str(r.get("event", ""))]
    assert len(warns) == 1  # only the first call warned


# ------- RoleAuthorizer -------


async def test_role_authorizer_allow_when_role_matches() -> None:
    a = RoleAuthorizer()
    decision = await a.authorize(
        _alice(roles=["admin", "ops"]),
        _FakeTool("t", roles=["admin"]),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert decision.decision == "ALLOW"
    assert decision.authorizer_name == "role"


async def test_role_authorizer_deny_when_no_overlap() -> None:
    a = RoleAuthorizer()
    decision = await a.authorize(
        _alice(roles=["dev"]),
        _FakeTool("t", roles=["admin"]),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert decision.decision == "DENY"


async def test_role_authorizer_allow_when_tool_declares_no_roles() -> None:
    """If tool doesn't declare any role requirement, RoleAuthorizer is permissive."""
    a = RoleAuthorizer()
    decision = await a.authorize(
        _alice(roles=[]),
        _FakeTool("t", roles=[]),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert decision.decision == "ALLOW"


# ------- DomainAuthorizer -------


async def test_domain_authorizer_allow_when_domain_overlaps() -> None:
    a = DomainAuthorizer()
    decision = await a.authorize(
        _alice(domains=["sre", "platform"]),
        _FakeTool("t", domains=["sre"]),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert decision.decision == "ALLOW"


async def test_domain_authorizer_deny_when_no_overlap() -> None:
    a = DomainAuthorizer()
    decision = await a.authorize(
        _alice(domains=["frontend"]),
        _FakeTool("t", domains=["backend"]),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert decision.decision == "DENY"


async def test_domain_authorizer_allow_when_tool_declares_no_domains() -> None:
    a = DomainAuthorizer()
    decision = await a.authorize(
        _alice(domains=[]),
        _FakeTool("t", domains=[]),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert decision.decision == "ALLOW"


# ------- ChainedAuthorizer -------


async def test_chained_authorizer_first_deny_wins() -> None:
    """When the first authorizer DENIES, that decision is returned immediately."""

    class _Deny:
        name = "deny"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="DENY", reason="no", authorizer_name=self.name)

    class _Allow:
        name = "allow"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="ALLOW", authorizer_name=self.name)

    chained = ChainedAuthorizer([_Deny(), _Allow()])  # type: ignore[list-item]
    decision = await chained.authorize(_alice(), _FakeTool("t"), None, None)  # type: ignore[arg-type]
    assert decision.decision == "DENY"
    assert "deny" in decision.reason.lower() or decision.authorizer_name == "deny"


async def test_chained_authorizer_all_allow_returns_allow() -> None:
    class _Allow:
        name = "allow"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="ALLOW", authorizer_name=self.name)

    chained = ChainedAuthorizer([_Allow(), _Allow()])  # type: ignore[list-item]
    decision = await chained.authorize(_alice(), _FakeTool("t"), None, None)  # type: ignore[arg-type]
    assert decision.decision == "ALLOW"


async def test_chained_authorizer_union_of_masked_fields() -> None:
    """When all authorizers ALLOW but some MASK, the chain returns MASK with the union of masked_fields."""

    class _MaskEmail:
        name = "mask_email"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(
                decision="MASK", masked_fields=["email"], authorizer_name=self.name
            )

    class _MaskSSN:
        name = "mask_ssn"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="MASK", masked_fields=["ssn"], authorizer_name=self.name)

    chained = ChainedAuthorizer([_MaskEmail(), _MaskSSN()])  # type: ignore[list-item]
    decision = await chained.authorize(_alice(), _FakeTool("t"), None, None)  # type: ignore[arg-type]
    assert decision.decision == "MASK"
    assert set(decision.masked_fields) == {"email", "ssn"}


async def test_chained_authorizer_empty_list_rejected() -> None:
    with pytest.raises(ValueError):
        ChainedAuthorizer([])


# ------- default_container binding -------


def test_default_container_binds_deny_by_default() -> None:
    from voussoir.auth.authorizers.deny_by_default import DenyByDefaultAuthorizer
    from voussoir.auth.protocol import Authorizer
    from voussoir.container.defaults import default_container

    c = default_container()
    authz = c.resolve(Authorizer)  # type: ignore[type-abstract]
    assert isinstance(authz, DenyByDefaultAuthorizer)
