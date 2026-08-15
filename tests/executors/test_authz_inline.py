"""Locks inline authz + broker integration in StandardExecutor.invoke (Phase 6 A4)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from voussoir.agent.policy import PolicyViolation, PolicyViolationError
from voussoir.auth import (
    AuthenticationFailedError,
    Authorizer,
    AuthRequirement,
    AuthType,
    AuthzDecision,
    CredentialBroker,
    Credentials,
    MissingCredentialError,
    Principal,
)
from voussoir.container import Container
from voussoir.executors.standard import StandardExecutor
from voussoir.tools import Capability, ToolContext, tool


def _alice() -> Principal:
    return Principal(user_id="alice", roles=["admin"], issued_at=datetime.now(UTC))


@tool(capability=Capability.READ_PUBLIC, name="ping_a4")
async def _ping() -> str:
    return "pong"


# ------- Authz -------


class _DenyAuthorizer:
    name = "deny-all"

    async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
        return AuthzDecision(decision="DENY", reason="forbidden", authorizer_name=self.name)


class _MaskAuthorizer:
    name = "mask-email"

    async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
        return AuthzDecision(decision="MASK", masked_fields=["email"], authorizer_name=self.name)


async def test_authz_default_allow_proceeds(make_container: Callable[..., Container]) -> None:
    """No Authorizer bound -> default AllowAll. tool.invoke runs."""
    c = make_container()
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(_ping, _ping.input_schema(), ctx)
    assert result == "pong"
    # AuthzDecision recorded with decision=ALLOW
    assert len(ctx.authz_decisions) == 1
    assert ctx.authz_decisions[0].decision == "ALLOW"


async def test_authz_deny_raises_policy_violation(make_container: Callable[..., Container]) -> None:
    c = make_container()
    c.bind(Authorizer, _DenyAuthorizer())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        await ex.invoke(_ping, _ping.input_schema(), ctx)
    assert excinfo.value.violation == PolicyViolation.AUTHZ_DENIED
    # v1.0.1: error message prepends authorizer name for debuggability
    assert "[deny-all]" in str(excinfo.value)
    assert "forbidden" in str(excinfo.value)
    # Decision was still recorded before raising
    assert len(ctx.authz_decisions) == 1
    assert ctx.authz_decisions[0].decision == "DENY"


async def test_authz_mask_redacts_fields(make_container: Callable[..., Container]) -> None:
    """MASK decision strips masked_fields from a dict result."""

    @tool(capability=Capability.READ_PUBLIC, name="user_record_a4")
    async def get_user() -> dict[str, str]:
        return {"name": "alice", "email": "a@b.c", "id": "1"}

    c = make_container()
    c.bind(Authorizer, _MaskAuthorizer())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(get_user, get_user.input_schema(), ctx)
    assert result["email"] == "[REDACTED]"
    assert result["name"] == "alice"  # unchanged
    assert result["id"] == "1"


async def test_authz_mask_redacts_nested_dict(make_container: Callable[..., Container]) -> None:
    """v1.0.1: MASK redaction recurses into nested dicts."""

    @tool(capability=Capability.READ_PUBLIC, name="user_wrapper_v101")
    async def get_wrapped() -> dict[str, Any]:
        return {
            "request_id": "r-1",
            "user": {"name": "alice", "ssn": "123-45-6789", "email": "a@b.c"},
        }

    class _MaskSSN:
        name = "mask-ssn"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="MASK", masked_fields=["ssn"], authorizer_name=self.name)

    c = make_container()
    c.bind(Authorizer, _MaskSSN())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(get_wrapped, get_wrapped.input_schema(), ctx)
    assert result["request_id"] == "r-1"
    assert result["user"]["name"] == "alice"
    assert result["user"]["email"] == "a@b.c"
    assert result["user"]["ssn"] == "[REDACTED]"


async def test_authz_mask_redacts_list_of_records(make_container: Callable[..., Container]) -> None:
    """v1.0.1: MASK redaction iterates over list elements."""

    @tool(capability=Capability.READ_PUBLIC, name="user_list_v101")
    async def list_users() -> list[dict[str, str]]:
        return [
            {"name": "alice", "ssn": "111-11-1111"},
            {"name": "bob", "ssn": "222-22-2222"},
        ]

    class _MaskSSN:
        name = "mask-ssn"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="MASK", masked_fields=["ssn"], authorizer_name=self.name)

    c = make_container()
    c.bind(Authorizer, _MaskSSN())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(list_users, list_users.input_schema(), ctx)
    assert len(result) == 2
    assert result[0]["name"] == "alice"
    assert result[0]["ssn"] == "[REDACTED]"
    assert result[1]["name"] == "bob"
    assert result[1]["ssn"] == "[REDACTED]"


async def test_authz_mask_redacts_pydantic_model(make_container: Callable[..., Container]) -> None:
    """v1.0.1: MASK redaction handles Pydantic BaseModel results + round-trips the class."""
    from pydantic import BaseModel

    class UserRecord(BaseModel):
        name: str
        ssn: str
        email: str

    @tool(capability=Capability.READ_PUBLIC, name="user_model_v101")
    async def get_user_model() -> UserRecord:
        return UserRecord(name="alice", ssn="123-45-6789", email="a@b.c")

    class _MaskSSN:
        name = "mask-ssn"

        async def authorize(self, principal: Any, tool: Any, args: Any, ctx: Any) -> AuthzDecision:
            return AuthzDecision(decision="MASK", masked_fields=["ssn"], authorizer_name=self.name)

    c = make_container()
    c.bind(Authorizer, _MaskSSN())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(get_user_model, get_user_model.input_schema(), ctx)
    # Pydantic round-trips back to the same class (or a dict if validation fails)
    if isinstance(result, UserRecord):
        assert result.ssn == "[REDACTED]"
        assert result.name == "alice"
    else:
        assert result["ssn"] == "[REDACTED]"
        assert result["name"] == "alice"


# ------- Broker -------


async def test_missing_credential_error_when_tool_requires_auth_no_broker(
    make_container: Callable[..., Container],
) -> None:
    req = AuthRequirement(auth_type=AuthType.BEARER, service="x")

    @tool(capability=Capability.READ_PUBLIC, name="needs_auth_a4", auth_requirement=req)
    async def needs_auth() -> str:
        return "ok"

    c = make_container()
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    with pytest.raises(MissingCredentialError):
        await ex.invoke(needs_auth, needs_auth.input_schema(), ctx)


async def test_broker_resolve_populates_credentials(
    make_container: Callable[..., Container],
) -> None:
    req = AuthRequirement(auth_type=AuthType.BEARER, service="x")
    target_creds = Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer tok"})

    @tool(capability=Capability.READ_PUBLIC, name="check_creds_a4", auth_requirement=req)
    async def check_creds(ctx: ToolContext) -> str:
        assert ctx.credentials is target_creds
        return "ok"

    class _StaticBroker:
        name = "static"

        async def resolve(
            self, requirement: AuthRequirement, principal: Principal, ctx: ToolContext
        ) -> Credentials:
            return target_creds

        async def refresh(self, creds: Credentials) -> Credentials:
            return creds

    c = make_container()
    c.bind(CredentialBroker, _StaticBroker())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(check_creds, check_creds.input_schema(), ctx)
    assert result == "ok"


async def test_authentication_failed_triggers_refresh_retry(
    make_container: Callable[..., Container],
) -> None:
    """First tool.invoke raises AuthenticationFailedError -> broker.refresh -> retry succeeds."""
    req = AuthRequirement(auth_type=AuthType.BEARER, service="x")
    creds1 = Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer old"})
    creds2 = Credentials(auth_type=AuthType.BEARER, headers={"Authorization": "Bearer new"})

    call_count: dict[str, int] = {"n": 0}

    @tool(capability=Capability.READ_PUBLIC, name="flaky_a4", auth_requirement=req)
    async def flaky(ctx: ToolContext) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise AuthenticationFailedError("401 stale token")
        return "ok-after-refresh"

    refresh_calls: dict[str, int] = {"n": 0}

    class _RefreshingBroker:
        name = "refreshing"

        async def resolve(
            self, requirement: AuthRequirement, principal: Principal, ctx: ToolContext
        ) -> Credentials:
            return creds1

        async def refresh(self, creds: Credentials) -> Credentials:
            refresh_calls["n"] += 1
            return creds2

    c = make_container()
    c.bind(CredentialBroker, _RefreshingBroker())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    result = await ex.invoke(flaky, flaky.input_schema(), ctx)
    assert result == "ok-after-refresh"
    assert call_count["n"] == 2
    assert refresh_calls["n"] == 1


async def test_authentication_failed_twice_propagates(
    make_container: Callable[..., Container],
) -> None:
    """Second AuthenticationFailedError propagates."""
    req = AuthRequirement(auth_type=AuthType.BEARER, service="x")

    @tool(capability=Capability.READ_PUBLIC, name="always_fail_a4", auth_requirement=req)
    async def always_fail(ctx: ToolContext) -> str:
        raise AuthenticationFailedError("403")

    class _NoopBroker:
        name = "noop"

        async def resolve(
            self, requirement: AuthRequirement, principal: Principal, ctx: ToolContext
        ) -> Credentials:
            return Credentials(auth_type=AuthType.BEARER, headers={})

        async def refresh(self, creds: Credentials) -> Credentials:
            return creds

    c = make_container()
    c.bind(CredentialBroker, _NoopBroker())
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    with pytest.raises(AuthenticationFailedError):
        await ex.invoke(always_fail, always_fail.input_schema(), ctx)


async def test_no_container_authz_defaults_to_allow() -> None:
    """ToolContext.container=None -> FallbackAllowAllAuthorizer allows invocation."""
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        # container=None is the default
    )
    result = await ex.invoke(_ping, _ping.input_schema(), ctx)
    assert result == "pong"
    assert ctx.authz_decisions[0].decision == "ALLOW"


# ------- sensitive_args log redaction (v1.0.1 T6) -------


async def test_executor_redacts_sensitive_args_in_log(
    make_container: Callable[..., Container], monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1.0.1: sensitive_args keys are replaced with [REDACTED] in the tool.invoke log."""
    import voussoir.executors.standard as _std_mod

    @tool(capability=Capability.READ_PUBLIC, name="db_query_v101", sensitive_args=["password"])
    async def db_query(host: str, password: str) -> str:
        return f"connected to {host}"

    c = make_container()
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r-1",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    args = db_query.input_schema(host="db.example.com", password="secret-value")

    captured_calls: list[dict[str, Any]] = []
    real_info = _std_mod._log.info

    def _capture_info(event: str, **kwargs: Any) -> None:
        captured_calls.append({"event": event, **kwargs})
        return real_info(event, **kwargs)

    monkeypatch.setattr(_std_mod._log, "info", _capture_info)
    result = await ex.invoke(db_query, args, ctx)

    assert result == "connected to db.example.com"
    invoke_logs = [r for r in captured_calls if r.get("event") == "tool.invoke"]
    assert invoke_logs, "expected one tool.invoke log entry"
    logged_args = invoke_logs[0]["args"]
    assert logged_args["host"] == "db.example.com"
    assert logged_args["password"] == "[REDACTED]"


async def test_executor_no_sensitive_args_logs_all_fields(
    make_container: Callable[..., Container], monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1.0.1: tools without sensitive_args log all fields (backward compat)."""
    import voussoir.executors.standard as _std_mod

    @tool(capability=Capability.READ_PUBLIC, name="plain_query_v101")
    async def plain_query(host: str, port: int) -> str:
        return f"{host}:{port}"

    c = make_container()
    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r-2",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    args = plain_query.input_schema(host="db.example.com", port=5432)

    captured_calls: list[dict[str, Any]] = []
    real_info = _std_mod._log.info

    def _capture_info(event: str, **kwargs: Any) -> None:
        captured_calls.append({"event": event, **kwargs})
        return real_info(event, **kwargs)

    monkeypatch.setattr(_std_mod._log, "info", _capture_info)
    result = await ex.invoke(plain_query, args, ctx)

    assert result == "db.example.com:5432"
    invoke_logs = [r for r in captured_calls if r.get("event") == "tool.invoke"]
    assert invoke_logs, "expected one tool.invoke log entry"
    logged_args = invoke_logs[0]["args"]
    assert logged_args["host"] == "db.example.com"
    assert logged_args["port"] == 5432
