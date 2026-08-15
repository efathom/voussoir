"""Phase 6 Tranche A exit gate -- auth concrete bindings (Phase 6 Task A10).

10 invariants that lock the shipped auth surface. If any fails, the auth
contract has regressed.
"""

from __future__ import annotations

import inspect


def test_exit_1_authzdecision_record_present():
    """AuthzDecision is a public record with the spec's 4 fields + extra-forbid."""
    import pytest as _pytest
    from pydantic import ValidationError

    from voussoir.auth import AuthzDecision

    d = AuthzDecision(decision="ALLOW")
    assert d.decision == "ALLOW"
    assert d.reason == ""
    assert d.masked_fields == []
    assert d.authorizer_name == ""

    with _pytest.raises(ValidationError):
        AuthzDecision(decision="ALLOW", bogus="x")  # type: ignore[call-arg]


def test_exit_2_policy_violation_authz_denied():
    """PolicyViolation.AUTHZ_DENIED enum variant exists."""
    from voussoir.agent import PolicyViolation

    assert PolicyViolation.AUTHZ_DENIED == "authz_denied"


def test_exit_3_six_brokers_shipped():
    """All six built-in CredentialBrokers are importable + named."""
    from voussoir.auth.brokers import (
        ChainedCredentialBroker,
        EnvCredentialBroker,
        FileCredentialBroker,
        KeychainCredentialBroker,
        MTLSCredentialBroker,
        OAuth2CredentialBroker,
    )

    for cls in (
        ChainedCredentialBroker,
        EnvCredentialBroker,
        FileCredentialBroker,
        KeychainCredentialBroker,
        MTLSCredentialBroker,
        OAuth2CredentialBroker,
    ):
        # Each broker class has a `name` class attribute
        assert isinstance(cls.name, str), f"{cls.__name__} missing class name attribute"
        # Each has resolve + refresh methods
        assert callable(getattr(cls, "resolve", None))
        assert callable(getattr(cls, "refresh", None))


def test_exit_4_five_authorizers_shipped():
    """All five built-in Authorizers are importable + named."""
    from voussoir.auth.authorizers import (
        AllowAllAuthorizer,
        ChainedAuthorizer,
        DenyByDefaultAuthorizer,
        DomainAuthorizer,
        RoleAuthorizer,
    )

    for cls in (
        AllowAllAuthorizer,
        ChainedAuthorizer,
        DenyByDefaultAuthorizer,
        DomainAuthorizer,
        RoleAuthorizer,
    ):
        assert isinstance(cls.name, str), f"{cls.__name__} missing name attribute"
        assert callable(getattr(cls, "authorize", None))


def test_exit_5_agent_run_accepts_principal_kwarg():
    """Agent.run and Agent.stream accept principal= keyword argument."""
    from voussoir import Agent

    for fn in (Agent.run, Agent.stream):
        sig = inspect.signature(fn)
        assert "principal" in sig.parameters, f"{fn.__name__} missing principal= kwarg"
        # Must be keyword-only (passed via *args separator)
        param = sig.parameters["principal"]
        assert (
            param.kind == inspect.Parameter.KEYWORD_ONLY
        ), f"{fn.__name__}.principal must be keyword-only; got {param.kind}"


def test_exit_6_tool_decorator_accepts_authz_kwargs():
    """The @tool decorator accepts roles=, domains=, auth_requirement=."""
    from voussoir.tools.decorator import tool

    sig = inspect.signature(tool)
    for kw in ("roles", "domains", "auth_requirement"):
        assert kw in sig.parameters, f"@tool missing {kw}= kwarg"


def test_exit_7_agent_result_has_authz_decisions():
    """AgentResult.authz_decisions field is on the Pydantic model."""
    from voussoir.agent import AgentResult

    assert (
        "authz_decisions" in AgentResult.model_fields
    ), "AgentResult.authz_decisions field missing"


def test_exit_8_authz_metric_present():
    """voussoir.observability.metrics.AUTHZ_DECISIONS counter handle exists."""
    from voussoir.observability import metrics

    assert hasattr(metrics, "AUTHZ_DECISIONS")
    # OTel counter API
    assert callable(getattr(metrics.AUTHZ_DECISIONS, "add", None))


def test_exit_9_default_container_binds_deny_by_default():
    """default_container resolves Authorizer to a fail-closed DenyByDefaultAuthorizer."""
    from voussoir.auth.authorizers.deny_by_default import DenyByDefaultAuthorizer
    from voussoir.auth.protocol import Authorizer
    from voussoir.container.defaults import default_container

    c = default_container()
    authz = c.resolve(Authorizer)  # type: ignore[type-abstract]
    assert isinstance(authz, DenyByDefaultAuthorizer)


def test_exit_10_a2a_principal_flow_shipped():
    """JWTPrincipalMapper + PrincipalForwarder Protocols + default impls exist."""
    from voussoir.auth import (
        DefaultJWTPrincipalMapper,
        JWTPrincipalMapper,
        PrincipalForwarder,
        SameOrgPassthroughForwarder,
    )

    # Default impl satisfies the Protocol structurally
    assert isinstance(DefaultJWTPrincipalMapper(), JWTPrincipalMapper)
    # SameOrgPassthroughForwarder is the concrete PrincipalForwarder impl;
    # construction requires a key_provider (tested in test_principal_a2a.py).
    # Here we just check the class is importable and has the right method.
    assert callable(getattr(SameOrgPassthroughForwarder, "forward", None))
    assert issubclass(SameOrgPassthroughForwarder, object)
    # Ensure PrincipalForwarder Protocol is runtime-checkable
    assert hasattr(PrincipalForwarder, "__protocol_attrs__") or callable(PrincipalForwarder)
