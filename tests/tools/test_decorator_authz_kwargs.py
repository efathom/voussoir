"""Locks @tool(roles=, domains=, auth_requirement=) decorator kwargs (Phase 6 A3)."""

from __future__ import annotations

from voussoir.auth import AuthRequirement, AuthType
from voussoir.tools import Capability, tool


def test_tool_default_roles_domains_empty():
    @tool(capability=Capability.READ_PUBLIC, name="t1")
    async def t() -> str:
        return "ok"

    assert t.roles == []
    assert t.domains == []
    assert t.auth_requirement is None


def test_tool_with_roles():
    @tool(capability=Capability.READ_PUBLIC, name="t2", roles=["admin", "ops"])
    async def t() -> str:
        return "ok"

    assert t.roles == ["admin", "ops"]


def test_tool_with_domains():
    @tool(capability=Capability.READ_PUBLIC, name="t3", domains=["sre", "platform"])
    async def t() -> str:
        return "ok"

    assert t.domains == ["sre", "platform"]


def test_tool_with_auth_requirement():
    req = AuthRequirement(auth_type=AuthType.BEARER, service="atlassian", scopes=["read"])

    @tool(capability=Capability.READ_PUBLIC, name="t4", auth_requirement=req)
    async def t() -> str:
        return "ok"

    assert t.auth_requirement is req
    assert t.auth_requirement.service == "atlassian"


def test_tool_with_sensitive_args() -> None:
    @tool(capability=Capability.READ_PUBLIC, name="t_sens", sensitive_args=["password", "api_key"])
    async def t() -> str:
        return "ok"

    assert t.sensitive_args == ["password", "api_key"]


def test_tool_default_sensitive_args_empty() -> None:
    @tool(capability=Capability.READ_PUBLIC, name="t_default_sens")
    async def t() -> str:
        return "ok"

    assert t.sensitive_args == []
