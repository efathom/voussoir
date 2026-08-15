"""Phase 4.5a Tranche A exit criteria — gates v0.4.0e-hardening."""

from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


# Exit 1 — WireAgentResult exists with redacted-summary fields only.
def test_exit_1_wire_agent_result_exists() -> None:
    from voussoir.a2a.wire import WireAgentResult

    fields = set(WireAgentResult.model_fields.keys())
    assert fields == {"output", "finish_reason", "tokens_in", "tokens_out", "duration_ms"}
    # The leaky fields MUST NOT appear.
    assert "steps" not in fields
    assert "delegation_chain" not in fields
    assert "trace_id" not in fields


# Exit 2 — JWT validation tightens iss against expected_issuers.
@pytest.mark.asyncio
async def test_exit_2_jwt_iss_validated(make_container, stub_llm, fresh_env, monkeypatch) -> None:
    monkeypatch.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    agent = Agent("r", container=c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))
    provider: KeyProvider = c.resolve(KeyProvider)  # type: ignore[type-abstract]
    secret = provider.jwt_secret()
    now = int(time.time())
    bad = jwt.encode(
        {"iss": "attacker", "aud": "r", "iat": now, "exp": now + 60, "nbf": now, "sub": "u"},
        secret,
        algorithm=provider.jwt_algorithm(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        resp = await client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "voussoir.delegate",
                "params": {"task": "x"},
                "id": "1",
            },
            headers={"Authorization": f"Bearer {bad}"},
        )
    assert resp.status_code == 401


# Exit 3 — AgentCards must declare jwks_uri; inline_jwk is no longer trusted.
def test_exit_3_inline_jwk_no_longer_resolved() -> None:
    """The resolver function in discovery.py reads jwks_uri, not inline_jwk."""
    from voussoir.a2a import discovery

    src = Path(discovery.__file__).read_text()
    # Phase 4.5a removed the inline_jwk lookup branch. The string may still
    # appear in comments / docstrings, but no `unverified_payload.get("inline_jwk")`
    # lookup remains.
    assert 'unverified_payload.get("inline_jwk")' not in src
    assert "unverified_payload.get('inline_jwk')" not in src


# Exit 4 — Plugin allow-list kwarg present.
def test_exit_4_load_delegate_plugins_has_allowed_names() -> None:
    from voussoir.agent.plugins import load_delegate_plugins

    sig = inspect.signature(load_delegate_plugins)
    assert "allowed_names" in sig.parameters
    assert sig.parameters["allowed_names"].default is None


# Exit 5 — EnvKeyProvider(allow_ephemeral=False) is the default; raises without secret.
def test_exit_5_env_key_provider_default_safety(fresh_env) -> None:
    from voussoir.a2a.keys import EnvKeyProvider

    p = EnvKeyProvider()
    assert p._allow_ephemeral is False  # noqa: SLF001
    with pytest.raises(RuntimeError, match="VOUSSOIR_A2A_JWT_SECRET"):
        p.jwt_secret()


# Exit 6 — dispatch_tool_calls uses gather (no as_completed / outcomes_by_id).
def test_exit_6_dispatch_uses_gather() -> None:
    """Parse the AST of dispatch.py and confirm gather is called inside
    dispatch_tool_calls, while as_completed is not. The string check is too
    coarse because the module docstring explains the migration from
    as_completed → gather (mentioning both names)."""
    from voussoir.agent import dispatch

    tree = ast.parse(Path(dispatch.__file__).read_text())
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "dispatch_tool_calls"
    )
    call_names: list[str] = []
    for sub in ast.walk(target):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            call_names.append(f"{sub.value.id}.{sub.attr}")
    assert "asyncio.gather" in call_names
    assert "asyncio.as_completed" not in call_names
    # outcomes_by_id was the pre-4.5a re-ordering dict. AST check: no Name
    # node references it in this function body.
    refs = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
    assert "outcomes_by_id" not in refs


# Exit 7 — Agent.stream raises with cascade configured.
@pytest.mark.asyncio
async def test_exit_7_stream_raises_with_cascade(make_container, stub_llm) -> None:
    from voussoir.agent.agent import Agent
    from voussoir.agent.cascade import Decision, RequestCascade
    from voussoir.agent.policy import PolicyViolation, PolicyViolationError
    from voussoir.agent.result import AgentResult

    class _AlwaysPass:
        name = "p"

        async def validate(self, result: AgentResult, *, task: str) -> Decision:
            return Decision.PASS

    c = make_container(stub_llm())
    agent = Agent(
        "x",
        cascade=RequestCascade(verifier=_AlwaysPass()),
        container=c,
    )
    with pytest.raises(PolicyViolationError) as exc_info:
        async for _ in agent.stream("test"):
            pass
    assert exc_info.value.violation is PolicyViolation.STREAMING_NOT_SUPPORTED


# Exit 8 — Layering: observability free of agent imports.
def test_exit_8_observability_does_not_import_agent() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "voussoir" / "observability"
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("voussoir.agent"), f"{py} imports {node.module}"


# Exit 9 — PolicyViolation enum has the two new variants.
def test_exit_9_policy_violation_new_variants() -> None:
    from voussoir.agent.policy import PolicyViolation

    assert hasattr(PolicyViolation, "DELEGATE_NOT_FOUND")
    assert hasattr(PolicyViolation, "STREAMING_NOT_SUPPORTED")


# Exit 10 — DelegationError hierarchy exists.
def test_exit_10_delegation_error_hierarchy() -> None:
    from voussoir.a2a.errors import (
        DelegationError,
        RemoteAuthFailed,
        RemoteMalformed,
        RemoteProtocolError,
        RemoteUnreachable,
    )

    assert issubclass(RemoteUnreachable, DelegationError)
    assert issubclass(RemoteAuthFailed, DelegationError)
    assert issubclass(RemoteProtocolError, DelegationError)
    assert issubclass(RemoteMalformed, DelegationError)


# Exit 11 — Agent.__init__ raises on string delegate without registry.
def test_exit_11_agent_init_fails_loud(make_container, stub_llm) -> None:
    from voussoir.agent.agent import Agent

    c = make_container(stub_llm())
    with pytest.raises(ValueError, match="AgentRegistry"):
        Agent("x", delegates=["foo"], container=c)


# Exit 12 — Bare AgentRef(card) then .delegate() raises.
@pytest.mark.asyncio
async def test_exit_12_agent_ref_bare_raises(
    make_container, stub_llm, fresh_env, monkeypatch
) -> None:
    monkeypatch.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent

    server_c = make_container(stub_llm(content="x"))
    agent = Agent("r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(agent, endpoint="https://test/a2a"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref_e = await AgentRef.discover("https://test", http_client=client)
    bare = AgentRef(ref_e.card)
    with pytest.raises(RuntimeError, match="async with"):
        await bare.delegate("hi", parent_ctx=None)  # type: ignore[arg-type]


# Exit 13 — bind_agent_registry default load_plugins=True (Phase 4.5a flip).
def test_exit_13_bind_agent_registry_load_plugins_default_true() -> None:
    from voussoir.agent.bootstrap import bind_agent_registry

    sig = inspect.signature(bind_agent_registry)
    assert "load_plugins" in sig.parameters
    assert sig.parameters["load_plugins"].default is True
    assert "allowed_plugin_names" in sig.parameters


# Exit 14 — voussoir top-level re-exports AgentRef + serve_a2a + make_a2a_router.
def test_exit_14_top_level_reexports() -> None:
    import voussoir

    assert hasattr(voussoir, "AgentRef")
    assert hasattr(voussoir, "make_a2a_router")
    assert hasattr(voussoir, "serve_a2a")
