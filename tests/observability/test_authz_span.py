"""Locks authz.check span emission + nesting (Phase 6 A6).

Uses the InMemorySpanExporter pattern from conftest.py (shared session-scoped
provider). After StandardExecutor.invoke fires, expect exactly one authz.check
span per tool call, nested as a child of tool.call.
"""

from __future__ import annotations

import pytest

from voussoir.auth import AuthzDecision
from voussoir.executors.standard import StandardExecutor
from voussoir.tools import Capability, ToolContext, tool

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@tool(capability=Capability.READ_PUBLIC, name="ping_a6")
async def _ping() -> str:
    return "pong"


class _AllowAuthorizer:
    name = "allow-a6"

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        return AuthzDecision(decision="ALLOW", reason="ok", authorizer_name=self.name)


class _DenyAuthorizer:
    name = "deny-a6"

    async def authorize(
        self,
        principal: object,
        tool: object,
        args: object,
        ctx: object,
    ) -> AuthzDecision:
        return AuthzDecision(decision="DENY", reason="forbidden", authorizer_name=self.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_authz_check_span_emitted_with_attrs(
    make_container: object,
    span_exporter: object,
) -> None:
    """A tool call emits exactly one authz.check span with required attrs."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from voussoir.auth.protocol import Authorizer

    assert isinstance(span_exporter, InMemorySpanExporter)

    c = make_container()  # type: ignore[operator]
    c.bind(Authorizer, _AllowAuthorizer())  # type: ignore[type-abstract]

    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    await ex.invoke(_ping, _ping.input_schema(), ctx)

    authz_spans = [s for s in span_exporter.get_finished_spans() if s.name == "authz.check"]
    assert len(authz_spans) == 1
    span = authz_spans[0]
    attrs = dict(span.attributes or {})
    assert attrs.get("authorizer_name") == "allow-a6"
    assert attrs.get("decision") == "ALLOW"
    assert attrs.get("reason") == "ok"


async def test_authz_check_span_nested_under_tool_call(
    make_container: object,
    span_exporter: object,
) -> None:
    """authz.check is a child of tool.call (sibling of capability.check / taint.check)."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from voussoir.auth.protocol import Authorizer

    assert isinstance(span_exporter, InMemorySpanExporter)

    c = make_container()  # type: ignore[operator]
    c.bind(Authorizer, _AllowAuthorizer())  # type: ignore[type-abstract]

    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    await ex.invoke(_ping, _ping.input_schema(), ctx)

    spans = span_exporter.get_finished_spans()
    tool_call = next((s for s in spans if s.name == "tool.call"), None)
    authz = next((s for s in spans if s.name == "authz.check"), None)

    assert tool_call is not None, "expected tool.call span"
    assert authz is not None, "expected authz.check span"

    # OTel parent linkage: child.parent.span_id == parent.context.span_id
    assert authz.parent is not None
    assert authz.parent.span_id == tool_call.context.span_id


async def test_authz_check_span_emitted_on_deny(
    make_container: object,
    span_exporter: object,
) -> None:
    """authz.check span is emitted even when the decision is DENY."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from voussoir.agent.policy import PolicyViolationError
    from voussoir.auth.protocol import Authorizer

    assert isinstance(span_exporter, InMemorySpanExporter)

    c = make_container()  # type: ignore[operator]
    c.bind(Authorizer, _DenyAuthorizer())  # type: ignore[type-abstract]

    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
        container=c,
    )
    with pytest.raises(PolicyViolationError):
        await ex.invoke(_ping, _ping.input_schema(), ctx)

    authz_spans = [s for s in span_exporter.get_finished_spans() if s.name == "authz.check"]
    assert len(authz_spans) == 1
    attrs = dict(authz_spans[0].attributes or {})
    assert attrs.get("decision") == "DENY"
    assert attrs.get("authorizer_name") == "deny-a6"
    assert attrs.get("reason") == "forbidden"
