"""Hard-invariant tests: sub-agent capability clamping (fail-loud).

Phase 5 spec §4.5. When a parent dispatches a sub-agent, the child runs
with `parent.allowed_capabilities & child.allowed_capabilities`. Tools
whose capability doesn't fit are filtered out. Empty-post-clamp registry
(when child had tools) raises CAPABILITY_CLAMPED_EMPTY at delegate-tool-
synthesis time (in the parent's run), not at first call.

Closes Tranche C #1.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from unittest.mock import MagicMock

import pytest

from voussoir import Agent
from voussoir.agent import PolicyViolation, PolicyViolationError
from voussoir.agent.context import AgentContext
from voussoir.agent.delegation import clamp_tools
from voussoir.tools import Capability, tool


@tool(capability=Capability.READ_PUBLIC, name="pub_search")
async def pub_search() -> str:
    return "pub"


@tool(capability=Capability.EXFILTRATION, name="send_email")
async def send_email(target: str) -> str:
    return f"sent to {target}"


def _make_ctx(
    make_container,
    *,
    allowed_capabilities: Capability = Capability.READ_PUBLIC | Capability.READ_PRIVATE,
    delegation_depth: int = 0,
    max_delegation_depth: int = 3,
) -> AgentContext:
    """Build a minimal AgentContext for clamping tests."""
    c = make_container()
    return AgentContext(
        container=c,
        run_id="r",
        trace_id="t",
        session_id="s",
        user_id="test-user",
        engine=MagicMock(),
        _stack=AsyncExitStack(),
        allowed_capabilities=allowed_capabilities,
        delegation_depth=delegation_depth,
        max_delegation_depth=max_delegation_depth,
    )


def test_clamp_tools_filters_to_subset(make_container):
    """Clamping yields only the tools whose caps fit the AND mask."""
    child = Agent(
        name="child",
        container=make_container(),
        tools=[pub_search, send_email],
        allowed_capabilities=Capability.READ_PUBLIC | Capability.EXFILTRATION,
    )
    # Parent allows only READ_PUBLIC → clamped = READ_PUBLIC → only pub_search survives.
    clamped = clamp_tools(child, parent_mask=Capability.READ_PUBLIC)
    assert [t.name for t in clamped] == ["pub_search"]
    # Explicitly confirm the EXFILTRATION tool was filtered out (defends against a
    # silent regression where empty results accidentally pass).
    assert "send_email" not in {t.name for t in clamped}
    assert len(clamped) == 1


def test_clamp_tools_empty_fail_loud(make_container):
    """When all child tools are clamped away, raise CAPABILITY_CLAMPED_EMPTY."""
    child = Agent(
        name="child",
        container=make_container(),
        tools=[send_email],
        allowed_capabilities=Capability.EXFILTRATION,
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        # Parent allows only READ_PUBLIC; child has only EXFILTRATION tool → empty.
        clamp_tools(child, parent_mask=Capability.READ_PUBLIC)
    assert excinfo.value.violation == PolicyViolation.CAPABILITY_CLAMPED_EMPTY


def test_clamp_tools_toolless_child_ok(make_container):
    """A child declared with tools=[] is a valid configuration; no raise."""
    child = Agent(
        name="child",
        container=make_container(),
        tools=[],
        allowed_capabilities=Capability.READ_PUBLIC,
    )
    clamped = clamp_tools(child, parent_mask=Capability.NONE)
    assert clamped == []


def test_build_delegate_tools_fails_loud_on_empty_clamp(make_container):
    """Parent's _build_delegate_tools raises if a child's clamping yields empty.

    This is the spec's "fail-loud at delegate-tool-list time" — the error
    surfaces in the parent's run, not as a per-invocation DELEGATION_REFUSED.
    Uses direct _build_delegate_tools call to avoid needing a live LLM.
    """
    child = Agent(
        name="child",
        container=make_container(),
        tools=[send_email],
        allowed_capabilities=Capability.EXFILTRATION,
    )
    parent = Agent(
        name="parent",
        container=make_container(),
        delegates=[child],
        allowed_capabilities=Capability.READ_PUBLIC,  # excludes EXFILTRATION
    )
    ctx = _make_ctx(
        make_container,
        allowed_capabilities=Capability.READ_PUBLIC,
        delegation_depth=0,
        max_delegation_depth=3,
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        parent._build_delegate_tools(ctx)
    assert excinfo.value.violation == PolicyViolation.CAPABILITY_CLAMPED_EMPTY
    assert "child" in str(excinfo.value)


async def test_delegate_invocation_runs_clamped_child(make_container, stub_llm):
    """When clamping is non-empty, the child runs with the filtered tool list.

    Calls child.delegate() directly with a parent_ctx carrying parent's mask.
    Child has READ_PUBLIC and EXFILTRATION tools; parent mask is READ_PUBLIC only.
    After clamping, the clone used in the run must carry exactly [pub_search] —
    captured via a spy on _with_container.
    """
    llm = stub_llm()
    child = Agent(
        name="child",
        container=make_container(llm),
        tools=[pub_search, send_email],
        allowed_capabilities=Capability.READ_PUBLIC | Capability.EXFILTRATION,
    )
    parent_ctx = _make_ctx(
        make_container,
        allowed_capabilities=Capability.READ_PUBLIC,
        delegation_depth=0,
        max_delegation_depth=3,
    )

    captured: list[Agent] = []
    original = child._with_container

    def spy(container: object, **kwargs: object) -> Agent:
        clone = original(container, **kwargs)  # type: ignore[arg-type]
        captured.append(clone)
        return clone

    child._with_container = spy  # type: ignore[method-assign]
    result = await child.delegate("go", parent_ctx=parent_ctx)

    assert result.output == "ok"
    assert len(captured) == 1
    clone = captured[0]
    # The clone the child ran with carries the clamped tool list.
    assert [t.name for t in clone.tools] == ["pub_search"]
    assert clone.allowed_capabilities == Capability.READ_PUBLIC
