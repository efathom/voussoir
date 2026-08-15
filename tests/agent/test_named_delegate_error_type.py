"""Phase 4.5a P1 #20 — NamedDelegate raises DELEGATE_NOT_FOUND not MAX_STEPS."""

from __future__ import annotations

import pytest

from voussoir.agent.context import AgentContext
from voussoir.agent.delegate import NamedDelegate
from voussoir.agent.policy import PolicyViolation, PolicyViolationError


@pytest.mark.asyncio
async def test_named_delegate_no_registry_raises_delegate_not_found(
    make_container, stub_llm
) -> None:
    c = make_container(stub_llm())
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        nd = NamedDelegate("nobody")
        with pytest.raises(PolicyViolationError) as exc_info:
            await nd.delegate("task", parent_ctx=ctx)
        assert exc_info.value.violation is PolicyViolation.DELEGATE_NOT_FOUND


@pytest.mark.asyncio
async def test_named_delegate_unknown_name_raises_delegate_not_found(
    make_container, stub_llm
) -> None:
    from voussoir.agent.registry import AgentRegistry

    c = make_container(stub_llm())
    c.bind(AgentRegistry, AgentRegistry())  # empty registry
    async with await AgentContext.open(container=c, run_id="r", session_id="s", user_id="u") as ctx:
        nd = NamedDelegate("nobody")
        with pytest.raises(PolicyViolationError) as exc_info:
            await nd.delegate("task", parent_ctx=ctx)
        assert exc_info.value.violation is PolicyViolation.DELEGATE_NOT_FOUND
