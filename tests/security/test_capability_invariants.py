"""Hard-invariant tests: Capability mask enforcement at StandardExecutor.invoke.

Phase 5 spec §4.4 Rule 1. A tool whose declared capability is not a subset of
the agent's allowed_capabilities cannot run. Fail-closed.
"""

import pytest

from voussoir.agent import PolicyViolation, PolicyViolationError
from voussoir.executors.standard import StandardExecutor
from voussoir.tools import Capability, ToolContext, tool


@tool(capability=Capability.EXFILTRATION, name="send_email")
async def send_email(target: str) -> str:
    return f"sent to {target}"


async def test_exfiltration_blocked_when_not_in_mask():
    ex = StandardExecutor()
    args = send_email.input_schema(target="x@example.com")
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC | Capability.READ_PRIVATE,
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        await ex.invoke(send_email, args, ctx)
    assert excinfo.value.violation == PolicyViolation.CAPABILITY_DENIED


async def test_exfiltration_allowed_when_in_mask():
    ex = StandardExecutor()
    args = send_email.input_schema(target="x@example.com")
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC | Capability.EXFILTRATION,
    )
    result = await ex.invoke(send_email, args, ctx)
    assert result == "sent to x@example.com"


def test_none_capability_tool_raises_at_decoration():
    """Since v1.0.2 D3, Capability.NONE is forbidden at @tool decoration time."""
    with pytest.raises(ValueError, match="capability="):

        @tool(capability=Capability.NONE, name="ping")
        async def ping() -> str:
            return "pong"
