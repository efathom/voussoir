"""Hard-invariant tests: Trust-vs-Capability gate + output tagging.

Phase 5 spec §4.4 Rule 2 + §4.3. EXFILTRATION refuses to run with UNTRUSTED
in taint; tool outputs add their Cap-derived Trust tag to ctx.taint.
"""

import pytest

from voussoir.agent import PolicyViolation, PolicyViolationError
from voussoir.executors.standard import StandardExecutor
from voussoir.guardrails import Trust
from voussoir.tools import Capability, ToolContext, tool


@tool(capability=Capability.EXFILTRATION, name="exfil")
async def exfil(target: str) -> str:
    return f"sent to {target}"


@tool(capability=Capability.READ_PUBLIC, name="fetch_public")
async def fetch_public() -> str:
    return "external data"


@tool(capability=Capability.READ_PRIVATE, name="fetch_private")
async def fetch_private() -> str:
    return "internal data"


@tool(capability=Capability.READ_PUBLIC, name="curated", output_trust=Trust.INTERNAL)
async def curated_public() -> str:
    return "trusted public cache"


async def test_exfiltration_blocked_when_untrusted_in_taint():
    ex = StandardExecutor()
    args = exfil.input_schema(target="x@example.com")
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.EXFILTRATION,
        taint={Trust.UNTRUSTED},
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        await ex.invoke(exfil, args, ctx)
    assert excinfo.value.violation == PolicyViolation.TAINT_EXFILTRATION


async def test_exfiltration_allowed_when_taint_only_internal():
    ex = StandardExecutor()
    args = exfil.input_schema(target="x@example.com")
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.EXFILTRATION,
        taint={Trust.INTERNAL, Trust.SYSTEM},
    )
    result = await ex.invoke(exfil, args, ctx)
    assert result == "sent to x@example.com"


async def test_read_public_records_untrusted_into_taint():
    """When a READ_PUBLIC tool returns, the run's taint set gains UNTRUSTED."""
    ex = StandardExecutor()
    args = fetch_public.input_schema()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
    )
    await ex.invoke(fetch_public, args, ctx)
    # Pydantic v2 deep-copies mutable args at construction time, so we check
    # ctx.taint (the live set the executor mutates), not a pre-construction ref.
    assert Trust.UNTRUSTED in ctx.taint


async def test_read_private_records_internal_into_taint():
    ex = StandardExecutor()
    args = fetch_private.input_schema()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PRIVATE,
    )
    await ex.invoke(fetch_private, args, ctx)
    assert Trust.INTERNAL in ctx.taint
    assert Trust.UNTRUSTED not in ctx.taint


async def test_output_trust_override_wins():
    """A tool with explicit output_trust=Trust.INTERNAL overrides the Cap default."""
    ex = StandardExecutor()
    args = curated_public.input_schema()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
    )
    await ex.invoke(curated_public, args, ctx)
    assert Trust.INTERNAL in ctx.taint
    assert Trust.UNTRUSTED not in ctx.taint
