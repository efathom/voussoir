"""Locks streaming cascade gate behavior (Phase 5 Task C7).

Phase 4.5a raised STREAMING_NOT_SUPPORTED unconditionally when a cascade was
configured. C7 lifts this for max_cascade_depth=1 (single-pass cascade after
the stream's `done` event). max_cascade_depth>1 (retry-capable cascade)
remains gated until a future task.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider

from voussoir import Agent
from voussoir.agent import PolicyViolation, PolicyViolationError
from voussoir.agent.cascade import Decision, RequestCascade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _streaming_llm(content: str) -> MagicMock:
    """Build a MagicMock ILLMProvider whose .stream yields a single chunk."""

    async def _gen(*args, **kwargs):  # type: ignore[no-untyped-def]
        yield content

    m = MagicMock(spec=ILLMProvider)
    m.name = "stub"
    m.stream = MagicMock(return_value=_gen())
    return m


class _PassValidator:
    name = "pass-validator"

    async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
        return Decision.PASS


class _FailValidator:
    name = "fail-validator"

    async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
        return Decision.FAIL


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_stream_max_depth_1_pass_emits_cascade_passed(make_container):
    cascade = RequestCascade(verifier=_PassValidator(), max_cascade_depth=1)
    a = Agent(name="x", container=make_container(_streaming_llm("hello")), cascade=cascade)
    events = []
    async for ev in a.stream("hi"):
        events.append(ev)
    kinds = [e.kind for e in events]
    # `done` should come first, then `cascade_passed`
    assert "done" in kinds, f"expected 'done' in {kinds}"
    assert "cascade_passed" in kinds, f"expected 'cascade_passed' in {kinds}"
    done_idx = kinds.index("done")
    cascade_idx = kinds.index("cascade_passed")
    assert cascade_idx > done_idx
    cascade_ev = events[cascade_idx]
    assert cascade_ev.payload["validator_name"] == "pass-validator"
    assert cascade_ev.payload["verdict"] == "PASS"


async def test_stream_max_depth_1_fail_emits_cascade_failed(make_container):
    cascade = RequestCascade(verifier=_FailValidator(), max_cascade_depth=1)
    a = Agent(name="x", container=make_container(_streaming_llm("oops")), cascade=cascade)
    events = []
    async for ev in a.stream("hi"):
        events.append(ev)
    kinds = [e.kind for e in events]
    assert "cascade_failed" in kinds, f"expected 'cascade_failed' in {kinds}"
    failed_ev = events[kinds.index("cascade_failed")]
    assert failed_ev.payload["validator_name"] == "fail-validator"
    assert failed_ev.payload["verdict"] == "fail"


async def test_stream_max_depth_gt_1_still_raises(make_container):
    """Retry-capable cascade still rejects streaming (deferred to future task)."""
    cascade = RequestCascade(verifier=_PassValidator(), max_cascade_depth=2)
    a = Agent(name="x", container=make_container(_streaming_llm("hello")), cascade=cascade)
    with pytest.raises(PolicyViolationError) as excinfo:
        async for _ in a.stream("hi"):
            pass
    assert excinfo.value.violation == PolicyViolation.STREAMING_NOT_SUPPORTED


async def test_stream_no_cascade_works_unchanged(make_container):
    """Without a cascade, stream emits the usual events; no cascade_passed/failed."""
    a = Agent(name="x", container=make_container(_streaming_llm("hi")))
    events = []
    async for ev in a.stream("hi"):
        events.append(ev)
    kinds = {e.kind for e in events}
    assert "cascade_passed" not in kinds
    assert "cascade_failed" not in kinds
