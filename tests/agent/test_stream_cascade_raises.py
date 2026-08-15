"""Phase 4.5a — Agent.stream raises STREAMING_NOT_SUPPORTED when cascade is set."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider

from voussoir.agent.agent import Agent
from voussoir.agent.cascade import Decision, RequestCascade
from voussoir.agent.policy import PolicyViolation, PolicyViolationError
from voussoir.agent.result import AgentResult
from voussoir.protocols import ILLMProvider as ILLMProviderProto


class _AlwaysPass:
    name = "always_pass"

    async def validate(self, result: AgentResult[Any], *, task: str) -> Decision:
        return Decision.PASS


@pytest.mark.asyncio
async def test_stream_raises_when_cascade_set(make_container, stub_llm) -> None:
    """An agent with a cascade configured raises STREAMING_NOT_SUPPORTED on
    .stream() — Phase 4.5a documents this limitation explicitly rather than
    silently bypassing the cascade gate (P1 #12)."""
    c = make_container(stub_llm())
    cascade = RequestCascade(verifier=_AlwaysPass(), escalation=None)
    agent = Agent("lead", instructions="", cascade=cascade, container=c)
    with pytest.raises(PolicyViolationError) as exc_info:
        async for _ in agent.stream("test"):
            pass  # should raise before the first yield
    assert exc_info.value.violation is PolicyViolation.STREAMING_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_stream_without_cascade_unchanged(make_container) -> None:
    """Sanity: agents without a cascade still stream normally."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"

    async def _stream_gen():
        yield "hello"

    llm.stream = MagicMock(return_value=_stream_gen())
    c = make_container(llm)
    c.bind(ILLMProviderProto, llm)
    agent = Agent("lead", container=c)  # no cascade
    events = [e async for e in agent.stream("test")]
    kinds = [e.kind for e in events]
    assert "token" in kinds and kinds[-1] == "done"
