"""Locks AgentResult.authz_decisions accumulation across run paths (Phase 6 A5)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Agent
from voussoir.auth import Authorizer, AuthzDecision
from voussoir.container import Container
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.protocols import IMemoryStore, ISessionStore
from voussoir.tools import Capability, ToolContext, tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOOL_NAME = "ping_a5"


@tool(capability=Capability.READ_PUBLIC, name=_TOOL_NAME)
async def _ping(ctx: ToolContext) -> str:
    return "pong"


class _AllowRecordingAuthorizer:
    """Authorizer that always ALLOWs and records every decision."""

    name = "allow-recording-a5"

    async def authorize(self, principal: Any, tool_fn: Any, args: Any, ctx: Any) -> AuthzDecision:
        return AuthzDecision(decision="ALLOW", authorizer_name=self.name)


def _make_tool_calling_container(llm: ILLMProvider) -> Container:
    """Container bound to an LLM that calls _ping once, then returns 'done'."""
    c = Container()
    c.bind(ILLMProviderProto, llm)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    c.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]

    from voussoir.a2a.keys import EnvKeyProvider, KeyProvider

    c.bind(KeyProvider, EnvKeyProvider(allow_ephemeral=True))  # type: ignore[type-abstract]
    return c


def _tool_use_llm() -> MagicMock:
    """Mock LLM: first response calls _ping, second response is 'done'."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="tool_use",
                raw_response={"tool_calls": [{"id": "tc_a5", "name": _TOOL_NAME, "arguments": {}}]},
            ),
            LLMResponse(
                content="done",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                finish_reason="end_turn",
                raw_response=None,
            ),
        ]
    )
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_agent_result_authz_decisions_populates() -> None:
    """After a tool call, AgentResult.authz_decisions contains one ALLOW record."""
    llm = _tool_use_llm()
    c = _make_tool_calling_container(llm)
    c.bind(Authorizer, _AllowRecordingAuthorizer())

    a = Agent(name="a5-agent", container=c, tools=[_ping])
    result = await a.run("ping please")

    assert result.authz_decisions, "authz_decisions should be non-empty after a tool call"
    assert result.authz_decisions[0].decision == "ALLOW"
    assert result.authz_decisions[0].authorizer_name == _AllowRecordingAuthorizer.name


async def test_agent_result_authz_decisions_default_empty_when_no_tool_call(
    make_container: Any, stub_llm: Any
) -> None:
    """A run with no tool_call still emits empty authz_decisions (not None)."""
    a = Agent(name="a5-no-tool", container=make_container(stub_llm(content="just a reply")))
    result = await a.run("hi")
    assert result.authz_decisions == []


async def test_agent_result_authz_decisions_blocked_run_empty() -> None:
    """When the input guardrail BLOCKs (no tool calls), authz_decisions is empty."""
    from voussoir.guardrails import (
        DefaultGuardrailChain,
        GuardrailPayload,
        GuardrailVerdict,
        IGuardrailChain,
    )

    class _BlockAll:
        name = "block-all"
        stage = "input"

        async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
            return GuardrailVerdict(verdict="BLOCK", reason="blocked")

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="stub",
            input_tokens=0,
            output_tokens=0,
            finish_reason="end_turn",
            raw_response=None,
        )
    )
    c = _make_tool_calling_container(llm)
    c.bind(IGuardrailChain, DefaultGuardrailChain([_BlockAll()]))  # type: ignore[type-abstract, list-item]

    a = Agent(name="a5-blocked", container=c)
    result = await a.run("bad input")

    assert result.finish_reason == "blocked"
    assert result.authz_decisions == []
