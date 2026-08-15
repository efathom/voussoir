"""End-to-end Agent with a mocked OpenAI provider + tool-call round-trip (v1.1.0 F7)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ILLMProvider, LLMResponse

from voussoir import Agent
from voussoir.tools import Capability, tool


@tool(capability=Capability.READ_PUBLIC, name="echo")
async def echo(text: str) -> str:
    return f"echoed: {text}"


def make_openai_two_turn_llm() -> MagicMock:
    """First turn: OpenAI-shaped tool_call for echo. Second turn: 'done'."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "openai"
    llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                model="gpt-5",
                input_tokens=10,
                output_tokens=2,
                finish_reason="tool_calls",
                raw_response={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_e1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"text": "hi"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            ),
            LLMResponse(
                content="done",
                model="gpt-5",
                input_tokens=20,
                output_tokens=3,
                finish_reason="end_turn",
                raw_response={"choices": [{"message": {"content": "done"}}]},
            ),
        ]
    )
    return llm


async def test_openai_agent_with_tools_runs_end_to_end(make_container: Any) -> None:
    """Agent(tools=[echo]) with an OpenAI LLM completes a tool-call round-trip."""
    llm = make_openai_two_turn_llm()
    c = make_container(llm)
    a = Agent(name="oa", container=c, tools=[echo])
    result = await a.run("say hi via echo")
    assert result.finish_reason == "completed"
    assert "done" in result.output


async def test_unsupported_provider_raises_clear_error(make_container: Any) -> None:
    """Non-Anthropic, non-OpenAI provider raises a clear NotImplementedError."""
    llm = MagicMock(spec=ILLMProvider)
    llm.name = "cohere"
    llm.chat = AsyncMock()
    c = make_container(llm)
    a = Agent(name="x", container=c, tools=[echo])
    with pytest.raises(NotImplementedError) as excinfo:
        await a.run("anything")
    msg = str(excinfo.value).lower()
    assert "cohere" in msg
    assert "anthropic" in msg
    assert "openai" in msg
