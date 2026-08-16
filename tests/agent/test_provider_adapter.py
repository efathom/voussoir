"""Locks OpenAIToolCalls adapter shape + dispatch (v1.1.0 F6)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from ctxforge.protocols.llm import LLMResponse

from voussoir.agent.turn_adapter import (
    AnthropicToolCalls,
    OpenAIToolCalls,
    adapter_for,
)
from voussoir.tools import Capability, tool


@tool(capability=Capability.READ_PUBLIC, name="search")
async def search(query: str) -> str:
    return f"results for {query}"


def test_openai_serialize_tool_uses_function_wrapper() -> None:
    adapter = OpenAIToolCalls()
    schema = adapter.serialize_tool(search)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert "parameters" in schema["function"]


def test_openai_extract_tool_calls_decodes_json_arguments() -> None:
    """OpenAI returns tool_calls[].function.arguments as a JSON STRING."""
    adapter = OpenAIToolCalls()
    response = LLMResponse(
        content="",
        model="gpt-5",
        input_tokens=1,
        output_tokens=1,
        finish_reason="tool_calls",
        raw_response={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "hello"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    calls = adapter.extract_tool_calls(response)
    assert calls == [{"id": "call_1", "name": "search", "arguments": {"query": "hello"}}]


def test_openai_build_tool_result_carries_tool_call_id() -> None:
    adapter = OpenAIToolCalls()
    msg = adapter.build_tool_result_message("call_1", "results", "search")
    # The exact ChatMessage shape depends on ctxforge; whatever it is,
    # "call_1" must be retrievable from the constructed message.
    serialized: Any = msg.__dict__ if hasattr(msg, "__dict__") else vars(msg)
    flat = repr(serialized)
    assert "call_1" in flat
    assert "results" in flat


def test_adapter_for_anthropic() -> None:
    llm = MagicMock()
    llm.name = "anthropic"
    assert isinstance(adapter_for(llm), AnthropicToolCalls)


def test_adapter_for_openai() -> None:
    llm = MagicMock()
    llm.name = "openai"
    assert isinstance(adapter_for(llm), OpenAIToolCalls)


def test_adapter_for_openrouter_reuses_openai_adapter() -> None:
    """OpenRouter is OpenAI-compatible, so it shares the OpenAI tool-calling shape."""
    llm = MagicMock()
    llm.name = "openrouter"
    assert isinstance(adapter_for(llm), OpenAIToolCalls)


def test_adapter_for_unknown_provider_raises() -> None:
    llm = MagicMock()
    llm.name = "cohere"
    with pytest.raises(NotImplementedError) as excinfo:
        adapter_for(llm)
    msg = str(excinfo.value)
    assert "cohere" in msg
    # All supported providers should be named in the message
    assert "anthropic" in msg.lower()
    assert "openai" in msg.lower()
    assert "openrouter" in msg.lower()
