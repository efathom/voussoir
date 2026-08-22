"""Per-provider tool-calling shape adapter (v1.1.0 F5, F6).

ILLMProvider.chat() takes a ``functions=[{...}]`` arg and returns an
LLMResponse whose tool-call shape differs between providers. This module
defines an internal Protocol that normalises the three things voussoir
cares about: serialising tools, extracting tool calls, and building
tool-result messages.

The adapter is selected once per run via ``llm.name``. Not exposed
publicly because per-provider quirks are an implementation detail; not
in ``voussoir.agent.__all__``. F5 ships ``AnthropicToolCalls``; F6 adds
``OpenAIToolCalls`` (also used by OpenRouter, which is OpenAI-compatible).
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from ctxforge.protocols.llm import ChatMessage, LLMResponse

from voussoir.observability.logging_setup import get_logger
from voussoir.tools.protocol import Tool

_log = get_logger(__name__)


@runtime_checkable
class ToolCallAdapter(Protocol):
    """Use this when adapting voussoir to a new LLM provider's tool-calling shape.

    Three methods cover the full surface: ``serialize_tool`` (when calling
    ``llm.chat`` with ``functions=[...]``), ``extract_tool_calls`` (when
    reading the response back), and ``build_tool_result_message`` (when
    appending the tool output to the message history for the next turn).
    """

    def serialize_tool(self, tool: Tool) -> dict[str, Any]:
        """Serialise a Tool into the dict shape expected by llm.chat(functions=[...])."""
        ...

    def extract_tool_calls(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Return normalised tool-call dicts: [{id, name, arguments: dict}].

        Returns an empty list when the response contains no tool calls.
        """
        ...

    def build_tool_result_message(
        self,
        tool_call_id: str,
        content: str,
        tool_name: str,
    ) -> ChatMessage:
        """Build the ChatMessage that carries a tool-result back to the LLM."""
        ...


class AnthropicToolCalls:
    """Anthropic shape: tool_use content blocks, tool_result via function role.

    ``llm.chat(functions=[...])`` accepts the OpenAI-style
    ``{name, description, parameters}`` shape; ``AnthropicLLMProvider``
    translates ``parameters`` → ``input_schema`` internally before the API
    call. Tool-call results travel as ``ChatMessage(role="function", ...)``
    with ``function_call={"tool_use_id": ...}``; ``AnthropicLLMProvider``
    wraps them into ``tool_result`` content blocks.
    """

    name = "anthropic"

    def serialize_tool(self, tool: Tool) -> dict[str, Any]:
        """Return the OpenAI-style function dict that llm.chat(functions=[...]) expects.

        AnthropicLLMProvider translates the ``parameters`` key to
        ``input_schema`` before passing to the Anthropic Messages API;
        voussoir never sends the Anthropic native shape directly.
        """
        schema = tool.input_schema.model_json_schema()
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        }

    def extract_tool_calls(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Extract tool calls from the LLMResponse raw_response dict.

        AnthropicLLMProvider surfaces tool-use blocks as
        ``raw_response["tool_calls"]`` — a list of ``{id, name, arguments}``
        dicts. The test-mock path uses the same shape (set by conftest's
        ``stub_llm``).
        """
        return (response.raw_response or {}).get("tool_calls") or []

    def build_tool_result_message(
        self,
        tool_call_id: str,
        content: str,
        tool_name: str,
    ) -> ChatMessage:
        """Build the function-role ChatMessage that carries a tool result.

        AnthropicLLMProvider reads ``function_call["tool_use_id"]`` to
        produce the ``tool_result`` content block it sends to the API.
        The ``name`` field on ChatMessage records the tool name for
        downstream inspection (e.g. audit logs, context compressors).
        """
        return ChatMessage(
            role="function",
            content=content,
            name=tool_name,
            function_call={"tool_use_id": tool_call_id},
        )


class OpenAIToolCalls:
    """OpenAI shape: function-wrapped tools, JSON-string arguments, tool_call_id result.

    Use this when voussoir runs against an OpenAI-compatible provider
    (gpt-4, gpt-5, etc.). OpenRouter (`ctxforge.llm.openrouter_provider`)
    speaks this same wire shape, so ``adapter_for`` routes its ``name ==
    "openrouter"`` here too. The wire shape diverges from Anthropic in
    three places: tool schemas are wrapped in a ``{type, function}``
    envelope, ``tool_calls`` live under
    ``choices[0].message.tool_calls``, and ``arguments`` arrive as a
    JSON string rather than a dict.

    Tool-result messages travel as ``ChatMessage(role="function", ...)``
    with ``function_call={"tool_call_id": ...}``.  ``ChatMessage`` has
    no dedicated ``tool_call_id`` field; the ``function_call`` dict is
    the extension hook used here (mirroring how ``AnthropicToolCalls``
    threads ``tool_use_id``).  A future ctxforge bump that adds
    ``role="tool"`` support would let this be expressed more precisely.

    Note: ``OpenAILLMProvider.chat()`` currently sets
    ``raw_response=None``, so ``extract_tool_calls`` will always return
    ``[]`` when called against the live provider.  F6 ships the adapter
    shape so tests and mocks work correctly; full live-provider wiring
    awaits a ctxforge fix.
    """

    name = "openai"

    def serialize_tool(self, tool: Tool) -> dict[str, Any]:
        """Return the function-wrapped dict that OpenAI's ``functions=`` param expects.

        OpenAI requires the ``{type: "function", function: {...}}`` envelope;
        this differs from the Anthropic path which uses a flat dict.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema.model_json_schema(),
            },
        }

    def extract_tool_calls(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Extract tool calls from the OpenAI raw_response shape.

        OpenAI surfaces ``tool_calls`` under
        ``choices[0].message.tool_calls`` where each entry's
        ``function.arguments`` is a JSON string.  This method JSON-decodes
        the arguments and returns the normalised
        ``[{id, name, arguments: dict}]`` list.
        """
        raw = response.raw_response or {}
        if not isinstance(raw, dict):
            return []
        choices = raw.get("choices") or []
        if not choices:
            return []
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict) or tc.get("type") != "function":
                continue
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                # Nothing dispatchable; skipping is better than a KeyError that
                # takes down the run (audit, minor).
                _log.warning("openai_tool_call_missing_name", tool_call=str(tc)[:200])
                continue
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args_dict: dict[str, Any] = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    # Malformed JSON arguments are a routine LLM failure mode.
                    # Raising here aborted the entire Agent.run/stream, unlike
                    # every other bad-args path — dispatch.py turns a schema
                    # failure into a per-tool BLOCK the model can recover from.
                    # Hand dispatch an empty dict so it produces that same
                    # BLOCK outcome instead (audit, minor).
                    _log.warning(
                        "openai_tool_call_bad_arguments_json",
                        tool_name=name,
                        arguments_preview=args_raw[:200],
                    )
                    args_dict = {}
            else:
                args_dict = args_raw or {}
            out.append({"id": tc.get("id", ""), "name": name, "arguments": args_dict})
        return out

    def build_tool_result_message(
        self,
        tool_call_id: str,
        content: str,
        tool_name: str,
    ) -> ChatMessage:
        """Build the ChatMessage carrying a tool result back to an OpenAI provider.

        ``ChatMessage`` has no ``tool_call_id`` field; the id is attached
        via ``function_call={"tool_call_id": ...}`` — the same extension
        hook ``AnthropicToolCalls`` uses for ``tool_use_id``.
        ``OpenAILLMProvider.chat()`` passes ``function_call`` through to
        the wire payload unchanged, so a provider that reads
        ``function_call["tool_call_id"]`` can reconstruct the binding.
        """
        return ChatMessage(
            role="function",
            content=content,
            name=tool_name,
            function_call={"tool_call_id": tool_call_id},
        )


def adapter_for(llm: Any) -> ToolCallAdapter:
    """Select the per-provider tool-calling adapter based on ``llm.name``.

    Dispatches on ``llm.name``. v1.1.0 ships Anthropic + OpenAI; OpenRouter
    reuses the OpenAI-compatible adapter. Unsupported providers raise a
    clear ``NotImplementedError`` naming the unknown value and listing
    supported options.
    """
    provider_name = getattr(llm, "name", "")
    if provider_name == "anthropic":
        return AnthropicToolCalls()
    if provider_name in ("openai", "openrouter"):
        return OpenAIToolCalls()
    raise NotImplementedError(
        f"Tool-calling adapter for LLM {provider_name!r} not yet shipped; "
        f"supported providers: anthropic, openai, openrouter"
    )
