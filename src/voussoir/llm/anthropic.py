"""Anthropic LLM provider — implements ctxforge's ILLMProvider contract.

This is what `default_container()` lazy-binds to ILLMProvider when ANTHROPIC_API_KEY
is set. Should eventually be upstreamed into ctxforge (see Phase 0 audit).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from ctxforge.protocols.llm import ChatMessage, LLMResponse


class AnthropicLLMProvider:
    """Adapts Anthropic's Messages API to ctxforge's ILLMProvider Protocol.

    Notes on Anthropic ↔ ctxforge translation:
    - Anthropic splits the system prompt out of `messages` into a top-level
      `system` arg. We extract role=='system' messages, concatenate them, pass
      as `system`.
    - Anthropic uses `tools` (with `input_schema`); ctxforge passes `functions`
      (with `parameters`). Same JSON Schema; we rename keys.
    - Anthropic's `stop_reason` becomes `LLMResponse.finish_reason` verbatim.
    - Tool-use blocks in the response surface via
      `raw_response["tool_calls"]` — a list of `{id, name, arguments}` dicts.
      The run loop reads this.
    """

    def __init__(
        self,
        *,
        client: AsyncAnthropic | None = None,
        default_model: str = "claude-opus-4-7",
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        # timeout_s bounds any single HTTP call so a hung provider cannot pin
        # a run forever; max_retries lets the SDK retry transient 429/5xx
        # failures with exponential backoff before the error propagates to the
        # caller (which may apply its own RetryMiddleware).
        self._client = client or AsyncAnthropic(timeout=timeout_s, max_retries=max_retries)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._default_model

    @classmethod
    def from_env(cls, *, default_model: str = "claude-opus-4-7") -> AnthropicLLMProvider:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return cls(client=AsyncAnthropic(api_key=api_key), default_model=default_model)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_parts: list[str] = []
        non_system: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue

            if m.role == "function":
                # ctxforge ChatMessage uses the OpenAI-legacy "function" role
                # for tool-call replies. Anthropic's API needs these wrapped
                # as a `tool_result` content block in a user message; the
                # tool_use_id comes from `function_call.tool_use_id` (set by
                # voussoir.agent.agent.Agent.run after a tool returns).
                tool_use_id = (m.function_call or {}).get("tool_use_id", "")
                block = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": m.content,
                }
                # Anthropic disallows consecutive user messages — merge adjacent
                # tool-result replies into one user message with multiple blocks.
                if (
                    non_system
                    and non_system[-1]["role"] == "user"
                    and isinstance(non_system[-1]["content"], list)
                    and non_system[-1]["content"]
                    and non_system[-1]["content"][0].get("type") == "tool_result"
                ):
                    non_system[-1]["content"].append(block)
                else:
                    non_system.append({"role": "user", "content": [block]})
                continue

            if m.role == "assistant" and m.function_call and "tool_calls" in m.function_call:
                # Assistant turn that carried tool calls. Anthropic represents
                # this with `tool_use` content blocks alongside any text.
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.function_call["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "input": tc.get("arguments", {}),
                        }
                    )
                non_system.append({"role": "assistant", "content": blocks})
                continue

            non_system.append({"role": m.role, "content": m.content})

        params: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": non_system,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
        }
        if temperature is not None:
            # Some Anthropic models (e.g. claude-opus-4-7) reject `temperature`.
            # Only forward it when the caller explicitly set one.
            params["temperature"] = temperature
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        if stop:
            params["stop_sequences"] = stop
        if functions:
            params["tools"] = [
                {
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "input_schema": f.get("parameters", {}),
                }
                for f in functions
            ]

        t0 = time.time()
        resp = await self._client.messages.create(**params)
        latency_ms = (time.time() - t0) * 1000

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        raw: dict[str, Any] = {}
        if tool_calls:
            raw["tool_calls"] = tool_calls

        return LLMResponse(
            content="".join(text_parts),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            finish_reason=resp.stop_reason,
            latency_ms=latency_ms,
            raw_response=raw or None,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        # The Protocol declares `async def stream(...) -> AsyncIterator[str]`,
        # which mypy reads as a coroutine that returns an iterator
        # (`Coroutine[..., AsyncIterator[str]]`). The standard streaming pattern
        # in Python is to use `async def + yield` (an async generator). We use
        # the standard pattern; consumers do `async for x in p.stream(...)`.
        # ctxforge's OpenAI provider has the same signature shape but raises
        # NotImplementedError, hiding the discrepancy. Worth a Protocol fix
        # upstream — until then, the ignore tells mypy this is intentional.
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        params: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": non_system,
            "max_tokens": max_tokens or 4096,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.chat(
            [ChatMessage(role="user", content=prompt)],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        # Prefer tiktoken (cl100k_base is a close approximation for Claude) when
        # installed; otherwise fall back to a 4-chars/token estimate. The word
        # split previously used here was a poor proxy (compressed code/tables
        # drastically under-count). Provider-reported usage is authoritative on
        # the chat() path; this only feeds stream-path estimation + budget checks.
        del model
        if not text:
            return 0
        try:
            import tiktoken  # noqa: PLC0415 — optional dep

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, (len(text) + 3) // 4)

    def count_message_tokens(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
    ) -> int:
        return sum(self.count_tokens(m.content) for m in messages)
