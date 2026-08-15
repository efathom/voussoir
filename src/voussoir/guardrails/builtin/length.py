"""Length-cap guardrails: input, args, tool output.

Use this when you want to bound the size of LLM-facing content. Each cap raises
BLOCK with a reason that names the size + threshold; nothing rewrites or trims.
"""

from __future__ import annotations

import json
from typing import Literal

from voussoir.guardrails.protocol import GuardrailPayload, GuardrailVerdict


class InputLengthCap:
    """Block input payloads larger than `max_chars`.

    Use this when the agent's initial user-input should be bounded (e.g. to
    prevent context-flooding attacks).
    """

    name = "input_length_cap"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "input"

    def __init__(self, *, max_chars: int = 100_000) -> None:
        self.max_chars = max_chars

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        if len(payload.content) > self.max_chars:
            return GuardrailVerdict(
                verdict="BLOCK",
                reason=f"input exceeds {self.max_chars} chars (got {len(payload.content)})",
            )
        return GuardrailVerdict(verdict="ALLOW")


class ArgsSizeCap:
    """Block tool calls whose serialized args exceed `max_args_bytes`.

    Use this when a tool's `tool_args` dict could become very large (e.g.
    embedded blobs); we cap the JSON-serialized byte length.
    """

    name = "args_size_cap"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_call"

    def __init__(self, *, max_args_bytes: int = 64 * 1024) -> None:
        self.max_args_bytes = max_args_bytes

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        size = len(json.dumps(payload.tool_args or {}))
        if size > self.max_args_bytes:
            return GuardrailVerdict(
                verdict="BLOCK",
                reason=f"args {size}B exceed {self.max_args_bytes}B",
            )
        return GuardrailVerdict(verdict="ALLOW")


class ToolOutputSizeCap:
    """Block tool outputs larger than `max_output_bytes`.

    Use this when a tool's response could be very large (web pages, file
    contents, search results); prevents downstream context flooding.
    """

    name = "tool_output_size_cap"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_output"

    def __init__(self, *, max_output_bytes: int = 1024 * 1024) -> None:
        self.max_output_bytes = max_output_bytes

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        if len(payload.content) > self.max_output_bytes:
            return GuardrailVerdict(
                verdict="BLOCK",
                reason=f"output {len(payload.content)}B exceeds {self.max_output_bytes}B",
            )
        return GuardrailVerdict(verdict="ALLOW")
