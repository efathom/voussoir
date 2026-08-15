"""stub_llm() + multi_turn_llm() — mocked ILLMProvider builders for tests."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

from ctxforge.protocols.llm import ILLMProvider, LLMResponse


def stub_llm(
    *,
    content: str = "ok",
    finish_reason: Literal["end_turn", "tool_use", "stop", "tool_calls"] = "end_turn",
    input_tokens: int = 1,
    output_tokens: int = 1,
    tool_calls: list[dict[str, Any]] | None = None,
    raw_response: dict[str, Any] | None = None,
    name: str = "anthropic",
    side_effect: Any = None,
) -> MagicMock:
    """Use this to script a single-turn LLM response.

    Defaults give you a clean end_turn 'ok' response; pass tool_calls= for
    Anthropic-shaped tool_use turns, or raw_response= to supply a fully
    custom response payload. For multi-step agents use multi_turn_llm instead.

    name: sets the .name attribute on the mock (default "anthropic").

    side_effect: when given, .chat is wired with AsyncMock(side_effect=...)
    instead of a normal return value. Pass an exception class/instance to
    test error paths, or a list of responses to script ordered turns
    (multi_turn_llm wraps this pattern with stricter typing).
    """
    llm = MagicMock(spec=ILLMProvider)
    llm.name = name
    if side_effect is not None:
        llm.chat = AsyncMock(side_effect=side_effect)
        return llm

    resolved_raw: dict[str, Any] = raw_response if raw_response is not None else {}
    if tool_calls:
        resolved_raw = dict(resolved_raw)
        resolved_raw["tool_calls"] = tool_calls

    response = LLMResponse(
        content=content,
        model="stub",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        raw_response=resolved_raw,
    )
    llm.chat = AsyncMock(return_value=response)
    return llm


def multi_turn_llm(turns: list[LLMResponse], *, name: str = "anthropic") -> MagicMock:
    """Use this to script a sequence of LLM responses for multi-step agents.

    Each turn is consumed in order via side_effect. After the list is
    exhausted, further .chat() calls raise StopIteration.

    name: sets the .name attribute on the mock (default "anthropic").
    """
    llm = MagicMock(spec=ILLMProvider)
    llm.name = name
    llm.chat = AsyncMock(side_effect=list(turns))
    return llm
