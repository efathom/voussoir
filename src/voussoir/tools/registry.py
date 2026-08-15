"""ToolRegistry — name → Tool resolution + function-call schema export."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from voussoir.tools.protocol import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # v1.0.2 D9 (b): cache the describe() output. Invalidated on every
        # register() — describe() is called per LLM turn and would otherwise
        # rebuild every tool's JSON schema each turn.
        self._describe_cache: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        # v1.0.2 D9 (b): drop the cache so the next describe() rebuilds.
        self._describe_cache = None

    def register_many(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.register(t)

    def resolve(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"No tool named {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> list[dict[str, Any]]:
        """Return tool descriptions in OpenAI function-call format.

        ctxforge's ILLMProvider.chat(functions=[...]) accepts this shape;
        AnthropicLLMProvider translates it to Anthropic's `tools` format.

        v1.0.2 D9 (b): result is cached and reused across calls; the cache
        is invalidated on register(). The returned list is the cached
        instance — callers must treat it as read-only. Mutating a registered
        tool's `description` or `input_schema` post-registration requires
        re-registration to refresh the cache.
        """
        if self._describe_cache is not None:
            return self._describe_cache
        out: list[dict[str, Any]] = []
        for t in self._tools.values():
            schema = t.input_schema.model_json_schema()
            out.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                }
            )
        self._describe_cache = out
        return out
