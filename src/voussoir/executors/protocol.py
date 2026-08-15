"""IToolExecutor Protocol — dispatch policy for tool calls (v1.1.0 F1 rename).

Use this when you need a custom dispatch policy (sandboxing,
instrumentation, alternate validation, mocked execution for tests).
Bind on the container, or pass per-run via agent.run(executor=...).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from voussoir.tools.protocol import Tool, ToolContext


@runtime_checkable
class IToolExecutor(Protocol):
    """Use this when implementing a custom Tool dispatch policy.

    StandardExecutor (voussoir.executors.standard) is the default impl bound
    on default_container(). Custom impls (sandboxed, instrumented, mocked)
    bind on the container via c.bind(IToolExecutor, MyExecutor()) or pass
    per-run via agent.run(executor=...).
    """

    name: str

    async def invoke(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> Any: ...


# Back-compat alias: pre-v1.1.0 code imported ToolExecutor. The IToolExecutor
# rename keeps the I-prefix convention symmetric with IGuardrailChain (v1.1.0
# F2) and IMemoryStore (ctxforge). The old name remains importable for one
# release cycle.
ToolExecutor = IToolExecutor
