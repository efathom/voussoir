"""@tool decorator — wraps an async function into a Tool implementation.

Generates a Pydantic input schema from the function's signature, attaches a
capability tag, and produces an object that satisfies the Tool Protocol from
voussoir.tools.protocol.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from voussoir.auth.credentials import AuthRequirement
from voussoir.guardrails.trust import Trust
from voussoir.tools.protocol import Capability, Tool, ToolContext


def tool(
    *,
    capability: Capability = Capability.NONE,
    name: str | None = None,
    description: str | None = None,
    output_trust: Trust | None = None,
    roles: list[str] | None = None,
    domains: list[str] | None = None,
    auth_requirement: AuthRequirement | None = None,
    sensitive_args: list[str] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """Decorator: turn an async function into a Tool.

    Usage::

        @tool(capability=Capability.READ_PUBLIC)
        async def web_search(query: str, top_k: int = 5) -> list[dict]:
            \"\"\"Search the web.\"\"\"
            ...

    `output_trust` overrides the default Capability→Trust mapping for the
    output tag (e.g., a `Capability.READ_PUBLIC` tool wrapping a curated
    trusted source can declare `output_trust=Trust.INTERNAL`).

    `roles`, `domains`, `auth_requirement`: see RoleAuthorizer / DomainAuthorizer
    (A8) and CredentialBroker.resolve (A7). Stored as attributes on the wrapped
    tool; consumed via ``getattr(tool, "roles", [])`` by executor / authorizers.

    Use ``sensitive_args=['password', 'api_key']`` to redact those keys from
    the INFO ``tool.invoke`` log line. Redaction is logging-only; the tool
    function always receives the real values.
    """

    def wrap(fn: Callable[..., Awaitable[Any]]) -> Tool:
        if capability == Capability.NONE:
            raise ValueError(
                f"@tool decorating {fn.__name__!r}: capability= is required. "
                "Pick the narrowest capability that allows the tool to function "
                "(READ_PUBLIC, READ_PRIVATE, WRITE_PRIVATE, EXFILTRATION). "
                "Capability.NONE bypasses the gate due to bitmask semantics "
                "(0 & X == 0) and was a security footgun."
            )
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@tool requires an async function; got {fn!r}")

        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "")
        input_schema = _schema_from_signature(fn, tool_name)

        decorated = _DecoratedTool(
            name=tool_name,
            description=tool_desc,
            input_schema=input_schema,
            capability=capability,
            fn=fn,
        )
        decorated._output_trust = output_trust  # private attr for executor read
        decorated.roles = roles if roles is not None else []
        decorated.domains = domains if domains is not None else []
        decorated.auth_requirement = auth_requirement
        decorated.sensitive_args = sensitive_args or []
        return decorated

    return wrap


def _schema_from_signature(
    fn: Callable[..., Awaitable[Any]],
    tool_name: str,
) -> type[BaseModel]:
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "ctx":
            # ToolContext is injected by the executor at invoke time, not a
            # tool input.
            continue
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"@tool {tool_name}: parameter {param_name!r} must have a type annotation"
            )
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (param.annotation, Field(default=default))
    return create_model(
        f"{tool_name.title()}Args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class _DecoratedTool:
    """A Tool that wraps a user function and validates inputs against the schema."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        capability: Capability,
        fn: Callable[..., Awaitable[Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.capability = capability
        self._fn = fn
        # v1.0.2 D9 (a): cache whether the wrapped fn accepts `ctx` once at
        # decoration time. `inspect.signature(fn)` is non-trivial and was
        # previously called on every invoke — but the answer never changes for
        # a given fn, so we compute it once here.
        self._wants_ctx: bool = "ctx" in inspect.signature(fn).parameters
        # Set by the @tool decorator after construction; None means "use default
        # Cap→Trust mapping in StandardExecutor". Not part of Tool Protocol.
        self._output_trust: Trust | None = None
        # Phase 6 A3: authz convenience attributes — set by @tool decorator.
        # Not part of the Tool Protocol; consumers use getattr(tool, "roles", []).
        self.roles: list[str] = []
        self.domains: list[str] = []
        self.auth_requirement: AuthRequirement | None = None
        # v1.0.1 T6: arg keys to redact from the INFO tool.invoke log line.
        # Not part of the Tool Protocol; consumers use getattr(tool, "sensitive_args", []).
        self.sensitive_args: list[str] = []

    async def invoke(self, args: BaseModel, ctx: ToolContext) -> Any:
        kwargs = args.model_dump()
        # v1.0.2 D9 (a): use the cached _wants_ctx flag instead of calling
        # inspect.signature on every invoke.
        if self._wants_ctx:
            return await self._fn(**kwargs, ctx=ctx)
        return await self._fn(**kwargs)
