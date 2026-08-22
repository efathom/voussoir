"""MCPTool — wraps a remote MCP server tool as a voussoir Tool.

`MCPTool.from_server(session, server_name=...)` returns a list of Tool
instances, one per descriptor returned by the MCP server's `tools/list`.
Each tool's name is namespaced as `<server_name>__<tool_name>` so
multiple servers can register without collision (Anthropic's tool name
regex doesn't accept dots, so we use double underscore).

Capability inference uses spec §10.3 conservative defaults; explicit
overrides via `capability_overrides={"name": Capability.X}` always win.

**Session lifetime contract.** An MCPTool holds a reference to the live
`mcp.ClientSession` — that session is owned by `MCPClient.connect_*(...)`,
which is an async context manager. The Tool is only valid while the CM
is open. Calling `tool.invoke()` after the CM has exited raises a
`RuntimeError` with a clear message instead of the opaque anyio
`ClosedResourceError`. Typical pattern:

```python
async with MCPClient.connect_stdio(...) as session:
    tools = await MCPTool.from_server(session, server_name="x")
    agent = Agent(tools=tools)
    result = await agent.run(...)   # OK — session still open
# session closed here; using the tool now raises a clear error.
```
"""

from __future__ import annotations

from typing import Any

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import ClientSession, MCPError, types
from pydantic import BaseModel

from voussoir.guardrails.builtin.injection import find_injection_pattern
from voussoir.mcp.schema import jsonschema_to_pydantic
from voussoir.observability.logging_setup import get_logger
from voussoir.tools.protocol import Capability, ToolContext

_log = get_logger(__name__)


def _model_visible_strings(
    name: str,
    description: str,
    input_schema: type[BaseModel],
) -> list[tuple[str, str]]:
    """Every server-controlled string that ends up in the model's context.

    Returns ``(label, text)`` pairs so the caller can name the offending field
    in its error. Covers the tool name, the tool description, and each
    parameter's ``description`` / ``title`` as they appear in the JSON schema
    the adapter sends to the provider.
    """
    out: list[tuple[str, str]] = [("name", name), ("description", description)]
    try:
        schema = input_schema.model_json_schema()
    except Exception:  # pragma: no cover — a malformed model shouldn't mask the rest
        return out
    for prop_name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        for key in ("description", "title"):
            value = prop.get(key)
            if isinstance(value, str) and value:
                out.append((f"parameter {prop_name!r} {key}", value))
    return out


_READ_PREFIXES = ("search_", "get_", "list_", "read_", "find_", "fetch_")
_WRITE_PREFIXES = ("create_", "send_", "delete_", "post_", "update_", "write_", "remove_")


def infer_capability(
    tool_name: str,
    overrides: dict[str, Capability] | None = None,
) -> Capability:
    """Infer the capability tier from a tool name.

    Spec §10.3:
      - read-only-looking names → READ_PUBLIC
      - write-looking names → EXFILTRATION (most restrictive)
      - everything else → READ_PRIVATE (safe default)
      - explicit override always wins
    """
    if overrides and tool_name in overrides:
        return overrides[tool_name]
    lower = tool_name.lower()
    if any(lower.startswith(p) for p in _READ_PREFIXES):
        return Capability.READ_PUBLIC
    if any(lower.startswith(p) for p in _WRITE_PREFIXES):
        return Capability.EXFILTRATION
    return Capability.READ_PRIVATE


class MCPTool:
    """A remote MCP tool exposed through the voussoir Tool Protocol.

    Holds a reference to the live ClientSession. The lifetime of the
    session is the caller's responsibility — typically the MCPClient async
    context manager that produced it.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        capability: Capability,
        session: ClientSession,
        remote_name: str,
    ) -> None:
        # v1.0.2 D4: screen for prompt-injection patterns before registering.
        #
        # Screening the description alone was not enough (audit M10): the
        # generated Pydantic model carries each PROPERTY's description, and
        # ToolCallAdapter.serialize_tool ships model_json_schema() straight to
        # the model — so a hostile server just moved its payload from the tool
        # description into a parameter description and reached the LLM
        # verbatim. The tool name reaches the model too. Screen all three.
        for field_label, text in _model_visible_strings(name, description, input_schema):
            matched = find_injection_pattern(text)
            if matched is None:
                continue
            _log.warning(
                "mcp_tool_injection_blocked",
                tool_name=name,
                field=field_label,
                pattern=matched.pattern,
                text_preview=text[:100],
            )
            raise ValueError(
                f"MCP tool {name!r} has a {field_label} matching prompt-injection "
                f"pattern {matched.pattern!r}; refusing to register. The MCP server "
                f"may be compromised. Preview: {text[:100]!r}"
            )

        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.capability = capability
        self._session = session
        self._remote_name = remote_name

    @classmethod
    async def from_server(
        cls,
        session: ClientSession,
        *,
        server_name: str,
        capability_overrides: dict[str, Capability] | None = None,
    ) -> list[MCPTool]:
        """List remote tools and wrap each as an MCPTool."""
        listing = await session.list_tools()
        out: list[MCPTool] = []
        for descriptor in listing.tools:
            input_model = jsonschema_to_pydantic(
                f"{server_name}_{descriptor.name}",
                descriptor.input_schema,
            )
            out.append(
                cls(
                    # Use `__` as the namespace separator: Anthropic's tool name
                    # validator is `^[a-zA-Z0-9_-]{1,128}$` — no dots allowed.
                    # Double underscore is collision-resistant (single underscores
                    # are common in tool names) and round-trips through every
                    # major LLM provider's tool-name regex.
                    name=f"{server_name}__{descriptor.name}",
                    description=descriptor.description or "",
                    input_schema=input_model,
                    capability=infer_capability(
                        descriptor.name,
                        overrides=capability_overrides,
                    ),
                    session=session,
                    remote_name=descriptor.name,
                )
            )
        return out

    async def invoke(self, args: BaseModel, ctx: ToolContext) -> Any:
        """Call the remote MCP tool and return a stringified text result.

        Phase 2 ignores ctx (MCP tools/call has no first-class trace_id);
        Task 2.5 wires progress callbacks for observability. Multi-block
        results are squashed into one string for v1; non-text blocks are
        dropped.

        Raises a clear RuntimeError if the underlying session has been
        closed (the surrounding `MCPClient.connect_*` async context
        manager has exited).
        """
        try:
            result = await self._session.call_tool(
                self._remote_name,
                arguments=args.model_dump(),
            )
        except (ClosedResourceError, BrokenResourceError, EndOfStream, MCPError) as exc:
            # v2 SDK reports a closed session as MCPError(code=CONNECTION_CLOSED)
            # instead of the v1 anyio resource errors. Translate both into a
            # clear error; any other MCPError (a live protocol error) re-raises.
            if isinstance(exc, MCPError) and exc.code != types.CONNECTION_CLOSED:
                raise
            raise RuntimeError(
                f"MCP tool {self.name!r} cannot be invoked: the underlying "
                "ClientSession is closed. The session is owned by the "
                "MCPClient.connect_*(...) async context manager — keep it "
                "open while using tools derived from it."
            ) from exc
        if result.is_error:
            text = _content_to_text(result.content) or "MCP tool returned error"
            raise RuntimeError(f"MCP tool {self.name} failed: {text}")
        return _content_to_text(result.content)


def _content_to_text(content: list[Any]) -> str:
    """Concatenate text blocks; ignore non-text blocks for v1."""
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)
