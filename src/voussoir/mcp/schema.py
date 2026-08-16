"""JSON Schema → Pydantic model generation for MCP tool inputs.

MCP servers emit each tool's `input_schema` as a JSON Schema dict. To plug
into voussoir's Tool Protocol (which uses a Pydantic input_schema class),
we synthesize a Pydantic model class at runtime.

Supported JSON Schema features (intentionally narrow):
- type: object with properties + required
- primitive types: string, number, integer, boolean, array, object
- enum constraint → Literal
- default values
- anything else → typing.Any (so we never reject a novel schema)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

_PRIMITIVES: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def jsonschema_to_pydantic(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model class from a JSON Schema 'object' definition."""
    properties: dict[str, dict[str, Any]] = schema.get("properties", {}) or {}
    required: set[str] = set(schema.get("required", []) or [])

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _resolve_type(prop_schema)
        is_required = prop_name in required
        if is_required:
            fields[prop_name] = (
                py_type,
                Field(..., description=prop_schema.get("description")),
            )
        else:
            default = prop_schema.get("default")
            fields[prop_name] = (
                py_type | None if default is None else py_type,
                Field(default=default, description=prop_schema.get("description")),
            )

    safe_name = "".join(c if c.isalnum() else "_" for c in name) or "MCPInput"
    return create_model(f"{safe_name}Args", **fields)


def _resolve_type(prop: dict[str, Any]) -> Any:
    enum = prop.get("enum")
    if enum:
        return Literal[tuple(enum)]

    json_type = prop.get("type")
    if json_type == "array":
        item = prop.get("items", {})
        item_t = _resolve_type(item) if isinstance(item, dict) else Any
        return list[item_t]  # type: ignore[valid-type]

    if isinstance(json_type, str) and json_type in _PRIMITIVES:
        return _PRIMITIVES[json_type]

    return Any
