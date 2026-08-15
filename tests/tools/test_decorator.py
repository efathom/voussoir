import pytest
from pydantic import BaseModel

from voussoir.tools import Capability, Tool, ToolContext, tool


@tool(capability=Capability.READ_PUBLIC)
async def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the public web."""
    return [{"q": query, "n": top_k}]


def test_decorated_function_satisfies_tool_protocol():
    assert isinstance(search, Tool)
    assert search.name == "search"
    assert "Search the public web." in search.description
    assert search.capability == Capability.READ_PUBLIC


def test_decorator_generates_input_schema():
    schema = search.input_schema
    assert issubclass(schema, BaseModel)
    fields = schema.model_fields
    assert "query" in fields
    assert "top_k" in fields
    assert fields["top_k"].default == 5


async def test_decorated_tool_invocable():
    args = search.input_schema(query="hello", top_k=3)
    ctx = ToolContext(run_id="r1", span_id="s1")
    result = await search.invoke(args, ctx)
    assert result == [{"q": "hello", "n": 3}]


def test_decorator_uses_explicit_name_when_provided():
    @tool(name="custom_name", capability=Capability.READ_PUBLIC)
    async def x(q: str) -> str:
        return q

    assert x.name == "custom_name"


def test_decorator_rejects_function_without_annotations():
    with pytest.raises(TypeError, match="must have a type annotation"):

        @tool(capability=Capability.READ_PUBLIC)
        async def bad(query):  # no annotation
            return query


def test_decorator_rejects_sync_function():
    with pytest.raises(TypeError, match="async function"):

        @tool(capability=Capability.READ_PUBLIC)
        def sync_fn(q: str) -> str:
            return q


def test_decorator_without_capability_raises():
    """Since v1.0.2 D3, omitting capability= is a ValueError (security footgun)."""
    with pytest.raises(ValueError, match="capability="):

        @tool()
        async def y(q: str) -> str:
            return q
