import pytest

from voussoir.tools import Capability, ToolRegistry, tool


@tool(capability=Capability.READ_PUBLIC)
async def alpha(q: str) -> str:
    """Tool A."""
    return q


@tool(capability=Capability.READ_PUBLIC)
async def beta(q: str) -> str:
    """Tool B."""
    return q


def test_register_and_resolve():
    r = ToolRegistry()
    r.register(alpha)
    r.register(beta)
    assert r.resolve("alpha") is alpha
    assert r.resolve("beta") is beta


def test_resolve_missing_raises():
    r = ToolRegistry()
    with pytest.raises(KeyError, match="alpha"):
        r.resolve("alpha")


def test_register_duplicate_raises():
    r = ToolRegistry()
    r.register(alpha)
    with pytest.raises(ValueError, match="already registered"):
        r.register(alpha)


def test_register_many():
    r = ToolRegistry()
    r.register_many([alpha, beta])
    assert sorted(r.names()) == ["alpha", "beta"]


def test_describe_emits_function_call_format():
    r = ToolRegistry()
    r.register(alpha)
    described = r.describe()
    assert len(described) == 1
    d = described[0]
    assert d["name"] == "alpha"
    assert "Tool A" in d["description"]
    assert d["parameters"]["type"] == "object"
    assert "q" in d["parameters"]["properties"]


def test_empty_registry_has_no_names():
    r = ToolRegistry()
    assert r.names() == []
    assert r.describe() == []
