import pytest
from pydantic import ValidationError

from voussoir.mcp.schema import jsonschema_to_pydantic


def test_simple_object_with_required_string():
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string", "description": "query"}},
        "required": ["q"],
    }
    model_cls = jsonschema_to_pydantic("Search", schema)
    obj = model_cls(q="hello")
    assert obj.q == "hello"
    with pytest.raises(ValidationError):
        model_cls()  # missing required


def test_optional_with_default():
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["q"],
    }
    model_cls = jsonschema_to_pydantic("Search", schema)
    obj = model_cls(q="x")
    assert obj.top_k == 5


def test_enum_becomes_literal():
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["fast", "deep"]}},
        "required": ["mode"],
    }
    model_cls = jsonschema_to_pydantic("Search", schema)
    model_cls(mode="fast")  # valid
    with pytest.raises(ValidationError):
        model_cls(mode="random")


def test_array_of_strings():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["tags"],
    }
    model_cls = jsonschema_to_pydantic("Tagged", schema)
    obj = model_cls(tags=["a", "b"])
    assert obj.tags == ["a", "b"]


def test_unknown_type_falls_back_to_any():
    schema = {
        "type": "object",
        "properties": {"weird": {"type": "geometry"}},
        "required": [],
    }
    model_cls = jsonschema_to_pydantic("Weird", schema)
    obj = model_cls(weird={"anything": "goes"})
    assert obj.weird == {"anything": "goes"}


def test_empty_schema_yields_empty_model():
    model_cls = jsonschema_to_pydantic("Empty", {"type": "object", "properties": {}})
    obj = model_cls()
    assert obj.model_dump() == {}
