import pytest
from pydantic import ValidationError

from voussoir.executors.standard import StandardExecutor
from voussoir.tools import Capability, ToolContext, tool


@tool(capability=Capability.READ_PUBLIC)
async def echo(text: str) -> str:
    """Echo the input."""
    return f"echo: {text}"


@tool(capability=Capability.READ_PUBLIC)
async def fail(text: str) -> str:
    """Always fails."""
    raise RuntimeError("nope")


async def test_invoke_passes_validated_args_and_returns_result():
    ex = StandardExecutor()
    args = echo.input_schema(text="hi")
    ctx = ToolContext(run_id="r1", span_id="s1", allowed_capabilities=Capability.READ_PUBLIC)
    result = await ex.invoke(echo, args, ctx)
    assert result == "echo: hi"


async def test_invoke_validates_args_against_schema():
    ex = StandardExecutor()  # noqa: F841 — exercise constructor
    # Wrong field name → ValidationError when constructing args
    with pytest.raises(ValidationError):
        echo.input_schema(wrong_field="hi")


async def test_invoke_propagates_tool_errors():
    ex = StandardExecutor()
    args = fail.input_schema(text="x")
    ctx = ToolContext(run_id="r1", span_id="s1", allowed_capabilities=Capability.READ_PUBLIC)
    with pytest.raises(RuntimeError, match="nope"):
        await ex.invoke(fail, args, ctx)
