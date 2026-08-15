import sys
from pathlib import Path

from voussoir.mcp.client import MCPClient
from voussoir.tools.mcp import MCPTool, infer_capability
from voussoir.tools.protocol import Capability, ToolContext

FAKE_SERVER = str(Path(__file__).resolve().parent.parent / "mcp" / "fake_mcp_server.py")


def test_capability_inference_read_only_names():
    assert infer_capability("search_web") == Capability.READ_PUBLIC
    assert infer_capability("get_user") == Capability.READ_PUBLIC
    assert infer_capability("list_files") == Capability.READ_PUBLIC
    assert infer_capability("read_doc") == Capability.READ_PUBLIC


def test_capability_inference_write_names():
    assert infer_capability("send_email") == Capability.EXFILTRATION
    assert infer_capability("create_record") == Capability.EXFILTRATION
    assert infer_capability("delete_thing") == Capability.EXFILTRATION
    assert infer_capability("post_message") == Capability.EXFILTRATION


def test_capability_inference_default_is_read_private():
    assert infer_capability("compute_stats") == Capability.READ_PRIVATE


def test_capability_override_wins():
    assert (
        infer_capability("send_email", overrides={"send_email": Capability.READ_PUBLIC})
        == Capability.READ_PUBLIC
    )


async def test_from_server_yields_tools_for_each_remote():
    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        tools = await MCPTool.from_server(session, server_name="fake")
        names = [t.name for t in tools]
        assert "fake__echo" in names
        echo = next(t for t in tools if t.name == "fake__echo")
        assert echo.capability == Capability.READ_PRIVATE  # default for "echo"


async def test_invoke_round_trips_via_session():
    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        tools = await MCPTool.from_server(session, server_name="fake")
        echo = next(t for t in tools if t.name == "fake__echo")
        args = echo.input_schema(text="from voussoir")
        result = await echo.invoke(args, ToolContext(run_id="r1", span_id="s1"))
        assert "echoed: from voussoir" in str(result)


async def test_capability_override_at_from_server():
    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        tools = await MCPTool.from_server(
            session,
            server_name="fake",
            capability_overrides={"echo": Capability.READ_PUBLIC},
        )
        echo = next(t for t in tools if t.name == "fake__echo")
        assert echo.capability == Capability.READ_PUBLIC


async def test_invoke_after_session_closed_raises_clear_error():
    """Using an MCPTool after the underlying ClientSession has been closed
    must raise a clear RuntimeError, not the opaque anyio.ClosedResourceError."""
    import pytest

    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        tools = await MCPTool.from_server(session, server_name="fake")
        echo = next(t for t in tools if t.name == "fake__echo")
    # Session is now closed (CM exited).
    args = echo.input_schema(text="after-close")
    with pytest.raises(RuntimeError, match="ClientSession is closed"):
        await echo.invoke(args, ToolContext(run_id="r", span_id="s"))
