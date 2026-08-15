import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import ClientSession

from voussoir.mcp.client import MCPClient

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


async def test_client_connect_lists_one_tool():
    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        result = await session.list_tools()
        names = [t.name for t in result.tools]
        assert "echo" in names


async def test_client_call_tool_round_trips():
    async with MCPClient.connect_stdio(command=sys.executable, args=[FAKE_SERVER]) as session:
        result = await session.call_tool("echo", {"text": "hi"})
        assert result.content[0].type == "text"
        assert "echoed: hi" in result.content[0].text
        assert result.isError is False


async def test_connect_http_initializes_session_with_url_and_headers():
    """Verify connect_http translates headers → httpx.AsyncClient(headers=...)
    and forwards it to streamable_http_client, then initializes the session."""
    fake_session = MagicMock(spec=ClientSession)
    fake_session.initialize = AsyncMock()
    fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    fake_transport_cm = MagicMock()
    fake_transport_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    fake_transport_cm.__aexit__ = AsyncMock(return_value=False)

    fake_http_client = MagicMock()
    fake_http_client.__aenter__ = AsyncMock(return_value=fake_http_client)
    fake_http_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "voussoir.mcp.client.httpx.AsyncClient",
            return_value=fake_http_client,
        ) as mock_http_client_cls,
        patch(
            "voussoir.mcp.client.streamable_http_client",
            return_value=fake_transport_cm,
        ) as mock_transport,
        patch("voussoir.mcp.client.ClientSession", return_value=fake_session),
    ):
        async with MCPClient.connect_http(
            url="https://mcp.example/v1",
            headers={"Authorization": "Bearer xxx"},
        ) as session:
            await session.list_tools()

    # httpx.AsyncClient constructed with the user's headers:
    mock_http_client_cls.assert_called_once()
    assert mock_http_client_cls.call_args.kwargs["headers"] == {"Authorization": "Bearer xxx"}

    # streamable_http_client got the URL and the AsyncClient we built:
    mock_transport.assert_called_once()
    assert mock_transport.call_args.args[0] == "https://mcp.example/v1"
    assert mock_transport.call_args.kwargs["http_client"] is fake_http_client

    # AsyncClient cleanup ran:
    fake_http_client.__aexit__.assert_awaited()
    fake_session.initialize.assert_awaited_once()
