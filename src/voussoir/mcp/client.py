"""MCPClient — wraps the official `mcp` SDK's transport + session lifecycle.

Phase 2.1 ships the stdio transport; HTTP arrives in Task 2.2. The same
`MCPClient.connect_*` family yields an initialized `mcp.ClientSession` so
callers do `async with MCPClient.connect_stdio(...) as session: ...`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


class MCPClient:
    """Factory namespace for MCP transport connections.

    All `connect_*` methods are class methods returning an async context
    manager that yields an initialized `mcp.ClientSession`.
    """

    @classmethod
    @asynccontextmanager
    async def connect_stdio(
        cls,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[ClientSession]:
        """Spawn an MCP server as a stdio subprocess and yield an initialized session."""
        params = StdioServerParameters(command=command, args=args or [], env=env)
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session

    @classmethod
    @asynccontextmanager
    async def connect_http(
        cls,
        *,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[ClientSession]:
        """Connect to a remote MCP server over streamable HTTP.

        Headers are translated into an httpx.AsyncClient that we own;
        the underlying SDK takes the client via http_client= and we
        clean it up on exit.
        """
        async with (
            httpx.AsyncClient(headers=headers) as http_client,
            streamable_http_client(url, http_client=http_client) as (read, write, _get_session_id),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session
