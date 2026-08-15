"""Minimal stdio MCP server used by tests. Single 'echo' tool."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("voussoir-fake")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input back."""
    return f"echoed: {text}"


if __name__ == "__main__":
    mcp.run("stdio")
