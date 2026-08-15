"""{{project_name}} tools -- web_search stub.

Replace the stub with a real implementation (e.g. SerpAPI, Brave Search).
"""

from __future__ import annotations

from voussoir.tools import Capability, tool


@tool(capability=Capability.READ_PUBLIC, name="web_search")
async def web_search(query: str) -> str:
    """Stub web_search -- returns canned text. Replace with a real search backend."""
    return f"[web_search stub] Results for {query!r}: no real backend wired."
