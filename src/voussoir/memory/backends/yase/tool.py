"""Tool factory — wraps a YaseRetriever as a voussoir @tool for agent tool-calling.

Use case: an agent that grounds answers in a yase-indexed corpus.
The agent calls `yase_search(query=..., top_k=...)` like any other tool;
this factory builds the Tool instance bound to a specific YaseRetriever.

Each call to make_yase_search_tool() creates a fresh Tool — useful when
an agent needs multiple yase backends (e.g., one for internal docs and
one for public web content).
"""

from __future__ import annotations

from voussoir.memory.backends.yase.retriever import YaseRetriever
from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability, Tool


def make_yase_search_tool(retriever: YaseRetriever) -> Tool:
    """Wrap a YaseRetriever as a voussoir Tool for agent tool-calling."""

    @tool(capability=Capability.READ_PUBLIC, name="yase_search")
    async def yase_search(query: str, top_k: int = 5) -> str:
        """Search the yase-indexed corpus for relevant document passages."""
        citations = await retriever.retrieve(query, top_k=top_k)
        if not citations:
            return "No results."
        lines = [
            f"[{i + 1}] {c.text} "
            f"(source: {c.source_url or 'unknown'}, score: {c.relevance_score:.2f})"
            for i, c in enumerate(citations)
        ]
        return "\n\n".join(lines)

    return yase_search
