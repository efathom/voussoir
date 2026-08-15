"""YaseRetriever — RAG-only retriever for agent tools.

Calls yase /v1/rag and returns voussoir-shaped Citations. NOT an
IRetriever impl: yase's API model is for tool-shaped RAG calls, not
the memory subsystem. See spec §8.1.3.

Wrap as a voussoir Tool via `make_yase_search_tool(retriever)` from
voussoir.memory.backends.yase.tool.
"""

from __future__ import annotations

from voussoir.memory.backends.yase.citation import Citation
from voussoir.memory.backends.yase.client import YaseClient


class YaseRetriever:
    """Plain async wrapper over yase /v1/rag — no Protocol contract."""

    def __init__(self, *, client: YaseClient, default_top_k: int = 5) -> None:
        self._client = client
        self._default_top_k = default_top_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[Citation]:
        """Hybrid retrieval via yase /v1/rag. Returns voussoir Citations."""
        out = await self._client.rag(
            query=query,
            top_k=top_k or self._default_top_k,
            filters=filters,
        )
        return [Citation.from_yase(c) for c in out.get("citations", [])]
