"""Citation — voussoir-side Pydantic model for one yase RAG citation.

Independent of MemoryItem deliberately: yase RAG returns external content
snippets retrieved on demand by an agent tool, not user-owned typed
memories. Conflating the two through MemoryItem was a category error in
earlier drafts (see spec §8.1.3 and the second yase audit in the Phase 2
plan).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Citation(BaseModel):
    """One yase /v1/rag citation."""

    doc_id: int
    text: str
    source_url: str | None = None
    relevance_score: float = 0.0
    bm25_score: float | None = None
    semantic_score: float | None = None
    metadata: dict[str, str] = {}

    @classmethod
    def from_yase(cls, payload: dict[str, Any]) -> Citation:
        """Build a Citation from a raw yase /v1/rag citation entry."""
        return cls(
            doc_id=int(payload.get("doc_id", 0)),
            text=str(payload.get("chunk_text", "")),
            source_url=payload.get("source_url"),
            relevance_score=float(payload.get("relevance_score", 0.0)),
            bm25_score=(
                float(payload["bm25_score"]) if payload.get("bm25_score") is not None else None
            ),
            semantic_score=(
                float(payload["semantic_score"])
                if payload.get("semantic_score") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )
