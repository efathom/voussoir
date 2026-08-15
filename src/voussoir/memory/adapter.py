"""Tier 0 in-memory IMemoryStore + ISessionStore implementations.

These satisfy ctxforge's Protocols for local dev / tests / the 5-line demo.
Persistence is process-only; nothing survives restart. Tier 1 (SQLite) lands in
Phase 2.

All methods are ``async`` to match ctxforge's Protocol signatures even though
the in-memory dict ops do no actual I/O. This lets downstream agent code await
them uniformly across tiers.
"""

from __future__ import annotations

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.core.session import Session
from ctxforge.protocols.storage import IMemoryStore, ISessionStore

__all__ = [
    "IMemoryStore",
    "ISessionStore",
    "InMemorySessionStore",
    "InMemoryStore",
]


class InMemoryStore(IMemoryStore):
    """Dict-backed IMemoryStore for Tier 0."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    async def add(self, item: MemoryItem) -> str:
        memory_id = str(item.memory_id)
        self._items[memory_id] = item
        return memory_id

    async def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> list[MemoryItem]:
        out = [
            i
            for i in self._items.values()
            if i.user_id == user_id and (include_inactive or i.is_active)
        ]
        return out[:limit]

    async def update(self, item: MemoryItem) -> bool:
        if item.memory_id not in self._items:
            return False
        self._items[item.memory_id] = item
        return True

    async def delete(self, memory_id: str) -> bool:
        return self._items.pop(memory_id, None) is not None

    async def count(self, user_id: str) -> int:
        return sum(1 for i in self._items.values() if i.user_id == user_id)

    async def keyword_search(
        self,
        user_id: str,
        keywords: list[str],
        limit: int = 10,
        filters: dict[str, list[str]] | None = None,
    ) -> list[MemoryItem]:
        # Simple substring-match against any keyword. Filters not honored at Tier 0.
        lowered = [k.lower() for k in keywords]
        matches = [
            i
            for i in self._items.values()
            if i.user_id == user_id and any(k in i.content.lower() for k in lowered)
        ]
        return matches[:limit]

    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        # MemoryQuery exposes user_id, query_text, and limit. Delegate to
        # keyword_search with query_text split into single-word keywords.
        words = (query.query_text or "").split() or [""]
        return await self.keyword_search(query.user_id, words, limit=query.limit)


class InMemorySessionStore(ISessionStore):
    """Dict-backed ISessionStore for Tier 0."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    async def load(self, session_id: str, user_id: str) -> Session:
        # ctxforge contract: returns Session (not Optional). Tier 0 fabricates a
        # fresh empty session if missing. Higher tiers may persist creation.
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        new = Session(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = new
        return new

    async def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Session]:
        out = [s for s in self._sessions.values() if s.user_id == user_id]
        return out[offset : offset + limit]
