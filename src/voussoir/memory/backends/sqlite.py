"""SQLite memory backend — voussoir's offline dev/CI helper, NOT a tier.

Production users bind Postgres + Qdrant via `bind_postgres_memory` /
`bind_qdrant_vector_store` (the Tier 1 path; see spec §8.1.1). This file
exists for offline workflows where infra isn't available — tests, single-
host demos, hacking on a plane.

Implements ctxforge's IMemoryStore Protocol with brute-force cosine
similarity in Python. ~10k memories before it gets slow.

Vector ANN strategy: brute-force cosine similarity in Python.

The original Phase 2 design used sqlite-vec for SQL-side ANN, but that
requires SQLite extension loading which Python's stdlib build often
disables (notably on macOS). At dev-helper scale (~10k memories) a
Python-side cosine loop is fast enough and removes the build-time dep.

Concurrency: all sqlite calls run in `asyncio.to_thread` so a blocking
`sqlite3` call never stalls the event loop (a write while the loop is busy
would otherwise freeze every concurrent agent). WAL mode allows concurrent
readers during writes.

Schema:
- memories(memory_id PK, user_id, content, metadata, type, confidence_score,
           is_active, created_at, embedding JSON)
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.utils.math import cosine_similarity

if TYPE_CHECKING:
    from ctxforge.protocols.llm import IEmbeddingProvider

T = TypeVar("T")


class SQLiteMemoryStore(IMemoryStore):
    """Tier 1 memory store. Single-process; one writer at a time."""

    def __init__(
        self,
        *,
        path: str,
        embedder: IEmbeddingProvider,
        embedding_dim: int = 384,
    ) -> None:
        self._path = path
        self._embedder = embedder
        self._dim = embedding_dim
        self._lock = asyncio.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path)
        # Fail loudly on misuse rather than hanging in a second thread.
        con.execute("PRAGMA busy_timeout = 5000")
        return con

    def _init_schema(self) -> None:
        con = self._connect()
        try:
            con.execute("PRAGMA journal_mode = WAL")
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id        TEXT PRIMARY KEY,
                    user_id          TEXT NOT NULL,
                    content          TEXT NOT NULL,
                    metadata         TEXT NOT NULL,
                    type             TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    is_active        INTEGER NOT NULL DEFAULT 1,
                    created_at       REAL NOT NULL,
                    embedding        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
                """
            )
            con.commit()
        finally:
            con.close()

    async def _run(self, fn: Callable[[], T]) -> T:
        """Run a blocking sqlite operation off the event loop."""
        return await asyncio.to_thread(fn)

    async def add(self, item: MemoryItem) -> str:
        async with self._lock:
            vec = await self._embedder.embed_single(item.content)

            def _add() -> None:
                con = self._connect()
                try:
                    con.execute(
                        """
                        INSERT OR REPLACE INTO memories
                          (memory_id, user_id, content, metadata, type,
                           confidence_score, is_active, created_at, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item.memory_id,
                            item.user_id,
                            item.content,
                            json.dumps(item.metadata or {}),
                            _type_value(item.type),
                            item.confidence_score,
                            1 if item.is_active else 0,
                            time.time(),
                            json.dumps(vec),
                        ),
                    )
                    con.commit()
                finally:
                    con.close()

            await self._run(_add)
            return str(item.memory_id)

    async def get(self, memory_id: str) -> MemoryItem | None:
        def _get() -> MemoryItem | None:
            con = self._connect()
            try:
                row = con.execute(
                    """
                    SELECT memory_id, user_id, content, metadata, type,
                           confidence_score, is_active
                      FROM memories WHERE memory_id = ?
                    """,
                    (memory_id,),
                ).fetchone()
                if row is None:
                    return None
                return _row_to_item(row)
            finally:
                con.close()

        return await self._run(_get)

    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> list[MemoryItem]:
        active_clause = "" if include_inactive else " AND is_active = 1"

        def _get_by_user() -> list[MemoryItem]:
            con = self._connect()
            try:
                rows = con.execute(
                    f"""
                    SELECT memory_id, user_id, content, metadata, type,
                           confidence_score, is_active
                      FROM memories WHERE user_id = ?{active_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
                return [_row_to_item(r) for r in rows]
            finally:
                con.close()

        return await self._run(_get_by_user)

    async def update(self, item: MemoryItem) -> bool:
        def _exists() -> bool:
            con = self._connect()
            try:
                row = con.execute(
                    "SELECT 1 FROM memories WHERE memory_id = ?", (item.memory_id,)
                ).fetchone()
                return row is not None
            finally:
                con.close()

        if not await self._run(_exists):
            return False
        await self.add(item)
        return True

    async def delete(self, memory_id: str) -> bool:
        async with self._lock:

            def _delete() -> bool:
                con = self._connect()
                try:
                    changed = con.execute(
                        "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
                    ).rowcount
                    con.commit()
                    return changed > 0
                finally:
                    con.close()

            return await self._run(_delete)

    async def count(self, user_id: str) -> int:
        def _count() -> int:
            con = self._connect()
            try:
                row = con.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id = ? AND is_active = 1",
                    (user_id,),
                ).fetchone()
                return int(row[0])
            finally:
                con.close()

        return await self._run(_count)

    async def keyword_search(
        self,
        user_id: str,
        keywords: list[str],
        limit: int = 10,
        filters: dict[str, list[str]] | None = None,
    ) -> list[MemoryItem]:
        if not keywords:
            return []
        del filters  # reserved for future filter support

        def _keyword_search() -> list[MemoryItem]:
            con = self._connect()
            try:
                # ESCAPE '\\' + escaping the wildcards: a keyword containing %
                # or _ otherwise matched far more than the caller asked for
                # (audit, minor).
                placeholders = " OR ".join(["LOWER(content) LIKE ? ESCAPE '\\'"] * len(keywords))
                params: list[Any] = [f"%{_escape_like(kw.lower())}%" for kw in keywords]
                params.append(user_id)
                rows = con.execute(
                    f"""
                    SELECT memory_id, user_id, content, metadata, type,
                           confidence_score, is_active
                      FROM memories
                     WHERE ({placeholders}) AND user_id = ? AND is_active = 1
                     LIMIT {int(limit)}
                    """,
                    params,
                ).fetchall()
                return [_row_to_item(r) for r in rows]
            finally:
                con.close()

        return await self._run(_keyword_search)

    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """Brute-force cosine-similarity ANN over the user's active memories."""
        if not query.query_text:
            return []
        qvec = await self._embedder.embed_single(query.query_text)

        def _search() -> list[tuple[Any, ...]]:
            con = self._connect()
            try:
                return con.execute(
                    """
                    SELECT memory_id, user_id, content, metadata, type,
                           confidence_score, is_active, embedding
                      FROM memories
                     WHERE user_id = ? AND is_active = 1
                    """,
                    (query.user_id,),
                ).fetchall()
            finally:
                con.close()

        rows = await self._run(_search)

        scored: list[tuple[float, tuple[Any, ...]]] = []
        for row in rows:
            vec = json.loads(row[7])
            scored.append((cosine_similarity(qvec, vec), row[:7]))

        # Higher cosine = more similar; sort descending.
        scored.sort(key=lambda t: t[0], reverse=True)
        return [_row_to_item(r) for _, r in scored[: query.limit]]


def _escape_like(text: str) -> str:
    """Escape SQL LIKE metacharacters so a keyword matches itself literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _type_value(t: Any) -> str:
    """Coerce a MemoryType enum (or str) to its string value for storage."""
    return t.value if hasattr(t, "value") else str(t)


def _row_to_item(row: tuple[Any, ...]) -> MemoryItem:
    (
        memory_id,
        user_id,
        content,
        metadata_json,
        type_str,
        confidence_score,
        is_active,
    ) = row
    return MemoryItem(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        metadata=json.loads(metadata_json),
        type=type_str,
        confidence_score=confidence_score,
        is_active=bool(is_active),
    )
