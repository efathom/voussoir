from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session

from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore


async def test_memory_store_add_and_get():
    store = InMemoryStore()
    item = MemoryItem(
        memory_id="m1",
        user_id="u1",
        content="alice likes hiking",
        type=MemoryType.SEMANTIC,
    )
    await store.add(item)
    got = await store.get("m1")
    assert got is not None
    assert got.content == "alice likes hiking"


async def test_memory_store_get_missing_returns_none():
    store = InMemoryStore()
    assert await store.get("nope") is None


async def test_memory_store_delete():
    store = InMemoryStore()
    item = MemoryItem(
        memory_id="m1",
        user_id="u1",
        content="x",
        type=MemoryType.SEMANTIC,
    )
    await store.add(item)
    assert await store.delete("m1") is True
    assert await store.get("m1") is None
    assert await store.delete("m1") is False


async def test_memory_store_keyword_search():
    store = InMemoryStore()
    for i, content in enumerate(["alice likes hiking", "bob likes biking", "carol writes code"]):
        await store.add(
            MemoryItem(
                memory_id=f"m{i}",
                user_id="u1",
                content=content,
                type=MemoryType.SEMANTIC,
            )
        )
    results = await store.keyword_search("u1", ["biking"])
    assert len(results) == 1
    assert "biking" in results[0].content


async def test_memory_store_count():
    store = InMemoryStore()
    for i in range(3):
        await store.add(
            MemoryItem(
                memory_id=f"m{i}",
                user_id="u1",
                content=f"item-{i}",
                type=MemoryType.SEMANTIC,
            )
        )
    await store.add(
        MemoryItem(
            memory_id="other",
            user_id="u2",
            content="other",
            type=MemoryType.SEMANTIC,
        )
    )
    assert await store.count("u1") == 3
    assert await store.count("u2") == 1


async def test_session_store_save_and_load():
    store = InMemorySessionStore()
    s = Session(session_id="s1", user_id="u1")
    await store.save(s)
    got = await store.load("s1", "u1")
    assert got is not None
    assert got.session_id == "s1"


async def test_session_store_load_missing_returns_default():
    """ctxforge ISessionStore.load returns Session (not Optional). Tier 0 returns
    a fresh empty session for unknown session_id rather than raising."""
    store = InMemorySessionStore()
    got = await store.load("nope", "u1")
    assert got.session_id == "nope"
    assert got.user_id == "u1"


async def test_session_store_exists():
    store = InMemorySessionStore()
    s = Session(session_id="s1", user_id="u1")
    await store.save(s)
    assert await store.exists("s1") is True
    assert await store.exists("nope") is False
