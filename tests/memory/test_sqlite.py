import pytest
from ctxforge.core.memory import MemoryItem, MemoryQuery

from voussoir.llm.fake_embedder import FakeEmbeddingProvider
from voussoir.memory.backends.sqlite import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "memory.db"
    return SQLiteMemoryStore(path=str(db), embedder=FakeEmbeddingProvider())


def _item(memory_id: str, user_id: str, content: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        type="semantic",
        confidence_score=0.9,
        is_active=True,
    )


async def test_add_and_get(store):
    item = _item("m1", "alice", "the sky is blue")
    mid = await store.add(item)
    assert mid == "m1"
    got = await store.get("m1")
    assert got is not None
    assert got.content == "the sky is blue"


async def test_get_missing_returns_none(store):
    assert await store.get("missing") is None


async def test_get_by_user_filters_active(store):
    await store.add(_item("m1", "alice", "x"))
    await store.add(_item("m2", "alice", "y"))
    await store.add(_item("m3", "bob", "z"))
    items = await store.get_by_user("alice")
    assert sorted(i.memory_id for i in items) == ["m1", "m2"]


async def test_search_returns_closest_first(store):
    await store.add(_item("m1", "alice", "blueberry pie"))
    await store.add(_item("m2", "alice", "apple turnover"))
    await store.add(_item("m3", "alice", "the sky is blue"))
    q = MemoryQuery(user_id="alice", query_text="blueberry pie", limit=2)
    results = await store.search(q)
    assert len(results) == 2
    assert results[0].memory_id == "m1"


async def test_search_isolates_users(store):
    await store.add(_item("m1", "alice", "alice secret"))
    await store.add(_item("m2", "bob", "bob secret"))
    q = MemoryQuery(user_id="bob", query_text="secret", limit=10)
    results = await store.search(q)
    assert {r.memory_id for r in results} == {"m2"}


async def test_persists_across_reopens(tmp_path):
    db = tmp_path / "memory.db"
    s1 = SQLiteMemoryStore(path=str(db), embedder=FakeEmbeddingProvider())
    await s1.add(_item("m1", "alice", "persistent fact"))

    s2 = SQLiteMemoryStore(path=str(db), embedder=FakeEmbeddingProvider())
    got = await s2.get("m1")
    assert got is not None
    assert got.content == "persistent fact"


async def test_delete_removes_item(store):
    await store.add(_item("m1", "alice", "x"))
    assert await store.delete("m1") is True
    assert await store.get("m1") is None
    assert await store.delete("m1") is False


async def test_count_per_user(store):
    await store.add(_item("m1", "alice", "x"))
    await store.add(_item("m2", "alice", "y"))
    await store.add(_item("m3", "bob", "z"))
    assert await store.count("alice") == 2
    assert await store.count("bob") == 1


async def test_update_existing_item_returns_true(store):
    """update() returns True when the item exists and overwrites it."""
    original = _item("m1", "alice", "old content")
    await store.add(original)

    updated = _item("m1", "alice", "new content")
    assert await store.update(updated) is True

    got = await store.get("m1")
    assert got is not None
    assert got.content == "new content"


async def test_update_missing_item_returns_false(store):
    """update() returns False (no-op) when the memory_id doesn't exist."""
    assert await store.update(_item("nope", "alice", "x")) is False


async def test_search_with_empty_query_text_returns_empty(store):
    """search() short-circuits to [] when MemoryQuery.query_text is empty."""
    from ctxforge.core.memory import MemoryQuery

    await store.add(_item("m1", "alice", "x"))
    q = MemoryQuery(user_id="alice", query_text="", limit=10)
    assert await store.search(q) == []


async def test_keyword_search_empty_keywords_returns_empty(store):
    """keyword_search() short-circuits to [] when keywords is empty."""
    await store.add(_item("m1", "alice", "x"))
    assert await store.keyword_search("alice", []) == []


async def test_keyword_search_matches_substring(store):
    await store.add(_item("m1", "alice", "the blueberry pie"))
    await store.add(_item("m2", "alice", "an apple turnover"))
    out = await store.keyword_search("alice", ["blueberry"])
    assert len(out) == 1
    assert out[0].memory_id == "m1"
