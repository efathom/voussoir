from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.vectorstores.protocol import VectorRecord

pytest.importorskip("qdrant_client")

from voussoir.memory.backends.qdrant import QdrantVectorStore


@pytest.fixture
def fake_client():
    c = MagicMock()
    c.upsert = AsyncMock()
    c.query_points = AsyncMock(return_value=MagicMock(points=[]))
    c.delete = AsyncMock()
    c.count = AsyncMock(return_value=MagicMock(count=42))
    c.retrieve = AsyncMock(return_value=[])
    c.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    c.create_collection = AsyncMock()
    c.collection_exists = AsyncMock(return_value=False)
    c.delete_collection = AsyncMock()
    c.close = AsyncMock()
    return c


def test_construct(fake_client):
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    assert store._default_collection == "memories"


async def test_upsert_calls_qdrant(fake_client):
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    n = await store.upsert(
        [VectorRecord(id="m1", embedding=[0.1] * 384, metadata={}, content="hello")]
    )
    assert n == 1
    fake_client.upsert.assert_awaited_once()
    kwargs = fake_client.upsert.await_args.kwargs
    assert kwargs["collection_name"] == "memories"


async def test_query_translates_results(fake_client):
    point = MagicMock(id="m1", score=0.95, payload={"content": "hello"}, vector=None)
    fake_client.query_points = AsyncMock(return_value=MagicMock(points=[point]))

    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    out = await store.query(embedding=[0.1] * 384, top_k=5)
    assert len(out) == 1
    assert out[0].id == "m1"
    assert out[0].score == 0.95


async def test_delete_uses_point_ids_list(fake_client):
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    n = await store.delete(ids=["m1", "m2"])
    assert n == 2
    fake_client.delete.assert_awaited_once()


async def test_count(fake_client):
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    n = await store.count()
    assert n == 42


async def test_initialize_creates_collection_when_missing(fake_client):
    fake_client.collection_exists = AsyncMock(return_value=False)
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    await store.initialize()
    fake_client.create_collection.assert_awaited_once()


async def test_initialize_skips_when_collection_exists(fake_client):
    fake_client.collection_exists = AsyncMock(return_value=True)
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    await store.initialize()
    fake_client.create_collection.assert_not_called()


async def test_namespace_overrides_default(fake_client):
    store = QdrantVectorStore(
        client=fake_client, default_collection="default_coll", vector_size=384
    )
    await store.upsert(
        [VectorRecord(id="m1", embedding=[0.0] * 384, metadata={}, content="x")],
        namespace="custom",
    )
    kwargs = fake_client.upsert.await_args.kwargs
    assert kwargs["collection_name"] == "custom"


async def test_close_calls_underlying_client(fake_client):
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    await store.close()
    fake_client.close.assert_awaited_once()


async def test_fetch_returns_records_keyed_by_id(fake_client):
    point = MagicMock(id="m1", payload={"content": "hello", "tag": "x"}, vector=[0.1] * 384)
    fake_client.retrieve = AsyncMock(return_value=[point])

    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    out = await store.fetch(ids=["m1"])
    assert "m1" in out
    rec = out["m1"]
    assert rec.id == "m1"
    assert rec.content == "hello"
    assert rec.metadata == {"tag": "x"}
    assert rec.embedding == [0.1] * 384


async def test_describe_lists_collections(fake_client):
    coll = MagicMock(name="memories")
    coll.name = "memories"
    fake_client.get_collections = AsyncMock(return_value=MagicMock(collections=[coll]))

    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    info = await store.describe()
    assert info == {"collections": ["memories"]}


async def test_list_namespaces(fake_client):
    coll = MagicMock(name="alpha")
    coll.name = "alpha"
    coll2 = MagicMock(name="beta")
    coll2.name = "beta"
    fake_client.get_collections = AsyncMock(return_value=MagicMock(collections=[coll, coll2]))

    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    assert await store.list_namespaces() == ["alpha", "beta"]


async def test_delete_namespace(fake_client):
    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    out = await store.delete_namespace("custom")
    assert out is True
    fake_client.delete_collection.assert_awaited_once_with("custom")


async def test_query_by_id_uses_recommend_query(fake_client):
    point = MagicMock(id="m2", score=0.8, payload={"content": "related"}, vector=None)
    fake_client.query_points = AsyncMock(return_value=MagicMock(points=[point]))

    store = QdrantVectorStore(client=fake_client, default_collection="memories", vector_size=384)
    out = await store.query_by_id(vector_id="m1", top_k=5)
    assert len(out) == 1
    assert out[0].id == "m2"
    fake_client.query_points.assert_awaited_once()


async def test_delete_by_filter_raises_phase_2():
    """Filter-based delete is wired in Phase 5 alongside guardrails."""
    fake = MagicMock()
    store = QdrantVectorStore(client=fake, default_collection="memories", vector_size=384)
    with pytest.raises(NotImplementedError, match="Phase 5"):
        await store.delete_by_filter(filters=[])
