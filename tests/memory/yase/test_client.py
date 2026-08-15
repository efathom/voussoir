import os

import httpx
import pytest
import respx

from voussoir.memory.backends.yase.client import YaseClient


@pytest.fixture
def client() -> YaseClient:
    return YaseClient(base_url="http://yase.test:8000", api_key="test-key")


@respx.mock
async def test_search_calls_correct_endpoint_with_auth(client):
    route = respx.post("http://yase.test:8000/v1/collections/notes/search").mock(
        return_value=httpx.Response(
            200,
            json={"status": "ok", "count": 1, "results": [{"id": 1, "text": "hi"}]},
        )
    )
    out = await client.search(collection_id="notes", query="hello", top_k=5)
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["X-API-Key"] == "test-key"
    body = sent.read().decode()
    assert '"query"' in body and '"hello"' in body
    assert out["results"][0]["text"] == "hi"


@respx.mock
async def test_search_with_filters(client):
    respx.post("http://yase.test:8000/v1/collections/notes/search").mock(
        return_value=httpx.Response(200, json={"status": "ok", "results": []}),
    )
    await client.search(
        collection_id="notes",
        query="x",
        top_k=10,
        filters={"tag": "important"},
    )


@respx.mock
async def test_rag_calls_v1_rag(client):
    respx.post("http://yase.test:8000/v1/rag").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ok",
                "query": "q",
                "context": "abc",
                "citations": [],
                "duration_ms": 12,
            },
        )
    )
    out = await client.rag(query="q", top_k=3)
    assert out["context"] == "abc"


@respx.mock
async def test_create_collection_round_trips(client):
    respx.post("http://yase.test:8000/v1/collections").mock(
        return_value=httpx.Response(
            201,
            json={"id": "notes", "name": "Notes", "status": "ready"},
        )
    )
    out = await client.create_collection(id="notes", name="Notes")
    assert out["id"] == "notes"


@respx.mock
async def test_get_collection(client):
    respx.get("http://yase.test:8000/v1/collections/notes").mock(
        return_value=httpx.Response(200, json={"id": "notes", "name": "Notes"}),
    )
    out = await client.get_collection("notes")
    assert out["id"] == "notes"


@respx.mock
async def test_delete_collection(client):
    respx.delete("http://yase.test:8000/v1/collections/notes").mock(
        return_value=httpx.Response(200, json={"status": "ok"}),
    )
    out = await client.delete_collection("notes")
    assert out["status"] == "ok"


@respx.mock
async def test_delete_documents(client):
    respx.request("DELETE", "http://yase.test:8000/v1/documents").mock(
        return_value=httpx.Response(200, json={"status": "ok", "deleted_count": 2}),
    )
    out = await client.delete_documents(doc_ids=[1, 2])
    assert out["deleted_count"] == 2


@respx.mock
async def test_retries_on_transient_error(client):
    """3 attempts with exponential backoff; succeed on attempt 3."""
    seq = [
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json={"status": "ok", "results": []}),
    ]
    respx.post("http://yase.test:8000/v1/collections/notes/search").mock(side_effect=seq)
    out = await client.search(collection_id="notes", query="x")
    assert out == {"status": "ok", "results": []}


@respx.mock
async def test_raises_after_exhausting_retries(client):
    respx.post("http://yase.test:8000/v1/collections/notes/search").mock(
        return_value=httpx.Response(503)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.search(collection_id="notes", query="x")


def test_rejects_both_api_key_and_bearer():
    with pytest.raises(ValueError, match="api_key OR bearer"):
        YaseClient(base_url="http://x", api_key="k", bearer="b")


@pytest.mark.skipif(
    "YASE_URL" not in os.environ,
    reason="integration — requires YASE_URL pointing at a docker-compose'd yase",
)
async def test_live_search_against_real_yase():
    base = os.environ["YASE_URL"]
    api_key = os.environ.get("YASE_API_KEY", "")
    c = YaseClient(base_url=base, api_key=api_key or None)
    out = await c.search(collection_id="_default", query="hello", top_k=1)
    assert "status" in out
