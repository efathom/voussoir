import httpx
import pytest
import respx

from voussoir.memory.backends.yase.citation import Citation
from voussoir.memory.backends.yase.client import YaseClient
from voussoir.memory.backends.yase.retriever import YaseRetriever


@pytest.fixture
def client():
    return YaseClient(base_url="http://yase.test:8000", api_key="key")


def test_citation_constructs_with_minimal_fields():
    c = Citation(doc_id=7, text="hello", relevance_score=0.95)
    assert c.doc_id == 7
    assert c.text == "hello"
    assert c.relevance_score == 0.95
    assert c.source_url is None
    assert c.metadata == {}


def test_citation_round_trips_yase_payload():
    c = Citation.from_yase(
        {
            "doc_id": 11,
            "chunk_text": "Claude by Anthropic.",
            "source_url": "https://anthropic.com",
            "relevance_score": 0.92,
            "bm25_score": 0.55,
            "semantic_score": 0.71,
            "metadata": {"source": "blog"},
        }
    )
    assert c.doc_id == 11
    assert c.text.startswith("Claude")
    assert c.source_url == "https://anthropic.com"
    assert c.relevance_score == 0.92
    assert c.bm25_score == 0.55
    assert c.semantic_score == 0.71
    assert c.metadata == {"source": "blog"}


@respx.mock
async def test_retrieve_returns_citations(client):
    respx.post("http://yase.test:8000/v1/rag").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ok",
                "query": "claude",
                "context": "[1] Claude is an AI by Anthropic.",
                "citations": [
                    {
                        "doc_id": 7,
                        "chunk_text": "Claude is an AI by Anthropic.",
                        "source_url": "https://anthropic.com",
                        "relevance_score": 0.95,
                        "metadata": {"source": "blog"},
                    }
                ],
                "duration_ms": 23,
            },
        )
    )
    r = YaseRetriever(client=client)
    citations = await r.retrieve("claude")
    assert len(citations) == 1
    c = citations[0]
    assert isinstance(c, Citation)
    assert c.doc_id == 7
    assert c.text.startswith("Claude")
    assert c.source_url == "https://anthropic.com"
    assert c.relevance_score == 0.95
    assert c.metadata == {"source": "blog"}


@respx.mock
async def test_retrieve_passes_top_k(client):
    route = respx.post("http://yase.test:8000/v1/rag").mock(
        return_value=httpx.Response(200, json={"citations": []}),
    )
    r = YaseRetriever(client=client)
    await r.retrieve("q", top_k=3)
    body = route.calls.last.request.read().decode()
    assert '"top_k":3' in body.replace(" ", "")


@respx.mock
async def test_retrieve_passes_filters(client):
    route = respx.post("http://yase.test:8000/v1/rag").mock(
        return_value=httpx.Response(200, json={"citations": []}),
    )
    r = YaseRetriever(client=client)
    await r.retrieve("q", filters={"tag": "important"})
    body = route.calls.last.request.read().decode()
    assert '"tag"' in body and '"important"' in body
