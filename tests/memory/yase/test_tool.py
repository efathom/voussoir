import httpx
import pytest
import respx

from voussoir.memory.backends.yase.client import YaseClient
from voussoir.memory.backends.yase.retriever import YaseRetriever
from voussoir.memory.backends.yase.tool import make_yase_search_tool
from voussoir.tools.protocol import Capability, ToolContext


@pytest.fixture
def retriever():
    return YaseRetriever(client=YaseClient(base_url="http://yase.test:8000", api_key="k"))


def test_tool_factory_returns_a_voussoir_tool(retriever):
    t = make_yase_search_tool(retriever)
    assert t.name == "yase_search"
    assert t.capability == Capability.READ_PUBLIC
    assert t.input_schema is not None  # Pydantic model class


@respx.mock
async def test_tool_invokes_retriever_and_formats_output(retriever):
    respx.post("http://yase.test:8000/v1/rag").mock(
        return_value=httpx.Response(
            200,
            json={
                "citations": [
                    {
                        "doc_id": 1,
                        "chunk_text": "First result.",
                        "source_url": "https://a.com",
                        "relevance_score": 0.9,
                    },
                    {
                        "doc_id": 2,
                        "chunk_text": "Second result.",
                        "source_url": None,
                        "relevance_score": 0.7,
                    },
                ],
            },
        )
    )
    t = make_yase_search_tool(retriever)
    args = t.input_schema(query="hello", top_k=2)
    out = await t.invoke(args, ToolContext(run_id="r1", span_id="s1"))
    assert "[1]" in out and "[2]" in out
    assert "First result." in out
    assert "Second result." in out
    assert "https://a.com" in out
    # The "unknown" placeholder appears for the citation with no source_url:
    assert "unknown" in out


@respx.mock
async def test_tool_returns_no_results_message_when_empty(retriever):
    respx.post("http://yase.test:8000/v1/rag").mock(
        return_value=httpx.Response(200, json={"citations": []}),
    )
    t = make_yase_search_tool(retriever)
    args = t.input_schema(query="x")
    out = await t.invoke(args, ToolContext(run_id="r1", span_id="s1"))
    assert out == "No results."
