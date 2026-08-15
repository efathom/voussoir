"""Phase 2 exit-criteria gates. If any fail, Phase 2 is not done."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def test_top_level_imports():
    """Each Phase 2 surface is reachable from voussoir.* without errors."""
    from voussoir import Agent, Container  # noqa: F401
    from voussoir.container.defaults import (  # noqa: F401
        bind_graph_store,
        bind_postgres_memory,
        bind_qdrant_vector_store,
        bind_sqlite_memory,
    )
    from voussoir.mcp.client import MCPClient  # noqa: F401
    from voussoir.memory.backends.qdrant import QdrantVectorStore  # noqa: F401
    from voussoir.memory.backends.sqlite import SQLiteMemoryStore  # noqa: F401  — dev helper
    from voussoir.memory.backends.yase.citation import Citation  # noqa: F401
    from voussoir.memory.backends.yase.client import YaseClient  # noqa: F401
    from voussoir.memory.backends.yase.retriever import YaseRetriever  # noqa: F401
    from voussoir.memory.backends.yase.tool import make_yase_search_tool  # noqa: F401
    from voussoir.skills.adapter import SkillActivationMiddleware  # noqa: F401
    from voussoir.tools.mcp import MCPTool, infer_capability  # noqa: F401


async def test_mcp_tool_round_trip_against_fake_server():
    """Exit criterion #1: MCPTool.from_server returns Tool instances
    indistinguishable from local @tool ones."""
    from voussoir.mcp.client import MCPClient
    from voussoir.tools.mcp import MCPTool
    from voussoir.tools.protocol import ToolContext

    fake = str(Path(__file__).parent / "mcp" / "fake_mcp_server.py")
    async with MCPClient.connect_stdio(command=sys.executable, args=[fake]) as session:
        tools = await MCPTool.from_server(session, server_name="fake")
        echo = next(t for t in tools if t.name == "fake__echo")
        args = echo.input_schema(text="phase2")
        out = await echo.invoke(args, ToolContext(run_id="r", span_id="s"))
        assert "echoed: phase2" in str(out)


async def test_sqlite_memory_persists_across_reopens(tmp_path):
    """Exit criterion #2: SQLite (dev helper) memory persists across runs.
    Production tier 1 = Postgres + Qdrant; this test exercises the SQLite
    backend used in tests + offline workflows."""
    from ctxforge.core.memory import MemoryItem

    from voussoir.llm.fake_embedder import FakeEmbeddingProvider
    from voussoir.memory.backends.sqlite import SQLiteMemoryStore

    db = tmp_path / "memory.db"
    s1 = SQLiteMemoryStore(path=str(db), embedder=FakeEmbeddingProvider())
    await s1.add(
        MemoryItem(
            memory_id="p1",
            user_id="alice",
            content="phase 2 fact",
            type="semantic",
            confidence_score=0.9,
            is_active=True,
        )
    )
    s2 = SQLiteMemoryStore(path=str(db), embedder=FakeEmbeddingProvider())
    got = await s2.get("p1")
    assert got is not None
    assert got.content == "phase 2 fact"


def test_skill_activation_middleware_importable():
    """Exit criterion #4: SkillActivationMiddleware exists."""
    from voussoir.skills.adapter import SkillActivationMiddleware  # noqa: F401


def test_qdrant_vector_store_importable():
    """Exit criterion (post-pivot): QdrantVectorStore is the canonical
    Tier 1 IVectorStore impl."""
    from voussoir.memory.backends.qdrant import QdrantVectorStore  # noqa: F401


def test_yase_is_rag_only_no_protocol_coupling():
    """Spec §8.1.3: yase artifacts are plain classes, not Protocol impls.

    YaseRetriever has a single `retrieve()` method returning list[Citation].
    It deliberately does NOT have IRetriever's other methods
    (retrieve_by_embedding / retrieve_related), nor IMemoryStore /
    IVectorStore / IEmbeddingProvider methods.
    """
    from voussoir.memory.backends.yase.retriever import YaseRetriever

    methods = {m for m in dir(YaseRetriever) if not m.startswith("_")}
    # The voussoir-shaped surface:
    assert "retrieve" in methods
    # IRetriever methods we deliberately don't carry:
    assert "retrieve_by_embedding" not in methods
    assert "retrieve_related" not in methods
    # IMemoryStore methods we don't carry:
    assert "add" not in methods
    assert "get_by_user" not in methods
    # IVectorStore methods we don't carry:
    assert "upsert" not in methods
    assert "query" not in methods
    # IEmbeddingProvider methods we don't carry:
    assert "embed" not in methods
    assert "embed_single" not in methods


@pytest.mark.skipif(
    "YASE_URL" not in os.environ,
    reason="exit criterion (live) — requires docker-compose'd yase",
)
async def test_yase_search_round_trip_live():
    """Exit criterion (live): yase RAG round-trips end-to-end."""
    from voussoir.memory.backends.yase.client import YaseClient
    from voussoir.memory.backends.yase.retriever import YaseRetriever

    client = YaseClient(
        base_url=os.environ["YASE_URL"],
        api_key=os.environ.get("YASE_API_KEY") or None,
    )
    retriever = YaseRetriever(client=client)
    citations = await retriever.retrieve("hello", top_k=1)
    # Don't assert on data shape — just verify the round-trip works:
    assert isinstance(citations, list)


def test_coverage_floor_phase2_packages():
    """Exit criterion: ≥85% on voussoir/mcp/, voussoir/memory/backends/,
    voussoir/skills/.

    Coverage is enforced by `make ci` running pytest --cov; this test reads
    the most-recent coverage data and asserts the floor.
    """
    pytest.importorskip("coverage")
    import coverage as cov_mod

    cov_file = Path(__file__).resolve().parent.parent / ".coverage"
    if not cov_file.exists():
        pytest.skip("no .coverage file — run `make ci` first")
    cov = cov_mod.Coverage(data_file=str(cov_file))
    cov.load()
    paths = [
        "src/voussoir/mcp",
        "src/voussoir/memory/backends",
        "src/voussoir/skills",
    ]
    total_stmts = 0
    total_miss = 0
    for f in cov.get_data().measured_files():
        if any(p in f for p in paths):
            _, stmts, miss, _ = cov.analysis2(f)[:4]
            total_stmts += len(stmts)
            total_miss += len(miss)
    if total_stmts == 0:
        pytest.skip("no measured files match paths")
    pct = (total_stmts - total_miss) / total_stmts
    assert pct >= 0.85, f"coverage {pct:.1%} on phase 2 packages is below 85% floor"
