# 02 — research agent (MCP + SQLite)

Demonstrates voussoir with two Phase 2 features:
- An **MCP-bound tool** from a local stdio MCP server (the test fixture
  `tests/mcp/fake_mcp_server.py`)
- **SQLite memory** with brute-force cosine ANN (the dev/CI helper —
  see `voussoir.memory.backends.sqlite`)

No docker required. Memory file (`memory.db`) is created in this directory
on first run; delete it to reset.

For the canonical Tier 1 production path (Postgres + Qdrant), see
`bind_postgres_memory` + `bind_qdrant_vector_store` in
`voussoir.container.defaults`.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install -e ".[mcp,mem-sqlite]"
python examples/02_research_agent/main.py
```

Expected output (rough):

```
output:         I'll use the echo tool: echoed: hello voussoir
steps:          [('llm_call', 'chat'), ('tool_call', 'fake.echo'), ('llm_call', 'chat')]
tokens_in:      87
tokens_out:     34
finish_reason:  completed
```
