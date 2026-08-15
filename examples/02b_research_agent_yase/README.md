# 02b — research agent (yase RAG)

Demonstrates the canonical yase wiring: a voussoir agent that calls the
`yase_search` tool when it needs to ground answers in the yase corpus.
Architecture per spec §8.1.3 — yase is a **RAG provider for agent tools**,
not a memory tier.

```
YaseClient (HTTP) → YaseRetriever → make_yase_search_tool → Agent(tools=[…])
```

## Setup

```bash
# 1. Start yase (single-node):
docker compose -f examples/02b_research_agent_yase/docker-compose.yml up -d
# wait ~10s for yase to be ready

# 2. Seed the corpus.
# The example assumes yase has ingested at least one document about
# "voussoir" — out of scope for this script to handle ingestion. In
# practice you'd `POST /v1/collections/{id}/connectors` against yase
# with a connector job that crawls your source.

# 3. Set up env:
export ANTHROPIC_API_KEY=sk-ant-...
export YASE_URL=http://localhost:8000
# Optional:
export YASE_API_KEY=...   # only if YASE_AUTH_ENABLED=true on the server
pip install -e ".[yase]"

# 4. Run:
python examples/02b_research_agent_yase/main.py
```

## Notes

- The example pins `claude-haiku-4-5-20251001` (consistent with the Phase
  1 live smoke test — opus 4.7 is overload-prone under casual use).
- Skipped in CI by default: yase requires docker-compose + a seeded
  corpus, neither of which we set up automatically.
- For a tier-1 production wiring (Postgres + Qdrant memory + yase RAG),
  combine this example with `bind_postgres_memory` + `bind_qdrant_vector_store`
  helpers in `voussoir.container.defaults`.
