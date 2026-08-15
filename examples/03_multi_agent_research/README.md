# 03 — multi-agent research (delegation)

Demonstrates Phase 3's hierarchical delegation:

```
lead ──[delegate_to_researcher(task=...)]──> researcher (own session, child container)
     ──[delegate_to_writer(task=research_notes)]──> writer (separate session)
     ── final synthesis text from writer ──> output
```

- `delegation_chain` populates with the actual call sequence.
- Each delegation runs in its own ctxforge session (independent memory).
- Costs and tokens aggregate into the lead's `AgentResult`.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/03_multi_agent_research/main.py
```

Expected output (rough):

```
output:            A polished paragraph synthesizing the research notes…
delegation_chain:  ['lead', 'researcher', 'writer']
steps:             [('llm_call', 'chat'), ('tool_call', 'delegate_to_researcher'),
                    ('delegation', 'researcher'), ('llm_call', 'chat'),
                    ('tool_call', 'delegate_to_writer'),
                    ('delegation', 'writer'), ('llm_call', 'chat')]
tokens_in:         ~1800
tokens_out:        ~600
cost_usd:          $0.0019
finish_reason:     completed
```
