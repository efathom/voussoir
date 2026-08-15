# 04 — Validator Judge

Phase 3.5 demo: ToolUseFaithfulness + LLMJudge fallback wired through `AmbiguousFallback`. The lead's `RequestCascade.verifier` is the composite; validator cost rolls into `AgentResult.cost_usd`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/04_validator_judge/main.py
```
