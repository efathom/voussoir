# 01 — hello agent

Smallest possible voussoir program. Builds an Agent with default container,
sends one user message, prints the response + token stats.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/01_hello_agent/main.py
```

Expected output (rough):

```
output:        Hi, hello, hello!
tokens_in:     19
tokens_out:    7
cost_usd:      $0.000211
duration_ms:   742.3
finish_reason: completed
```
