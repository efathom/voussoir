# 05 — Observability

Demonstrates Phase 5's OpenTelemetry instrumentation in voussoir:

| Demo | What it shows |
|---|---|
| `console_exporter_demo.py` | Tier 0 default: spans printed at end of run via ConsoleSpanExporter |
| `otlp_phoenix_demo.py` | Tier 1: spans exported via OTLP HTTP to a local Phoenix instance |
| `streaming_cascade_demo.py` | Streaming agent + single-pass cascade; observe `cascade_passed` / `cascade_failed` events after `done` |

## Run a demo

```bash
.venv/bin/python examples/05_observability/console_exporter_demo.py
```

## Disable OTel entirely

```bash
VOUSSOIR_OTEL_DISABLED=1 python examples/05_observability/console_exporter_demo.py
# or the official OTel env var:
OTEL_SDK_DISABLED=true python examples/05_observability/console_exporter_demo.py
```

## Phoenix setup (for `otlp_phoenix_demo.py`)

```bash
docker run -p 6006:6006 arizephoenix/phoenix:latest
# Then run the demo. Spans appear in the Phoenix UI at http://localhost:6006
```
