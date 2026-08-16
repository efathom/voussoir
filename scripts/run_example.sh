#!/usr/bin/env bash
# Load .env (if present) then run an example under examples/<dir>/<script>.
#   scripts/run_example.sh 01_hello_agent
#   scripts/run_example.sh 04_a2a_peer publisher.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "warning: no .env found at $ROOT/.env — using already-exported vars" >&2
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "error: no LLM API key set. Copy .env.example to .env and fill in a key." >&2
    exit 1
fi

for k in ANTHROPIC_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY; do
    v="${!k:-}"
    if [[ -n "$v" && "$v" == *"..."* ]]; then
        echo "warning: $k still holds the .env.example placeholder — replace it in .env" >&2
    fi
done

example="${1:?usage: scripts/run_example.sh <example-dir> [script] — script defaults to main.py}"
script="${2:-main.py}"

exec .venv/bin/python "examples/$example/$script"