"""Phase 4b A2A example — publisher.

Run in terminal 1:
    export VOUSSOIR_A2A_JWT_SECRET=$(python -c "import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())")
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/04_a2a_peer/publisher.py

Then in terminal 2 (with the SAME VOUSSOIR_A2A_JWT_SECRET in the env):
    python examples/04_a2a_peer/caller.py
"""

from __future__ import annotations

import os

from voussoir.a2a.publisher import serve_a2a
from voussoir.agent.agent import Agent
from voussoir.container.defaults import default_container


def main() -> None:
    if not any(
        os.environ.get(k)
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    ):
        print("Set an LLM API key (ANTHROPIC_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY) to run the publisher.")
        return
    c = default_container()
    agent = Agent(
        name="researcher",
        description="Surveys recent reading on a topic and produces a brief summary.",
        instructions=(
            "You are a research assistant. Given a topic, produce 3-5 bullet "
            "points capturing key recent developments. Be concise."
        ),
        model="deepseek/deepseek-v4-flash-0731",
        container=c,
    )
    print("Publishing 'researcher' on http://127.0.0.1:8765")
    print("Card: http://127.0.0.1:8765/.well-known/agent-card.json")
    serve_a2a(agent, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
