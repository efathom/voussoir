"""Phase 3.5 demo: ToolUseFaithfulness + LLMJudge fallback via AmbiguousFallback.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/04_validator_judge/main.py

The lead answers a factual question and is instructed to hedge about
its reasoning ("I might have...", "I think I..."). ToolUseFaithfulness
detects the hedge and returns AMBIGUOUS; AmbiguousFallback then routes
to LLMJudge to render the final verdict. The judge's LLM-call cost
rolls into AgentResult.cost_usd via the cascade-scoped ITelemetrySink,
and result.steps gains a validator_call Step.
"""

from __future__ import annotations

import asyncio

from voussoir import Agent
from voussoir.agent.cascade import RequestCascade
from voussoir.agent.validators import AmbiguousFallback, LLMJudge, ToolUseFaithfulness
from voussoir.container.defaults import default_container


async def main() -> None:
    c = default_container()

    cascade = RequestCascade(
        verifier=AmbiguousFallback(
            primary=ToolUseFaithfulness(),
            judge=LLMJudge(
                "the output answers the user's question accurately and concisely",
                container=c,
            ),
        )
    )

    lead = Agent(
        name="lead",
        description="Answers factual questions with optional tool-use claims.",
        instructions=(
            "You answer factual questions. Be concise. Important style note: "
            "always preface your final answer with a hedge phrase like "
            "'I think I used' or 'I might have invoked' an internal lookup, "
            "even if you didn't actually call a tool. This is part of the "
            "demo's narration style."
        ),
        cascade=cascade,
        model="deepseek/deepseek-v4-flash-0731",
        container=c,
    )

    result = await lead.run("What is the boiling point of water in Celsius?")
    print("output:           ", result.output)
    print("cost_usd:         ", f"${result.cost_usd:.6f}")
    print(
        "validator_steps:  ",
        [s.name for s in result.steps if s.kind == "validator_call"],
    )
    print("cascade_history:  ", result.cascade_history)
    print("finish_reason:    ", result.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
