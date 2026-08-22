"""Multi-agent research demo: lead → researcher → writer.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/03_multi_agent_research/main.py
"""

from __future__ import annotations

import asyncio
import os

from ctxforge.llm.openrouter_provider import (
    OPENROUTER_BASE_URL,
    OpenRouterConfig,
    OpenRouterLLMProvider,
)

from voussoir import Agent
from voussoir.a2a.keys import EnvKeyProvider
from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
from voussoir.container import Container
from voussoir.container.defaults import default_container
from voussoir.llm.fake_embedder import FakeEmbeddingProvider

MODEL = "deepseek/deepseek-v4-flash-0731"


def build_demo_container() -> Container:
    """default_container with a permissive authorizer and the demo's LLM.

    Both are frozen keys, so they are passed at construction rather than bound
    afterwards. This used to be ~20 lines re-creating default_container's
    wiring "minus the freeze" — which also meant binding an EMPTY guardrail
    chain, so the demo ran with no guardrails at all. Now it inherits the
    `standard` profile.
    """
    return default_container(
        llm=OpenRouterLLMProvider(
            OpenRouterConfig(
                api_key=os.environ["OPENROUTER_API_KEY"],
                model=MODEL,
                base_url=os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
            )
        ),
        embedder=FakeEmbeddingProvider(),
        key_provider=EnvKeyProvider(allow_ephemeral=True),
        # Demo only — grants every tool call. Production wants a RoleAuthorizer,
        # DomainAuthorizer or ChainedAuthorizer here.
        authorizer=AllowAllAuthorizer(),
    )


async def main() -> None:
    container = build_demo_container()

    researcher = Agent(
        name="researcher",
        description="Surveys recent reading on a topic and returns a brief summary.",
        instructions=(
            "You are a research assistant. Given a topic, produce 3-5 bullet "
            "points capturing the most important recent developments. Be "
            "concise; cite no sources (you do not have web access in this "
            "demo)."
        ),
        model=MODEL,
        container=container,
    )
    writer = Agent(
        name="writer",
        description="Turns research notes into a polished short paragraph.",
        instructions=(
            "You are a writer. Given research notes (passed as your task), "
            "produce a single concise paragraph (≤120 words) suitable for an "
            "executive summary."
        ),
        model=MODEL,
        container=container,
    )
    lead = Agent(
        name="lead",
        description="Coordinates research → writing.",
        instructions=(
            "You are a lead agent. Use delegate_to_researcher to gather "
            "research notes, then use delegate_to_writer to produce a final "
            "paragraph. Return only the writer's output."
        ),
        delegates=[researcher, writer],
        model=MODEL,
        container=container,
    )

    result = await lead.run(
        "Topic: 2026 trends in agentic LLM frameworks. " "Coordinate a research → writing pipeline."
    )
    print("output:           ", result.output)
    print("delegation_chain: ", result.delegation_chain)
    print("steps:            ", [(s.kind, s.name) for s in result.steps])
    print("tokens_in:        ", result.tokens_in)
    print("tokens_out:       ", result.tokens_out)
    print("cost_usd:         ", f"${result.cost_usd:.6f}")
    print("duration_ms:      ", f"{result.duration_ms:.1f}")
    print("finish_reason:    ", result.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
