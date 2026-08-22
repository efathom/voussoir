"""Phase 4b A2A example — caller.

Discovers the publisher's AgentCard, wraps it in an AgentRef, and uses it
as a delegate from a local 'lead' agent.
"""

from __future__ import annotations

import asyncio
import os

from ctxforge.llm.openrouter_provider import (
    OPENROUTER_BASE_URL,
    OpenRouterConfig,
    OpenRouterLLMProvider,
)

from voussoir.a2a import AgentRef
from voussoir.a2a.keys import EnvKeyProvider
from voussoir.agent.agent import Agent
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
    if not any(
        os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    ):
        print(
            "Set an LLM API key (ANTHROPIC_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY) to run the caller."
        )
        return
    if not os.environ.get("VOUSSOIR_A2A_JWT_SECRET"):
        print(
            "Set VOUSSOIR_A2A_JWT_SECRET to the same value as the publisher "
            "(see examples/04_a2a_peer/README.md)."
        )
        return

    c = build_demo_container()
    ref = await AgentRef.discover("http://127.0.0.1:8765")
    print(f"Discovered: {ref.name} — {ref.description}")

    # Phase 4.5a P0 #8: bare AgentRef(card) has _http=None. Wrap in
    # `async with ref:` so the httpx client is open for .delegate() calls.
    async with ref:
        lead = Agent(
            name="lead",
            instructions=(
                "You orchestrate research. Use delegate_to_researcher to gather "
                "notes on the user's topic, then return a single concise paragraph "
                "summarizing them."
            ),
            delegates=[ref],
            model=MODEL,
            container=c,
        )
        result = await lead.run(
            "Topic: 2026 advances in agent-to-agent protocols. "
            "Get research notes and produce a final summary."
        )
        print()
        print("Final output:")
        print(result.output)
        print()
        print(f"delegation_chain: {result.delegation_chain}")
        print(f"cost_usd: ${result.cost_usd:.6f}")
        print(f"finish_reason: {result.finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
