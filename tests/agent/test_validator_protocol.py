"""Pin the Validator protocol shape. Phase 3.5 added the keyword-only `task` kwarg."""

from __future__ import annotations

import inspect

import pytest

from voussoir.agent.agent import Agent
from voussoir.agent.cascade import Decision, RequestCascade, Validator


def test_validator_validate_takes_task_kwarg() -> None:
    sig = inspect.signature(Validator.validate)
    params = sig.parameters
    assert "task" in params, "Validator.validate must accept a `task` parameter"
    assert params["task"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.asyncio
async def test_cascade_passes_task_to_validator(make_container, stub_llm) -> None:
    """A spy validator captures `task`; cascade must pass task=input."""
    captured: dict[str, object] = {}

    class SpyValidator:
        name = "spy"

        async def validate(self, result, *, task):  # type: ignore[no-untyped-def]
            captured["task"] = task
            return Decision.PASS

    c = make_container(stub_llm())
    agent = Agent(
        "lead",
        instructions="",
        container=c,
        cascade=RequestCascade(verifier=SpyValidator()),
    )
    await agent.run("research X and report back")
    assert captured["task"] == "research X and report back"
