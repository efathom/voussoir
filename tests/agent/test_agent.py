from unittest.mock import AsyncMock, MagicMock

from voussoir.agent import Agent, AgentResult


async def test_agent_run_returns_agent_result(make_container, stub_llm):
    llm = stub_llm(content="hi from stub", input_tokens=10, output_tokens=5)
    agent = Agent(name="test", container=make_container(llm))
    result = await agent.run("hello")
    assert isinstance(result, AgentResult)
    assert result.output == "hi from stub"
    assert result.tokens_in == 10
    assert result.tokens_out == 5
    assert result.finish_reason == "completed"
    assert result.delegation_chain == []


async def test_agent_run_calls_llm_with_user_message(make_container, stub_llm):
    llm = stub_llm()
    agent = Agent(name="test", container=make_container(llm))
    await agent.run("question?")
    call_args = llm.chat.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    assert messages[-1].role == "user"
    assert messages[-1].content == "question?"


async def test_agent_run_uses_instructions_as_system_message(make_container, stub_llm):
    llm = stub_llm()
    agent = Agent(name="test", instructions="be terse", container=make_container(llm))
    await agent.run("hi")
    messages = llm.chat.call_args.kwargs.get("messages") or llm.chat.call_args.args[0]
    assert messages[0].role == "system"
    assert "be terse" in messages[0].content


async def test_agent_auto_wires_skill_activation_when_skill_store_bound(make_container, stub_llm):
    """When agent.skills is set and ISkillStore is bound, the SkillActivation
    middleware is auto-built per run — no manual agent.middleware setup."""
    from unittest.mock import patch

    from ctxforge.core.skill import Skill, SkillMatch, SkillMetadata, SkillScope
    from ctxforge.storage.memory.skill import InMemorySkillStore

    skill = Skill(
        name="incident-response",
        description="",
        scope=SkillScope.PROJECT,
        scope_id="default",
        content="page oncall.",
    )

    store = InMemorySkillStore()
    await store.save(skill)

    llm = stub_llm()
    c = make_container(llm)
    from voussoir.protocols import ISkillStore

    c.bind(ISkillStore, store)

    incident_meta = SkillMetadata(
        name="incident-response",
        description="",
        scope=SkillScope.PROJECT,
        scope_id="default",
    )
    with patch(
        "ctxforge.engine.services.skill_matcher.SkillMatcher.match",
        new=AsyncMock(
            return_value=[
                SkillMatch(
                    skill=incident_meta,
                    confidence=0.9,
                    matched_trigger=None,
                    match_reason="",
                )
            ]
        ),
    ):
        agent = Agent(name="ops", skills=["incident-response"], container=c)
        # Note: agent.middleware is empty — auto-wiring should provide the middleware.
        assert agent.middleware == []
        await agent.run("alert!")

    messages = llm.chat.call_args.kwargs["messages"]
    assert any(m.role == "system" and "page oncall" in m.content for m in messages)


async def test_agent_skips_auto_wire_when_no_skill_store(make_container, stub_llm):
    """Skills set but ISkillStore not bound: log a warning, run normally."""
    llm = stub_llm()
    c = make_container(llm)
    # ISkillStore is NOT bound on `c`.
    agent = Agent(name="ops", skills=["whatever"], container=c)
    result = await agent.run("hi")
    # Run completed without crashing:
    assert result.finish_reason == "completed"


async def test_agent_skips_auto_wire_when_user_already_provided_skill_middleware(
    make_container, stub_llm
):
    """If the caller manually attaches a SkillActivationMiddleware,
    auto-wiring doesn't add a second one."""
    from voussoir.skills.adapter import SkillActivationMiddleware

    llm = stub_llm()
    c = make_container(llm)

    user_mw = MagicMock(spec=SkillActivationMiddleware)
    user_mw.before_run = AsyncMock()
    user_mw.after_step = AsyncMock()
    user_mw.after_run = AsyncMock(side_effect=lambda ctx, r: r)
    user_mw.on_error = AsyncMock(side_effect=lambda ctx, e: e)

    agent = Agent(name="ops", skills=["x"], container=c)
    agent.middleware = [user_mw]

    # No ISkillStore bound — but it doesn't matter, the user provided their own.
    await agent.run("hi")
    user_mw.before_run.assert_awaited_once()


async def test_agent_run_prepends_skill_content_when_active(make_container, stub_llm):
    """SkillActivationMiddleware → ctx.skill_content → leading system messages.

    Verifies the end-to-end wiring: agent.middleware runs before_run, the
    middleware writes ctx.skill_content, and _build_messages prepends
    each entry as a system message before the user input.
    """
    from ctxforge.core.skill import Skill, SkillMatch, SkillMetadata, SkillScope

    from voussoir.skills.adapter import SkillActivationMiddleware

    incident_meta = SkillMetadata(
        name="incident-response",
        description="",
        scope=SkillScope.PROJECT,
        scope_id="default",
    )
    matcher = MagicMock()
    matcher.match = AsyncMock(
        return_value=[
            SkillMatch(
                skill=incident_meta,
                confidence=0.9,
                matched_trigger=None,
                match_reason="",
            )
        ]
    )
    store = MagicMock()
    store.list_all_metadata = AsyncMock(return_value=[incident_meta])
    store.get = AsyncMock(
        return_value=Skill(
            name="incident-response",
            description="",
            scope=SkillScope.PROJECT,
            scope_id="default",
            content="When alerts fire: page oncall.",
        )
    )

    llm = stub_llm()
    agent = Agent(
        name="ops",
        skills=["incident-response"],
        container=make_container(llm),
    )
    agent.middleware = [
        SkillActivationMiddleware(
            matcher=matcher,
            skill_store=store,
            agent_skills_hint=["incident-response"],
        ),
    ]
    await agent.run("alert!")

    messages = llm.chat.call_args.kwargs["messages"]
    assert any(m.role == "system" and "page oncall" in m.content for m in messages)


async def test_agent_with_container_returns_clone(make_container, stub_llm):
    """`_with_container` returns a shallow copy with the container field
    swapped. Other fields share references with the original."""
    from voussoir import Container

    llm = stub_llm()
    c1 = make_container(llm)
    c2 = Container()  # different instance; not bound to anything for this test

    agent = Agent(
        name="orig",
        instructions="be terse",
        model="claude-haiku-4-5-20251001",
        container=c1,
    )

    swapped = agent._with_container(c2)
    assert swapped is not agent
    assert swapped._container is c2
    assert agent._container is c1  # original unchanged
    # Non-overridden fields share by reference (shallow copy semantics):
    assert swapped.name == agent.name
    assert swapped.instructions == agent.instructions
    assert swapped.model == agent.model
