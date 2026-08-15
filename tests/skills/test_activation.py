from unittest.mock import AsyncMock, MagicMock

from ctxforge.core.skill import Skill, SkillMatch, SkillMetadata, SkillScope

from voussoir.skills.adapter import SkillActivationMiddleware


def _meta(name: str, *, description: str = "") -> SkillMetadata:
    return SkillMetadata(
        name=name,
        description=description,
        scope=SkillScope.PROJECT,
        scope_id="default",
    )


def _skill(name: str, *, content: str) -> Skill:
    return Skill(
        name=name,
        description="",
        scope=SkillScope.PROJECT,
        scope_id="default",
        content=content,
    )


def _match(meta: SkillMetadata, *, confidence: float) -> SkillMatch:
    return SkillMatch(
        skill=meta,
        confidence=confidence,
        matched_trigger=None,
        match_reason="test",
    )


async def test_before_run_records_matched_skills_on_ctx():
    incident_meta = _meta("incident-response")
    runbook_meta = _meta("ops-runbooks")

    matcher = MagicMock()
    matcher.match = AsyncMock(
        return_value=[
            _match(incident_meta, confidence=0.9),
            _match(runbook_meta, confidence=0.85),
        ]
    )
    skill_store = MagicMock()
    skill_store.list_all_metadata = AsyncMock(return_value=[incident_meta, runbook_meta])
    skill_store.get = AsyncMock(
        side_effect=lambda name, scope, scope_id: _skill(name, content=f"{name} body")
    )

    mw = SkillActivationMiddleware(
        matcher=matcher,
        skill_store=skill_store,
        agent_skills_hint=["ops-runbooks"],
    )
    ctx = MagicMock()
    ctx.skills_active = []
    ctx.skill_content = []
    await mw.before_run(ctx, "the cluster is on fire")

    assert sorted(ctx.skills_active) == ["incident-response", "ops-runbooks"]


async def test_before_run_loads_skill_content_for_messages():
    incident_meta = _meta("incident-response")
    matcher = MagicMock()
    matcher.match = AsyncMock(return_value=[_match(incident_meta, confidence=0.9)])
    skill_store = MagicMock()
    skill_store.list_all_metadata = AsyncMock(return_value=[incident_meta])
    skill_store.get = AsyncMock(
        return_value=_skill("incident-response", content="run incident-response runbook")
    )

    mw = SkillActivationMiddleware(
        matcher=matcher,
        skill_store=skill_store,
        agent_skills_hint=[],
    )
    ctx = MagicMock()
    ctx.skills_active = []
    ctx.skill_content = []
    await mw.before_run(ctx, "the cluster is on fire")

    assert ctx.skill_content == ["run incident-response runbook"]


async def test_match_below_threshold_is_skipped():
    low_meta = _meta("low-signal")
    matcher = MagicMock()
    matcher.match = AsyncMock(return_value=[_match(low_meta, confidence=0.3)])
    skill_store = MagicMock()
    skill_store.list_all_metadata = AsyncMock(return_value=[low_meta])
    skill_store.get = AsyncMock()

    mw = SkillActivationMiddleware(
        matcher=matcher,
        skill_store=skill_store,
        agent_skills_hint=[],
        threshold=0.7,
    )
    ctx = MagicMock()
    ctx.skills_active = []
    ctx.skill_content = []
    await mw.before_run(ctx, "vague question")

    assert ctx.skills_active == []
    assert ctx.skill_content == []
    skill_store.get.assert_not_called()


async def test_hint_skill_activated_even_without_matcher_mention():
    """An agent_skills_hint entry should be activated unconditionally,
    even if the matcher returns nothing for it."""
    hint_meta = _meta("manual-hint")
    matcher = MagicMock()
    matcher.match = AsyncMock(return_value=[])  # No matches at all
    skill_store = MagicMock()
    skill_store.list_all_metadata = AsyncMock(return_value=[hint_meta])
    skill_store.get = AsyncMock(return_value=_skill("manual-hint", content="manual content"))

    mw = SkillActivationMiddleware(
        matcher=matcher,
        skill_store=skill_store,
        agent_skills_hint=["manual-hint"],
    )
    ctx = MagicMock()
    ctx.skills_active = []
    ctx.skill_content = []
    await mw.before_run(ctx, "anything")

    assert ctx.skills_active == ["manual-hint"]
    assert ctx.skill_content == ["manual content"]


async def test_passthroughs_dont_break():
    """after_step / after_run / on_error should be no-ops."""
    matcher = MagicMock()
    skill_store = MagicMock()
    mw = SkillActivationMiddleware(
        matcher=matcher,
        skill_store=skill_store,
        agent_skills_hint=[],
    )
    ctx = MagicMock()
    await mw.after_step(ctx, MagicMock())
    result_in = MagicMock()
    result_out = await mw.after_run(ctx, result_in)
    assert result_out is result_in
    exc = RuntimeError("x")
    out = await mw.on_error(ctx, exc)
    assert out is exc
