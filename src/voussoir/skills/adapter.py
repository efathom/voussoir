"""SkillActivationMiddleware — bridges ctxforge skill matching into voussoir.

On before_run, queries the bound SkillMatcher for the run's input plus any
agent-hinted skill names, then attaches the matched skill names + their
content to `ctx` so the agent loop's _build_messages can prepend them.

Phase 2 ships matching + ctx attachment. The actual prepend-to-messages
step is handled by Agent.run via the new `skills` parameter (Task 2.15).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voussoir.agent.context import AgentContext
from voussoir.observability.logging_setup import get_logger

if TYPE_CHECKING:
    from ctxforge.core.skill import SkillMetadata
    from ctxforge.engine.services.skill_matcher import SkillMatcher
    from ctxforge.protocols.skill import ISkillStore

_log = get_logger(__name__)


class SkillActivationMiddleware:
    """Activates ctxforge skills based on user input + Agent.skills hint.

    The hint list is activated unconditionally; SkillMatcher matches above
    threshold are merged in. Skill content is loaded via ISkillStore.get
    using each skill's own scope/scope_id from its SkillMetadata.
    """

    def __init__(
        self,
        *,
        matcher: SkillMatcher,
        skill_store: ISkillStore,
        agent_skills_hint: list[str],
        threshold: float = 0.7,
    ) -> None:
        self._matcher = matcher
        self._store = skill_store
        self._hint = list(agent_skills_hint)
        self._threshold = threshold

    async def before_run(self, ctx: AgentContext, input: Any) -> Any | None:
        available: list[SkillMetadata] = await self._store.list_all_metadata()
        matches = await self._matcher.match(
            query=str(input),
            available_skills=available,
            threshold=self._threshold,
        )

        # Build a name → metadata map of skills to activate.
        # Hint skills first (unconditional), then matched skills above threshold.
        active_metas: dict[str, SkillMetadata] = {}
        hint_set = set(self._hint)
        for meta in available:
            if meta.name in hint_set:
                active_metas[meta.name] = meta
        for m in matches:
            if m.confidence >= self._threshold:
                active_metas[m.skill.name] = m.skill

        # Load content for each active skill (best-effort — never fail the run).
        contents: list[str] = []
        for name, meta in sorted(active_metas.items()):
            try:
                skill = await self._store.get(name, meta.scope, meta.scope_id)
            except Exception as exc:  # noqa: BLE001 — never break the run on skill load
                _log.warning("skill.load_failed", skill=name, error=str(exc))
                continue
            if skill is not None and getattr(skill, "content", None):
                contents.append(str(skill.content))

        ctx.skills_active = sorted(active_metas.keys())
        ctx.skill_content = contents
        return None

    async def after_step(self, ctx: AgentContext, step: Any) -> None:
        return None

    async def after_run(self, ctx: AgentContext, result: Any) -> Any:
        return result

    async def on_error(self, ctx: AgentContext, exc: BaseException) -> Any:
        return exc
