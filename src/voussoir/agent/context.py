"""AgentContext — per-run handle to the ctxforge engine + container scope.

This is voussoir's facade over ctxforge's actual API. It hides the ctxforge
method names and lifecycle from the run loop, so future ctxforge churn doesn't
ripple through agent.py.

ctxforge engine methods are async; AgentContext mirrors that on its facade
methods so callers `await` uniformly.
"""

from __future__ import annotations

import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.factory import EngineFactory
from pydantic import BaseModel

from voussoir.agent.result import GuardrailDecision
from voussoir.auth.decision import AuthzDecision
from voussoir.auth.principal import Principal, default_principal
from voussoir.container import Container
from voussoir.guardrails.chain import DefaultGuardrailChain
from voussoir.guardrails.protocol import IGuardrailChain
from voussoir.guardrails.trust import Trust
from voussoir.protocols import IMemoryStore, ISessionStore
from voussoir.tools.protocol import Capability
from voussoir.tools.registry import ToolRegistry


@dataclass
class AgentContext:
    container: Container
    run_id: str
    trace_id: str
    session_id: str
    user_id: str
    engine: CtxForge
    _stack: AsyncExitStack = field(repr=False)
    # Middleware-populated state (e.g. SkillActivationMiddleware writes here).
    skills_active: list[str] = field(default_factory=list)
    skill_content: list[str] = field(default_factory=list)
    # Delegation lineage.
    agent_name: str = ""
    parent_run_id: str | None = None
    delegation_depth: int = 0
    delegation_chain: list[str] = field(default_factory=list)
    max_delegation_depth: int = 3

    # Security / guardrail state.
    # Fail-closed default: if a future code path constructs AgentContext directly
    # without going through Agent, it defaults to denying all capabilities.
    allowed_capabilities: Capability = Capability.NONE
    taint: set[Trust] = field(default_factory=set)
    guardrail_decisions: list[GuardrailDecision] = field(default_factory=list)
    # Phase 6 A2: principal + authz audit log.
    principal: Principal = field(default_factory=default_principal)
    authz_decisions: list[AuthzDecision] = field(default_factory=list)
    # v1.0.2 D8: cached guardrail chain. Threaded from Agent at run-start so
    # _dispatch_one (per tool call) uses the SAME chain instance as the
    # input/output-stage screening in agent.py. Previously dispatch
    # re-resolved DefaultGuardrailChain from the container per tool call,
    # which would diverge from the agent's cached chain if anything rebound
    # the chain on the container mid-run (asymmetric enforcement).
    # v1.1.0 F2: annotation widened to IGuardrailChain (Protocol); the
    # default_factory still returns a DefaultGuardrailChain([]) (the concrete
    # impl), but custom chains satisfying IGuardrailChain can be threaded in.
    guardrail_chain: IGuardrailChain = field(default_factory=lambda: DefaultGuardrailChain([]))
    # Per-turn tool registry, set by Agent._run_setup. Backs the
    # `tool_input_schema` accessor that ArgsSchemaCheck reads (audit M13).
    tool_registry: ToolRegistry | None = field(default=None, repr=False)

    @classmethod
    async def open(
        cls,
        *,
        container: Container,
        run_id: str | None = None,
        session_id: str = "default",
        user_id: str = "local",
        allowed_capabilities: Capability = Capability.NONE,
        principal: Principal | None = None,
        guardrail_chain: IGuardrailChain | None = None,
    ) -> _AgentContextCM:
        """Build an async context manager that opens a Container RUN scope + ctxforge engine.

        Usage::

            async with await AgentContext.open(container=c, ...) as ctx:
                ...

        The `allowed_capabilities` parameter defaults to NONE (fail-closed); callers
        that need tool dispatch must pass the agent's allowed_capabilities explicitly
        OR overwrite `ctx.allowed_capabilities` post-construction. Agent._run_normal
        and Agent.stream currently use the post-construction assignment pattern
        (matching how they set other run-time fields like `agent_name`); future
        callers may prefer the kwarg path. Both are equivalent.

        The `principal` parameter (Phase 6 A2) carries the authenticated identity
        into the context. Defaults to the system principal when not provided.

        The `guardrail_chain` parameter (v1.0.2 D8) threads the agent's cached
        chain into the context so `_dispatch_one` can read it from ctx instead
        of re-resolving from the container per tool call. Defaults to an empty
        chain (no-op) when not provided.
        """
        rid = run_id or str(uuid.uuid4())
        return _AgentContextCM(
            container=container,
            run_id=rid,
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            allowed_capabilities=allowed_capabilities,
            principal=principal or default_principal(),
            guardrail_chain=(
                guardrail_chain if guardrail_chain is not None else DefaultGuardrailChain([])
            ),
        )

    def tool_input_schema(self, tool_name: str) -> type[BaseModel] | None:
        """Return the Pydantic input schema for `tool_name`, or None if unknown.

        `ArgsSchemaCheck` (the only guardrail in the "off" profile, and present
        in all three) looks this up with `getattr(ctx, "tool_input_schema",
        None)` and returns ALLOW when it's absent. AgentContext never defined
        it — only a test stub did — so the guardrail was a permanent no-op in
        production, contrary to its own docstring's promise that "the B5 task
        wires AgentContext with a real tool_input_schema accessor" (audit M13).

        The registry is per-turn, so Agent sets `tool_registry` at run setup;
        before that (or on a context built by hand) this returns None and the
        guardrail falls through to ALLOW exactly as before.
        """
        registry = self.tool_registry
        if registry is None:
            return None
        try:
            return registry.resolve(tool_name).input_schema
        except KeyError:
            return None

    async def record_user_message(self, content: str) -> None:
        await self.engine.record_user_message(self.session_id, self.user_id, content)

    async def record_assistant_message(self, content: str) -> None:
        await self.engine.record_assistant_message(self.session_id, self.user_id, content)

    async def record_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        tool_call_id: str | None = None,
    ) -> None:
        await self.engine.record_tool_use(
            self.session_id,
            self.user_id,
            tool_name,
            tool_input,
            tool_output,
            tool_call_id=tool_call_id,
        )

    async def prepare_context_for(self, user_input: str) -> Any:
        """Wrap ctxforge.prepare_context with our session/user defaults."""
        return await self.engine.prepare_context(
            session_id=self.session_id,
            user_id=self.user_id,
            user_input=user_input,
        )


class _AgentContextCM:
    """Async-context-manager helper produced by AgentContext.open()."""

    def __init__(
        self,
        *,
        container: Container,
        run_id: str,
        trace_id: str,
        session_id: str,
        user_id: str,
        allowed_capabilities: Capability = Capability.NONE,
        principal: Principal | None = None,
        guardrail_chain: IGuardrailChain | None = None,
    ) -> None:
        self._container = container
        self._run_id = run_id
        self._trace_id = trace_id
        self._session_id = session_id
        self._user_id = user_id
        self._allowed_capabilities = allowed_capabilities
        self._principal = principal or default_principal()
        self._guardrail_chain: IGuardrailChain = (
            guardrail_chain if guardrail_chain is not None else DefaultGuardrailChain([])
        )
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> AgentContext:
        # Open RUN scope on the container (sync context manager via ExitStack).
        scope_cm = self._container.run_scope(self._run_id)
        self._stack.enter_context(scope_cm)

        # Resolve the bound stores and build a CtxForge engine.
        session_store = self._container.resolve(ISessionStore)
        memory_store = self._container.resolve(IMemoryStore)
        engine = await EngineFactory.create_minimal(
            session_store=session_store,
            memory_store=memory_store,
        )
        # Ensure the session exists (Tier 0 InMemorySessionStore.load auto-creates).
        await engine.get_session(self._session_id, self._user_id)
        return AgentContext(
            container=self._container,
            run_id=self._run_id,
            trace_id=self._trace_id,
            session_id=self._session_id,
            user_id=self._user_id,
            engine=engine,
            _stack=self._stack,
            allowed_capabilities=self._allowed_capabilities,
            principal=self._principal,
            guardrail_chain=self._guardrail_chain,
        )

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self._stack.aclose()
