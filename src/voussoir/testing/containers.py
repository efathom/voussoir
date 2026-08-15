"""make_container() and make_runtime_container() -- fresh Container factories.

make_container is for unit-testing voussoir-ecosystem code (custom
guardrails, brokers, executors, A2A peers). It binds the minimal set of
defaults needed to construct an Agent and run it against a stub LLM.

make_runtime_container (v1.2 G3) is for in-process runtimes (e.g. impost)
that host voussoir agents and need to override security-critical Protocols
(IToolExecutor, ILLMProvider, ITelemetrySink, IGuardrailChain) which
``default_container()`` freezes. Same minimal-default surface as
``make_container`` plus runtime-mediated kwargs; no freeze list, so
runtime layers can rebind any Protocol after construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from voussoir.container import Container

if TYPE_CHECKING:
    from ctxforge.protocols.llm import ILLMProvider

    from voussoir.executors import IToolExecutor
    from voussoir.guardrails import IGuardrailChain
    from voussoir.observability.sink import ITelemetrySink


def make_container(llm: ILLMProvider | None = None) -> Container:
    """Use this when unit-testing voussoir-ecosystem code.

    Builds a fresh Container() (NOT default_container -- no freeze) with
    the minimal bindings mirroring the shared conftest fixture:
    ILLMProvider (optional), IMemoryStore, ISessionStore, ITelemetrySink,
    and KeyProvider (allow_ephemeral=True so tests run without
    VOUSSOIR_A2A_JWT_SECRET set).

    llm: bind as ILLMProvider. None leaves the binding empty.
    """
    from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
    from voussoir.protocols import ILLMProvider as ILLMProviderProto
    from voussoir.protocols import IMemoryStore, ISessionStore

    container = Container()

    if llm is not None:
        container.bind(ILLMProviderProto, llm)

    container.bind(IMemoryStore, InMemoryStore())
    container.bind(ISessionStore, InMemorySessionStore())
    container.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]
    container.bind(KeyProvider, EnvKeyProvider(allow_ephemeral=True))  # type: ignore[type-abstract]

    return container


def make_runtime_container(
    *,
    llm: ILLMProvider | None = None,
    tool_executor: IToolExecutor | None = None,
    guardrail_chain: IGuardrailChain | None = None,
    telemetry_sink: ITelemetrySink | None = None,
    extra_bindings: Mapping[type, Any] | None = None,
) -> Container:
    """Use this when an in-process runtime hosts voussoir agents (v1.2).

    Companion to ``make_container``: same minimal default bindings, plus
    runtime-mediated override kwargs for the four Protocols a runtime
    typically needs to intercept (LLM, executor, guardrail chain,
    telemetry sink). Returns a fresh ``Container()`` -- NOT
    ``default_container()`` -- so the freeze list does not block
    rebinding. ``extra_bindings`` registers additional first-party
    Protocol bindings without re-implementing the helper.

    For tests, prefer ``make_container``. This factory exists for runtime
    layers (e.g. impost) that mediate every tool/LLM/telemetry call and
    cannot tolerate the freeze enforced by ``default_container()``.
    """
    from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
    from voussoir.executors import IToolExecutor as IToolExecutorProto
    from voussoir.executors import StandardExecutor
    from voussoir.guardrails import DefaultGuardrailChain
    from voussoir.guardrails import IGuardrailChain as IGuardrailChainProto
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.observability.sink import ITelemetrySink as ITelemetrySinkProto
    from voussoir.observability.sink import NullTelemetrySink
    from voussoir.protocols import ILLMProvider as ILLMProviderProto
    from voussoir.protocols import IMemoryStore, ISessionStore

    container = Container()

    if llm is not None:
        container.bind(ILLMProviderProto, llm)

    container.bind(IMemoryStore, InMemoryStore())
    container.bind(ISessionStore, InMemorySessionStore())
    container.bind(KeyProvider, EnvKeyProvider(allow_ephemeral=True))  # type: ignore[type-abstract]

    container.bind(
        ITelemetrySinkProto,  # type: ignore[type-abstract]
        telemetry_sink if telemetry_sink is not None else NullTelemetrySink(),
    )
    container.bind(
        IToolExecutorProto,  # type: ignore[type-abstract]
        tool_executor if tool_executor is not None else StandardExecutor(),
    )
    container.bind(
        IGuardrailChainProto,  # type: ignore[type-abstract]
        guardrail_chain if guardrail_chain is not None else DefaultGuardrailChain([]),
    )

    if extra_bindings is not None:
        for protocol, impl in extra_bindings.items():
            container.bind(protocol, impl)

    return container
