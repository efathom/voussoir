"""Default container — wires sensible defaults so `Agent("x").run("hi")` works.

Import-style note: leaf Protocol imports (IMemoryStore, ISessionStore,
ILLMProvider, IEmbeddingProvider) sit at module level. Concrete impls
(AnthropicLLMProvider, SentenceTransformersEmbedder, SQLiteMemoryStore,
ctxforge OpenAI provider, etc.) are imported inside the helper that
binds them — they're heavy or optional and we only want the cost/probe
when the user actually triggers that branch.
"""

from __future__ import annotations

import os
from typing import Any

from voussoir.auth.authorizers.deny_by_default import DenyByDefaultAuthorizer
from voussoir.auth.protocol import Authorizer
from voussoir.container.container import Container
from voussoir.observability.logging_setup import configure_logging
from voussoir.observability.tracer import configure_otel
from voussoir.protocols import (
    IEmbeddingProvider,
    IGraphStore,
    ILLMProvider,
    IMemoryStore,
    ISessionStore,
    ISkillStore,
    IVectorStore,
)


def default_container() -> Container:
    """Return a container pre-bound with framework defaults.

    The defaults populate (lazily where possible):
    - OpenTelemetry (TracerProvider + MeterProvider; no-op when disabled)
    - logging (JSON in prod via VOUSSOIR_LOG_FORMAT, otherwise dev)
    - LLM provider (Anthropic if ANTHROPIC_API_KEY else OpenAI if OPENAI_API_KEY
      else OpenRouter if OPENROUTER_API_KEY)
    - in-memory MemoryStore
    - in-memory SessionStore
    - fail-closed Authorizer (DenyByDefaultAuthorizer — tool calls deny until granted)
    - standard guardrail chain (length caps, injection heuristic, exfil scan)
    - chained env+keychain CredentialBroker

    Each binding is added via lambda so heavy modules import lazily.
    """
    # Wire OTel early so every subsequent span/metric emission has a provider.
    # configure_otel is idempotent and honors VOUSSOIR_OTEL_DISABLED /
    # OTEL_SDK_DISABLED (set by the test suite's autouse fixture).
    configure_otel()
    configure_logging()

    c = Container()
    _bind_default_stores(c)
    _bind_llm_provider(c)
    _bind_default_embedder(c)
    _bind_default_telemetry_sink(c)
    _bind_default_key_provider(c)
    _bind_default_authorizer(c)
    _bind_default_executor(c)
    _bind_default_guardrail_chain(c)

    # v1.0.2 D6: freeze security-critical keys against plugin rebind.
    # voussoir.delegates plugins load AFTER default_container returns;
    # freezing prevents a hostile plugin from swapping the Authorizer /
    # KeyProvider / ITelemetrySink behind the agent's back.
    # v1.0.4 E3: extends the freeze list to 5 keys — ILLMProvider and
    # IMemoryStore are equally critical attack surfaces: a hostile plugin
    # rebinding ILLMProvider silently replaces every agent's model; rebinding
    # IMemoryStore injects false history into every subsequent agent run.
    from voussoir.a2a.keys import KeyProvider
    from voussoir.executors import IToolExecutor  # v1.1.0 F1
    from voussoir.observability.sink import ITelemetrySink

    c.freeze(Authorizer)
    c.freeze(KeyProvider)
    c.freeze(ITelemetrySink)
    c.freeze(ILLMProvider)  # v1.0.4 E3: prevent hostile plugin from swapping the LLM
    c.freeze(IMemoryStore)  # v1.0.4 E3: prevent hostile plugin from poisoning history
    c.freeze(ISessionStore)  # v1.0.4 E3: prevent hostile plugin from poisoning session state
    c.freeze(IToolExecutor)  # v1.1.0 F1: prevent hostile plugin from swapping the executor
    return c


def _bind_default_stores(c: Container) -> None:
    """Tier 0 in-memory session + memory stores. Phase 1 default; tier-up via bind()."""
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore

    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())


def _bind_default_authorizer(c: Container) -> None:
    """Bind DenyByDefaultAuthorizer as the default Authorizer (v1.3.0).

    The default is fail-closed: every tool invocation denies until an operator
    binds a concrete Authorizer (RoleAuthorizer, DomainAuthorizer,
    ChainedAuthorizer) that grants it. This is the zero-trust posture for
    production; `AllowAllAuthorizer` remains available for dev/experiments.
    """
    c.bind(Authorizer, DenyByDefaultAuthorizer())  # type: ignore[type-abstract]


def _bind_default_executor(c: Container) -> None:
    """Bind StandardExecutor as the default IToolExecutor (v1.1.0 F1)."""
    from voussoir.executors import IToolExecutor, StandardExecutor

    c.bind(IToolExecutor, StandardExecutor())  # type: ignore[type-abstract]


def _bind_default_guardrail_chain(c: Container) -> None:
    """Bind the 'standard' guardrail profile by default.

    The empty chain previously bound here meant the framework's advertised
    "safe by default" posture was actually opt-in: injection / exfil / length
    guards only ran if an operator explicitly called bind_default_guardrails.
    'standard' registers length caps, the prompt-injection heuristic, args
    schema + size caps, and the exfil pattern scan. 'strict' (PII + URL
    allowlist) remains opt-in via bind_default_guardrails(c, profile="strict").
    """
    from voussoir.guardrails.bootstrap import bind_default_guardrails

    bind_default_guardrails(c, profile="standard")


def _bind_default_key_provider(c: Container) -> None:
    """Default A2A KeyProvider. Lazy-bound so processes that never use A2A
    don't pay the ephemeral-keygen cost."""
    from voussoir.a2a.keys import EnvKeyProvider, KeyProvider

    def _make_provider() -> KeyProvider:
        # Phase 4.5a P0 #5: explicit allow_ephemeral=False matches the new
        # default; production deployments must set VOUSSOIR_A2A_JWT_SECRET.
        return EnvKeyProvider(allow_ephemeral=False)

    c.bind(KeyProvider, _make_provider)  # type: ignore[type-abstract]


def _bind_default_telemetry_sink(c: Container) -> None:
    """Default no-op sink. Phase 3.5 cascade scopes a buffer over the top
    during validator.validate() calls; production deployments override with
    an OTel-backed sink. Tests bind InMemoryTelemetrySink for assertion."""
    from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink

    # mypy strict mode forbids using a Protocol as `type[T]`. ctxforge
    # Protocols (IMemoryStore, ILLMProvider, etc.) escape this because
    # ctxforge.* is `ignore_missing_imports = true` in pyproject.toml, so
    # they read as Any. ITelemetrySink is first-party and mypy sees it as
    # an abstract Protocol; the runtime contract (any value satisfying the
    # Protocol) is what we want, so suppress here.
    c.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]


def _bind_llm_provider(c: Container) -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):

        def _make_anthropic() -> ILLMProvider:
            # Built in Phase 1 — voussoir owns the Anthropic adapter because
            # ctxforge ships only OpenAI / Azure / Mock providers as of 2026-05.
            #
            # `type: ignore[return-value]`: AnthropicLLMProvider's `stream` is
            # an async generator (`async def + yield`), the idiomatic Python
            # streaming pattern. ctxforge's ILLMProvider Protocol declares
            # `stream` with the wrong signature shape — `async def -> AsyncIterator`
            # without `yield`, which mypy reads as `Coroutine[..., AsyncIterator]`
            # (i.e., `iter = await p.stream(); async for x in iter`). Both forms
            # produce a usable iterator at runtime via `async for`, so the
            # behavioral contract is honored even though the static signatures
            # disagree. Worth fixing upstream in ctxforge.
            from voussoir.llm.anthropic import AnthropicLLMProvider

            return AnthropicLLMProvider()

        c.bind(ILLMProvider, _make_anthropic)
    elif os.environ.get("OPENAI_API_KEY"):

        def _make_openai() -> ILLMProvider:
            from ctxforge.llm.openai_provider import OpenAIConfig, OpenAILLMProvider

            return OpenAILLMProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))

        c.bind(ILLMProvider, _make_openai)
    elif os.environ.get("OPENROUTER_API_KEY"):
        # OpenRouter exposes an OpenAI-compatible API, so ctxforge's provider
        # subclasses OpenAILLMProvider and only swaps the base URL (+ optional
        # attribution headers). No embeddings endpoint is offered, so
        # `_bind_default_embedder` still falls back to a local/fake embedder.
        def _make_openrouter() -> ILLMProvider:
            from ctxforge.llm.openrouter_provider import (
                OPENROUTER_BASE_URL,
                OpenRouterConfig,
                OpenRouterLLMProvider,
            )

            return OpenRouterLLMProvider(
                OpenRouterConfig(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    model=os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o-mini",
                    base_url=os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
                    http_referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
                    site_title=os.environ.get("OPENROUTER_SITE_TITLE"),
                )
            )

        c.bind(ILLMProvider, _make_openrouter)
    else:
        raise RuntimeError(
            "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
            "OPENROUTER_API_KEY, or bind a custom ILLMProvider via Container.bind()."
        )


def _bind_default_embedder(c: Container) -> None:
    """Pick an IEmbeddingProvider impl by preference order.

    1. OPENAI_API_KEY set → ctxforge's OpenAIEmbeddingProvider (production)
    2. sentence-transformers installed → SentenceTransformersEmbedder (offline fallback)
    3. Fall back to FakeEmbeddingProvider (warns; tests only)

    Production wiring uses ctxforge primitives when an API key is available;
    the local sentence-transformers fallback exists for offline dev workflows.
    Users override via `c.bind(IEmbeddingProvider, MyImpl())` after
    `default_container()` returns, or by constructing their own Container.
    """
    if os.environ.get("OPENAI_API_KEY"):

        def _make_openai_embedder() -> IEmbeddingProvider:
            from ctxforge.llm.openai_provider import (
                OpenAIConfig,
                OpenAIEmbeddingProvider,
            )

            return OpenAIEmbeddingProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))

        c.bind(IEmbeddingProvider, _make_openai_embedder)
        return

    try:
        import sentence_transformers  # noqa: F401  — probe install
    except ImportError:
        pass
    else:
        from voussoir.llm.sentence_transformers_embedder import (
            SentenceTransformersEmbedder,
        )

        c.bind(IEmbeddingProvider, SentenceTransformersEmbedder())
        return

    from voussoir.llm.fake_embedder import FakeEmbeddingProvider
    from voussoir.observability.logging_setup import get_logger

    get_logger(__name__).warning(
        "No embedder available; using FakeEmbeddingProvider (development only). "
        "Set OPENAI_API_KEY or install voussoir[mem-sqlite] for sentence-transformers."
    )
    c.bind(IEmbeddingProvider, FakeEmbeddingProvider())


def bind_postgres_memory(c: Container, *, config: Any | None = None) -> None:
    """Bind ctxforge's PostgresMemoryStore + PostgresSessionStore.

    `config` is an optional `ctxforge.storage.connection.PostgresConfig`. If
    omitted, the stores read DSN/credentials from the standard env vars
    (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`,
    `POSTGRES_PASSWORD`).
    """
    from ctxforge.storage.postgres.memory import PostgresMemoryStore
    from ctxforge.storage.postgres.session import PostgresSessionStore

    if config is None:
        c.bind(IMemoryStore, PostgresMemoryStore())
        c.bind(ISessionStore, PostgresSessionStore())
    else:
        c.bind(IMemoryStore, PostgresMemoryStore(config=config))
        c.bind(ISessionStore, PostgresSessionStore(config=config))


def bind_qdrant_vector_store(
    c: Container,
    *,
    host: str = "localhost",
    port: int = 6333,
    api_key: str | None = None,
    default_collection: str = "memories",
    vector_size: int = 384,
) -> None:
    """Bind a Qdrant-backed IVectorStore on the container.

    Constructs a fresh `qdrant_client.AsyncQdrantClient`; voussoir owns
    the lifetime — call `c.resolve(IVectorStore).close()` at shutdown.
    """
    from qdrant_client import AsyncQdrantClient

    from voussoir.memory.backends.qdrant import QdrantVectorStore

    # `check_compatibility=False` suppresses the version-probe round-trip the
    # client makes on construction; we don't need to talk to qdrant until a
    # method is called, and the probe causes a 30s+ delay when binding against
    # a not-yet-running qdrant (e.g., in tests or before docker-compose is up).
    client = AsyncQdrantClient(host=host, port=port, api_key=api_key, check_compatibility=False)
    store = QdrantVectorStore(
        client=client,
        default_collection=default_collection,
        vector_size=vector_size,
    )
    c.bind(IVectorStore, store)


def bind_skill_store(c: Container, *, store: ISkillStore | None = None) -> None:
    """Bind a ctxforge ISkillStore on the container.

    `store` defaults to ctxforge's InMemorySkillStore (in-process). With this
    bound, `Agent(skills=[...])` triggers auto-wiring of
    SkillActivationMiddleware on each run; without it, agent.skills is logged
    and ignored.
    """
    if store is None:
        from ctxforge.storage.memory.skill import InMemorySkillStore

        store = InMemorySkillStore()
    c.bind(ISkillStore, store)


def bind_graph_store(c: Container, *, store: IGraphStore | None = None) -> None:
    """Bind a ctxforge IGraphStore on the container (Tier 3 graph wiring).

    `store` defaults to ctxforge's InMemoryGraphStore (zero-arg, in-process).
    Pass a Neo4j-backed store for production:

        from ctxforge.graph.stores.neo4j import Neo4jGraphStore
        bind_graph_store(c, store=Neo4jGraphStore(uri=..., user=..., password=...))
    """
    if store is None:
        from ctxforge.graph.stores.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
    c.bind(IGraphStore, store)


def bind_sqlite_memory(c: Container, *, path: str, embedding_dim: int = 384) -> None:
    """Tier-up the bound IMemoryStore to SQLite.

    v1.0.4 E3: must be called on a fresh `Container()`, NOT on a
    `default_container()` result — the latter freezes `IMemoryStore` to
    prevent plugin-driven history poisoning. Build your own Container,
    bind the embedder, then call this helper; or call this BEFORE
    `default_container()` if you can wire the container yourself.

    Resolves the IEmbeddingProvider that's currently bound on the container.
    To use a custom embedder, bind it BEFORE calling this helper.
    """
    from voussoir.memory.backends.sqlite import SQLiteMemoryStore

    embedder = c.resolve(IEmbeddingProvider)
    store = SQLiteMemoryStore(path=path, embedder=embedder, embedding_dim=embedding_dim)
    c.bind(IMemoryStore, store)
