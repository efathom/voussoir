import pytest

from voussoir.container.defaults import default_container


def test_default_container_returns_container_with_logging_bound(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    default_container()
    # Logging configured (no public surface to assert; just must not raise):
    from voussoir.observability.logging_setup import get_logger

    log = get_logger("voussoir.test")
    log.info("ok")


def test_default_container_picks_anthropic_when_key_present(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    c = default_container()
    # Resolution happens lazily via ctxforge; we just confirm a binding exists.
    from ctxforge.protocols.llm import ILLMProvider

    binding_keys = [p for (p, _) in c.list_bindings()]
    assert ILLMProvider in binding_keys


def test_default_container_picks_openai_when_only_openai_key_present(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    c = default_container()
    from ctxforge.protocols.llm import ILLMProvider

    binding_keys = [p for (p, _) in c.list_bindings()]
    assert ILLMProvider in binding_keys


def test_default_container_picks_openrouter_when_only_openrouter_key_present(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    from ctxforge.protocols.llm import ILLMProvider

    c = default_container()
    binding_keys = [p for (p, _) in c.list_bindings()]
    assert ILLMProvider in binding_keys


def test_default_container_openrouter_binds_ctxforge_provider(monkeypatch):
    """OPENROUTER_API_KEY wires ctxforge's OpenAI-compatible OpenRouter provider.

    The bound factory must produce an OpenRouterLLMProvider (not the plain
    OpenAI provider) so attribution headers + the OpenRouter base URL apply.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    from ctxforge.protocols.llm import ILLMProvider

    c = default_container()
    impl = c.resolve(ILLMProvider)
    assert type(impl).__name__ == "OpenRouterLLMProvider"
    assert impl.name == "openrouter"
    assert impl.default_model == "anthropic/claude-sonnet-4"
    assert impl._config.http_referer == "https://example.com"


def test_default_container_raises_when_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No LLM API key"):
        default_container()


def test_default_container_binds_session_and_memory_stores(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    c = default_container()
    from voussoir.protocols import IMemoryStore, ISessionStore

    binding_keys = {p for (p, _) in c.list_bindings()}
    assert IMemoryStore in binding_keys
    assert ISessionStore in binding_keys


def test_default_container_resolves_anthropic_provider_factory_lazily(monkeypatch):
    """Binding registers without invoking the factory (which would import voussoir.llm.anthropic)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    c = default_container()
    # We don't resolve; we just confirm the binding exists.
    from voussoir.protocols import ILLMProvider

    assert any(p is ILLMProvider for (p, _) in c.list_bindings())


def test_default_container_picks_openai_embedder_when_key_set(monkeypatch):
    """OpenAI is preferred when OPENAI_API_KEY is set, even if
    sentence-transformers is installed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from voussoir.protocols import IEmbeddingProvider

    c = default_container()
    impl = c.resolve(IEmbeddingProvider)
    # ctxforge's OpenAIEmbeddingProvider — check by class name to avoid heavy import:
    assert type(impl).__name__ == "OpenAIEmbeddingProvider"


def test_default_container_falls_back_to_sentence_transformers_when_no_openai_key(monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from voussoir.llm.sentence_transformers_embedder import SentenceTransformersEmbedder
    from voussoir.protocols import IEmbeddingProvider

    c = default_container()
    impl = c.resolve(IEmbeddingProvider)
    assert isinstance(impl, SentenceTransformersEmbedder)


def test_default_container_falls_back_to_fake_when_st_missing_and_no_openai(monkeypatch):
    """Simulate sentence-transformers not installed and OpenAI not configured."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from voussoir.llm.fake_embedder import FakeEmbeddingProvider
    from voussoir.protocols import IEmbeddingProvider

    c = default_container()
    impl = c.resolve(IEmbeddingProvider)
    assert isinstance(impl, FakeEmbeddingProvider)


def test_user_can_override_embedder(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from voussoir.llm.fake_embedder import FakeEmbeddingProvider
    from voussoir.protocols import IEmbeddingProvider

    c = default_container()
    custom = FakeEmbeddingProvider()
    c.bind(IEmbeddingProvider, custom)
    assert c.resolve(IEmbeddingProvider) is custom


def test_bind_sqlite_memory_uses_bound_embedder(monkeypatch, tmp_path):
    # v1.0.4 E3: default_container() now freezes IMemoryStore; test the helper
    # against a fresh Container() so the tier-up bind is not blocked.
    from voussoir.container import Container
    from voussoir.container.defaults import bind_sqlite_memory
    from voussoir.llm.fake_embedder import FakeEmbeddingProvider
    from voussoir.memory.adapter import InMemoryStore
    from voussoir.memory.backends.sqlite import SQLiteMemoryStore
    from voussoir.protocols import IEmbeddingProvider, IMemoryStore

    c = Container()
    custom = FakeEmbeddingProvider()
    c.bind(IEmbeddingProvider, custom)
    c.bind(IMemoryStore, InMemoryStore())

    bind_sqlite_memory(c, path=str(tmp_path / "memory.db"))
    impl = c.resolve(IMemoryStore)
    assert isinstance(impl, SQLiteMemoryStore)
    # Store uses the embedder bound on the container, not a hardcoded fake:
    assert impl._embedder is custom


def test_bind_sqlite_memory_with_default_embedder(monkeypatch, tmp_path):
    # v1.0.4 E3: default_container() now freezes IMemoryStore; test the helper
    # against a fresh Container() so the tier-up bind is not blocked.
    from voussoir.container import Container
    from voussoir.container.defaults import bind_sqlite_memory
    from voussoir.llm.sentence_transformers_embedder import SentenceTransformersEmbedder
    from voussoir.memory.adapter import InMemoryStore
    from voussoir.memory.backends.sqlite import SQLiteMemoryStore
    from voussoir.protocols import IEmbeddingProvider, IMemoryStore

    c = Container()
    c.bind(IEmbeddingProvider, SentenceTransformersEmbedder())
    c.bind(IMemoryStore, InMemoryStore())

    bind_sqlite_memory(c, path=str(tmp_path / "memory.db"))
    impl = c.resolve(IMemoryStore)
    assert isinstance(impl, SQLiteMemoryStore)
    # Whatever the picker chose lives on the store:
    assert impl._embedder is c.resolve(IEmbeddingProvider)


def test_bind_skill_store_uses_inmemory_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from ctxforge.storage.memory.skill import InMemorySkillStore

    from voussoir.container.defaults import bind_skill_store
    from voussoir.protocols import ISkillStore

    c = default_container()
    bind_skill_store(c)
    impl = c.resolve(ISkillStore)
    assert isinstance(impl, InMemorySkillStore)


def test_bind_skill_store_accepts_custom_store(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from ctxforge.storage.memory.skill import InMemorySkillStore

    from voussoir.container.defaults import bind_skill_store
    from voussoir.protocols import ISkillStore

    c = default_container()
    custom = InMemorySkillStore()
    bind_skill_store(c, store=custom)
    assert c.resolve(ISkillStore) is custom


def test_bind_graph_store_uses_inmemory_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from ctxforge.graph.stores.memory import InMemoryGraphStore

    from voussoir.container.defaults import bind_graph_store
    from voussoir.protocols import IGraphStore

    c = default_container()
    bind_graph_store(c)
    impl = c.resolve(IGraphStore)
    assert isinstance(impl, InMemoryGraphStore)


def test_bind_graph_store_accepts_custom_store(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from ctxforge.graph.stores.memory import InMemoryGraphStore

    from voussoir.container.defaults import bind_graph_store
    from voussoir.protocols import IGraphStore

    c = default_container()
    custom = InMemoryGraphStore()
    bind_graph_store(c, store=custom)
    assert c.resolve(IGraphStore) is custom


def test_bind_qdrant_vector_store_binds_qdrant_impl(monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from voussoir.container.defaults import bind_qdrant_vector_store
    from voussoir.memory.backends.qdrant import QdrantVectorStore
    from voussoir.protocols import IVectorStore

    c = default_container()
    bind_qdrant_vector_store(c, host="localhost", port=6333)
    impl = c.resolve(IVectorStore)
    assert isinstance(impl, QdrantVectorStore)


def test_bind_postgres_memory_binds_postgres_impls(monkeypatch):
    """Smoke test — actual Postgres connection isn't made until a method is called.

    v1.0.4 E3: default_container() now freezes IMemoryStore; test the helper
    against a fresh Container() so the tier-up bind is not blocked.
    """
    from ctxforge.storage.connection import PostgresConfig
    from ctxforge.storage.postgres.memory import PostgresMemoryStore
    from ctxforge.storage.postgres.session import PostgresSessionStore

    from voussoir.container import Container
    from voussoir.container.defaults import bind_postgres_memory
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.protocols import IMemoryStore, ISessionStore

    c = Container()
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    cfg = PostgresConfig(
        host="localhost",
        port=5432,
        database="test",
        user="test",
        password="test",
    )
    bind_postgres_memory(c, config=cfg)
    assert isinstance(c.resolve(IMemoryStore), PostgresMemoryStore)
    assert isinstance(c.resolve(ISessionStore), PostgresSessionStore)
