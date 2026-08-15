def test_protocols_re_exports_ctxforge_core():
    from voussoir.protocols import (
        IEmbeddingProvider,
        ILLMProvider,
        IMemoryStore,
        IRetriever,
        ISessionStore,
        ISkillStore,
        ITokenizerProvider,
        IVectorStore,
    )

    # Each must be a Protocol class:
    for proto in [
        IEmbeddingProvider,
        ILLMProvider,
        IMemoryStore,
        IRetriever,
        ISessionStore,
        ISkillStore,
        ITokenizerProvider,
        IVectorStore,
    ]:
        assert isinstance(proto, type)
