"""Re-export ctxforge Protocols so voussoir consumers have one import surface.

voussoir does NOT redefine these. They are the canonical ctxforge contracts;
voussoir merely surfaces them under its own namespace for ergonomic imports.
"""

from ctxforge.protocols.compactor import IContextAssembler
from ctxforge.protocols.expertise import IExpertiseRetriever, IExpertiseStore
from ctxforge.protocols.extractor import IMemoryExtractor
from ctxforge.protocols.graph import IGraphStore
from ctxforge.protocols.llm import IEmbeddingProvider, ILLMProvider
from ctxforge.protocols.retriever import IRetriever
from ctxforge.protocols.skill import ISkillStore
from ctxforge.protocols.storage import IMemoryStore, ISessionStore
from ctxforge.protocols.tokenizer import ITokenizerProvider
from ctxforge.protocols.vectorstore import IVectorStore

__all__ = [
    "IContextAssembler",
    "IEmbeddingProvider",
    "IExpertiseRetriever",
    "IExpertiseStore",
    "IGraphStore",
    "ILLMProvider",
    "IMemoryExtractor",
    "IMemoryStore",
    "IRetriever",
    "ISessionStore",
    "ISkillStore",  # noqa: kept here; already in alpha-order
    "ITokenizerProvider",
    "IVectorStore",
]
