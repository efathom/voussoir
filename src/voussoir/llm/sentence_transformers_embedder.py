"""SentenceTransformersEmbedder — local IEmbeddingProvider, default for Tier 1.

Lazy-loads the model on first call. Model files (~80MB for the default
all-MiniLM-L6-v2) are cached by sentence-transformers under the user's
HF cache dir; first run requires network. Subsequent runs are offline.

The synchronous .encode() call runs in an executor so the event loop
isn't blocked during inference.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ctxforge.protocols.llm import EmbeddingResponse

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbedder:
    """Local sentence-transformers IEmbeddingProvider (Tier 1 default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return "sentence-transformers"

    @property
    def default_model(self) -> str:
        return self._model_name

    def _ensure_model(self) -> Any:
        if self._model is None:
            # Imported lazily so unit tests that patch the symbol don't
            # need it loaded at module import time.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **_kwargs: Any,
    ) -> EmbeddingResponse:
        m = self._ensure_model()

        def _encode() -> list[list[float]]:
            arr = m.encode(texts)
            return [list(map(float, row)) for row in arr]

        embeddings = await asyncio.to_thread(_encode)
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self._model_name,
            total_tokens=sum(len(t.split()) for t in texts),
        )

    async def embed_single(
        self,
        text: str,
        model: str | None = None,
        **_kwargs: Any,
    ) -> list[float]:
        m = self._ensure_model()
        result = await asyncio.to_thread(m.encode, text)
        return [float(x) for x in result]
