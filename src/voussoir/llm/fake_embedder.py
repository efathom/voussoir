"""FakeEmbeddingProvider — deterministic, no-network embeddings for tests/dev.

Returns 384-dim float vectors derived from a hash of the input text. Two
identical strings get identical vectors; different strings get vectors
with small but non-zero distance.

This avoids depending on sentence-transformers (which downloads weights
on first use) in the test suite.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ctxforge.protocols.llm import EmbeddingResponse


class FakeEmbeddingProvider:
    """384-dim deterministic embedder. Implements IEmbeddingProvider."""

    DIM = 384

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-384d"

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **_kwargs: Any,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[self._vec(t) for t in texts],
            model="fake-384d",
            total_tokens=sum(len(t.split()) for t in texts),
        )

    async def embed_single(
        self,
        text: str,
        model: str | None = None,
        **_kwargs: Any,
    ) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        out: list[float] = []
        i = 0
        while len(out) < self.DIM:
            byte = h[i % len(h)]
            out.append((byte / 127.5) - 1.0)
            i += 1
        return out
