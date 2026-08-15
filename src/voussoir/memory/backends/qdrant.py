"""QdrantVectorStore — IVectorStore over qdrant-client (production Tier 1).

Voussoir's only first-party IVectorStore impl; ctxforge ships Chroma /
Pinecone / Weaviate adapters but not Qdrant. This adapter maps voussoir
namespaces to Qdrant collections and translates VectorRecord/VectorQueryResult
to/from Qdrant's PointStruct/ScoredPoint shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ctxforge.vectorstores.protocol import (
    IVectorStore,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
)

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient


class QdrantVectorStore(IVectorStore):
    """Async Qdrant-backed IVectorStore.

    `default_collection` is used when callers pass namespace=None. To bind
    multiple collections, construct multiple instances.
    """

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        default_collection: str = "memories",
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> None:
        self._client = client
        self._default_collection = default_collection
        self._vector_size = vector_size
        self._distance = distance

    def _coll(self, namespace: str | None) -> str:
        return namespace or self._default_collection

    async def initialize(self) -> None:
        """Create the default collection if it doesn't already exist."""
        from qdrant_client.http.models import Distance, VectorParams

        if not await self._client.collection_exists(self._default_collection):
            await self._client.create_collection(
                collection_name=self._default_collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=getattr(Distance, self._distance.upper()),
                ),
            )

    async def upsert(
        self,
        vectors: list[VectorRecord],
        namespace: str | None = None,
    ) -> int:
        from qdrant_client.http.models import PointStruct

        points = [
            PointStruct(
                id=v.id,
                vector=v.embedding,
                payload={"content": v.content, **(v.metadata or {})},
            )
            for v in vectors
        ]
        await self._client.upsert(
            collection_name=self._coll(namespace),
            points=points,
        )
        return len(points)

    async def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filters: list[QueryFilter] | None = None,
        include_embedding: bool = False,
        include_metadata: bool = True,
    ) -> list[VectorQueryResult]:
        # qdrant-client deprecated `search()` in favor of `query_points()`.
        response = await self._client.query_points(
            collection_name=self._coll(namespace),
            query=embedding,
            limit=top_k,
            with_payload=include_metadata,
            with_vectors=include_embedding,
        )
        out: list[VectorQueryResult] = []
        for p in response.points:
            payload = dict(p.payload or {})
            content = str(payload.pop("content", ""))
            out.append(
                VectorQueryResult(
                    id=str(p.id),
                    score=float(p.score),
                    metadata=payload,
                    content=content,
                    embedding=list(p.vector) if include_embedding and p.vector else None,
                )
            )
        return out

    async def delete(self, ids: list[str], namespace: str | None = None) -> int:
        from qdrant_client.http.models import PointIdsList

        await self._client.delete(
            collection_name=self._coll(namespace),
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    async def count(
        self,
        namespace: str | None = None,
        filters: list[QueryFilter] | None = None,
    ) -> int:
        result = await self._client.count(collection_name=self._coll(namespace))
        return int(result.count)

    async def fetch(
        self,
        ids: list[str],
        namespace: str | None = None,
    ) -> dict[str, VectorRecord]:
        points = await self._client.retrieve(
            collection_name=self._coll(namespace),
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )
        out: dict[str, VectorRecord] = {}
        for p in points:
            payload = dict(p.payload or {})
            content = str(payload.pop("content", ""))
            out[str(p.id)] = VectorRecord(
                id=str(p.id),
                embedding=list(p.vector) if p.vector else [],
                metadata=payload,
                content=content,
            )
        return out

    async def delete_by_filter(
        self,
        filters: list[QueryFilter],
        namespace: str | None = None,
    ) -> int:
        # Phase 2 keeps the QueryFilter → Qdrant Filter translation minimal.
        # Wired in Phase 5 when guardrails need filter-based pruning.
        raise NotImplementedError(
            "filter-based delete lands in Phase 5; use delete(ids=...) for now"
        )

    async def delete_namespace(self, namespace: str) -> bool:
        await self._client.delete_collection(namespace)
        return True

    async def describe(self) -> dict[str, Any]:
        info = await self._client.get_collections()
        return {"collections": [c.name for c in info.collections]}

    async def list_namespaces(self) -> list[str]:
        info = await self._client.get_collections()
        return [c.name for c in info.collections]

    async def query_by_id(
        self,
        vector_id: str,
        top_k: int = 10,
        namespace: str | None = None,
        filters: list[QueryFilter] | None = None,
        include_embedding: bool = False,
    ) -> list[VectorQueryResult]:
        # qdrant-client deprecated `recommend()`; query_points() accepts a
        # point id as the query and treats it as a recommendation.
        from qdrant_client.http.models import RecommendInput, RecommendQuery

        response = await self._client.query_points(
            collection_name=self._coll(namespace),
            query=RecommendQuery(recommend=RecommendInput(positive=[vector_id])),
            limit=top_k,
            with_vectors=include_embedding,
        )
        out: list[VectorQueryResult] = []
        for p in response.points:
            payload = dict(p.payload or {})
            content = str(payload.pop("content", ""))
            out.append(
                VectorQueryResult(
                    id=str(p.id),
                    score=float(p.score),
                    metadata=payload,
                    content=content,
                    embedding=list(p.vector) if include_embedding and p.vector else None,
                )
            )
        return out

    async def close(self) -> None:
        await self._client.close()
