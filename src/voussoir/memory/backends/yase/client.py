"""YaseClient — hand-written async httpx wrapper for the yase HTTP API.

Phase 2 covers only the endpoints voussoir consumes: search, rag,
collection CRUD, document delete. Add a new wrapper method as new
endpoints land. If the surface grows past ~15 endpoints, revisit
the openapi-python-client codegen path that Task 2.8 deferred.

Adds on top of raw httpx:
- API-key / bearer-token header injection
- Exponential-backoff retry on 5xx and connect/read timeout
- structlog + tracer spans around each call
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from voussoir.observability import span as otel_span
from voussoir.observability.logging_setup import get_logger

_log = get_logger(__name__)


class YaseClient:
    """Async HTTP client for yase. One instance per (base_url, auth) pair."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        bearer: str | None = None,
        timeout_s: float = 30.0,
        retry_attempts: int = 3,
        retry_base_delay_s: float = 0.5,
    ) -> None:
        if api_key and bearer:
            raise ValueError("YaseClient: pass api_key OR bearer, not both")
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
        )
        self._retries = retry_attempts
        self._retry_delay = retry_base_delay_s

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---- search / rag -----------------------------------------------------

    async def search(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if filters:
            body["filters"] = filters
        return await self._post(f"/v1/collections/{collection_id}/search", body)

    async def rag(
        self,
        *,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        include_text: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_text": include_text,
        }
        if filters:
            body["filters"] = filters
        return await self._post("/v1/rag", body)

    # ---- collection CRUD --------------------------------------------------

    async def create_collection(
        self,
        *,
        id: str,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"id": id, "name": name}
        if description:
            body["description"] = description
        return await self._post("/v1/collections", body)

    async def get_collection(self, collection_id: str) -> dict[str, Any]:
        return await self._get(f"/v1/collections/{collection_id}")

    async def delete_collection(self, collection_id: str) -> dict[str, Any]:
        return await self._delete(f"/v1/collections/{collection_id}")

    # ---- document writes --------------------------------------------------

    async def delete_documents(self, doc_ids: list[int]) -> dict[str, Any]:
        return await self._delete("/v1/documents", json={"doc_ids": doc_ids})

    # ---- transport --------------------------------------------------------

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._with_retry(lambda: self._http.post(path, json=body), op=f"POST {path}")

    async def _get(self, path: str) -> dict[str, Any]:
        return await self._with_retry(lambda: self._http.get(path), op=f"GET {path}")

    async def _delete(
        self,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._with_retry(
            lambda: self._http.request("DELETE", path, json=json),
            op=f"DELETE {path}",
        )

    async def _with_retry(
        self,
        call: Callable[[], Awaitable[httpx.Response]],
        *,
        op: str,
    ) -> dict[str, Any]:
        last: BaseException | None = None
        for attempt in range(self._retries):
            with otel_span("yase.call", op=op, attempt=attempt):
                try:
                    resp = await call()
                    if resp.status_code >= 500 and attempt < self._retries - 1:
                        await asyncio.sleep(self._retry_delay * (2**attempt))
                        continue
                    resp.raise_for_status()
                    data: dict[str, Any] = resp.json()
                    return data
                except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                    last = exc
                    if attempt < self._retries - 1:
                        await asyncio.sleep(self._retry_delay * (2**attempt))
                        continue
                    raise
                except httpx.HTTPStatusError as exc:
                    last = exc
                    if attempt < self._retries - 1 and exc.response.status_code >= 500:
                        await asyncio.sleep(self._retry_delay * (2**attempt))
                        continue
                    raise
        assert last is not None
        raise last
