"""Container scopes: SINGLETON, RUN, TRANSIENT.

A SINGLETON instance is cached for the entire process. A RUN-scoped instance
is cached for the duration of a single agent run (entered via `run_scope`).
A TRANSIENT instance is constructed every time it is requested.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any

_run_id_ctx: ContextVar[str | None] = ContextVar("voussoir_run_id", default=None)


class Scope(StrEnum):
    """Lifetime label for a container-managed dependency.

    Use this when registering a factory with the Container to declare how
    long the constructed value should live: SINGLETON caches one instance
    for the whole process, RUN caches one instance per agent run (scoped
    via `Container.run_scope`), and TRANSIENT rebuilds on every request.
    """

    SINGLETON = "singleton"
    RUN = "run"
    TRANSIENT = "transient"


class ScopeContext:
    """Holds the per-process singleton cache and per-run-id RUN caches."""

    def __init__(self) -> None:
        self._singletons: dict[str, Any] = {}
        self._run_caches: dict[str, dict[str, Any]] = {}

    @contextmanager
    def run_scope(self, run_id: str) -> Iterator[None]:
        token = _run_id_ctx.set(run_id)
        self._run_caches.setdefault(run_id, {})
        try:
            yield
        finally:
            self._run_caches.pop(run_id, None)
            _run_id_ctx.reset(token)

    def get_or_create(self, key: str, factory: Callable[[], Any], *, scope: Scope) -> Any:
        if scope is Scope.TRANSIENT:
            return factory()
        if scope is Scope.SINGLETON:
            if key not in self._singletons:
                self._singletons[key] = factory()
            return self._singletons[key]
        # RUN
        run_id = _run_id_ctx.get()
        if run_id is None:
            raise LookupError("no active run scope; wrap call in `container.run_scope(...)`.")
        cache = self._run_caches[run_id]
        if key not in cache:
            cache[key] = factory()
        return cache[key]
