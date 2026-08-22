"""DI container — Hollywood-Principle wiring."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from voussoir.container.scopes import Scope, ScopeContext

T = TypeVar("T")

_Sentinel = object()


def _scope_key(protocol: type, name: str | None) -> str:
    """Cache key for a (protocol, name) binding. Shared by bind() and resolve()
    so an eviction on rebind targets exactly the entry resolve() would read."""
    return f"{protocol.__module__}.{protocol.__qualname__}#{name or ''}"


class Container:
    """Lightweight DI container.

    Bindings map a Protocol (or any type) → an instance or a factory callable.
    Resolution walks named bindings first, then default bindings, then any
    `default=` arg passed to `resolve`.
    """

    def __init__(self) -> None:
        self._bindings: dict[tuple[type, str | None], _Binding] = {}
        self._frozen: set[tuple[type, str | None]] = set()
        self._scopes = ScopeContext()

    # ---- binding ---------------------------------------------------------

    def bind(
        self,
        protocol: type[T],
        impl: T | Callable[[], T],
        *,
        name: str | None = None,
        scope: Scope = Scope.SINGLETON,
    ) -> None:
        key = (protocol, name)
        if key in self._frozen:
            named = f" (name={name!r})" if name else ""
            raise RuntimeError(
                f"Container key {protocol.__qualname__!r}{named} is frozen; "
                f"rebind rejected. This is a security feature -- "
                f"voussoir.delegates plugins must not replace security-critical "
                f"bindings (Authorizer, KeyProvider, ITelemetrySink). To swap "
                f"the binding in a test, construct a fresh Container() without "
                f"default_container()."
            )
        is_factory = callable(impl) and not _looks_like_instance(impl, protocol)
        self._bindings[key] = _Binding(impl=impl, is_factory=is_factory, scope=scope)
        # Rebinding must invalidate any instance the previous binding already
        # produced, otherwise `resolve()` keeps serving the old impl from the
        # singleton cache and the rebind is a silent no-op.
        self._scopes.evict(_scope_key(protocol, name))

    def freeze(self, protocol: type, *, name: str | None = None) -> None:
        """Freeze a (protocol, name) binding key against subsequent bind().

        Use this when binding security-critical Protocols (Authorizer,
        KeyProvider, ITelemetrySink) that plugins MUST NOT replace. Once
        frozen, any bind() to the same key raises RuntimeError.
        Idempotent — calling freeze() twice on the same key is a no-op.
        Frozen keys can still be resolve()d normally; only the default
        binding name is affected unless `name=` is passed (named bindings
        are independent freeze targets).
        """
        self._frozen.add((protocol, name))

    # ---- resolution ------------------------------------------------------

    def resolve(
        self,
        protocol: type[T],
        *,
        name: str | None = None,
        default: Any = _Sentinel,
    ) -> T:
        binding = self._bindings.get((protocol, name))
        if binding is None:
            # Only the sentinel means "no default supplied". An explicit
            # `default=None` is a legitimate "return None when unbound" —
            # LLMGuardrailJudge / LLMJudge both rely on it to make
            # ITelemetrySink optional.
            if default is _Sentinel:
                known = [n for (p, n) in self._bindings if p is protocol]
                hint = f" (known names: {known})" if known else ""
                raise LookupError(
                    f"No binding registered for {protocol.__name__}"
                    + (f" name={name!r}" if name else "")
                    + hint
                )
            return default  # type: ignore[no-any-return]

        key = _scope_key(protocol, name)
        impl = binding.impl
        is_factory = binding.is_factory

        def factory() -> Any:
            return impl() if is_factory else impl

        return self._scopes.get_or_create(key, factory, scope=binding.scope)  # type: ignore[no-any-return]

    # ---- introspection ---------------------------------------------------

    def list_bindings(self) -> list[tuple[type, str | None]]:
        return list(self._bindings.keys())

    def has(self, protocol: type, *, name: str | None = None) -> bool:
        """True iff a binding exists for `protocol` (optionally under `name`)."""
        return (protocol, name) in self._bindings

    # ---- run-scope passthrough ------------------------------------------

    @contextmanager
    def run_scope(self, run_id: str) -> Iterator[None]:
        with self._scopes.run_scope(run_id):
            yield

    # ---- delegation ------------------------------------------------------

    def child(self) -> Container:
        """Return a child container that inherits this container's bindings.

        The child starts from a snapshot of the parent's bindings and of the
        singletons the parent has already constructed, so an unmodified child
        keeps sharing the parent's instances. The snapshot is a *copy*: a
        child that rebinds a key gets its own instance for it, and neither
        that binding nor the instance it produces leaks back into the parent.
        Pre-fix the cache was shared by reference and keyed by protocol alone,
        so parent and child collided — whichever resolved first won, in both
        directions (audit H1).

        The child also has its own RUN-scope context. Useful for delegation:
        a sub-agent's container is a child of its parent's.
        """
        c = Container()
        c._bindings = dict(self._bindings)
        # D6: child inherits parent's frozen keys so plugin rebind via a
        # child container is rejected the same way.
        c._frozen = set(self._frozen)
        c._scopes._singletons = dict(self._scopes._singletons)
        return c


class _Binding:
    __slots__ = ("impl", "is_factory", "scope")

    def __init__(self, impl: Any, is_factory: bool, scope: Scope) -> None:
        self.impl = impl
        self.is_factory = is_factory
        self.scope = scope


def _looks_like_instance(obj: Any, protocol: type) -> bool:
    """Tell apart `bind(P, impl_instance)` from `bind(P, factory_callable)`.

    A bound *instance* may itself be callable (e.g. an Agent whose `__call__`
    runs it). We treat anything that is an instance of `protocol` as an
    instance; otherwise a callable is a factory.
    """
    try:
        return isinstance(obj, protocol)
    except TypeError:
        # Non-runtime-checkable Protocol: fall back to "callables are factories,
        # non-callables are instances." A callable instance (e.g. has __call__)
        # gets misclassified here — `@runtime_checkable` your Protocol if you
        # need that case to work.
        return not callable(obj)
