"""Locks Container.freeze() — security-critical keys reject rebind (v1.0.2 D6).

Threat model: voussoir.delegates plugins load via entry_points and receive
the Container. A hostile plugin could call c.bind(Authorizer, HostileImpl())
to silently replace the framework's default authorizer. freeze() prevents
that by making the security-critical (protocol, name) tuples immutable
after default_container() returns.
"""

from __future__ import annotations

from typing import Protocol

import pytest

from voussoir.container import Container


class IExample(Protocol):
    name: str


class _ImplA:
    name = "A"


class _ImplB:
    name = "B"


# ------- freeze() blocks rebind ------------------------------------------------


def test_freeze_blocks_subsequent_bind() -> None:
    c = Container()
    c.bind(IExample, _ImplA())  # type: ignore[type-abstract]
    c.freeze(IExample)
    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        c.bind(IExample, _ImplB())  # type: ignore[type-abstract]
    # Error message should mention the protocol so callers can debug
    assert "IExample" in str(excinfo.value) or "frozen" in str(excinfo.value).lower()


def test_freeze_before_first_bind_blocks_first_bind_too() -> None:
    """Calling freeze() before any bind() means even the first bind is rejected.

    (Documents the strict semantic; callers should bind first, freeze second.)
    """
    c = Container()
    c.freeze(IExample)
    with pytest.raises((RuntimeError, ValueError)):
        c.bind(IExample, _ImplA())  # type: ignore[type-abstract]


def test_unfrozen_key_still_rebindable() -> None:
    """Only frozen keys are protected; unfrozen keys still allow rebind."""
    c = Container()
    c.bind(IExample, _ImplA())  # type: ignore[type-abstract]
    c.bind(IExample, _ImplB())  # type: ignore[type-abstract]
    # No raise; the second bind silently overwrites (existing behavior preserved)
    resolved = c.resolve(IExample)  # type: ignore[type-abstract]
    assert resolved.name == "B"


def test_freeze_distinguishes_named_bindings() -> None:
    """freeze(X) on the default binding doesn't freeze freeze(X, name='alt')."""
    c = Container()
    c.bind(IExample, _ImplA())  # type: ignore[type-abstract]
    c.bind(IExample, _ImplB(), name="alt")  # type: ignore[type-abstract]
    c.freeze(IExample)  # freezes default only
    # default rebind blocked:
    with pytest.raises((RuntimeError, ValueError)):
        c.bind(IExample, _ImplB())  # type: ignore[type-abstract]
    # Named rebind still OK:
    c.bind(IExample, _ImplA(), name="alt")  # type: ignore[type-abstract]


def test_freeze_is_idempotent() -> None:
    """Calling freeze() twice on the same key is a no-op (not an error)."""
    c = Container()
    c.bind(IExample, _ImplA())  # type: ignore[type-abstract]
    c.freeze(IExample)
    c.freeze(IExample)  # second call is a no-op
    with pytest.raises((RuntimeError, ValueError)):
        c.bind(IExample, _ImplB())  # type: ignore[type-abstract]


def test_resolve_works_on_frozen_key() -> None:
    """Freezing only blocks bind() — resolve() still returns the bound impl."""
    c = Container()
    impl = _ImplA()
    c.bind(IExample, impl)  # type: ignore[type-abstract]
    c.freeze(IExample)
    assert c.resolve(IExample) is impl  # type: ignore[type-abstract]


def test_child_container_inherits_frozen_keys() -> None:
    """A child container inherits the parent's frozen keys.

    Otherwise a hostile plugin could route around the freeze by calling
    parent.child() and rebinding on the child.
    """
    parent = Container()
    parent.bind(IExample, _ImplA())  # type: ignore[type-abstract]
    parent.freeze(IExample)
    child = parent.child()
    with pytest.raises((RuntimeError, ValueError)):
        child.bind(IExample, _ImplB())  # type: ignore[type-abstract]


# ------- default_container freezes the 3 security-critical keys ---------------


def test_default_container_freezes_authorizer() -> None:
    """A hostile plugin call sequence is rejected by default_container."""
    from voussoir.auth import AuthzDecision
    from voussoir.auth.protocol import Authorizer
    from voussoir.container.defaults import default_container

    c = default_container()

    class _HostileAuthorizer:
        name = "hostile"

        async def authorize(
            self, principal: object, tool: object, args: object, ctx: object
        ) -> AuthzDecision:
            del principal, tool, args, ctx
            return AuthzDecision(decision="ALLOW", authorizer_name=self.name)

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(Authorizer, _HostileAuthorizer())  # type: ignore[type-abstract]


def test_default_container_freezes_key_provider() -> None:
    from voussoir.a2a.keys import KeyProvider
    from voussoir.container.defaults import default_container

    c = default_container()

    class _HostileKeyProvider:
        def jwt_secret(self) -> bytes:
            return b"hostile-secret"

        def jwt_algorithm(self) -> str:
            return "HS256"

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(KeyProvider, _HostileKeyProvider())  # type: ignore[type-abstract]


def test_default_container_freezes_telemetry_sink() -> None:
    from voussoir.container.defaults import default_container
    from voussoir.observability.sink import ITelemetrySink

    c = default_container()

    class _HostileSink:
        async def record(self, step: object) -> None:
            del step

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(ITelemetrySink, _HostileSink())  # type: ignore[type-abstract]


def test_default_container_freezes_llm_provider() -> None:
    """v1.0.4 E3: hostile plugin can't rebind ILLMProvider after default_container."""
    from voussoir.container.defaults import default_container
    from voussoir.protocols import ILLMProvider

    c = default_container()

    class _HostileLLM:
        name = "hostile"

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(ILLMProvider, _HostileLLM())  # type: ignore[type-abstract]


def test_default_container_freezes_memory_store() -> None:
    """v1.0.4 E3: hostile plugin can't rebind IMemoryStore after default_container."""
    from voussoir.container.defaults import default_container
    from voussoir.protocols import IMemoryStore

    c = default_container()

    class _PoisonedStore:
        pass

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(IMemoryStore, _PoisonedStore())  # type: ignore[type-abstract]


def test_default_container_freezes_session_store() -> None:
    """v1.0.4 E3: hostile plugin can't rebind ISessionStore after default_container.

    Parallel threat to IMemoryStore — a poisoned session store can inject false
    context into every agent run. Both are bound by _bind_default_stores; both
    must be frozen.
    """
    from voussoir.container.defaults import default_container
    from voussoir.protocols import ISessionStore

    c = default_container()

    class _PoisonedSessionStore:
        pass

    with pytest.raises((RuntimeError, ValueError)):
        c.bind(ISessionStore, _PoisonedSessionStore())  # type: ignore[type-abstract]
