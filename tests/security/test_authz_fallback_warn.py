"""Locks the _FallbackAllowAllAuthorizer warn-once behavior (v1.0.2 D3)."""

from __future__ import annotations

import structlog


def test_fallback_warns_once_when_container_none():
    """First call to _resolve_authorizer(None) emits the structlog warning."""
    from voussoir.executors import standard as _exec

    _exec._fallback_warned = False  # type: ignore[attr-defined]
    with structlog.testing.capture_logs() as captured:
        _exec._resolve_authorizer(None)
        _exec._resolve_authorizer(None)  # second call should NOT re-warn
    warns = [r for r in captured if r.get("event") == "authz_fallback_used"]
    assert len(warns) == 1


def test_fallback_warns_once_on_lookup_error(make_container):
    """When container has no Authorizer bound, fallback fires + warns once."""
    from voussoir.container import Container
    from voussoir.executors import standard as _exec

    # make_container() does NOT bind Authorizer, so LookupError path fires.
    # Build a plain Container to be sure.
    c = Container()

    _exec._fallback_warned = False  # type: ignore[attr-defined]
    with structlog.testing.capture_logs() as captured:
        _exec._resolve_authorizer(c)
        _exec._resolve_authorizer(c)
    warns = [r for r in captured if r.get("event") == "authz_fallback_used"]
    assert len(warns) == 1


def test_fallback_does_not_warn_when_authorizer_bound():
    """When the container resolves a real Authorizer, no warn fires."""
    from voussoir.container.defaults import default_container
    from voussoir.executors import standard as _exec

    c = default_container()  # binds DenyByDefaultAuthorizer (fail-closed)

    _exec._fallback_warned = False  # type: ignore[attr-defined]
    with structlog.testing.capture_logs() as captured:
        _exec._resolve_authorizer(c)
    warns = [r for r in captured if r.get("event") == "authz_fallback_used"]
    assert warns == []
