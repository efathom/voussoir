import pytest

from voussoir.container.scopes import Scope, ScopeContext


def test_scope_enum_has_three_values():
    assert {s.value for s in Scope} == {"singleton", "run", "transient"}


def test_scope_context_singleton_caches_instance():
    ctx = ScopeContext()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return object()

    a = ctx.get_or_create("k", factory, scope=Scope.SINGLETON)
    b = ctx.get_or_create("k", factory, scope=Scope.SINGLETON)
    assert a is b
    assert factory_calls == 1


def test_scope_context_transient_returns_new_each_time():
    ctx = ScopeContext()
    a = ctx.get_or_create("k", object, scope=Scope.TRANSIENT)
    b = ctx.get_or_create("k", object, scope=Scope.TRANSIENT)
    assert a is not b


def test_scope_context_run_scope_isolated_per_run():
    ctx = ScopeContext()
    with ctx.run_scope("run-1"):
        a = ctx.get_or_create("k", object, scope=Scope.RUN)
        b = ctx.get_or_create("k", object, scope=Scope.RUN)
    with ctx.run_scope("run-2"):
        c = ctx.get_or_create("k", object, scope=Scope.RUN)
    assert a is b
    assert a is not c


def test_scope_context_run_scope_requires_active_run():
    ctx = ScopeContext()
    with pytest.raises(LookupError, match="no active run scope"):
        ctx.get_or_create("k", object, scope=Scope.RUN)
