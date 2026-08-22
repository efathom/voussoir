from typing import Protocol

import pytest

from voussoir.container import Container, Scope


class Greeter(Protocol):
    def greet(self) -> str: ...


class Hello:
    def greet(self) -> str:
        return "hello"


class Howdy:
    def greet(self) -> str:
        return "howdy"


def test_bind_and_resolve():
    c = Container()
    c.bind(Greeter, Hello())
    assert c.resolve(Greeter).greet() == "hello"


def test_resolve_unbound_raises():
    c = Container()
    with pytest.raises(LookupError, match="No binding registered"):
        c.resolve(Greeter)


def test_default_scope_is_singleton():
    c = Container()
    c.bind(Greeter, lambda: Hello())
    a = c.resolve(Greeter)
    b = c.resolve(Greeter)
    assert a is b


def test_transient_scope_returns_new_each_time():
    c = Container()
    c.bind(Greeter, lambda: Hello(), scope=Scope.TRANSIENT)
    assert c.resolve(Greeter) is not c.resolve(Greeter)


def test_run_scope_cached_per_run():
    c = Container()
    c.bind(Greeter, lambda: Hello(), scope=Scope.RUN)
    with c.run_scope("r1"):
        a = c.resolve(Greeter)
        b = c.resolve(Greeter)
    with c.run_scope("r2"):
        d = c.resolve(Greeter)
    assert a is b
    assert a is not d


def test_bind_with_default_param_overrides_at_resolve():
    c = Container()
    assert c.resolve(Greeter, default=Hello()).greet() == "hello"


def test_resolve_default_none_returns_none_for_optional_dependencies():
    """`default=None` is "return None when unbound", not "raise".

    LLMGuardrailJudge.screen and LLMJudge.validate both do
    `resolve(ITelemetrySink, default=None)` and then guard with
    `if sink is not None:`. While the sentinel check also treated an explicit
    None as "no default", both judges raised LookupError mid-validation on any
    container that didn't bind a telemetry sink (audit H2).
    """
    c = Container()
    assert c.resolve(Greeter, default=None) is None


def test_rebinding_evicts_the_previously_cached_singleton():
    """A rebind after a resolve must take effect, not be masked by the cache."""
    c = Container()
    c.bind(Greeter, Hello())
    assert c.resolve(Greeter).greet() == "hello"

    class Goodbye:
        def greet(self) -> str:
            return "goodbye"

    c.bind(Greeter, Goodbye())
    assert c.resolve(Greeter).greet() == "goodbye"


def test_resolve_missing_with_helpful_message():
    c = Container()
    c.bind(Greeter, Hello(), name="english")
    with pytest.raises(LookupError) as exc:
        c.resolve(Greeter)
    assert "english" in str(exc.value)  # mentions known names


def test_list_bindings_returns_registered_protocols():
    c = Container()
    c.bind(Greeter, Hello(), name="english")
    c.bind(Greeter, Howdy(), name="texan")
    bindings = c.list_bindings()
    assert (Greeter, "english") in bindings
    assert (Greeter, "texan") in bindings


def test_child_container_inherits_singletons_but_has_own_run_scope():
    parent = Container()
    parent.bind(Greeter, Hello())  # singleton
    parent.bind(Howdy, lambda: Howdy(), scope=Scope.RUN)

    child = parent.child()
    assert child.resolve(Greeter) is parent.resolve(Greeter)  # shared singleton

    with parent.run_scope("p"):  # noqa: SIM117 — nesting is the test
        with child.run_scope("c"):
            child_h = child.resolve(Howdy)
            assert child_h is not None
        # parent run scope is still alive but child's is gone
