"""Locks D9 perf caches: _wants_ctx + describe() (v1.0.2 D9).

Two hot-path perf wins:

(a) `_DecoratedTool.invoke` used to call `inspect.signature(self._fn)` on every
    invocation; now the `ctx`-wanted flag is computed once at decoration time.
(b) `ToolRegistry.describe` used to rebuild every tool's JSON schema on every
    call (per LLM turn); now the list is cached and invalidated on register().

(c) `ToolContext` default_factory=list was investigated and the per-construction
    cost is ~2.3μs — below the 0.05ms threshold, so no change was made there.
"""

from __future__ import annotations

import asyncio
import inspect

from voussoir.tools import Capability, ToolContext, tool
from voussoir.tools.registry import ToolRegistry

# ------- (a) _wants_ctx cached at decoration time -------


def test_decorated_tool_caches_wants_ctx() -> None:
    @tool(capability=Capability.READ_PUBLIC, name="with_ctx")
    async def with_ctx(ctx: ToolContext, x: int) -> int:
        return x

    @tool(capability=Capability.READ_PUBLIC, name="no_ctx")
    async def no_ctx(x: int) -> int:
        return x

    # The cached flag must exist on the decorated tool.
    assert getattr(with_ctx, "_wants_ctx", None) is True
    assert getattr(no_ctx, "_wants_ctx", None) is False


def test_invoke_does_not_call_inspect_signature_per_call(monkeypatch) -> None:
    """_DecoratedTool.invoke must not call inspect.signature on every invoke."""

    @tool(capability=Capability.READ_PUBLIC, name="t")
    async def t(x: int) -> int:
        return x

    call_count = {"n": 0}
    real_signature = inspect.signature

    def _spy(fn, *a, **kw):
        call_count["n"] += 1
        return real_signature(fn, *a, **kw)

    # Patch on the module decorator.py uses (it imported inspect itself).
    import voussoir.tools.decorator as dec

    monkeypatch.setattr(dec.inspect, "signature", _spy)

    args = t.input_schema(x=1)
    ctx = ToolContext(run_id="r", span_id="s", allowed_capabilities=Capability.READ_PUBLIC)
    asyncio.run(t.invoke(args, ctx))
    asyncio.run(t.invoke(args, ctx))
    asyncio.run(t.invoke(args, ctx))
    # The cached flag means inspect.signature is NEVER called during invoke.
    assert call_count["n"] == 0


# ------- (b) describe() caches; invalidates on register -------


def test_registry_describe_caches() -> None:
    """Repeated describe() calls return identical content (cache hit)."""

    @tool(capability=Capability.READ_PUBLIC, name="x")
    async def x() -> str:
        return "ok"

    reg = ToolRegistry()
    reg.register(x)
    d1 = reg.describe()
    d2 = reg.describe()
    assert d1 == d2


def test_registry_describe_returns_same_object_when_unchanged() -> None:
    """describe() called many times without register() returns the SAME list.

    Identity equality proves no rebuild happened between calls.
    """

    @tool(capability=Capability.READ_PUBLIC, name="t1")
    async def t1() -> str:
        return "ok"

    reg = ToolRegistry()
    reg.register(t1)
    d1 = reg.describe()
    d2 = reg.describe()
    d3 = reg.describe()
    assert d1 is d2 is d3


def test_registry_describe_invalidates_on_register() -> None:
    """Adding a new tool invalidates the cache; next describe reflects it."""

    @tool(capability=Capability.READ_PUBLIC, name="a")
    async def a() -> str:
        return "ok"

    @tool(capability=Capability.READ_PUBLIC, name="b")
    async def b() -> str:
        return "ok"

    reg = ToolRegistry()
    reg.register(a)
    d1 = reg.describe()
    assert len(d1) == 1
    assert d1[0]["name"] == "a"

    reg.register(b)
    d2 = reg.describe()
    assert len(d2) == 2
    assert {t["name"] for t in d2} == {"a", "b"}
    # Cache rebuilt — not the same list.
    assert d1 is not d2


def test_registry_describe_invalidates_on_each_register() -> None:
    """Cache must invalidate on EVERY register, not just the first."""

    @tool(capability=Capability.READ_PUBLIC, name="t_a")
    async def t_a() -> str:
        return "ok"

    @tool(capability=Capability.READ_PUBLIC, name="t_b")
    async def t_b() -> str:
        return "ok"

    @tool(capability=Capability.READ_PUBLIC, name="t_c")
    async def t_c() -> str:
        return "ok"

    reg = ToolRegistry()
    reg.register(t_a)
    reg.describe()  # warm
    reg.register(t_b)
    d2 = reg.describe()
    assert len(d2) == 2
    reg.register(t_c)
    d3 = reg.describe()
    assert len(d3) == 3
    assert {t["name"] for t in d3} == {"t_a", "t_b", "t_c"}


def test_registry_describe_register_many_invalidates() -> None:
    """register_many must also leave the cache invalidated."""

    @tool(capability=Capability.READ_PUBLIC, name="m_a")
    async def m_a() -> str:
        return "ok"

    @tool(capability=Capability.READ_PUBLIC, name="m_b")
    async def m_b() -> str:
        return "ok"

    reg = ToolRegistry()
    reg.describe()  # warm with empty
    reg.register_many([m_a, m_b])
    d = reg.describe()
    assert {t["name"] for t in d} == {"m_a", "m_b"}


def test_registry_describe_cache_isolated_per_instance() -> None:
    """Each ToolRegistry has its own cache — no global state."""

    @tool(capability=Capability.READ_PUBLIC, name="iso_a")
    async def iso_a() -> str:
        return "ok"

    @tool(capability=Capability.READ_PUBLIC, name="iso_b")
    async def iso_b() -> str:
        return "ok"

    r1 = ToolRegistry()
    r2 = ToolRegistry()
    r1.register(iso_a)
    r2.register(iso_b)
    assert {t["name"] for t in r1.describe()} == {"iso_a"}
    assert {t["name"] for t in r2.describe()} == {"iso_b"}
