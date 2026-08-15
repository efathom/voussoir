"""Phase 4.5b B1 — autouse ContextVar reset between tests.

The two `test_a_*` / `test_b_*` functions form a single proof: the first
dirties the ContextVars, the second asserts they're clean at entry. Their
alphabetical names lock collection order under pytest's default collector.
If a future change adds collection randomization (e.g., pytest-randomly),
the registration check below (test_autouse_fixture_registered) remains the
definitive guarantee — the behavioral pair becomes illustrative.
"""

from __future__ import annotations

import pytest

from voussoir.agent.dispatch import last_sub_result_var, parent_ctx_var


def test_a_dirties_contextvars() -> None:
    """First half of the proof: dirty the ContextVars with sentinels."""
    parent_ctx_var.set("dirty-parent")
    last_sub_result_var.set("dirty-result")
    assert parent_ctx_var.get() == "dirty-parent"
    assert last_sub_result_var.get() == "dirty-result"


def test_b_sees_clean_contextvars() -> None:
    """Second half: if the autouse fixture works, sentinels from
    `test_a_dirties_contextvars` are cleared by the time this runs.
    """
    assert parent_ctx_var.get() is None
    assert last_sub_result_var.get() is None


def test_autouse_fixture_registered(request: pytest.FixtureRequest) -> None:
    """The autouse fixture is active in this test's scope. If autouse=True
    is ever dropped, reset_dispatch_contextvars won't appear in fixturenames
    and this test fails — order-independent guarantee.
    """
    assert "reset_dispatch_contextvars" in request.fixturenames
