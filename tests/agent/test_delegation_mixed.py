"""check_delegate_collisions accepts pre-extracted name strings.

Locks the contract Task 9 depends on: Agent.__init__ can pass a list
of names regardless of whether each delegate entry was an Agent or a
string — the helper only ever sees the name list.
"""

from __future__ import annotations

import pytest

from voussoir.agent.delegation import check_delegate_collisions


def test_distinct_names_pass() -> None:
    check_delegate_collisions(["foo", "bar"])  # no raise


def test_duplicate_names_raise() -> None:
    with pytest.raises(ValueError, match="collision"):
        check_delegate_collisions(["foo", "foo"])


def test_distinct_names_sanitizing_to_same_collide() -> None:
    """Casing/whitespace difference can collapse via sanitize_for_tool_name."""
    with pytest.raises(ValueError, match="collision"):
        check_delegate_collisions(["Foo", "foo"])
