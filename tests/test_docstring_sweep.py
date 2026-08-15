"""Phase 4.5b B6 — every public symbol has a non-empty docstring.

Walks each public package's __all__ and asserts each name's __doc__ is
non-empty. For classes, also asserts the class itself has a docstring
(not just the methods).

Why a sweep test: docstrings rot silently when code moves around. This
test fails CI if a future addition to __all__ forgets a docstring, so
the contract is enforced going forward — not just at this tranche's
boundary.
"""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "voussoir",
    "voussoir.agent",
    "voussoir.a2a",
    "voussoir.observability",
    "voussoir.middleware",
]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_symbols_have_docstrings(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    public = getattr(mod, "__all__", None)
    if public is None:
        pytest.skip(f"{module_name} has no __all__")
    missing: list[str] = []
    for name in public:
        sym = getattr(mod, name, None)
        if sym is None:
            missing.append(f"{module_name}.{name} (not resolvable)")
            continue
        doc = getattr(sym, "__doc__", None)
        if not doc or not doc.strip():
            missing.append(f"{module_name}.{name}")
    assert not missing, (
        f"Public symbols missing docstrings: {missing}. "
        "Every name in __all__ should have at least a one-line docstring "
        "explaining when to use it."
    )
