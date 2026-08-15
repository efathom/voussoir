"""Phase 4.5a — programmatic layering tests.

observability and middleware are LOWER layers than agent; they must not
import from voussoir.agent.* at *runtime*. This test parses every Python
file in those two packages and asserts no such imports exist outside a
``TYPE_CHECKING`` guard. Prevents regression of P0 #9 from the post-Phase-4d
review.

v1.1.0 F12: TYPE_CHECKING-guarded imports are allowed — they create no
runtime dependency and are the standard pattern for typing lower layers
against higher-layer types.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "voussoir"


def _is_type_checking_block(node: ast.If) -> bool:
    """Return True if *node* is ``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:``."""
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def _collect_type_checking_nodes(tree: ast.Module) -> set[int]:
    """Return the ids of all AST nodes nested inside TYPE_CHECKING blocks."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_block(node):
            for child in ast.walk(node):
                guarded.add(id(child))
    return guarded


def _agent_imports_in(path: Path) -> list[str]:
    """Return every voussoir.agent.* module imported from `path` at runtime.

    Imports that appear only inside ``if TYPE_CHECKING:`` blocks are excluded
    because they create no runtime dependency.
    """
    tree = ast.parse(path.read_text())
    guarded = _collect_type_checking_nodes(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("voussoir.agent"):
                out.append(node.module)
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("voussoir.agent"):
                    out.append(n.name)
    return out


def test_observability_does_not_import_agent() -> None:
    root = _SRC_ROOT / "observability"
    for py in root.rglob("*.py"):
        bad = _agent_imports_in(py)
        assert not bad, (
            f"{py.relative_to(_SRC_ROOT)} imports {bad}; "
            "observability must not depend on agent (Phase 4.5a P0 #9)."
        )


def test_middleware_does_not_import_agent() -> None:
    root = _SRC_ROOT / "middleware"
    for py in root.rglob("*.py"):
        bad = _agent_imports_in(py)
        assert not bad, (
            f"{py.relative_to(_SRC_ROOT)} imports {bad}; "
            "middleware must not depend on agent (Phase 4.5a P0 #9)."
        )
