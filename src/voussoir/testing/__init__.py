"""voussoir.testing — public test helpers for downstream library authors.

Use these when unit-testing voussoir-ecosystem code (custom guardrails,
brokers, executors, A2A peers). Mirrors the conftest fixtures in
voussoir's own test suite but exposed as plain callables (no pytest
dependency at import time).

Install via ``pip install voussoir[testing]`` for the recommended
pytest + pytest-asyncio toolchain; the helpers themselves only depend
on unittest.mock.
"""

from __future__ import annotations

from voussoir.testing.containers import make_container, make_runtime_container
from voussoir.testing.keys import make_key_provider
from voussoir.testing.llm import multi_turn_llm, stub_llm

__all__ = [
    "make_container",
    "make_key_provider",
    "make_runtime_container",
    "multi_turn_llm",
    "stub_llm",
]
