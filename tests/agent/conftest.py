"""Agent-suite conftest.

Phase 3.5 lifted `make_container` and `stub_llm` to the top-level
`tests/conftest.py` so cross-suite tests (e.g. tests/test_phase35_exit.py)
can use them. This file is intentionally tiny; the fixtures are inherited.
"""

from __future__ import annotations
