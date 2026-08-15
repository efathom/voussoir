"""Shared fixtures for A2A tests.

`fresh_env` is defined in the top-level tests/conftest.py so cross-suite
exit tests can use it too.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def make_key_provider() -> Callable[..., Any]:
    """Thin wrapper around voussoir.testing.make_key_provider (v1.1.0 F11).

    Returns the public helper callable so tests calling
    ``make_key_provider(jwt_secret=b"...")`` continue to work.

    Note: the public helper defaults jwt_secret to None (ephemeral) whereas
    the old fixture defaulted to b"test-secret-32bytes-long-padded!".
    No a2a tests currently call this fixture so the default change is safe.
    """
    from voussoir.testing import make_key_provider as helper

    return helper
