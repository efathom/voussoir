"""Voussoir root-level exception base.

`VoussoirError` is the shared ancestor of all voussoir-domain exceptions —
catch this at application boundaries to handle any voussoir-side failure
uniformly. Domain-specific subclasses (AuthError, DelegationError,
PolicyViolationError, etc.) provide finer-grained discrimination.
"""

from __future__ import annotations


class VoussoirError(Exception):
    """Base for all voussoir-domain exceptions.

    Use this when you need a single `except` clause that covers any voussoir
    failure (auth, delegation, policy, A2A, etc.). For more specific error
    types, catch the dedicated subclasses listed in their respective
    modules — voussoir.auth.errors, voussoir.a2a.errors, voussoir.agent.policy.
    """
