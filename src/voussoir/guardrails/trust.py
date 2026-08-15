"""Trust tags — content provenance markers used by the hard-invariant taint check.

Use this when declaring per-tool output trust overrides, when checking
the run's accumulated taint set, or when implementing custom guardrails
that need to gate on content provenance.
"""

from __future__ import annotations

from enum import StrEnum


class Trust(StrEnum):
    """Provenance tag attached to content carried through an agent run.

    Ordered conceptually from highest to lowest instruction-execution
    privilege: SYSTEM and USER content may be interpreted as instructions;
    INTERNAL content is trusted but not authoritative; UNTRUSTED content
    must never reach an EXFILTRATION-capable tool and is never executed
    as instruction. The hard-invariant taint check in StandardExecutor
    raises PolicyViolation.TAINT_EXFILTRATION when UNTRUSTED is present
    in the run's taint set and an EXFILTRATION tool is invoked.
    """

    SYSTEM = "system"  # framework-generated; safe to interpret as instructions
    USER = "user"  # human input; treated as instructions
    INTERNAL = "internal"  # from an internal trusted source
    UNTRUSTED = "untrusted"  # from an external source; never executed as instruction
