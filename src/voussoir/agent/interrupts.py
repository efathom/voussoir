"""Cooperative interrupt protocol for tools that need to pause for external input.

A tool calls ``ctx.interrupt(kind, payload)`` to raise an ``InterruptRequest``,
which propagates up through ``Agent.run`` via the existing ``on_error``
middleware fanout (no special-cased agent loop handling needed). Runtime
layers — impost is the canonical consumer — catch the exception in a
``Middleware.on_error`` hook, persist the interrupt to durable storage,
and resume the agent later. On resume, a cached payload short-circuits
the second ``ctx.interrupt`` call so the agent proceeds as if nothing
happened.

In a vanilla voussoir process (no runtime catching the exception),
``InterruptRequest`` simply bubbles out of ``Agent.run`` to the caller.
No state is preserved; the caller decides what to do.

The ``IInterruptable`` Protocol exists so runtime layers can install
custom implementations via container DI (for replay) without leaking the
exception into the happy path.

``InterruptRequest`` intentionally omits the "Error" suffix — it reads
naturally as ``except InterruptRequest:`` and is a control-flow signal,
not a failure (cf. ``StopIteration``, ``GeneratorExit``). Suppressing
pep8-naming N818 for this module only — same precedent as
``voussoir.a2a.errors``'s DelegationError subclasses.
"""

# ruff: noqa: N818

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from voussoir.errors import VoussoirError


class InterruptRequest(VoussoirError):
    """Signal raised by ``ctx.interrupt(...)`` to pause the agent.

    ``kind`` is an operator-defined label (e.g. ``"manager_approval_required"``).
    ``payload`` carries typed data for the operator to act on. Both are
    surfaced unchanged through the agent loop to the runtime / caller.

    Subclass of ``VoussoirError`` so application boundaries catch it via a
    single ``except VoussoirError`` clause alongside auth, delegation, and
    policy failures.
    """

    def __init__(self, kind: str, payload: Mapping[str, Any]) -> None:
        super().__init__(f"interrupt requested: {kind}")
        self.kind = kind
        self.payload = dict(payload)


@runtime_checkable
class IInterruptable(Protocol):
    """Anything that can yield control to await external input.

    Implemented by ``ToolContext`` (default impl raises ``InterruptRequest``
    unconditionally). Runtime layers may bind a custom impl on the container
    to short-circuit via a cached payload during replay.
    """

    async def interrupt(self, kind: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Pause the agent and surface ``(kind, payload)`` to the runtime.

        Returns the operator-supplied output payload on resume. The default
        ``ToolContext.interrupt`` raises ``InterruptRequest`` unconditionally;
        replay-aware runtimes intercept the exception in middleware and
        re-enter the tool with a cached return value.
        """
        ...
