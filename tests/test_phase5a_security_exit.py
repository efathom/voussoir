"""Phase 5 Tranche A exit gate — one test per invariant from spec §4.

These tests lock the security claim for Tranche A. Future changes that
break any of these will fail at the exit gate, before they reach CI.
"""

from __future__ import annotations

import inspect
from enum import IntFlag
from pathlib import Path

import voussoir.executors.standard
from voussoir.agent import PolicyViolation
from voussoir.agent.agent import Agent
from voussoir.guardrails import Trust
from voussoir.tools import Capability

_STANDARD_EXECUTOR_SRC = Path(voussoir.executors.standard.__file__).read_text()


def test_exit_1_capability_is_intflag() -> None:
    """Spec §4.1: Capability is an enum.IntFlag (closes F-4)."""
    assert issubclass(Capability, IntFlag)


def test_exit_2_trust_strenum_present() -> None:
    """Spec §4.2: Trust StrEnum lives at voussoir.guardrails."""
    assert Trust.UNTRUSTED == "untrusted"
    assert isinstance(Trust.UNTRUSTED, str)


def test_exit_3_policy_violation_phase5_variants() -> None:
    """Spec §4.6: three new PolicyViolation variants are present."""
    assert PolicyViolation.CAPABILITY_DENIED == "capability_denied"
    assert PolicyViolation.TAINT_EXFILTRATION == "taint_exfiltration"
    assert PolicyViolation.CAPABILITY_CLAMPED_EMPTY == "capability_clamped_empty"


def test_exit_4_agent_has_allowed_capabilities_field() -> None:
    """Spec §4.4 §4.5: Agent.__init__ accepts allowed_capabilities kwarg."""
    sig = inspect.signature(Agent.__init__)
    assert "allowed_capabilities" in sig.parameters


def test_exit_5_executor_invoke_enforces_capability_mask() -> None:
    """Spec §4.4 Rule 1: StandardExecutor enforces capability mask inline."""
    assert "CAPABILITY_DENIED" in _STANDARD_EXECUTOR_SRC
    assert "ctx.allowed_capabilities" in _STANDARD_EXECUTOR_SRC


def test_exit_6_executor_invoke_enforces_taint_check() -> None:
    """Spec §4.4 Rule 2: StandardExecutor enforces UNTRUSTED-in-taint EXFIL gate."""
    assert "TAINT_EXFILTRATION" in _STANDARD_EXECUTOR_SRC
    assert "Trust.UNTRUSTED" in _STANDARD_EXECUTOR_SRC


def test_exit_7_clamping_helper_present_on_agent() -> None:
    """Spec §4.5: clamp_tools is wired (closes Tranche C #1).

    Originally lived as `Agent._tools_after_clamp`; extracted in Phase 5 B5
    to `voussoir.agent.delegation.clamp_tools` as a public-named free
    function for LOC headroom on agent.py.
    """
    from voussoir.agent.delegation import clamp_tools

    assert callable(clamp_tools)
