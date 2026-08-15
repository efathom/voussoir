"""Locks voussoir.authz.decisions metric handle existence (Phase 6 A6)."""

from __future__ import annotations


def test_authz_decisions_handle_present() -> None:
    from voussoir.observability import metrics

    assert hasattr(metrics, "AUTHZ_DECISIONS")
    # Confirm it has a .add(...) method (OTel Counter API)
    assert callable(getattr(metrics.AUTHZ_DECISIONS, "add", None))


def test_metrics_module_defines_10_handles() -> None:
    """All 10 spec-required metrics are exported from voussoir.observability.metrics."""
    from voussoir.observability import metrics

    for name in (
        "TOKENS_IN",
        "TOKENS_OUT",
        "COST_USD",
        "DURATION_MS",
        "TOOL_CALLS",
        "GUARDRAIL_DECISIONS",
        "CAPABILITY_DENIALS",
        "TAINT_EXFIL_BLOCKS",
        "CASCADE_ESCALATIONS",
        "AUTHZ_DECISIONS",
    ):
        assert hasattr(metrics, name), f"missing metric: {name}"
