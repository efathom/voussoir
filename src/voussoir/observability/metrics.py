"""Voussoir metric definitions — counters + histograms emitted at span-close.

Use this when adding a new metric; centralized so backend dashboards stay
stable. Imports cheap (single Meter instance); module-level constants.
"""

from __future__ import annotations

from opentelemetry import metrics

from voussoir.observability.tracer import PKG_VERSION

_meter = metrics.get_meter("voussoir", PKG_VERSION)

TOKENS_IN = _meter.create_counter(
    "voussoir.tokens.in",
    description="LLM input tokens consumed per agent run.",
)
TOKENS_OUT = _meter.create_counter(
    "voussoir.tokens.out",
    description="LLM output tokens generated per agent run.",
)
COST_USD = _meter.create_counter(
    "voussoir.cost_usd",
    description="Per-run LLM cost in USD (provider-reported).",
)
DURATION_MS = _meter.create_histogram(
    "voussoir.duration_ms",
    description="End-to-end duration of an Agent.run() in milliseconds.",
)
TOOL_CALLS = _meter.create_counter(
    "voussoir.tool_calls.count",
    description="Tool invocations dispatched per agent run.",
)
GUARDRAIL_DECISIONS = _meter.create_counter(
    "voussoir.guardrail.decisions",
    description="Guardrail chain verdicts emitted; counts by stage + verdict + guardrail name.",
)
CAPABILITY_DENIALS = _meter.create_counter(
    "voussoir.capability.denials",
    description="Hard-invariant capability-mask denials raised by StandardExecutor.",
)
TAINT_EXFIL_BLOCKS = _meter.create_counter(
    "voussoir.taint.exfil_blocks",
    description="EXFILTRATION tool calls blocked because UNTRUSTED is in the run's taint set.",
)
CASCADE_ESCALATIONS = _meter.create_counter(
    "voussoir.cascade.escalations",
    description="Cascade escalations (verdict != PASS) per agent run.",
)
AUTHZ_DECISIONS = _meter.create_counter(
    "voussoir.authz.decisions",
    description="Authorizer decisions emitted during tool dispatch; counts by decision + authorizer_name + tool_name.",
)
