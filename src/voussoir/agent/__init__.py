"""voussoir.agent — Core Agent abstraction, run loop, result types, and cascade gates.

Public API:
  Agent              — the primary entrypoint; typed generic over (In, Out)
  AgentBuilder       — fluent builder alternative to the Agent constructor
  AgentContext       — per-run state bag (principal, taint, trace, steps)
  AgentPolicy        — stop conditions (max_steps, budgets)
  AgentResult        — structured run output (steps, decisions, tokens, cost)
  AgentEvent         — streaming event emitted by Agent.stream()
  FinishReason       — Literal alias for AgentResult.finish_reason (8 variants)
  Step               — one reasoning step (LLM turn, tool call, or delegation)
  GuardrailDecision  — audit record of a guardrail verdict that fired
  CascadeOutcome     — audit record of a cascade escalation
  RequestCascade     — policy for SAS-first + MAS escalation
  Decision           — PASS / FAIL / AMBIGUOUS return type for Validators
  Validator          — Protocol for cascade gate logic
  ToolUseFaithfulness — built-in verifier: checks output is grounded in tool results
  AmbiguousFallback  — wraps a Validator to resolve AMBIGUOUS via LLMJudge
  LLMJudge           — LLM-backed binary verdict for ambiguous validation cases
  IDelegate          — Protocol common to Agent and AgentRef (local and remote peers)
  IInterruptable     — Protocol for cooperative pause/resume (ctx.interrupt)
  InterruptRequest   — exception raised by ctx.interrupt(...) for runtime to catch
  NamedDelegate      — delegate reference by registry name
  AgentRegistry      — registry of named agents, loaded from config or code
  PolicyViolation    — StrEnum of violation codes used in PolicyViolationError
  PolicyViolationError — raised on hard invariant violations (capability, taint, authz)
  bind_agent_registry — helper to register agents from VoussoirConfig into the container
  register_agent     — register a single agent in the registry
  load_delegate_plugins — discover IDelegate implementations via entry points
  BudgetMiddleware   — enforces token/cost/time budgets from AgentPolicy
  LoggingMiddleware  — structured logging on every run event
  RetryMiddleware    — exponential backoff on transient LLM/tool errors
  ENTRY_POINT_GROUP  — entry-point group name for third-party IDelegate plugins
  DEFAULT_LLM_JUDGE_PROMPT — default system prompt for LLMJudge
"""

from voussoir.agent.agent import Agent
from voussoir.agent.agent_builder import AgentBuilder
from voussoir.agent.bootstrap import bind_agent_registry
from voussoir.agent.cascade import Decision, RequestCascade, Validator
from voussoir.agent.context import AgentContext
from voussoir.agent.delegate import IDelegate, NamedDelegate
from voussoir.agent.interrupts import IInterruptable, InterruptRequest
from voussoir.agent.middleware import BudgetMiddleware, LoggingMiddleware, RetryMiddleware
from voussoir.agent.plugins import ENTRY_POINT_GROUP, load_delegate_plugins
from voussoir.agent.policy import AgentPolicy, PolicyViolation, PolicyViolationError
from voussoir.agent.registry import AgentRegistry, register_agent
from voussoir.agent.result import (
    AgentEvent,
    AgentResult,
    CascadeOutcome,
    FinishReason,
    GuardrailDecision,
    Step,
)
from voussoir.agent.validators import (
    DEFAULT_LLM_JUDGE_PROMPT,
    AmbiguousFallback,
    LLMJudge,
    ToolUseFaithfulness,
)

__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentContext",
    "AgentEvent",
    "AgentPolicy",
    "AgentRegistry",
    "AgentResult",
    "AmbiguousFallback",
    "BudgetMiddleware",
    "CascadeOutcome",
    "DEFAULT_LLM_JUDGE_PROMPT",
    "ENTRY_POINT_GROUP",
    "Decision",
    "FinishReason",
    "GuardrailDecision",
    "IDelegate",
    "IInterruptable",
    "InterruptRequest",
    "LLMJudge",
    "LoggingMiddleware",
    "NamedDelegate",
    "PolicyViolation",
    "PolicyViolationError",
    "RequestCascade",
    "RetryMiddleware",
    "Step",
    "ToolUseFaithfulness",
    "Validator",
    "bind_agent_registry",
    "load_delegate_plugins",
    "register_agent",
]
