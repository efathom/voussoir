"""voussoir — the wedge stone in your agent arch.

A Python LLM agent framework: single-agent first, hierarchical delegation,
typed-by-default, MCP/A2A native, deterministic-first guardrails.
"""

from __future__ import annotations

from voussoir.a2a.agent_ref import AgentRef
from voussoir.a2a.wire import WireAgentResult
from voussoir.agent import Agent, AgentResult
from voussoir.agent.interrupts import IInterruptable, InterruptRequest
from voussoir.agent.policy import PolicyViolation, PolicyViolationError
from voussoir.auth.errors import AuthError
from voussoir.config import AgentConfig, VoussoirConfig, load_voussoir_config
from voussoir.container import Container, Scope
from voussoir.container.defaults import default_container
from voussoir.errors import VoussoirError
from voussoir.executors import IToolExecutor, StandardExecutor
from voussoir.guardrails import IGuardrailChain

__version__ = "1.3.0"

# make_a2a_router and serve_a2a are lazily resolved via __getattr__ so that
# importing voussoir (or any voussoir.* submodule) does NOT eagerly pull in
# fastapi/uvicorn. The `a2a` extra installs those deps; users who don't install
# it and never call make_a2a_router/serve_a2a pay no import cost.
_LAZY_A2A_ATTRS = frozenset({"make_a2a_router", "serve_a2a"})


def __getattr__(name: str) -> object:
    if name in _LAZY_A2A_ATTRS:
        from voussoir.a2a.publisher import make_a2a_router, serve_a2a  # noqa: PLC0415

        globals()["make_a2a_router"] = make_a2a_router
        globals()["serve_a2a"] = serve_a2a
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentRef",
    "AgentResult",
    "AuthError",
    "Container",
    "IGuardrailChain",
    "IInterruptable",
    "IToolExecutor",
    "InterruptRequest",
    "PolicyViolation",
    "PolicyViolationError",
    "Scope",
    "StandardExecutor",
    "VoussoirConfig",
    "VoussoirError",
    "WireAgentResult",
    "__version__",
    "default_container",
    "load_voussoir_config",
    "make_a2a_router",
    "serve_a2a",
]
