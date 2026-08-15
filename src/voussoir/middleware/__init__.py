"""Agent-agnostic middleware namespace.

Phase 4.5a Task 1: built-in middlewares (LoggingMiddleware, RetryMiddleware,
BudgetMiddleware) moved to voussoir.agent.middleware because they reference
AgentPolicy / AgentEvent / AgentResult. This package now contains only the
Middleware Protocol — Agent-agnostic — reserved for future generic middlewares.
"""

from voussoir.middleware.protocol import Middleware

__all__ = ["Middleware"]
