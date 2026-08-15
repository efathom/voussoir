"""Logging setup helpers for voussoir.

Use this when you need a structlog-bound stdlib logger from anywhere in the
framework. ``configure_logging()`` installs the structlog/stdlib bridge once
at process start; ``get_logger(name)`` returns a bound ``BoundLogger`` ready
to emit structured events.

``configure_logging(format="json")`` is appropriate for production; ``"dev"``
is human-readable for local work. Always pass keyword args to log calls — the
JSON renderer requires them to produce structured payloads.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal, cast

import structlog


def _default_format() -> Literal["dev", "json"]:
    """Production-safe log format from env.

    Honors VOUSSOIR_LOG_FORMAT=json|dev (default dev) and
    VOUSSOIR_LOG_LEVEL (default INFO). JSON is recommended for production so
    structured events survive log aggregation; `dev` is human-readable.
    """
    fmt = os.environ.get("VOUSSOIR_LOG_FORMAT", "dev").strip().lower()
    return "json" if fmt == "json" else "dev"


def _default_level() -> str:
    return os.environ.get("VOUSSOIR_LOG_LEVEL", "INFO").strip() or "INFO"


def configure_logging(
    level: str | None = None,
    format: Literal["dev", "json"] | None = None,
) -> None:
    # Defaults resolve from env (VOUSSOIR_LOG_FORMAT / VOUSSOIR_LOG_LEVEL) so
    # `default_container()` no longer hard-codes the dev format — operators can
    # switch to JSON logs without code changes.
    level = level or _default_level()
    format = format or _default_format()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
