"""Profile-based default guardrail wiring.

Use this when configuring an agent's soft-policy chain. Profiles:
  - "off":      ArgsSchemaCheck only (defense in depth past LLM tool-call schema).
  - "standard": length caps + injection heuristic + schema + size caps + exfil scan.
  - "strict":   standard + PIIDetector + URLAllowlist (requires url_allowlist kwarg).

Binds a single `DefaultGuardrailChain` instance under the `IGuardrailChain` key — Agent
resolves it via `container.resolve(IGuardrailChain, default=DefaultGuardrailChain([]))`.

Design note: voussoir's Container has a single `bind(protocol, impl)` / `resolve(protocol)`
API — no `bind_many`/`resolve_all`. The chain wraps the guardrail list internally, so
one binding is all that's needed; all profile guardrails are encapsulated in the
`DefaultGuardrailChain` instance rather than as separate container bindings.
"""

from __future__ import annotations

from typing import Literal

import structlog

from voussoir.container import Container
from voussoir.guardrails.chain import DefaultGuardrailChain
from voussoir.guardrails.protocol import Guardrail, IGuardrailChain

_log = structlog.get_logger(__name__)


def bind_default_guardrails(
    c: Container,
    *,
    profile: Literal["off", "standard", "strict"] = "off",
    url_allowlist: list[str] | None = None,
) -> None:
    """Bind a `DefaultGuardrailChain` to the container under its protocol key.

    Lazy-imports the built-in guardrails so this module doesn't drag the
    full set into every import of `voussoir.guardrails`.
    """
    # Lazy imports — built-ins land in Task B3. mypy's
    # `[[tool.mypy.overrides]] module = "voussoir.guardrails.builtin.*"`
    # with `ignore_missing_imports = true` keeps strict mode green until
    # those modules exist; once they ship, the override becomes a no-op.
    from voussoir.guardrails.builtin.schema import ArgsSchemaCheck

    guardrails: list[Guardrail] = [ArgsSchemaCheck()]

    if profile in ("standard", "strict"):
        from voussoir.guardrails.builtin.injection import PromptInjectionHeuristic
        from voussoir.guardrails.builtin.length import (
            ArgsSizeCap,
            InputLengthCap,
            ToolOutputSizeCap,
        )
        from voussoir.guardrails.builtin.urls import ExfilPatternScan

        guardrails += [
            InputLengthCap(),
            PromptInjectionHeuristic(stage="input"),
            PromptInjectionHeuristic(stage="tool_output"),
            ArgsSizeCap(),
            ToolOutputSizeCap(),
            ExfilPatternScan(),
        ]

    if profile == "strict":
        from voussoir.guardrails.builtin.pii import PIIDetector
        from voussoir.guardrails.builtin.urls import URLAllowlist

        guardrails.append(PIIDetector())
        if url_allowlist:
            # Two instances: input stage guards user-supplied URLs, tool_output
            # guards URLs returned by external tools (spec §5.3 dual-stage).
            guardrails.append(URLAllowlist(allowlist=url_allowlist, stage="input"))
            guardrails.append(URLAllowlist(allowlist=url_allowlist, stage="tool_output"))
        else:
            _log.warning(
                "bind_default_guardrails strict profile: url_allowlist "
                "requires an explicit non-empty list; URLAllowlist omitted "
                "from chain",
                profile=profile,
            )

    c.bind(IGuardrailChain, DefaultGuardrailChain(guardrails))  # type: ignore[type-abstract]
