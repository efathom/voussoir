"""Per-model LLM pricing (USD per 1M tokens).

Use this to compute run-level cost estimates from token deltas. Prices are
snapshots and should be reviewed periodically; provider-reported costs (when a
provider surfaces them) always take precedence over this table. Keys are
matched by longest-prefix so a specific `claude-opus-4-7` beats the family
fallback `claude-opus`.
"""

from __future__ import annotations

# model prefix -> (input_usd_per_1m, output_usd_per_1m)
_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    # Anthropic (per 2026 pricing snapshots)
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "o1": (15.00, 60.00),
    "o3": (2.00, 8.00),
}

# Conservative default when the model isn't in the table.
_FALLBACK = (3.00, 15.00)


def _strip_vendor(model: str) -> str:
    """Drop a leading ``vendor/`` segment from a namespaced model id.

    OpenRouter (and several proxies) namespace ids as ``openai/gpt-4o-mini`` or
    ``anthropic/claude-sonnet-4``. Anchored prefix matching found none of them,
    so every OpenRouter run priced at ``_FALLBACK`` — including
    ``default_container()``'s own default of ``openai/gpt-4o-mini``, which
    costs $0.15/$0.60 but was billed at $3.00/$15.00, a ~24x overestimate. That
    is not just a wrong number on the result: `AgentPolicy._violation` compares
    the estimate against `max_cost_usd`, so runs were cut short with
    ``finish_reason="max_cost"`` at a fraction of real spend (audit M7).
    """
    head, sep, tail = model.partition("/")
    # Only treat it as a vendor namespace when both halves are non-empty;
    # a bare "/" or a leading slash is not a namespace.
    return tail if sep and head and tail else model


def _best_match(model: str) -> tuple[float, float]:
    """Longest-prefix match against the price table.

    Tries the id as given first, then again with any ``vendor/`` namespace
    stripped, so ``openai/gpt-4o-mini`` prices the same as ``gpt-4o-mini``.
    """
    for candidate in (model.lower(), _strip_vendor(model.lower())):
        best: tuple[float, float] | None = None
        best_len = -1
        for prefix, price in _PRICES_PER_1M.items():
            if candidate.startswith(prefix) and len(prefix) > best_len:
                best = price
                best_len = len(prefix)
        if best is not None:
            return best
    return _FALLBACK


def compute_cost(model: str | None, tokens_in: int, tokens_out: int) -> float:
    """Compute an estimated USD cost for a run.

    `model=None` uses the fallback price. The formula is
    ``(tokens_in * in_price + tokens_out * out_price) / 1_000_000``.
    """
    in_price, out_price = _best_match(model or "")
    return (tokens_in * in_price + tokens_out * out_price) / 1_000_000.0
