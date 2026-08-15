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


def _best_match(model: str) -> tuple[float, float]:
    """Longest-prefix match against the price table."""
    best: tuple[float, float] | None = None
    best_len = -1
    lower = model.lower()
    for prefix, price in _PRICES_PER_1M.items():
        if lower.startswith(prefix) and len(prefix) > best_len:
            best = price
            best_len = len(prefix)
    return best if best is not None else _FALLBACK


def compute_cost(model: str | None, tokens_in: int, tokens_out: int) -> float:
    """Compute an estimated USD cost for a run.

    `model=None` uses the fallback price. The formula is
    ``(tokens_in * in_price + tokens_out * out_price) / 1_000_000``.
    """
    in_price, out_price = _best_match(model or "")
    return (tokens_in * in_price + tokens_out * out_price) / 1_000_000.0
