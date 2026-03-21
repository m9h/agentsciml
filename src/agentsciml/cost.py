"""Cost tracking and model routing for swarm-optimized agent calls.

Applies the swarm learning insight: use cheap models (Haiku) for ~80% of calls,
escalate to expensive models (Sonnet/Opus) only when needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pricing per million tokens (as of 2026-03)
_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_M, output_per_M)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
}

# Fallback for unknown models
_DEFAULT_PRICING = (3.00, 15.00)


@dataclass
class CostTracker:
    """Track token usage and estimated costs across agent calls."""

    budget_usd: float = 5.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls_by_model: dict[str, int] = field(default_factory=dict)
    tokens_by_model: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record a single API call's token usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

        prev_in, prev_out = self.tokens_by_model.get(model, (0, 0))
        self.tokens_by_model[model] = (prev_in + input_tokens, prev_out + output_tokens)

        logger.debug(
            "Cost: +%d/%d tokens (%s), total $%.4f",
            input_tokens,
            output_tokens,
            model,
            self.estimated_cost_usd,
        )

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate total cost from token usage."""
        total = 0.0
        for model, (inp, out) in self.tokens_by_model.items():
            in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
            total += (inp / 1_000_000) * in_price + (out / 1_000_000) * out_price
        return total

    @property
    def budget_remaining(self) -> float:
        return self.budget_usd - self.estimated_cost_usd

    @property
    def within_budget(self) -> bool:
        return self.budget_remaining > 0

    @property
    def total_calls(self) -> int:
        return sum(self.calls_by_model.values())

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "budget_remaining_usd": round(self.budget_remaining, 4),
            "calls_by_model": dict(self.calls_by_model),
        }
