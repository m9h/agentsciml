"""Tests for cost tracking."""

from agentsciml.cost import CostTracker


def test_cost_tracking():
    tracker = CostTracker(budget_usd=1.0)

    # Simulate a haiku call: 1000 input, 500 output tokens
    tracker.add("claude-haiku-4-5-20251001", input_tokens=1000, output_tokens=500)

    assert tracker.total_calls == 1
    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 500

    # Haiku: $1/M input, $5/M output
    expected = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 5.0
    assert abs(tracker.estimated_cost_usd - expected) < 1e-10


def test_budget_check():
    tracker = CostTracker(budget_usd=0.001)

    # This call costs more than the budget
    tracker.add("claude-sonnet-4-6", input_tokens=100_000, output_tokens=10_000)
    assert not tracker.within_budget


def test_multiple_models():
    tracker = CostTracker(budget_usd=10.0)

    tracker.add("claude-haiku-4-5-20251001", input_tokens=5000, output_tokens=2000)
    tracker.add("claude-sonnet-4-6", input_tokens=3000, output_tokens=1000)

    assert tracker.total_calls == 2
    assert "claude-haiku-4-5-20251001" in tracker.calls_by_model
    assert "claude-sonnet-4-6" in tracker.calls_by_model

    summary = tracker.summary()
    assert summary["total_calls"] == 2
    assert summary["estimated_cost_usd"] > 0


def test_empty_tracker():
    tracker = CostTracker(budget_usd=5.0)
    assert tracker.estimated_cost_usd == 0.0
    assert tracker.within_budget
    assert tracker.total_calls == 0
