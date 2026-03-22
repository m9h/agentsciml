"""Shared test fixtures for agentsciml."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentsciml.cost import CostTracker


@pytest.fixture
def tmp_tree_path(tmp_path: Path) -> Path:
    return tmp_path / "tree.json"


@pytest.fixture
def knowledge_dir() -> Path:
    return Path(__file__).parent.parent / "knowledge"


@pytest.fixture
def cost_tracker() -> CostTracker:
    return CostTracker(budget_usd=5.0)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client that returns configurable responses."""
    client = MagicMock()

    def _make_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = text
        response.content = [content_block]
        response.usage = MagicMock()
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    client._make_response = _make_response
    # Default: return empty JSON
    client.messages.create.return_value = _make_response("{}")
    return client


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for adapter/orchestrator tests."""
    autoresearch = tmp_path / "autoresearch"
    autoresearch.mkdir()
    (autoresearch / "results.tsv").write_text(
        "commit\tmodel\ttopology\tN\tT\tmethod\tadvantage\n"
        "abc123\tSK\tcomplete\t10\t0.5\tclassical\t0.0\n"
        "abc123\tSK\tcomplete\t10\t0.5\tpimc\t0.042\n"
    )
    (autoresearch / "experiment.py").write_text(
        "from prepare import print_result, log_result\n"
        "def run_experiment():\n"
        "    print('RESULT|SK|complete|N=10|advantage=0.042')\n"
        "if __name__ == '__main__':\n"
        "    run_experiment()\n"
    )
    (autoresearch / "program.md").write_text(
        "Find parameter regimes where quantum methods outperform classical."
    )
    return tmp_path
