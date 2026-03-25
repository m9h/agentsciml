"""Meta-Architecture Parameter Golf Adapter.

Evolves the Orchestrator settings themselves to maximize (Best Score / Cost).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from .base import ProjectAdapter
from ..orchestrator import Orchestrator
from .qcccm import QCCCMAdapter

class MetaSciMLAdapter(ProjectAdapter):
    """Adapter for Meta-Architectural optimization of AgenticSciML."""

    def __init__(self, project_root: Path | None = None) -> None:
        # The project root for the meta-experiment is the agentsciml package itself
        root = project_root or Path.home() / "dev" / "agentsciml"
        super().__init__(root)

    def get_results_history(self) -> str:
        """Return results history from the target project's autoresearch output."""
        results_tsv = self.project_root / "autoresearch" / "results.tsv"
        if results_tsv.exists():
            return results_tsv.read_text()
        return ""

    def get_current_experiment(self) -> str:
        """Return the current meta-hypothesis YAML as the 'experiment'."""
        hyp = self.project_root / "autoresearch" / "workspace" / "meta_hypothesis.yaml"
        if hyp.exists():
            return hyp.read_text()
        return ""

    def get_context(self) -> str:
        return (
            "Meta-Architecture Parameter Golf: Optimize the Orchestrator settings "
            "to achieve the highest scientific yield (Quantum Advantage) at the "
            "lowest LLM cost. The 'experiment' is a configuration of the orchestrator "
            "running on the QCCCM Minority Game project."
        )

    def get_available_api(self) -> str:
        return """\
Meta-Architecture Knobs (YAML format):
    name: string
    rationale: string
    config:
        debate_rounds: int (1-10)
        max_generations: int (1-5 for meta-tests)
        budget_usd: float (1.0-5.0)
        model_overrides:
            proposer: string (claude-haiku-4-5-20251001, claude-sonnet-4-6)
            engineer: string (claude-haiku-4-5-20251001, claude-sonnet-4-6)
"""

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract Efficiency = Advantage / Cost from RESULT| lines."""
        for line in result_lines:
            if "RESULT|" in line and "efficiency=" in line:
                try:
                    return float(line.split("efficiency=")[1].split("|")[0])
                except:
                    continue
        return 0.0

    def get_metric_name(self) -> str:
        return "efficiency"

    def get_result_metric_key(self) -> str:
        return "efficiency"
