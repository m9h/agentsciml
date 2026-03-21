"""Abstract project adapter interface.

Each target project (qcccm, alf, jaxctrl, setae) implements this interface
to bridge its autoresearch infrastructure with the AgenticSciML orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ProjectAdapter(ABC):
    """Interface between AgenticSciML and a specific scientific project."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    @abstractmethod
    def get_context(self) -> str:
        """Return project description and research goals for agent prompts."""

    @abstractmethod
    def get_results_history(self) -> str:
        """Return contents of results.tsv (or equivalent) for DataAnalyst."""

    @abstractmethod
    def get_current_experiment(self) -> str:
        """Return current experiment.py content."""

    @abstractmethod
    def get_available_api(self) -> str:
        """Return the prepare.py API surface the Engineer must use."""

    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the primary metric (e.g. 'quantum_advantage')."""

    @abstractmethod
    def get_result_metric_key(self) -> str:
        """Return the key used in RESULT| lines for the primary metric."""

    @abstractmethod
    def parse_score(self, result_lines: list[str]) -> float:
        """Extract primary metric score from RESULT| lines."""

    @property
    def experiment_path(self) -> Path:
        """Path to the mutable experiment file."""
        return self.project_root / "autoresearch" / "experiment.py"

    @property
    def results_path(self) -> Path:
        """Path to the results TSV file."""
        return self.project_root / "autoresearch" / "results.tsv"
