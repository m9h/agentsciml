"""qcccm (quantum cognition) project adapter.

Bridges the quantum-cognition autoresearch infrastructure with AgenticSciML.
Reads prepare.py API surface, results.tsv, and program.md.
"""

from __future__ import annotations

from pathlib import Path

from .base import ProjectAdapter


class QCCCMAdapter(ProjectAdapter):
    """Adapter for the quantum-cognition (qcccm) project."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.home() / "dev" / "quantum-cognition"
        super().__init__(root)

    def get_context(self) -> str:
        """Return program.md as the research context."""
        program = self.project_root / "autoresearch" / "program.md"
        if program.exists():
            return program.read_text()
        return (
            "Find sociophysics models and parameter regimes where quantum methods "
            "outperform classical methods for finding ground states of disordered "
            "multi-agent systems. Primary metric: quantum_advantage = "
            "(E_classical - E_quantum) / |E_exact|"
        )

    def get_results_history(self) -> str:
        """Return results.tsv contents."""
        if self.results_path.exists():
            return self.results_path.read_text()
        return ""

    def get_current_experiment(self) -> str:
        """Return current experiment.py."""
        if self.experiment_path.exists():
            return self.experiment_path.read_text()
        return ""

    def get_available_api(self) -> str:
        """Return the prepare.py API surface for the Engineer."""
        return """\
Available imports from prepare.py (DO NOT MODIFY prepare.py):

from prepare import (
    ExperimentResult,       # Dataclass: commit, model, topology, disorder, n_agents,
                            #   temperature, transverse_field, frustration, seed,
                            #   method, E_best, E_exact, quantum_advantage, q_EA,
                            #   wall_time, status, magnetization, frustration_index,
                            #   binder, metadata
    compute_quantum_advantage,  # (E_classical, E_quantum, E_exact) -> float
    exact_ground_state,     # (params: SocialSpinGlassParams) -> (energy, spins)
                            #   Only for N <= 20
    get_commit_hash,        # () -> str
    log_result,             # (result: ExperimentResult) -> None  # appends to results.tsv
    print_result,           # (result: ExperimentResult) -> None  # prints RESULT| line
    run_classical,          # (params, n_steps=5000) -> (SolverResult, wall_time)
    run_pimc,               # (params, n_trotter=8, n_steps=5000) -> (SolverResult, wall_time)
    run_vqe,                # (params, n_layers=2, max_steps=200) -> (SolverResult, wall_time)
    run_qaoa,               # (params, depth=3, max_steps=200) -> (SolverResult, wall_time)
)

Available from qcccm.spin_glass.hamiltonians:
    SocialSpinGlassParams   # Dataclass: n_agents, adjacency, J, fields,
                            #   temperature, transverse_field, seed
    sk_couplings            # (N, seed) -> (adjacency, J)
    ea_couplings            # (N, topology, disorder, seed) -> (adjacency, J)
    frustration_index       # (adjacency, J) -> float

Available from qcccm.spin_glass.order_params:
    edwards_anderson_q      # (trajectory) -> float
    overlap                 # (spins1, spins2) -> float
    overlap_distribution    # (overlaps) -> array
    binder_cumulant         # (overlaps) -> float
    glass_susceptibility    # (overlaps) -> float

Experiment structure:
    - Define run_experiment() function
    - Call it in if __name__ == "__main__"
    - Each result must call print_result() AND log_result()
    - Use SocialSpinGlassParams to configure models
    - 10-minute timeout per experiment
"""

    def get_metric_name(self) -> str:
        return "quantum_advantage"

    def get_result_metric_key(self) -> str:
        return "advantage"

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract best quantum_advantage from RESULT| lines."""
        best = float("-inf")
        for line in result_lines:
            parts = line.split("|")
            for part in parts:
                if part.strip().startswith("advantage="):
                    try:
                        val = float(part.strip().split("=", 1)[1])
                        best = max(best, val)
                    except ValueError:
                        continue
        return best if best > float("-inf") else 0.0
