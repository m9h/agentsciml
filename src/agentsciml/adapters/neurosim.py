"""Neurosim (bl1) cortical culture simulation adapter.

Bridges the bl1 DishBrain-inspired spiking neural network simulator
with AgenticSciML for automated discovery of plasticity rules,
network topologies, and stimulation protocols.
"""

from __future__ import annotations

from pathlib import Path

from .base import ProjectAdapter


class NeurosimAdapter(ProjectAdapter):
    """Adapter for the bl1 cortical culture simulator."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.home() / "dev" / "bl1"
        super().__init__(root)

    def get_context(self) -> str:
        program = self.project_root / "autoresearch" / "program.md"
        if program.exists():
            return program.read_text()
        return (
            "Discover plasticity rules, network topologies, and stimulation "
            "protocols that maximize learning speed in simulated cortical "
            "cultures. The bl1 simulator models dissociated cortical cultures "
            "on multi-electrode arrays (DishBrain-inspired) using JAX-based "
            "spiking neural networks with STDP. Primary metric: learning_speed "
            "(inverse of episodes to criterion performance on a game task). "
            "Secondary metrics: information_transfer (bits/s across MEA), "
            "firing_regularity (CV of ISI), biological_plausibility (constraint)."
        )

    def get_results_history(self) -> str:
        if self.results_path.exists():
            return self.results_path.read_text()
        return ""

    def get_current_experiment(self) -> str:
        if self.experiment_path.exists():
            return self.experiment_path.read_text()
        return ""

    def get_available_api(self) -> str:
        return """\
Available imports from prepare.py (DO NOT MODIFY prepare.py):

from prepare import (
    ExperimentResult,       # Dataclass: commit, topology, n_neurons, stdp_rule,
                            #   learning_rate, tau_plus, tau_minus, a_plus, a_minus,
                            #   connectivity, stim_protocol, game_task,
                            #   episodes_to_criterion, learning_speed,
                            #   information_transfer, firing_rate_mean,
                            #   firing_rate_cv, wall_time, status, metadata
    create_network,         # (n_neurons, topology, connectivity, seed) -> Network
                            #   topology: 'random', 'small_world', 'scale_free',
                            #   'distance_dependent', 'cortical_column'
    configure_stdp,         # (rule, tau_plus, tau_minus, a_plus, a_minus,
                            #   **kwargs) -> STDPConfig
                            #   rule: 'pair', 'triplet', 'voltage_dependent',
                            #   'reward_modulated'
    configure_stimulation,  # (protocol, electrodes, **kwargs) -> StimConfig
                            #   protocol: 'random', 'patterned', 'closed_loop',
                            #   'reward_modulated', 'activity_dependent'
    run_culture,            # (network, stdp, stim, game_task, max_episodes=500,
                            #   seed=0) -> (CultureResult, wall_time)
    compute_learning_speed, # (episodes_to_criterion, max_episodes) -> float
    compute_info_transfer,  # (mea_recordings) -> float  # bits/s
    log_result,             # (result: ExperimentResult) -> None
    print_result,           # (result: ExperimentResult) -> None  # RESULT| line
)

Experiment structure:
    - Define run_experiment() function
    - Call it in if __name__ == "__main__"
    - Each result must call print_result() AND log_result()
    - 30-minute timeout per experiment
    - Simulation runs at ~5.3x realtime for 10K neurons on A100
"""

    def get_metric_name(self) -> str:
        return "learning_speed"

    def get_result_metric_key(self) -> str:
        return "learning_speed"

    def get_score_direction(self) -> str:
        return "maximize"

    def get_constraints(self) -> str:
        return (
            "CONSTRAINTS for biologically plausible cortical culture simulation:\n"
            "1. Neuron count must be <= 50,000 (A100 memory limit at dt=0.1ms).\n"
            "2. STDP time constants must be in biologically plausible range:\n"
            "   tau_plus: 5-40ms, tau_minus: 10-80ms.\n"
            "3. Learning rates (a_plus, a_minus) must be < 0.1 to avoid "
            "   runaway potentiation.\n"
            "4. Connectivity density must be <= 0.3 for networks > 10K neurons "
            "   (memory constraint).\n"
            "5. Stimulation amplitude must not exceed 500 uA (electrode safety).\n"
            "6. Maximum simulation duration: 30 minutes wall-clock.\n"
            "7. Firing rates outside 0.1-100 Hz indicate pathological dynamics "
            "   (epileptiform or silent) — penalize these configurations.\n"
        )

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract best learning_speed from RESULT| lines."""
        best = float("-inf")
        for line in result_lines:
            parts = line.split("|")
            for part in parts:
                if part.strip().startswith("learning_speed="):
                    try:
                        val = float(part.strip().split("=", 1)[1])
                        best = max(best, val)
                    except ValueError:
                        continue
        return best if best > float("-inf") else 0.0
