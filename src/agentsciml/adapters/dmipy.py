"""dmipy (diffusion MRI) project adapter.

Bridges the dmipy microstructure estimation toolbox with AgenticSciML.
Primary optimization target: minimize median fiber orientation error
(degrees) for neural posterior estimation of Ball+2Stick model.

Since the evolutionary tree maximizes score, fiber error is negated so that
less negative = lower error = better.
"""

from __future__ import annotations

from pathlib import Path

from .base import ProjectAdapter


class DmipyAdapter(ProjectAdapter):
    """Adapter for the dmipy diffusion MRI microstructure project."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.home() / "dev" / "dmipy"
        super().__init__(root)

    def get_context(self) -> str:
        """Return program.md as the research context."""
        program = self.project_root / "autoresearch" / "program.md"
        if program.exists():
            return program.read_text()
        return (
            "Optimize neural posterior estimation for diffusion MRI "
            "microstructure using the Ball + 2-Stick forward model in dmipy. "
            "Primary metric: median fiber orientation error (degrees). "
            "Search space: network architecture (MLP/E3), training hyperparams, "
            "noise schedules, sampler strategies. "
            "Baseline: MLP score ~15.5 deg (30k steps), Flow ~3.2 deg (200k steps). "
            "Target: <5 deg with efficient training (<50k steps)."
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
        """Return the dmipy API surface the Engineer must use."""
        return """\
Available imports from prepare.py (DO NOT MODIFY prepare.py):

from prepare import (
    ExperimentResult,       # Dataclass for structured results
                            #   Required fields: commit, model_name, architecture,
                            #   dataset, fiber1_median_deg (PRIMARY METRIC)
                            #   Optional: fiber1_mean_deg, d_stick_r, f1_r,
                            #   final_loss, train_time_s, n_steps, hidden_dim,
                            #   learning_rate, seed, status, metadata
    get_commit_hash,        # () -> str  (short git hash)
    print_result,           # (result: ExperimentResult) -> None
                            #   Prints RESULT|model|dataset|arch|fiber_error_deg=X|...
    log_result,             # (result: ExperimentResult) -> None
                            #   Appends to autoresearch/results.tsv
    make_hcp_acquisition,   # (n_b0=6, n_b1000=30, n_b2000=30, n_b3000=24, seed=0)
                            #   -> JaxAcquisition  (90-dir multi-shell, SI units)
    build_simulator,        # (acquisition=None, snr=30.0, snr_range=None)
                            #   -> ModelSimulator  (Ball+2Stick, 10-D params)
                            #   Parameters: [d_ball, d_stick, f1, f2,
                            #                mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
    angular_error_deg,      # (v1, v2) -> float  (degrees, antipodal-safe)
    compute_fiber_errors,   # (theta_true, preds, n_eval) -> array of errors
    safe_pearson_r,         # (a, b) -> float  (NaN-safe Pearson r)
)

ModelSimulator API (from build_simulator):
    sim.theta_dim           # 10 (parameter dimensionality)
    sim.signal_dim          # 90 (number of measurements)
    sim.parameter_names     # list of 10 parameter names
    sim.parameter_ranges    # dict {name: (low, high)}
    sim.acquisition         # JaxAcquisition object
    sim.prior_sampler(key, n) -> (n, 10) array
    sim.simulate(key, theta) -> (n, 90) array (clean signal)
    sim.add_noise(key, signal) -> (n, 90) array (noisy)
    sim.sample_and_simulate(key, n) -> (theta, noisy_signals)
    sim.snr_range = (10.0, 50.0)  # set for variable-SNR training

Available from dmipy_jax.inference.score_posterior:
    MLPScoreNet             # MLP score network (param_dim, signal_dim, hidden_dim, depth, cond_dim)
    VPSchedule              # VP-SDE noise schedule (beta_min, beta_max)
    train_score_posterior   # (key, net, simulator_fn, prior_fn, schedule, ...)
    sample_posterior        # (key, net, signal, schedule, n_samples, n_steps, ...)

Available from dmipy_jax.pipeline.train:
    train_sbi               # MDN/Flow training pipeline

Experiment structure:
    - Define run_experiment() function
    - Call it in if __name__ == "__main__"
    - Each result must call print_result() AND log_result()
    - RESULT format: RESULT|model|dataset|arch|fiber_error_deg=val|d_stick_r=val|...
    - 10-minute timeout per experiment
"""

    def get_metric_name(self) -> str:
        return "fiber_orientation_error"

    def get_result_metric_key(self) -> str:
        return "fiber_error_deg"

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract best (lowest) fiber error from RESULT| lines, negated for tree maximization.

        The evolutionary tree maximizes score, so we negate fiber error:
        lower error -> less negative score -> better.
        Returns 0.0 if no fiber_error_deg values found.
        """
        best_error = float("inf")
        found = False
        for line in result_lines:
            parts = line.split("|")
            for part in parts:
                if part.strip().startswith("fiber_error_deg="):
                    try:
                        val = float(part.strip().split("=", 1)[1])
                        best_error = min(best_error, val)
                        found = True
                    except ValueError:
                        continue
        if not found:
            return 0.0
        return -best_error
