"""brain-fwi (Brain Full Waveform Inversion) project adapter.

Bridges the brain-fwi research infrastructure with AgenticSciML.
Evolves FWI strategies (frequency bands, parameterization, loss functions).
"""

from __future__ import annotations

from pathlib import Path

from .base import ProjectAdapter


class BrainFWIAdapter(ProjectAdapter):
    """Adapter for the brain-fwi project."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.home() / "Workspace" / "brain-fwi"
        super().__init__(root)

    def get_context(self) -> str:
        program = self.project_root / "autoresearch" / "program.md"
        context = ""
        if program.exists():
            context = program.read_text()
        else:
            context = (
                "Optimize Full Waveform Inversion (FWI) strategies for brain imaging. "
                "Explore frequency band scheduling, parameterization (Voxel vs SIREN), "
                "loss functions (L2, envelope, multiscale), and handling of skull "
                "heterogeneity and attenuation. Metric: brain_rmse (minimize)."
            )
        
        # Inject high-fidelity physical priors from neuro-kb / ITRUSST
        context += "\n\nPHYSICAL PRIORS (ITRUSST Benchmark BM3):\n"
        context += "- Water/CSF: c=1500 m/s, rho=1000 kg/m3, alpha=0.0 dB/cm/MHz\n"
        context += "- Grey/White Matter: c=1560 m/s, rho=1040 kg/m3, alpha=0.6 dB/cm/MHz\n"
        context += "- Skull (Cortical): c=2800 m/s, rho=1850 kg/m3, alpha=4.0 dB/cm/MHz\n"
        context += "- Skull (Trabecular): c=2300 m/s, rho=1700 kg/m3, alpha=8.0 dB/cm/MHz\n"
        context += "\nANATOMICAL CONTEXT:\n"
        context += "The human skull is a trilayer structure: Outer Cortical (~2mm), Diploe/Trabecular (~4mm), and Inner Cortical (~1mm). "
        context += "FWI must resolve these thin interfaces to accurately recover the internal brain velocity map."
        
        return context

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
    FWIConfig,              # Configuration class for FWI parameters
    run_fwi_experiment,     # (config: FWIConfig) -> ExperimentResult
    ExperimentResult,       # Dataclass with brain_rmse, skull_rmse, loss_history, wall_time
    print_result,           # (result) -> None  # prints RESULT| line
    log_result,             # (result) -> None  # appends to results.tsv
    get_commit_hash,        # () -> str
)

FWIConfig parameters:
    freq_bands: list[tuple[float, float]]  # Frequency bands in Hz
    n_iters_per_band: int                  # Iterations per band
    shots_per_iter: int                    # Source shots per iteration
    learning_rate: float                   # Learning rate (voxel path)
    parameterization: str                  # "voxel" or "siren"
    siren_lr: float                        # Adam LR for SIREN
    loss_fn: str                           # "l2", "multiscale", "envelope"
    gradient_smooth_sigma: float           # Gaussian smoothing for gradient
    mask_type: str                         # "head", "brain", "none"
    dx: float                              # Grid spacing (default 0.002)
    grid_size: int                         # N^3 grid size (default 64-128)

Experiment structure:
    - Define run_experiment() function
    - Call it in if __name__ == "__main__"
    - Each result must call print_result() AND log_result()
    - 30-minute timeout — allow for deeper sweeps and higher-fidelity grids (up to 128^3).
    - Primary metric: brain_rmse (RMSE in the brain region)
    - Target: brain_rmse < 50.0 m/s
"""

    def get_metric_name(self) -> str:
        return "brain_rmse"

    def get_result_metric_key(self) -> str:
        return "brain_rmse"

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract brain_rmse from RESULT| lines.

        Returns a normalized score where lower RMSE is better.
        Score = 1000 / (10 + brain_rmse)
        So RMSE=0 -> 100, RMSE=40 -> 20, RMSE=100 -> 9.
        """
        best_rmse = float("inf")
        for line in result_lines:
            parts = line.split("|")
            for part in parts:
                if part.strip().startswith("brain_rmse="):
                    try:
                        val = float(part.strip().split("=", 1)[1])
                        best_rmse = min(best_rmse, val)
                    except ValueError:
                        continue
        if best_rmse == float("inf"):
            return 0.0
        return 1000.0 / (10.0 + best_rmse)

    def get_constraints(self) -> str:
        return (
            "Hard constraints:\n"
            "1. Total experiment wall time < 30 minutes.\n"
            "2. Keep grid_size <= 128 and iters <= 50 to avoid timeouts.\n"
            "3. DO NOT modify prepare.py.\n"
            "4. Use deterministic seeds for reproducibility.\n"
            "5. Must call print_result() AND log_result() for every experiment.\n"
            "6. SIREN parameterization requires careful learning rate selection (e.g. 1e-4 to 1e-3).\n"
            "7. Voxel path learning rate is typically 10.0 to 100.0 (m/s max update).\n"
        )
