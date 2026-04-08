"""vbjax (neural mass model) project adapter.

Bridges the vbjax autoresearch infrastructure with AgenticSciML.
Evolves spectral fitting strategies for Bayesian model comparison.
"""

from __future__ import annotations

from pathlib import Path

from .base import ProjectAdapter


class VBJaxAdapter(ProjectAdapter):
    """Adapter for the vbjax neural mass model project."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.home() / "dev" / "vbjax"
        super().__init__(root)

    def get_context(self) -> str:
        program = self.project_root / "autoresearch" / "program.md"
        if program.exists():
            return program.read_text()
        return (
            "Optimize spectral fitting strategies for neural mass models "
            "(Liley, CMC, RRW, CBEI) so that Bayesian model comparison "
            "produces well-separated free energies. Metric: spectral_fit "
            "= negative mean spectral loss across all models."
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
    ExperimentResult,           # Dataclass: commit, strategy, model_name,
                                #   n_free_params, n_subjects, n_opt_steps, lr,
                                #   noise_sigma, dt_s, n_steps, spectral_loss,
                                #   free_energy, theta_map, wall_time, status
    generate_synthetic_eeg,     # (n_subjects=2, seed=0) -> jnp.ndarray (n_sub, n_freq)
                                #   Generates ground-truth EEG PSDs from Liley model
    fit_model,                  # (model_name, free_param_names, target_psd, **kwargs)
                                #   -> ExperimentResult
                                #   kwargs: n_opt_steps, lr, noise_sigma, dt_s, n_steps,
                                #   n_warmup, theta_init, noise_seed, grad_clip,
                                #   prior_std, lr_schedule ("constant"|"cosine"|"warmup")
    run_comparison,             # (models_config, target_psds, **fit_kwargs)
                                #   -> list[ExperimentResult]
                                #   models_config: [{'name': str, 'free_params': [str]}]
    print_result,               # (result) -> None  # prints RESULT| line
    log_result,                 # (result) -> None  # appends to results.tsv
    get_commit_hash,            # () -> str
    make_model_dfun,            # (model_name, free_param_names) -> (dfun, n_states, defaults)
    N_FREQ_BINS,                # int: number of frequency bins in the PSD
)

Model names: 'liley' (14D), 'cmc' (8D), 'rrw' (8D), 'cbei' (8D)

Available free parameters per model:
  liley: p_ee, p_ei, sigma_e, sigma_i, Gamma_e, Gamma_i, gamma_e, gamma_i,
         tau_e, tau_i, N_ee_b, N_ei_b, N_ie_b, N_ii_b, Lambda, v_e
  cmc:   I, He, Hi, a, b, g_ss_sp, g_sp_ii, g_sp_dp, g_dp_ii, g_dp_sp,
         g_ii_ss, g_ii_sp, g_ii_dp
  rrw:   I, Q_max, theta, sigma_prime, gamma_e, alpha, beta, nu_ee, nu_ei,
         nu_es, nu_se, nu_sr, nu_sn, nu_re, nu_rs
  cbei:  I, kappa_ee, kappa_ei, kappa_ie, kappa_ii, tau_m_e, tau_m_i,
         tau_s_e, tau_s_i, Delta_e, Delta_i, eta_e, eta_i

Experiment structure:
    - Define run_experiment() function
    - Call it in if __name__ == "__main__"
    - Each result must call print_result() AND log_result()
    - 5-minute timeout — keep total fits under ~32 (4 models x 2 subjects x 4 param sets)
    - theta is log-deviations from defaults: actual_param = exp(theta[i]) * default[i]
    - spectral_fit = -spectral_loss (higher is better)
    - Explore: lr schedules, more/fewer free params, different param combos,
      multi-start, warm restarts, adaptive noise, initialization strategies
"""

    def get_metric_name(self) -> str:
        return "spectral_fit"

    def get_result_metric_key(self) -> str:
        return "spectral_fit"

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract best spectral_fit from RESULT| lines.

        Returns a normalized score in [0, 1] for compatibility with the
        orchestrator's sanity check (|score| > 1.0 = suspicious).
        Normalization: score = 1 / (1 + mean_loss / 100)
        So loss=0 -> score=1.0, loss=100 -> 0.5, loss=1000 -> 0.09.
        """
        losses = []
        for line in result_lines:
            parts = line.split("|")
            for part in parts:
                if part.strip().startswith("loss="):
                    try:
                        val = float(part.strip().split("=", 1)[1])
                        losses.append(val)
                    except ValueError:
                        continue
        if not losses:
            return 0.0
        mean_loss = sum(losses) / len(losses)
        return 1.0 / (1.0 + mean_loss / 100.0)

    def get_constraints(self) -> str:
        return (
            "Hard constraints:\n"
            "1. Total experiment wall time < 5 minutes (RTX 2080, 8GB).\n"
            "2. DO NOT modify prepare.py.\n"
            "3. Use deterministic seeds for reproducibility.\n"
            "4. Must fit ALL 4 models (liley, cmc, rrw, cbei).\n"
            "5. Must call print_result() AND log_result() for every fit.\n"
            "6. theta values are log-deviations: param = exp(theta) * default.\n"
            "   Values outside [-3, 3] are suspicious (exp(3) ≈ 20x default).\n"
        )
