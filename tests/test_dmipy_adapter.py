"""Tests for DmipyAdapter.

Validates the adapter interface, context/API surface, metric configuration,
and RESULT| line parsing for the dmipy diffusion MRI project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentsciml.adapters.base import ProjectAdapter

# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------

class TestDmipyAdapterImport:

    @pytest.mark.unit
    def test_can_import(self):
        from agentsciml.adapters.dmipy import DmipyAdapter
        assert DmipyAdapter is not None

    @pytest.mark.unit
    def test_is_project_adapter(self):
        from agentsciml.adapters.dmipy import DmipyAdapter
        assert issubclass(DmipyAdapter, ProjectAdapter)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dmipy_project(tmp_path: Path) -> Path:
    """Create a minimal dmipy project structure for testing."""
    autoresearch = tmp_path / "autoresearch"
    autoresearch.mkdir()

    (autoresearch / "program.md").write_text(
        "Optimize neural posterior estimation for diffusion MRI microstructure.\n"
        "Primary metric: median fiber orientation error (degrees).\n"
        "Target: <5 deg with efficient training."
    )

    (autoresearch / "results.tsv").write_text(
        "commit\tmodel_name\tarchitecture\tdataset\tfiber1_median_deg\t"
        "fiber1_mean_deg\td_stick_r\tf1_r\tfinal_loss\ttrain_time_s\t"
        "n_steps\thidden_dim\tlearning_rate\tseed\tstatus\n"
        "abc123\tBall2Stick\tMLPScore_h256_d6\tsynthetic_hcp\t15.5\t"
        "18.2\t0.85\t0.91\t-1.2345\t120.0\t5000\t256\t0.0003\t0\tok\n"
        "def456\tBall2Stick\tMLPScore_h512_d4\tsynthetic_hcp\t12.3\t"
        "14.8\t0.89\t0.93\t-1.5678\t180.0\t10000\t512\t0.0003\t0\tok\n"
    )

    (autoresearch / "experiment.py").write_text(
        "from prepare import (\n"
        "    ExperimentResult, get_commit_hash, print_result, log_result,\n"
        "    make_hcp_acquisition, build_simulator, compute_fiber_errors,\n"
        "    safe_pearson_r,\n"
        ")\n"
        "def run_experiment():\n"
        "    acq = make_hcp_acquisition()\n"
        "    sim = build_simulator(acq, snr=30.0)\n"
        "    # ... train and evaluate ...\n"
        "    pass\n"
        "if __name__ == '__main__':\n"
        "    run_experiment()\n"
    )

    return tmp_path


@pytest.fixture
def dmipy_adapter(dmipy_project):
    from agentsciml.adapters.dmipy import DmipyAdapter
    return DmipyAdapter(dmipy_project)


# ---------------------------------------------------------------------------
# Interface implementation tests
# ---------------------------------------------------------------------------

class TestDmipyAdapterInterface:

    @pytest.mark.unit
    def test_project_root(self, dmipy_adapter, dmipy_project):
        assert dmipy_adapter.project_root == dmipy_project

    @pytest.mark.unit
    def test_experiment_path(self, dmipy_adapter, dmipy_project):
        assert dmipy_adapter.experiment_path == dmipy_project / "autoresearch" / "experiment.py"

    @pytest.mark.unit
    def test_results_path(self, dmipy_adapter, dmipy_project):
        assert dmipy_adapter.results_path == dmipy_project / "autoresearch" / "results.tsv"


# ---------------------------------------------------------------------------
# Context and API surface tests
# ---------------------------------------------------------------------------

class TestDmipyContext:

    @pytest.mark.unit
    def test_get_context_from_program_md(self, dmipy_adapter):
        context = dmipy_adapter.get_context()
        assert "diffusion" in context.lower() or "fiber" in context.lower()

    @pytest.mark.unit
    def test_get_context_fallback(self, tmp_path):
        """Falls back to hardcoded context if program.md missing."""
        (tmp_path / "autoresearch").mkdir()
        from agentsciml.adapters.dmipy import DmipyAdapter
        adapter = DmipyAdapter(tmp_path)
        context = adapter.get_context()
        assert len(context) > 50
        assert "microstructure" in context.lower() or "diffusion" in context.lower()

    @pytest.mark.unit
    def test_get_available_api(self, dmipy_adapter):
        api = dmipy_adapter.get_available_api()
        assert "ExperimentResult" in api
        assert "build_simulator" in api
        assert "ModelSimulator" in api
        assert "fiber_error_deg" in api

    @pytest.mark.unit
    def test_get_results_history(self, dmipy_adapter):
        history = dmipy_adapter.get_results_history()
        assert "fiber1_median_deg" in history
        assert "15.5" in history

    @pytest.mark.unit
    def test_get_results_history_empty(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        from agentsciml.adapters.dmipy import DmipyAdapter
        adapter = DmipyAdapter(tmp_path)
        assert adapter.get_results_history() == ""

    @pytest.mark.unit
    def test_get_current_experiment(self, dmipy_adapter):
        code = dmipy_adapter.get_current_experiment()
        assert "run_experiment" in code

    @pytest.mark.unit
    def test_get_current_experiment_empty(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        from agentsciml.adapters.dmipy import DmipyAdapter
        adapter = DmipyAdapter(tmp_path)
        assert adapter.get_current_experiment() == ""


# ---------------------------------------------------------------------------
# Metric configuration tests
# ---------------------------------------------------------------------------

class TestDmipyMetrics:

    @pytest.mark.unit
    def test_metric_name(self, dmipy_adapter):
        name = dmipy_adapter.get_metric_name()
        assert name == "fiber_orientation_error"

    @pytest.mark.unit
    def test_result_metric_key(self, dmipy_adapter):
        key = dmipy_adapter.get_result_metric_key()
        assert key == "fiber_error_deg"

    @pytest.mark.unit
    def test_parse_score_from_result_lines(self, dmipy_adapter):
        """Score should extract fiber_error_deg from RESULT| lines.

        Since lower error is better but the tree maximizes score,
        the adapter should negate fiber error.
        """
        lines = [
            "RESULT|Ball2Stick|synthetic_hcp|MLPScore_h256_d6|fiber_error_deg=15.5000|d_stick_r=0.8500|f1_r=0.9100|loss=-1.234500|time=120.0|steps=5000",
            "RESULT|Ball2Stick|synthetic_hcp|MLPScore_h512_d4|fiber_error_deg=12.3000|d_stick_r=0.8900|f1_r=0.9300|loss=-1.567800|time=180.0|steps=10000",
        ]
        score = dmipy_adapter.parse_score(lines)
        # Best fiber error is 12.3; negated -> -12.3
        assert score == pytest.approx(-12.3)

    @pytest.mark.unit
    def test_parse_score_empty(self, dmipy_adapter):
        assert dmipy_adapter.parse_score([]) == 0.0

    @pytest.mark.unit
    def test_parse_score_no_fiber_error_key(self, dmipy_adapter):
        lines = ["RESULT|Ball2Stick|synthetic_hcp|MLP|d_stick_r=0.85"]
        assert dmipy_adapter.parse_score(lines) == 0.0

    @pytest.mark.unit
    def test_parse_score_multiple_results_takes_best(self, dmipy_adapter):
        """With multiple fiber error values, take the best (lowest, most negative)."""
        lines = [
            "RESULT|Ball2Stick|synthetic_hcp|MLPScore_h256_d6|fiber_error_deg=15.5",
            "RESULT|Ball2Stick|synthetic_hcp|MLPScore_h512_d4|fiber_error_deg=8.2",
        ]
        score = dmipy_adapter.parse_score(lines)
        # Best fiber error = 8.2, negated = -8.2
        assert score == pytest.approx(-8.2)


# ---------------------------------------------------------------------------
# Default project root test
# ---------------------------------------------------------------------------

class TestDmipyDefaults:

    @pytest.mark.unit
    def test_default_project_root(self):
        from agentsciml.adapters.dmipy import DmipyAdapter
        adapter = DmipyAdapter()
        assert adapter.project_root == Path.home() / "dev" / "dmipy"
