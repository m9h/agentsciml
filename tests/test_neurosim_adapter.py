"""Tests for the NeurosimAdapter (bl1 cortical culture simulation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentsciml.adapters.neurosim import NeurosimAdapter


@pytest.fixture
def neurosim_project(tmp_path: Path) -> Path:
    """Create a minimal bl1 project structure."""
    autoresearch = tmp_path / "autoresearch"
    autoresearch.mkdir()
    (autoresearch / "experiment.py").write_text(
        "from prepare import print_result, log_result\n"
        "def run_experiment():\n"
        "    print('RESULT|small_world|n=10000|learning_speed=0.035')\n"
        "if __name__ == '__main__':\n"
        "    run_experiment()\n"
    )
    (autoresearch / "results.tsv").write_text(
        "topology\tn_neurons\tstdp_rule\tlearning_speed\n"
        "random\t5000\tpair\t0.012\n"
        "small_world\t10000\tpair\t0.035\n"
    )
    (autoresearch / "program.md").write_text(
        "Discover plasticity rules and network topologies that maximize "
        "learning speed in simulated cortical cultures."
    )
    return tmp_path


class TestNeurosimAdapter:

    @pytest.mark.unit
    def test_score_direction(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        assert adapter.get_score_direction() == "maximize"

    @pytest.mark.unit
    def test_metric_name(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        assert adapter.get_metric_name() == "learning_speed"

    @pytest.mark.unit
    def test_get_constraints(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        constraints = adapter.get_constraints()
        assert "50,000" in constraints
        assert "tau_plus" in constraints
        assert "firing rate" in constraints.lower() or "Firing" in constraints

    @pytest.mark.unit
    def test_parse_score(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        lines = [
            "RESULT|random|n=5000|learning_speed=0.012",
            "RESULT|small_world|n=10000|learning_speed=0.035",
        ]
        assert adapter.parse_score(lines) == pytest.approx(0.035)

    @pytest.mark.unit
    def test_parse_score_empty(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        assert adapter.parse_score([]) == 0.0

    @pytest.mark.unit
    def test_get_current_experiment(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        code = adapter.get_current_experiment()
        assert "run_experiment" in code

    @pytest.mark.unit
    def test_get_context_from_program_md(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        context = adapter.get_context()
        assert "plasticity" in context.lower() or "learning" in context.lower()

    @pytest.mark.unit
    def test_get_context_fallback(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = NeurosimAdapter(tmp_path)
        context = adapter.get_context()
        assert "cortical" in context.lower() or "learning_speed" in context

    @pytest.mark.unit
    def test_get_results_history(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        history = adapter.get_results_history()
        assert "0.035" in history

    @pytest.mark.unit
    def test_get_available_api(self, neurosim_project):
        adapter = NeurosimAdapter(neurosim_project)
        api = adapter.get_available_api()
        assert "create_network" in api
        assert "configure_stdp" in api
        assert "run_culture" in api

    @pytest.mark.unit
    def test_default_project_root(self):
        adapter = NeurosimAdapter()
        assert adapter.project_root == Path.home() / "dev" / "bl1"
