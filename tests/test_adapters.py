"""Tests for project adapters — base interface and QCCCMAdapter.

TDD: verify adapters implement the ProjectAdapter contract correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentsciml.adapters.base import ProjectAdapter
from agentsciml.adapters.qcccm import QCCCMAdapter

# ---------------------------------------------------------------------------
# ProjectAdapter ABC tests
# ---------------------------------------------------------------------------

class TestProjectAdapterInterface:

    @pytest.mark.unit
    def test_cannot_instantiate_abstract(self):
        """ProjectAdapter is abstract — can't be instantiated directly."""
        with pytest.raises(TypeError):
            ProjectAdapter(Path("/tmp"))

    @pytest.mark.unit
    def test_experiment_path_property(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        assert adapter.experiment_path == tmp_project / "autoresearch" / "experiment.py"

    @pytest.mark.unit
    def test_results_path_property(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        assert adapter.results_path == tmp_project / "autoresearch" / "results.tsv"


# ---------------------------------------------------------------------------
# QCCCMAdapter tests
# ---------------------------------------------------------------------------

class TestQCCCMAdapter:

    @pytest.mark.unit
    def test_get_context_from_program_md(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        context = adapter.get_context()
        assert "quantum" in context.lower()

    @pytest.mark.unit
    def test_get_context_fallback(self, tmp_path):
        """If program.md doesn't exist, returns hardcoded context."""
        (tmp_path / "autoresearch").mkdir()
        adapter = QCCCMAdapter(tmp_path)
        context = adapter.get_context()
        assert "quantum" in context.lower() or "classical" in context.lower()

    @pytest.mark.unit
    def test_get_results_history(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        history = adapter.get_results_history()
        assert "advantage" in history
        assert "0.042" in history

    @pytest.mark.unit
    def test_get_results_history_empty(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = QCCCMAdapter(tmp_path)
        assert adapter.get_results_history() == ""

    @pytest.mark.unit
    def test_get_current_experiment(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        code = adapter.get_current_experiment()
        assert "run_experiment" in code

    @pytest.mark.unit
    def test_get_current_experiment_empty(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = QCCCMAdapter(tmp_path)
        assert adapter.get_current_experiment() == ""

    @pytest.mark.unit
    def test_get_available_api(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        api = adapter.get_available_api()
        assert "prepare" in api
        assert "ExperimentResult" in api
        assert "run_vqe" in api

    @pytest.mark.unit
    def test_metric_name(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        assert adapter.get_metric_name() == "quantum_advantage"

    @pytest.mark.unit
    def test_result_metric_key(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        assert adapter.get_result_metric_key() == "advantage"

    @pytest.mark.unit
    def test_parse_score_from_result_lines(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        lines = [
            "RESULT|SK|complete|N=10|advantage=0.042",
            "RESULT|SK|complete|N=12|advantage=0.087",
        ]
        assert adapter.parse_score(lines) == pytest.approx(0.087)

    @pytest.mark.unit
    def test_parse_score_empty(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        assert adapter.parse_score([]) == 0.0

    @pytest.mark.unit
    def test_parse_score_no_advantage_key(self, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        lines = ["RESULT|SK|complete|N=10|energy=-5.2"]
        assert adapter.parse_score(lines) == 0.0
