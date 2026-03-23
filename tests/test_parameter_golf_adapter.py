"""Tests for the ParameterGolfAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentsciml.adapters.parameter_golf import ParameterGolfAdapter


@pytest.fixture
def golf_project(tmp_path: Path) -> Path:
    """Create a minimal parameter-golf project structure."""
    autoresearch = tmp_path / "autoresearch"
    autoresearch.mkdir()
    (tmp_path / "train_gpt.py").write_text(
        "import torch\n"
        "def train():\n"
        "    print('val_bpb=1.2244')\n"
        "if __name__ == '__main__':\n"
        "    train()\n"
    )
    (autoresearch / "results.tsv").write_text(
        "run\tval_bpb\tartifact_mb\n"
        "baseline\t1.2244\t15.86\n"
    )
    return tmp_path


class TestParameterGolfAdapter:

    @pytest.mark.unit
    def test_score_direction(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        assert adapter.get_score_direction() == "minimize"

    @pytest.mark.unit
    def test_metric_name(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        assert adapter.get_metric_name() == "val_bpb"

    @pytest.mark.unit
    def test_get_constraints(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        constraints = adapter.get_constraints()
        assert "16,000,000" in constraints
        assert "600 seconds" in constraints
        assert "paid prefix" in constraints.lower()

    @pytest.mark.unit
    def test_parse_score_from_result_lines(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        lines = [
            "RESULT|gpt|val_bpb=1.2244",
            "RESULT|gpt|val_bpb=1.1428",
        ]
        assert adapter.parse_score(lines) == pytest.approx(1.1428)

    @pytest.mark.unit
    def test_parse_score_plain_output(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        lines = ["val_bpb=1.1500"]
        assert adapter.parse_score(lines) == pytest.approx(1.15)

    @pytest.mark.unit
    def test_parse_score_empty(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        assert adapter.parse_score([]) == 0.0

    @pytest.mark.unit
    def test_get_current_experiment_from_train_gpt(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        code = adapter.get_current_experiment()
        assert "train" in code
        assert "torch" in code

    @pytest.mark.unit
    def test_experiment_path_prefers_train_gpt(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        assert adapter.experiment_path == golf_project / "train_gpt.py"

    @pytest.mark.unit
    def test_experiment_path_falls_back(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = ParameterGolfAdapter(tmp_path)
        assert adapter.experiment_path == tmp_path / "autoresearch" / "experiment.py"

    @pytest.mark.unit
    def test_get_available_api(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        api = adapter.get_available_api()
        assert "16,000,000" in api or "16MB" in api or "16 MB" in api
        assert "val_bpb" in api

    @pytest.mark.unit
    def test_get_context_fallback(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = ParameterGolfAdapter(tmp_path)
        context = adapter.get_context()
        assert "parameter golf" in context.lower() or "16 MB" in context

    @pytest.mark.unit
    def test_get_results_history(self, golf_project):
        adapter = ParameterGolfAdapter(golf_project)
        history = adapter.get_results_history()
        assert "1.2244" in history
