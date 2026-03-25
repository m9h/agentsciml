"""Tests for MetaSciMLAdapter.

Validates the meta-architecture adapter interface, context/API surface,
metric configuration, and RESULT| line parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentsciml.adapters.base import ProjectAdapter
from agentsciml.adapters.meta import MetaSciMLAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def meta_project(tmp_path: Path) -> Path:
    """Create a minimal meta-adapter project structure."""
    autoresearch = tmp_path / "autoresearch"
    autoresearch.mkdir()
    (autoresearch / "results.tsv").write_text(
        "run\tadvantage\tcost\tefficiency\n"
        "baseline\t0.042\t1.50\t0.028\n"
    )
    workspace = autoresearch / "workspace"
    workspace.mkdir()
    (workspace / "meta_hypothesis.yaml").write_text(
        "name: test\nconfig:\n  debate_rounds: 6\n"
    )
    return tmp_path


@pytest.fixture
def meta_adapter(meta_project: Path) -> MetaSciMLAdapter:
    return MetaSciMLAdapter(project_root=meta_project)


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestMetaAdapterInterface:

    @pytest.mark.unit
    def test_is_project_adapter(self):
        assert issubclass(MetaSciMLAdapter, ProjectAdapter)

    @pytest.mark.unit
    def test_can_instantiate(self, meta_project):
        adapter = MetaSciMLAdapter(project_root=meta_project)
        assert adapter.project_root == meta_project

    @pytest.mark.unit
    def test_default_project_root(self):
        adapter = MetaSciMLAdapter()
        assert adapter.project_root == Path.home() / "dev" / "agentsciml"


# ---------------------------------------------------------------------------
# Context and API surface tests
# ---------------------------------------------------------------------------

class TestMetaContext:

    @pytest.mark.unit
    def test_get_context(self, meta_adapter):
        context = meta_adapter.get_context()
        assert "Meta-Architecture" in context
        assert "QCCCM" in context

    @pytest.mark.unit
    def test_get_available_api(self, meta_adapter):
        api = meta_adapter.get_available_api()
        assert "debate_rounds" in api
        assert "budget_usd" in api
        assert "model_overrides" in api

    @pytest.mark.unit
    def test_get_results_history(self, meta_adapter):
        history = meta_adapter.get_results_history()
        assert "efficiency" in history
        assert "0.028" in history

    @pytest.mark.unit
    def test_get_results_history_empty(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = MetaSciMLAdapter(project_root=tmp_path)
        assert adapter.get_results_history() == ""

    @pytest.mark.unit
    def test_get_current_experiment(self, meta_adapter):
        exp = meta_adapter.get_current_experiment()
        assert "debate_rounds" in exp
        assert "6" in exp

    @pytest.mark.unit
    def test_get_current_experiment_empty(self, tmp_path):
        (tmp_path / "autoresearch").mkdir()
        adapter = MetaSciMLAdapter(project_root=tmp_path)
        assert adapter.get_current_experiment() == ""


# ---------------------------------------------------------------------------
# Metric configuration tests
# ---------------------------------------------------------------------------

class TestMetaMetrics:

    @pytest.mark.unit
    def test_metric_name(self, meta_adapter):
        assert meta_adapter.get_metric_name() == "efficiency"

    @pytest.mark.unit
    def test_result_metric_key(self, meta_adapter):
        assert meta_adapter.get_result_metric_key() == "efficiency"

    @pytest.mark.unit
    def test_parse_score(self, meta_adapter):
        lines = [
            "RESULT|advantage=0.5000|cost=1.2000|efficiency=0.416667",
        ]
        assert meta_adapter.parse_score(lines) == pytest.approx(0.416667)

    @pytest.mark.unit
    def test_parse_score_multiple_takes_first(self, meta_adapter):
        lines = [
            "RESULT|advantage=0.5|cost=1.2|efficiency=0.416667",
            "RESULT|advantage=0.8|cost=2.0|efficiency=0.400000",
        ]
        assert meta_adapter.parse_score(lines) == pytest.approx(0.416667)

    @pytest.mark.unit
    def test_parse_score_empty(self, meta_adapter):
        assert meta_adapter.parse_score([]) == 0.0

    @pytest.mark.unit
    def test_parse_score_no_efficiency_key(self, meta_adapter):
        lines = ["RESULT|advantage=0.5|cost=1.2"]
        assert meta_adapter.parse_score(lines) == 0.0
