"""Tests for orchestrator.py — the evolutionary multi-agent loop.

TDD: mock all external dependencies (Anthropic API, sandbox execution)
to test the orchestration logic in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agentsciml.adapters.base import ProjectAdapter
from agentsciml.adapters.qcccm import QCCCMAdapter
from agentsciml.orchestrator import (
    DEFAULT_DEBATE_ROUNDS,
    Orchestrator,
)
from agentsciml.protocols import AnalysisReport, MutationProposal
from agentsciml.sandbox import ExecutionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis_json(**overrides) -> str:
    data = {
        "summary": "Test analysis",
        "best_score": 0.05,
        "best_config": "SK N=10 PIMC",
        "worst_configs": [],
        "unexplored": ["N>12"],
        "n_experiments": 2,
    }
    data.update(overrides)
    return json.dumps(data)


def _make_proposal_json(**overrides) -> str:
    data = {
        "strategy": "Increase N to 14",
        "changes": ["N: 10 -> 14"],
        "expected_impact": "Better quantum advantage at larger N",
        "technique_used": None,
        "code_outline": "for N in [12, 14]: ...",
    }
    data.update(overrides)
    return json.dumps(data)


def _make_scored_result_json(**overrides) -> str:
    data = {
        "score": 0.08,
        "result_lines": ["RESULT|SK|complete|N=14|advantage=0.08"],
        "comparison": "Improved over previous best of 0.05",
        "insights": "Larger N shows better quantum advantage",
        "status": "ok",
    }
    data.update(overrides)
    return json.dumps(data)


def _make_exec_result(status="ok", advantage=0.08) -> ExecutionResult:
    return ExecutionResult(
        stdout=f"RESULT|SK|complete|N=14|advantage={advantage}\n",
        stderr="",
        returncode=0 if status == "ok" else 1,
        wall_time=5.0,
        result_lines=[f"RESULT|SK|complete|N=14|advantage={advantage}"] if status == "ok" else [],
        status=status,
    )


# ---------------------------------------------------------------------------
# Orchestrator init tests
# ---------------------------------------------------------------------------

class TestOrchestratorInit:

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_creates_with_defaults(self, mock_anthropic_cls, tmp_project, knowledge_dir):
        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(
            adapter,
            knowledge_file=knowledge_dir / "qcccm_techniques.yaml",
        )
        assert orch.max_generations == 20
        assert orch.cost.budget_usd == 5.0
        assert len(orch.techniques) > 0

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_custom_budget_and_generations(self, mock_anthropic_cls, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, budget_usd=1.0, max_generations=5)
        assert orch.cost.budget_usd == 1.0
        assert orch.max_generations == 5

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_tree_path_default(self, mock_anthropic_cls, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter)
        assert orch.tree._path == tmp_project / "autoresearch" / "tree.json"

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_missing_knowledge_file(self, mock_anthropic_cls, tmp_project):
        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(
            adapter,
            knowledge_file=Path("/nonexistent/techniques.yaml"),
        )
        assert orch.techniques == []


# ---------------------------------------------------------------------------
# Phase 2: Root solution tests
# ---------------------------------------------------------------------------

class TestGenerateRoot:

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.run_experiment")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_existing_experiment_used_as_root(
        self, mock_anthropic_cls, mock_run_exp, tmp_project
    ):
        mock_run_exp.return_value = _make_exec_result(status="ok", advantage=0.042)

        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, max_generations=0)
        best = orch.run()

        # Should have executed the existing experiment.py
        mock_run_exp.assert_called_once()
        assert len(orch.tree) == 1
        assert best is not None
        assert best.generation == 0

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.run_experiment")
    @patch("agentsciml.orchestrator.call_agent")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_generates_root_when_no_experiment(
        self, mock_anthropic_cls, mock_call_agent, mock_run_exp, tmp_path
    ):
        # Empty project — no existing experiment.py
        (tmp_path / "autoresearch").mkdir()
        (tmp_path / "autoresearch" / "experiment.py").write_text("")

        mock_call_agent.return_value = '```python\nprint("RESULT|test|advantage=0.01")\n```'
        mock_run_exp.return_value = _make_exec_result(status="ok", advantage=0.01)

        adapter = QCCCMAdapter(tmp_path)
        orch = Orchestrator(adapter, max_generations=0)
        best = orch.run()

        # Should have called engineer to generate root
        mock_call_agent.assert_called_once()
        assert best is not None


# ---------------------------------------------------------------------------
# Phase 3: Evolution tests
# ---------------------------------------------------------------------------

class TestEvolution:

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.run_experiment")
    @patch("agentsciml.orchestrator.call_agent")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_one_generation(
        self, mock_anthropic_cls, mock_call_agent, mock_run_exp, tmp_project
    ):
        # Root execution
        root_result = _make_exec_result(status="ok", advantage=0.042)
        child_result = _make_exec_result(status="ok", advantage=0.08)
        mock_run_exp.side_effect = [root_result, child_result, child_result]

        # Agent responses for mutation pipeline
        call_count = [0]
        def mock_agent_side_effect(role, docs, **kwargs):
            call_count[0] += 1
            if role == "data_analyst":
                return _make_analysis_json()
            elif role == "retriever":
                return '{"selected": null}'
            elif role == "proposer":
                return _make_proposal_json()
            elif role == "critic":
                return (
                    '{"flaws": [], "suggestions": [],'
                    ' "feasibility": "feasible", "revised_strategy": ""}'
                )
            elif role == "engineer":
                return '```python\nprint("RESULT|SK|complete|N=14|advantage=0.08")\n```'
            elif role == "result_analyst":
                return _make_scored_result_json()
            return "{}"

        mock_call_agent.side_effect = mock_agent_side_effect

        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, max_generations=1, budget_usd=10.0)
        best = orch.run()

        assert len(orch.tree) >= 2  # root + at least 1 child
        assert best is not None

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.run_experiment")
    @patch("agentsciml.orchestrator.call_agent")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_budget_exhaustion_stops_evolution(
        self, mock_anthropic_cls, mock_call_agent, mock_run_exp, tmp_project
    ):
        mock_run_exp.return_value = _make_exec_result(status="ok", advantage=0.042)
        mock_call_agent.return_value = _make_analysis_json()

        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, max_generations=100, budget_usd=0.0001)

        # Pre-exhaust budget
        orch.cost.add("claude-sonnet-4-6", 1_000_000, 1_000_000)
        orch.run()

        # Should stop very quickly after budget exhaustion
        assert len(orch.tree) <= 2


# ---------------------------------------------------------------------------
# Debug retry tests
# ---------------------------------------------------------------------------

class TestDebugRetries:

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.run_experiment")
    @patch("agentsciml.orchestrator.call_agent")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_debug_retry_on_crash(
        self, mock_anthropic_cls, mock_call_agent, mock_run_exp, tmp_project
    ):
        # Root succeeds, first child crashes, debug fixes it
        root_result = _make_exec_result(status="ok", advantage=0.042)
        crash_result = ExecutionResult(
            stdout="", stderr="ImportError: No module named 'foo'",
            returncode=1, wall_time=1.0, result_lines=[], status="crash",
        )
        fixed_result = _make_exec_result(status="ok", advantage=0.06)

        mock_run_exp.side_effect = [root_result, crash_result, fixed_result, fixed_result]

        def mock_agent_side_effect(role, docs, **kwargs):
            if role == "data_analyst":
                return _make_analysis_json()
            elif role == "retriever":
                return '{"selected": null}'
            elif role == "proposer":
                return _make_proposal_json()
            elif role == "critic":
                return (
                    '{"flaws": [], "suggestions": [],'
                    ' "feasibility": "feasible", "revised_strategy": ""}'
                )
            elif role == "engineer":
                return '```python\nprint("RESULT|test|advantage=0.06")\n```'
            elif role == "debugger":
                return '```python\nprint("RESULT|test|advantage=0.06")\n```'
            return "{}"

        mock_call_agent.side_effect = mock_agent_side_effect

        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, max_generations=1, budget_usd=10.0)
        orch.run()

        # Debugger should have been called
        debugger_calls = [
            c for c in mock_call_agent.call_args_list
            if c.args[0] == "debugger"
        ]
        assert len(debugger_calls) >= 1


# ---------------------------------------------------------------------------
# Debate protocol tests
# ---------------------------------------------------------------------------

class TestDebateProtocol:

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.call_agent")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_debate_round_count(
        self, mock_anthropic_cls, mock_call_agent, tmp_project
    ):
        """Verify debate_rounds of proposer calls + (N-1) critic calls."""
        call_log = []

        def mock_agent_side_effect(role, docs, **kwargs):
            call_log.append(role)
            if role == "proposer":
                return _make_proposal_json()
            elif role == "critic":
                return (
                    '{"flaws": [], "suggestions": [],'
                    ' "feasibility": "feasible", "revised_strategy": ""}'
                )
            return "{}"

        mock_call_agent.side_effect = mock_agent_side_effect

        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, budget_usd=10.0)

        analysis = AnalysisReport(
            summary="test", best_score=0.05, best_config="SK N=10"
        )
        proposal = orch._run_debate(
            analysis=analysis,
            technique_text="",
            parent_code="print('test')",
            context="test context",
        )

        proposer_calls = [r for r in call_log if r == "proposer"]
        critic_calls = [r for r in call_log if r == "critic"]
        assert len(proposer_calls) == orch.debate_rounds
        assert len(critic_calls) == orch.debate_rounds - 1  # no critic on final round

        assert isinstance(proposal, MutationProposal)

    @pytest.mark.integration
    @patch("agentsciml.orchestrator.call_agent")
    @patch("agentsciml.orchestrator.anthropic.Anthropic")
    def test_custom_debate_rounds(
        self, mock_anthropic_cls, mock_call_agent, tmp_project
    ):
        """Verify custom debate_rounds are respected."""
        call_log = []

        def mock_agent_side_effect(role, docs, **kwargs):
            call_log.append(role)
            if role == "proposer":
                return _make_proposal_json()
            elif role == "critic":
                return (
                    '{"flaws": [], "suggestions": [],'
                    ' "feasibility": "feasible", "revised_strategy": ""}'
                )
            return "{}"

        mock_call_agent.side_effect = mock_agent_side_effect

        adapter = QCCCMAdapter(tmp_project)
        orch = Orchestrator(adapter, budget_usd=10.0, debate_rounds=6)
        assert orch.debate_rounds == 6

        analysis = AnalysisReport(
            summary="test", best_score=0.05, best_config="SK N=10"
        )
        orch._run_debate(
            analysis=analysis,
            technique_text="",
            parent_code="print('test')",
            context="test context",
        )

        proposer_calls = [r for r in call_log if r == "proposer"]
        critic_calls = [r for r in call_log if r == "critic"]
        assert len(proposer_calls) == 6
        assert len(critic_calls) == 5


# ---------------------------------------------------------------------------
# load_adapter tests
# ---------------------------------------------------------------------------

class TestLoadAdapter:

    @pytest.mark.unit
    def test_load_valid_adapter(self, tmp_path):
        """load_adapter discovers and instantiates a ProjectAdapter subclass."""
        adapter_file = tmp_path / "my_adapter.py"
        adapter_file.write_text(
            "from pathlib import Path\n"
            "from agentsciml.adapters.base import ProjectAdapter\n"
            "\n"
            "class TestAdapter(ProjectAdapter):\n"
            "    def __init__(self):\n"
            "        super().__init__(Path('/tmp'))\n"
            "    def get_context(self): return 'test'\n"
            "    def get_results_history(self): return ''\n"
            "    def get_current_experiment(self): return ''\n"
            "    def get_available_api(self): return ''\n"
            "    def get_metric_name(self): return 'x'\n"
            "    def get_result_metric_key(self): return 'x'\n"
            "    def parse_score(self, result_lines): return 0.0\n"
        )
        adapter = Orchestrator.load_adapter(str(adapter_file))
        assert isinstance(adapter, ProjectAdapter)
        assert adapter.get_context() == "test"

    @pytest.mark.unit
    def test_load_nonexistent_file_raises(self):
        with pytest.raises((ImportError, FileNotFoundError)):
            Orchestrator.load_adapter("/nonexistent/path/adapter.py")

    @pytest.mark.unit
    def test_load_file_without_adapter_raises(self, tmp_path):
        """A module with no ProjectAdapter subclass raises AttributeError."""
        no_adapter = tmp_path / "empty.py"
        no_adapter.write_text("x = 42\n")
        with pytest.raises(AttributeError, match="No ProjectAdapter subclass"):
            Orchestrator.load_adapter(str(no_adapter))
