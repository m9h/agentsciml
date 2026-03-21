"""Tests for protocol models."""

import json

from agentsciml.protocols import (
    AnalysisReport,
    CodePatch,
    CriticReport,
    MutationProposal,
    ScoredResult,
    SelectionVote,
    SolutionRecord,
    TechniqueCard,
)


def test_analysis_report_roundtrip():
    report = AnalysisReport(
        summary="Good progress on SK model",
        best_score=0.05,
        best_config="SK N=10 T=0.5 Gamma=1.0",
        worst_configs=["chain N=4"],
        unexplored=["scale-free topology", "VQE solver"],
        n_experiments=50,
    )
    data = json.loads(report.model_dump_json())
    restored = AnalysisReport.model_validate(data)
    assert restored.best_score == 0.05
    assert len(restored.unexplored) == 2


def test_technique_card():
    card = TechniqueCard(
        name="Multi-start VQE",
        category="optimization",
        description="Run VQE from multiple starts",
        applicable_when="VQE stuck in local minima",
        code_pattern="results = [run_vqe(...) for _ in range(5)]",
        tags=["vqe", "optimization"],
    )
    assert card.name == "Multi-start VQE"
    assert "vqe" in card.tags


def test_mutation_proposal():
    proposal = MutationProposal(
        strategy="Increase frustration via EA bimodal",
        changes=["Switch from SK to EA bimodal", "Use square lattice"],
        expected_impact="Higher frustration should increase quantum advantage",
    )
    assert proposal.technique_used is None
    assert len(proposal.changes) == 2


def test_critic_report():
    report = CriticReport(
        flaws=["N=4 is too small for meaningful frustration"],
        suggestions=["Use N=12 or larger"],
        feasibility="feasible",
    )
    assert report.feasibility == "feasible"


def test_code_patch():
    patch = CodePatch(
        code='print("hello")',
        description="Test experiment",
    )
    assert "hello" in patch.code


def test_scored_result():
    result = ScoredResult(
        score=0.042,
        comparison="Better than parent (0.01)",
        insights="EA bimodal at T=0.8 shows quantum advantage",
        status="ok",
    )
    assert result.score == 0.042


def test_selection_vote():
    vote = SelectionVote(
        exploit_pick="abc123",
        explore_picks=["def456", "ghi789"],
        reasoning="abc123 has highest score, def456 is underexplored",
    )
    assert len(vote.explore_picks) == 2


def test_solution_record():
    record = SolutionRecord(
        id="test123",
        parent_id=None,
        generation=0,
        code='print("root")',
        score=0.0,
        status="ok",
    )
    data = json.loads(record.model_dump_json())
    restored = SolutionRecord.model_validate(data)
    assert restored.id == "test123"
    assert restored.parent_id is None
