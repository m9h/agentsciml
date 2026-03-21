"""Pydantic models for inter-agent document passing.

Each agent produces and consumes typed documents. The orchestrator assembles
relevant documents into each agent's prompt context — no chat history accumulation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AnalysisReport(BaseModel):
    """Produced by DataAnalyst: summary of experimental history."""

    summary: str = Field(description="High-level patterns in results so far")
    best_score: float = Field(description="Best primary metric value observed")
    best_config: str = Field(description="Description of best-scoring configuration")
    worst_configs: list[str] = Field(
        default_factory=list,
        description="Configurations that performed worst (avoid repeating)",
    )
    unexplored: list[str] = Field(
        default_factory=list,
        description="Regions of parameter space not yet tried",
    )
    n_experiments: int = Field(default=0, description="Total experiments run so far")


class TechniqueCard(BaseModel):
    """A curated SciML technique from the knowledge base."""

    name: str
    category: str
    description: str
    applicable_when: str
    code_pattern: str = Field(default="", description="Example code snippet")
    tags: list[str] = Field(default_factory=list)


class MutationProposal(BaseModel):
    """Produced by Proposer after structured debate: what to change and why."""

    strategy: str = Field(description="High-level approach for this mutation")
    changes: list[str] = Field(description="Specific changes to make to parent code")
    expected_impact: str = Field(description="Why this should improve the score")
    technique_used: str | None = Field(
        default=None, description="Knowledge base technique applied, if any"
    )
    code_outline: str = Field(
        default="",
        description="Pseudocode or outline of the key implementation change",
    )


class CriticReport(BaseModel):
    """Produced by Critic: challenges and refinements to a proposal."""

    flaws: list[str] = Field(
        default_factory=list, description="Issues found in the proposal"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Alternative approaches or fixes"
    )
    feasibility: str = Field(
        default="feasible",
        description="Assessment: feasible, risky, or infeasible",
    )
    revised_strategy: str = Field(
        default="",
        description="Refined strategy incorporating critique (if needed)",
    )


class CodePatch(BaseModel):
    """Produced by Engineer: the complete experiment.py content."""

    code: str = Field(description="Full experiment.py file content")
    description: str = Field(description="Short description of what changed")


class ResultLine(BaseModel):
    """A single parsed RESULT| line from experiment output."""

    fields: dict[str, Any] = Field(default_factory=dict)


class ScoredResult(BaseModel):
    """Produced by ResultAnalyst: evaluation of an experiment run."""

    score: float = Field(description="Primary metric value")
    result_lines: list[ResultLine] = Field(default_factory=list)
    comparison: str = Field(
        default="", description="How this compares to parent and best-ever"
    )
    insights: str = Field(
        default="", description="What we learned from this experiment"
    )
    status: str = Field(default="ok", description="ok, crash, or timeout")
    stderr_snippet: str = Field(
        default="", description="First 500 chars of stderr if error"
    )


class SelectionVote(BaseModel):
    """Produced by one member of the SelectorEnsemble."""

    exploit_pick: str = Field(description="Node ID of best-scoring solution to exploit")
    explore_picks: list[str] = Field(
        default_factory=list,
        description="Node IDs of promising or underexplored solutions",
    )
    reasoning: str = Field(default="", description="Why these were chosen")


class SolutionRecord(BaseModel):
    """Serializable record of a solution node in the evolutionary tree."""

    id: str
    parent_id: str | None = None
    generation: int = 0
    code: str = ""
    code_hash: str = ""
    mutation_description: str = ""
    technique_used: str | None = None
    score: float = float("inf")
    status: str = "pending"
    wall_time: float = 0.0
    llm_cost: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent_reports: dict[str, str] = Field(default_factory=dict)
