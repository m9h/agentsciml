"""Tests for the 'Retry & Refine' recovery protocol.
Follows Red-Green TDD to ensure robustness of infrastructure failure handling.
"""

import pytest
from pathlib import Path
from agentsciml.tree import SolutionTree
from agentsciml.protocols import SolutionRecord

def test_select_parents_picks_crash_node_when_no_ok_nodes():
    """RED: Assert that if only crashes exist, we still pick a parent."""
    tree = SolutionTree(direction="maximize")
    
    # Add a crashed root node
    tree.add(
        code="print('failed')",
        score=0.0,
        parent_id=None,
        generation=0,
        mutation_description="crashed root",
        status="crash"
    )
    
    # Act: Select parents for generation 1
    parents = tree.select_parents(n=1)
    
    # Assert
    assert len(parents) == 1
    assert parents[0].status == "crash"
    assert parents[0].mutation_description == "crashed root"

def test_select_parents_prefers_ok_over_crash():
    """GREENish: Assert that if both exist, we prefer the successful one."""
    tree = SolutionTree(direction="maximize")
    
    # Add a crashed node
    tree.add(
        code="print('failed')",
        score=0.0,
        parent_id=None,
        generation=0,
        mutation_description="crashed",
        status="crash"
    )
    
    # Add an OK node with low score
    tree.add(
        code="print('low score')",
        score=10.0,
        parent_id=None,
        generation=0,
        mutation_description="ok node",
        status="ok"
    )
    
    # Act
    parents = tree.select_parents(n=1)
    
    # Assert
    assert len(parents) == 1
    assert parents[0].status == "ok"
    assert parents[0].score == 10.0

def test_orchestrator_diagnoses_crashes():
    """RED: Assert that the orchestrator adds a diagnosis to crashed nodes.
    This feature is NOT implemented yet, so this test should fail.
    """
    from agentsciml.orchestrator import Orchestrator
    from agentsciml.adapters.base import ProjectAdapter
    from unittest.mock import MagicMock, patch
    
    # Setup mock adapter
    adapter = MagicMock(spec=ProjectAdapter)
    adapter.project_root = Path("/tmp/fake_project")
    adapter.get_score_direction.return_value = "maximize"
    adapter.get_context.return_value = "context"
    adapter.get_available_api.return_value = "api"
    
    orch = Orchestrator(adapter)
    orch.tree = MagicMock(spec=SolutionTree)
    
    # Mock tree.add to return a SolutionRecord
    def mock_add(**kwargs):
        return SolutionRecord(id="test_node", **kwargs)
    orch.tree.add.side_effect = mock_add
    
    # Mock run_experiment to return a crash
    with patch("agentsciml.orchestrator.run_experiment") as mock_run, \
         patch("agentsciml.orchestrator.call_agent") as mock_agent:
        
        from agentsciml.sandbox import ExecutionResult
        mock_run.return_value = ExecutionResult(
            stdout="some output",
            stderr="Traceback: Out of Memory",
            returncode=1,
            wall_time=1.0,
            result_lines=[],
            status="crash"
        )
        
        # Mock diagnostician response
        mock_agent.return_value = '{"category": "infrastructure", "reason": "GPU Out of Memory", "suggested_fix": "reduce batch size"}'
        
        # Execute and record
        node = orch._execute_and_record(
            code="print('oom')",
            parent_id=None,
            generation=0,
            mutation_description="oom test"
        )
        
        # Assert: Expect a diagnosis in the reports
        assert "crash_diagnosis" in node.agent_reports
        assert "Memory" in node.agent_reports["crash_diagnosis"]
        assert "[INFRASTRUCTURE]" in node.agent_reports["crash_diagnosis"]
