"""Tests for agents.py — agent registry, call_agent, parse_json_response, extract_code.

TDD: these tests define the expected behavior of the agent system.
Run first to confirm RED, then fix any issues to reach GREEN.
"""

from __future__ import annotations

import json

import pytest

from agentsciml.agents import (
    HAIKU,
    REGISTRY,
    SONNET,
    AgentConfig,
    call_agent,
    extract_code,
    parse_json_response,
)
from agentsciml.protocols import AnalysisReport, MutationProposal

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Verify all 8 agent roles are registered with correct config."""

    EXPECTED_ROLES = [
        "data_analyst", "retriever", "proposer", "critic",
        "engineer", "debugger", "result_analyst", "selector",
    ]

    @pytest.mark.unit
    def test_all_roles_registered(self):
        for role in self.EXPECTED_ROLES:
            assert role in REGISTRY, f"Missing agent role: {role}"

    @pytest.mark.unit
    def test_no_extra_roles(self):
        assert set(REGISTRY.keys()) == set(self.EXPECTED_ROLES)

    @pytest.mark.unit
    def test_haiku_roles(self):
        """Most roles should use cheap Haiku."""
        haiku_roles = {
            "data_analyst", "retriever", "critic", "debugger",
            "result_analyst", "selector",
        }
        for role in haiku_roles:
            assert REGISTRY[role].model == HAIKU, f"{role} should use Haiku"

    @pytest.mark.unit
    def test_sonnet_roles(self):
        """Creative roles use Sonnet."""
        sonnet_roles = {"proposer", "engineer"}
        for role in sonnet_roles:
            assert REGISTRY[role].model == SONNET, f"{role} should use Sonnet"

    @pytest.mark.unit
    def test_agent_config_fields(self):
        for role, config in REGISTRY.items():
            assert isinstance(config, AgentConfig)
            assert config.role == role
            assert config.max_tokens > 0
            assert 0.0 <= config.temperature <= 1.0
            assert len(config.system_prompt) > 50, f"{role} system prompt too short"


# ---------------------------------------------------------------------------
# call_agent tests
# ---------------------------------------------------------------------------

class TestCallAgent:
    """Test call_agent with mocked Anthropic client."""

    @pytest.mark.unit
    def test_basic_call(self, mock_anthropic_client):
        response_text = '{"summary": "test", "best_score": 0.5}'
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response(response_text)
        )

        result = call_agent(
            "data_analyst",
            {"results_history": "some data", "metric_name": "advantage"},
            client=mock_anthropic_client,
        )
        assert result == response_text
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.unit
    def test_documents_assembled_as_xml(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response("ok")
        )

        call_agent(
            "data_analyst",
            {"doc_a": "content_a", "doc_b": "content_b"},
            client=mock_anthropic_client,
        )

        call_args = mock_anthropic_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "<doc_a>" in user_msg
        assert "content_a" in user_msg
        assert "<doc_b>" in user_msg
        assert "content_b" in user_msg

    @pytest.mark.unit
    def test_model_from_registry(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response("ok")
        )

        call_agent("data_analyst", {"x": "y"}, client=mock_anthropic_client)
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["model"] == HAIKU

    @pytest.mark.unit
    def test_model_override(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response("ok")
        )

        call_agent(
            "data_analyst", {"x": "y"},
            client=mock_anthropic_client,
            model_override=SONNET,
        )
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["model"] == SONNET

    @pytest.mark.unit
    def test_cost_tracking(self, mock_anthropic_client, cost_tracker):
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response("ok", input_tokens=1000, output_tokens=500)
        )

        call_agent(
            "data_analyst", {"x": "y"},
            client=mock_anthropic_client,
            cost_tracker=cost_tracker,
        )
        assert cost_tracker.total_input_tokens == 1000
        assert cost_tracker.total_output_tokens == 500
        assert cost_tracker.total_calls == 1

    @pytest.mark.unit
    def test_invalid_role_raises(self, mock_anthropic_client):
        with pytest.raises(KeyError):
            call_agent("nonexistent_role", {}, client=mock_anthropic_client)

    @pytest.mark.unit
    def test_system_prompt_passed(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response("ok")
        )

        call_agent("engineer", {"x": "y"}, client=mock_anthropic_client)
        call_args = mock_anthropic_client.messages.create.call_args
        assert "Engineer" in call_args.kwargs["system"]

    @pytest.mark.unit
    def test_max_tokens_and_temperature_passed(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = (
            mock_anthropic_client._make_response("ok")
        )

        call_agent("engineer", {"x": "y"}, client=mock_anthropic_client)
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == REGISTRY["engineer"].max_tokens
        assert call_args.kwargs["temperature"] == REGISTRY["engineer"].temperature


# ---------------------------------------------------------------------------
# parse_json_response tests
# ---------------------------------------------------------------------------

class TestParseJsonResponse:

    @pytest.mark.unit
    def test_plain_json(self):
        text = '{"summary": "test", "best_score": 1.5, "best_config": "SK N=10"}'
        result = parse_json_response(text, AnalysisReport)
        assert result.summary == "test"
        assert result.best_score == 1.5

    @pytest.mark.unit
    def test_markdown_wrapped_json(self):
        text = '```json\n{"summary": "test", "best_score": 0.5, "best_config": "x"}\n```'
        result = parse_json_response(text, AnalysisReport)
        assert result.summary == "test"

    @pytest.mark.unit
    def test_mutation_proposal(self):
        data = {
            "strategy": "Increase Trotter steps",
            "changes": ["n_trotter 8->16"],
            "expected_impact": "Better convergence",
            "technique_used": "Trotter optimization",
            "code_outline": "for n in [8,16,32]: ...",
        }
        result = parse_json_response(json.dumps(data), MutationProposal)
        assert result.strategy == "Increase Trotter steps"
        assert len(result.changes) == 1

    @pytest.mark.unit
    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            parse_json_response("not json at all", AnalysisReport)

    @pytest.mark.unit
    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            parse_json_response('{"summary": "only summary"}', AnalysisReport)


# ---------------------------------------------------------------------------
# extract_code tests
# ---------------------------------------------------------------------------

class TestExtractCode:

    @pytest.mark.unit
    def test_python_code_block(self):
        text = 'Here is the code:\n```python\nprint("hello")\n```\nDone.'
        assert extract_code(text) == 'print("hello")'

    @pytest.mark.unit
    def test_multiple_code_blocks(self):
        text = '```python\nline1\n```\ntext\n```python\nline2\n```'
        code = extract_code(text)
        assert "line1" in code
        assert "line2" in code

    @pytest.mark.unit
    def test_no_code_block_fallback(self):
        text = "print('fallback')"
        assert extract_code(text) == text.strip()

    @pytest.mark.unit
    def test_preserves_indentation(self):
        text = '```python\ndef foo():\n    return 42\n```'
        code = extract_code(text)
        assert "    return 42" in code
