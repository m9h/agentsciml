"""Tests for knowledge base loading."""

from agentsciml.knowledge import format_techniques_for_prompt, load_techniques


def test_load_qcccm_techniques(knowledge_dir):
    path = knowledge_dir / "qcccm_techniques.yaml"
    techniques = load_techniques(path)
    assert len(techniques) >= 10
    assert all(t.name for t in techniques)
    assert all(t.category for t in techniques)
    assert all(t.description for t in techniques)


def test_format_techniques(knowledge_dir):
    path = knowledge_dir / "qcccm_techniques.yaml"
    techniques = load_techniques(path)
    text = format_techniques_for_prompt(techniques)
    assert "[1]" in text
    assert "Multi-start VQE" in text
    assert "Use when:" in text
