"""Tests for the solution tree."""

import random

from agentsciml.tree import SolutionTree


def test_add_and_best(tmp_tree_path):
    tree = SolutionTree(path=tmp_tree_path)

    n1 = tree.add(code="print(1)", score=0.1, mutation_description="first")
    n2 = tree.add(
        code="print(2)", score=0.5, parent_id=n1.id,
        generation=1, mutation_description="second",
    )
    tree.add(
        code="print(3)", score=0.3, parent_id=n1.id,
        generation=1, mutation_description="third",
    )

    assert len(tree) == 3
    assert tree.best().id == n2.id
    assert tree.best().score == 0.5


def test_children_and_max(tmp_tree_path):
    tree = SolutionTree(path=tmp_tree_path)
    root = tree.add(code="root", score=0.0, mutation_description="root")

    for i in range(SolutionTree.MAX_CHILDREN):
        tree.add(code=f"child_{i}", score=float(i), parent_id=root.id, generation=1)

    assert len(tree.children_of(root.id)) == SolutionTree.MAX_CHILDREN
    assert not tree.can_mutate(root.id)


def test_select_parents(tmp_tree_path):
    tree = SolutionTree(path=tmp_tree_path)
    tree.add(code="a", score=0.1, mutation_description="a")
    n2 = tree.add(code="b", score=0.9, mutation_description="b")
    tree.add(code="c", score=0.5, mutation_description="c")

    rng = random.Random(42)
    parents = tree.select_parents(n=2, rng=rng)

    # Best (n2) should always be first
    assert parents[0].id == n2.id
    assert len(parents) == 2


def test_persistence(tmp_tree_path):
    tree1 = SolutionTree(path=tmp_tree_path)
    tree1.add(code="persist me", score=1.23, mutation_description="test")

    # Load from same path
    tree2 = SolutionTree(path=tmp_tree_path)
    assert len(tree2) == 1
    assert tree2.best().score == 1.23


def test_top_k(tmp_tree_path):
    tree = SolutionTree(path=tmp_tree_path)
    for i in range(10):
        tree.add(code=f"n{i}", score=float(i), mutation_description=f"n{i}")

    top = tree.top_k(3)
    assert len(top) == 3
    assert top[0].score == 9.0
    assert top[1].score == 8.0
    assert top[2].score == 7.0


def test_summary(tmp_tree_path):
    tree = SolutionTree(path=tmp_tree_path)
    tree.add(code="ok", score=0.5, status="ok", llm_cost=0.01)
    tree.add(code="crash", score=0.0, status="crash", llm_cost=0.02)

    s = tree.summary()
    assert s["total_nodes"] == 2
    assert s["ok_nodes"] == 1
    assert s["crashed_nodes"] == 1
    assert s["total_llm_cost"] == 0.03


def test_empty_tree(tmp_tree_path):
    tree = SolutionTree(path=tmp_tree_path)
    assert len(tree) == 0
    assert tree.best() is None
    assert tree.select_parents() == []
    assert tree.top_k(5) == []


def test_minimize_direction(tmp_tree_path):
    """Tree with direction='minimize' selects lowest score as best."""
    tree = SolutionTree(path=tmp_tree_path, direction="minimize")
    tree.add(code="a", score=1.5, mutation_description="a")
    n2 = tree.add(code="b", score=0.8, mutation_description="b")
    tree.add(code="c", score=1.2, mutation_description="c")

    assert tree.best().id == n2.id
    assert tree.best().score == 0.8

    top = tree.top_k(2)
    assert top[0].score == 0.8
    assert top[1].score == 1.2


def test_minimize_select_parents(tmp_tree_path):
    """select_parents in minimize mode picks lowest-scoring node first."""
    tree = SolutionTree(path=tmp_tree_path, direction="minimize")
    tree.add(code="a", score=5.0, mutation_description="a")
    n2 = tree.add(code="b", score=1.0, mutation_description="b")
    tree.add(code="c", score=3.0, mutation_description="c")

    rng = random.Random(42)
    parents = tree.select_parents(n=2, rng=rng)
    assert parents[0].id == n2.id


def test_invalid_direction(tmp_tree_path):
    """Invalid direction raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="direction"):
        SolutionTree(path=tmp_tree_path, direction="invalid")


def test_direction_property(tmp_tree_path):
    tree_max = SolutionTree(path=tmp_tree_path, direction="maximize")
    assert tree_max.direction == "maximize"

    tree_min = SolutionTree(direction="minimize")
    assert tree_min.direction == "minimize"
