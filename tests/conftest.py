"""Shared test fixtures for agentsciml."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_tree_path(tmp_path: Path) -> Path:
    return tmp_path / "tree.json"


@pytest.fixture
def knowledge_dir() -> Path:
    return Path(__file__).parent.parent / "knowledge"
