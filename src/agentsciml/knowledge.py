"""Knowledge base: curated SciML technique entries loaded from YAML.

The knowledge base is small enough (<50 entries per project domain) that
the Retriever agent receives the full list and selects 0-1 per mutation.
No vector DB needed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .protocols import TechniqueCard


def load_techniques(path: Path) -> list[TechniqueCard]:
    """Load technique entries from a YAML file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list in {path}, got {type(raw).__name__}")
    return [TechniqueCard.model_validate(entry) for entry in raw]


def format_techniques_for_prompt(techniques: list[TechniqueCard]) -> str:
    """Format all techniques as text for inclusion in agent prompts."""
    lines = []
    for i, t in enumerate(techniques, 1):
        lines.append(f"[{i}] {t.name} ({t.category})")
        lines.append(f"    {t.description}")
        lines.append(f"    Use when: {t.applicable_when}")
        if t.code_pattern:
            lines.append(f"    Pattern: {t.code_pattern}")
        lines.append("")
    return "\n".join(lines)
