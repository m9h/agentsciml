"""Solution tree: tracks the evolutionary search over experiment mutations.

Each node represents a complete experiment.py variant that was executed and scored.
Parent selection balances exploitation (best score) with exploration (ensemble vote).
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocols import SolutionRecord


class SolutionTree:
    """Append-only tree of solution nodes with parent selection."""

    MAX_CHILDREN = 10

    def __init__(self, path: Path | None = None) -> None:
        self._nodes: dict[str, SolutionRecord] = {}
        self._children: dict[str, list[str]] = {}  # parent_id -> child ids
        self._path = path
        if path and path.exists():
            self._load(path)

    @property
    def nodes(self) -> dict[str, SolutionRecord]:
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def add(
        self,
        code: str,
        score: float,
        *,
        parent_id: str | None = None,
        generation: int = 0,
        mutation_description: str = "",
        technique_used: str | None = None,
        status: str = "ok",
        wall_time: float = 0.0,
        llm_cost: float = 0.0,
        agent_reports: dict[str, str] | None = None,
    ) -> SolutionRecord:
        """Add a scored solution to the tree."""
        node_id = uuid.uuid4().hex[:12]
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        node = SolutionRecord(
            id=node_id,
            parent_id=parent_id,
            generation=generation,
            code=code,
            code_hash=code_hash,
            mutation_description=mutation_description,
            technique_used=technique_used,
            score=score,
            status=status,
            wall_time=wall_time,
            llm_cost=llm_cost,
            agent_reports=agent_reports or {},
        )
        self._nodes[node_id] = node
        if parent_id is not None:
            self._children.setdefault(parent_id, []).append(node_id)
        self._save()
        return node

    def best(self) -> SolutionRecord | None:
        """Return the node with the highest score (best = max for quantum_advantage)."""
        ok_nodes = [n for n in self._nodes.values() if n.status == "ok"]
        if not ok_nodes:
            return None
        return max(ok_nodes, key=lambda n: n.score)

    def children_of(self, node_id: str) -> list[SolutionRecord]:
        """Return all children of a node."""
        child_ids = self._children.get(node_id, [])
        return [self._nodes[cid] for cid in child_ids if cid in self._nodes]

    def can_mutate(self, node_id: str) -> bool:
        """Check if a node hasn't reached its child limit."""
        return len(self._children.get(node_id, [])) < self.MAX_CHILDREN

    def select_parents(
        self,
        n: int = 2,
        p_exploit: float = 0.7,
        rng: random.Random | None = None,
    ) -> list[SolutionRecord]:
        """Select parents for the next generation.

        Always includes the best-scoring node (exploitation).
        Additional parents chosen by weighted random from eligible nodes (exploration).
        """
        rng = rng or random.Random()
        eligible = [
            n for n in self._nodes.values()
            if n.status == "ok" and self.can_mutate(n.id)
        ]
        if not eligible:
            return []

        # Exploitation: always include the best
        best = max(eligible, key=lambda n: n.score)
        parents = [best]
        remaining = [n for n in eligible if n.id != best.id]

        # Exploration: random from remaining
        needed = min(n - 1, len(remaining))
        if needed > 0:
            parents.extend(rng.sample(remaining, needed))

        return parents

    def top_k(self, k: int = 5) -> list[SolutionRecord]:
        """Return top-k scoring nodes for selector ensemble context."""
        ok_nodes = [n for n in self._nodes.values() if n.status == "ok"]
        ok_nodes.sort(key=lambda n: n.score, reverse=True)
        return ok_nodes[:k]

    def summary(self) -> dict[str, Any]:
        """Return a summary of the tree state."""
        ok_nodes = [n for n in self._nodes.values() if n.status == "ok"]
        crashed = [n for n in self._nodes.values() if n.status == "crash"]
        best = self.best()
        return {
            "total_nodes": len(self._nodes),
            "ok_nodes": len(ok_nodes),
            "crashed_nodes": len(crashed),
            "best_score": best.score if best else None,
            "best_id": best.id if best else None,
            "generations": max((n.generation for n in self._nodes.values()), default=0),
            "total_llm_cost": sum(n.llm_cost for n in self._nodes.values()),
        }

    def _save(self) -> None:
        if self._path is None:
            return
        data = [node.model_dump() for node in self._nodes.values()]
        self._path.write_text(json.dumps(data, indent=2))

    def _load(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        for entry in raw:
            node = SolutionRecord.model_validate(entry)
            self._nodes[node.id] = node
            if node.parent_id is not None:
                self._children.setdefault(node.parent_id, []).append(node.id)
