"""Orchestrator: the evolutionary multi-agent loop.

Implements the three phases from AgenticSciML:
  Phase 1 (Init): Load project context, knowledge base
  Phase 2 (Root): Generate and evaluate initial experiment
  Phase 3 (Evolution): Iterative tree expansion with structured debate

The orchestrator is a plain Python loop — no framework needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
import importlib.util
import sys

import anthropic

from .adapters.base import ProjectAdapter
from .agents import (
    call_agent,
    extract_code,
    parse_json_response,
)
from .cost import CostTracker
from .knowledge import format_techniques_for_prompt, load_techniques
from .protocols import (
    AnalysisReport,
    MutationProposal,
    SolutionRecord,
    TechniqueCard,
)
from .sandbox import run_experiment
from .tree import SolutionTree

logger = logging.getLogger(__name__)

DEFAULT_DEBATE_ROUNDS = 4
MAX_DEBUG_RETRIES = 3
DEFAULT_KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"


class Orchestrator:
    """Multi-agent evolutionary search for SciML solutions."""

    @staticmethod
    def load_adapter(adapter_path: str) -> ProjectAdapter:
        """Dynamically load a ProjectAdapter from a file path."""
        path = Path(adapter_path).resolve()
        spec = importlib.util.spec_from_file_location("external_adapter", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load adapter from {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["external_adapter"] = module
        spec.loader.exec_module(module)

        # Look for a subclass of ProjectAdapter in the module
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type)
                    and issubclass(obj, ProjectAdapter)
                    and obj is not ProjectAdapter):
                return obj()
        raise AttributeError(f"No ProjectAdapter subclass found in {adapter_path}")

    def __init__(
        self,
        adapter: ProjectAdapter,
        *,
        budget_usd: float = 5.0,
        max_generations: int = 20,
        debate_rounds: int = 4,
        knowledge_file: Path | None = None,
        tree_path: Path | None = None,
        slurm_config: dict | None = None,
    ) -> None:
        self.adapter = adapter
        self.client = anthropic.Anthropic()
        self.cost = CostTracker(budget_usd=budget_usd)
        self.max_generations = max_generations
        self.debate_rounds = debate_rounds

        # Solution tree
        tree_path = tree_path or (adapter.project_root / "autoresearch" / "tree.json")
        self.tree = SolutionTree(
            path=tree_path,
            direction=adapter.get_score_direction(),
        )

        # Knowledge base — auto-detect from adapter type if not specified
        kf = knowledge_file
        if kf is None:
            adapter_name = type(adapter).__name__.lower()
            if "dmipy" in adapter_name:
                kf = DEFAULT_KNOWLEDGE_DIR / "dmipy_techniques.yaml"
            elif "golf" in adapter_name or "parametergolf" in adapter_name:
                kf = DEFAULT_KNOWLEDGE_DIR / "parameter_golf_techniques.yaml"
            else:
                kf = DEFAULT_KNOWLEDGE_DIR / "qcccm_techniques.yaml"
        self.techniques: list[TechniqueCard] = []
        if kf.exists():
            self.techniques = load_techniques(kf)
            logger.info("Loaded %d techniques from %s", len(self.techniques), kf)

    def run(self) -> SolutionRecord | None:
        """Execute the full evolutionary loop. Returns the best solution."""
        logger.info("Starting AgenticSciML orchestrator")
        logger.info(
            "Budget: $%.2f | Max generations: %d",
            self.cost.budget_usd, self.max_generations,
        )

        # Phase 1: Init — load context
        context = self.adapter.get_context()
        api_surface = self.adapter.get_available_api()

        # Phase 2: Root solution
        if len(self.tree) == 0:
            logger.info("Phase 2: Generating root solution")
            root = self._generate_root(context, api_surface)
            if root is None:
                logger.error("Failed to generate root solution")
                return None
            logger.info("Root solution: score=%.6f status=%s", root.score, root.status)

        # Phase 3: Evolutionary tree expansion
        for gen in range(1, self.max_generations + 1):
            if not self.cost.within_budget:
                logger.info("Budget exhausted ($%.4f spent)", self.cost.estimated_cost_usd)
                break

            logger.info(
                "Generation %d/%d | Tree: %d nodes | Best: %.6f | Cost: $%.4f",
                gen,
                self.max_generations,
                len(self.tree),
                (self.tree.best().score if self.tree.best() else float("-inf")),
                self.cost.estimated_cost_usd,
            )

            parents = self.tree.select_parents(n=2)
            if not parents:
                logger.warning("No eligible parents — stopping")
                break

            for parent in parents:
                if not self.cost.within_budget:
                    break
                try:
                    self._mutate_parent(parent, gen, context, api_surface)
                except Exception:
                    logger.exception("Mutation failed for parent %s, continuing", parent.id)

        best = self.tree.best()
        logger.info("Done. Best score: %.6f", best.score if best else float("-inf"))
        logger.info("Cost summary: %s", self.cost.summary())
        return best

    def _generate_root(self, context: str, api_surface: str) -> SolutionRecord | None:
        """Phase 2: generate initial experiment from scratch or use existing."""
        existing_code = self.adapter.get_current_experiment()
        if existing_code.strip():
            # Use existing experiment.py as root — just run it
            logger.info("Using existing experiment.py as root")
            return self._execute_and_record(
                code=existing_code,
                parent_id=None,
                generation=0,
                mutation_description="existing experiment (root)",
            )

        # Generate from scratch via Engineer
        docs = {
            "project_context": context,
            "api_surface": api_surface,
            "instructions": (
                "Write a complete experiment.py that serves as an initial baseline. "
                "Start simple: compare classical Metropolis vs one quantum method "
                "on a small system. Follow the API surface exactly."
            ),
        }
        response = call_agent(
            "engineer", docs, client=self.client, cost_tracker=self.cost
        )
        code = extract_code(response)
        return self._execute_and_record(
            code=code,
            parent_id=None,
            generation=0,
            mutation_description="AI-generated root experiment",
        )

    def _mutate_parent(
        self,
        parent: SolutionRecord,
        generation: int,
        context: str,
        api_surface: str,
    ) -> SolutionRecord | None:
        """Run the full agent pipeline to mutate a parent into a child."""

        # Step 1: DataAnalyst — analyze results history (truncated to last 100 lines)
        results_history = self.adapter.get_results_history()
        lines = results_history.split("\n")
        if len(lines) > 101:  # header + 100 data lines
            results_history = "\n".join(lines[:1] + lines[-100:])
            results_history = f"[truncated to last 100 of {len(lines)-1} rows]\n" + results_history
        analysis_text = call_agent(
            "data_analyst",
            {"results_history": results_history, "metric_name": self.adapter.get_metric_name()},
            client=self.client,
            cost_tracker=self.cost,
        )
        try:
            analysis = parse_json_response(analysis_text, AnalysisReport)
        except Exception:
            logger.warning("Failed to parse DataAnalyst response, using fallback")
            analysis = AnalysisReport(
                summary="Analysis unavailable",
                best_score=parent.score,
                best_config=parent.mutation_description,
            )

        # Step 2: Retriever — select a technique
        technique_text = ""
        if self.techniques:
            kb_text = format_techniques_for_prompt(self.techniques)
            retriever_response = call_agent(
                "retriever",
                {"analysis": analysis.model_dump_json(), "knowledge_base": kb_text},
                client=self.client,
                cost_tracker=self.cost,
            )
            try:
                import json
                sel = json.loads(retriever_response.strip().strip("`").strip())
                idx = sel.get("selected")
                if idx is not None and 1 <= idx <= len(self.techniques):
                    technique = self.techniques[idx - 1]
                    technique_text = technique.model_dump_json()
            except Exception:
                logger.debug("Retriever returned no valid selection")

        # Step 3: Structured debate — Proposer + Critic
        constraints = self.adapter.get_constraints()
        proposal = self._run_debate(
            analysis=analysis,
            technique_text=technique_text,
            parent_code=parent.code,
            context=context,
            constraints=constraints,
        )

        # Step 4: Engineer — implement the proposal
        docs = {
            "project_context": context,
            "api_surface": api_surface,
            "parent_code": parent.code,
            "proposal": proposal.model_dump_json(),
            "instructions": (
                "Implement the proposed mutation. Write a complete experiment.py. "
                "Do not change the imports from prepare or the RESULT format."
            ),
        }
        engineer_response = call_agent(
            "engineer", docs, client=self.client, cost_tracker=self.cost
        )
        code = extract_code(engineer_response)

        # Step 5: Execute with debug retries
        result = self._execute_with_retries(
            code=code,
            parent=parent,
            generation=generation,
            proposal=proposal,
            api_surface=api_surface,
        )
        return result

    def _run_debate(
        self,
        analysis: AnalysisReport,
        technique_text: str,
        parent_code: str,
        context: str,
        constraints: str = "",
    ) -> MutationProposal:
        """Run N-round structured debate between Proposer and Critic."""
        debate_history = ""

        for round_num in range(1, self.debate_rounds + 1):
            if round_num <= self.debate_rounds - 2:
                # Reasoning rounds
                round_instruction = (
                    f"ROUND {round_num}/{self.debate_rounds} — REASONING ONLY.\n"
                    "Analyze the problem deeply. What patterns do you see in the results? "
                    "What parameter regimes or solver configurations could improve the metric? "
                    "DO NOT propose a strategy yet — just reason."
                )
            elif round_num == self.debate_rounds - 1:
                # Synthesis round
                round_instruction = (
                    f"ROUND {round_num}/{self.debate_rounds} — SYNTHESIS.\n"
                    "Based on your reasoning, synthesize a concrete implementation plan. "
                    "Output a JSON object with: strategy, changes, expected_impact, "
                    "technique_used (or null), code_outline."
                )
            else:
                # Finalization round
                round_instruction = (
                    f"ROUND {round_num}/{self.debate_rounds} — FINALIZATION.\n"
                    "Produce your final implementation-ready proposal as JSON. "
                    "Incorporate any valid critic feedback. Output ONLY the JSON."
                )

            # Proposer
            proposer_docs = {
                "analysis": analysis.model_dump_json(),
                "parent_code": parent_code,
                "debate_history": debate_history,
                "round_instruction": round_instruction,
            }
            if technique_text:
                proposer_docs["technique"] = technique_text
            if constraints:
                proposer_docs["constraints"] = constraints

            proposer_response = call_agent(
                "proposer", proposer_docs, client=self.client, cost_tracker=self.cost
            )
            debate_history += f"\n--- Proposer Round {round_num} ---\n{proposer_response}\n"

            # Critic (skip on final round)
            if round_num < self.debate_rounds:
                critic_docs = {
                    "analysis": analysis.model_dump_json(),
                    "proposer_output": proposer_response,
                    "round_instruction": (
                        f"Round {round_num}/{self.debate_rounds}."
                        " Challenge the proposer's reasoning."
                    ),
                }
                if constraints:
                    critic_docs["constraints"] = constraints
                critic_response = call_agent(
                    "critic", critic_docs, client=self.client, cost_tracker=self.cost
                )
                debate_history += f"\n--- Critic Round {round_num} ---\n{critic_response}\n"

        # Parse the final proposer response
        try:
            proposal = parse_json_response(proposer_response, MutationProposal)
        except Exception:
            logger.warning("Failed to parse proposal, using fallback")
            proposal = MutationProposal(
                strategy="Vary parameters based on analysis",
                changes=["Modify experiment parameters"],
                expected_impact="Explore new region of parameter space",
            )

        return proposal

    def _execute_with_retries(
        self,
        code: str,
        parent: SolutionRecord,
        generation: int,
        proposal: MutationProposal,
        api_surface: str,
    ) -> SolutionRecord | None:
        """Execute code, debug on crash, retry up to MAX_DEBUG_RETRIES."""
        for attempt in range(MAX_DEBUG_RETRIES + 1):
            result = self._execute_and_record(
                code=code,
                parent_id=parent.id,
                generation=generation,
                mutation_description=proposal.strategy,
                technique_used=proposal.technique_used,
            )
            if result is None:
                return None

            if result.status == "ok":
                return result

            if attempt >= MAX_DEBUG_RETRIES:
                logger.warning("Max debug retries reached for generation %d", generation)
                return result

            # Debug: ask Debugger to fix
            logger.info("Debug attempt %d/%d", attempt + 1, MAX_DEBUG_RETRIES)
            debug_docs = {
                "crashed_code": code,
                "stderr": result.agent_reports.get("stderr", "")[:2000],
                "api_surface": api_surface,
            }
            debug_response = call_agent(
                "debugger", debug_docs, client=self.client, cost_tracker=self.cost
            )
            code = extract_code(debug_response)

        return None

    def _execute_and_record(
        self,
        code: str,
        parent_id: str | None,
        generation: int,
        mutation_description: str,
        technique_used: str | None = None,
    ) -> SolutionRecord | None:
        """Execute code in sandbox, parse results, add to tree."""
        exec_result = run_experiment(code, self.adapter.project_root)

        score = 0.0
        if exec_result.status == "ok":
            score = self.adapter.parse_score(exec_result.result_lines)
            # Sanity check: quantum_advantage > 1.0 is physically suspicious
            if abs(score) > 1.0:
                logger.warning(
                    "Suspicious score %.4f (|score| > 1.0) — likely a bug in generated code. "
                    "Clamping to 0.0.",
                    score,
                )
                score = 0.0
                exec_result = exec_result.__class__(
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    returncode=exec_result.returncode,
                    wall_time=exec_result.wall_time,
                    result_lines=exec_result.result_lines,
                    status="suspicious",
                )

        agent_reports = {
            "stdout_preview": exec_result.stdout[:1000],
            "stderr": exec_result.stderr[:1000],
            "result_lines": "\n".join(exec_result.result_lines[:20]),
        }

        node = self.tree.add(
            code=code,
            score=score,
            parent_id=parent_id,
            generation=generation,
            mutation_description=mutation_description,
            technique_used=technique_used,
            status=exec_result.status,
            wall_time=exec_result.wall_time,
            llm_cost=self.cost.estimated_cost_usd,
            agent_reports=agent_reports,
        )
        return node
