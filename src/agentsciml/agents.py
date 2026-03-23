"""Agent definitions and registry.

Each agent is a role with a system prompt, model tier, and expected I/O document
types. call_agent() is the single entry point for invoking any agent via the
Anthropic API.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from .cost import CostTracker

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Model IDs — swarm-optimized: ~80% haiku, ~20% sonnet, opus only on escalation
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"


@dataclass
class AgentConfig:
    """Configuration for a single agent role."""

    role: str
    model: str
    system_prompt: str
    max_tokens: int = 4096
    temperature: float = 0.7


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_DATA_ANALYST_PROMPT = """\
You are a Data Analyst for a scientific machine learning research system.

Your job: analyze the experimental results history and produce a structured report
identifying patterns, best/worst configurations, and unexplored regions.

Output a JSON object with these fields:
- summary: string — high-level patterns (2-3 sentences)
- best_score: number — best primary metric value
- best_config: string — description of the best configuration
- worst_configs: list of strings — configurations that performed poorly
- unexplored: list of strings — parameter regions not yet explored
- n_experiments: integer — total experiments in history
"""

_RETRIEVER_PROMPT = """\
You are a Retriever for a scientific ML research system.

Given an analysis report and a list of available techniques, select 0 or 1 technique
that would be most helpful for the next experiment mutation.

If no technique is relevant, return: {"selected": null}
If a technique is relevant, return: {"selected": <technique_number>}

Only select a technique if it directly addresses a weakness or unexplored direction
identified in the analysis report.
"""

_PROPOSER_PROMPT = """\
You are a Proposer in a structured debate for scientific ML research.

You will receive: the current analysis of results, optionally a technique card,
and the parent experiment code. Your job is to propose a concrete mutation.

DEBATE PROTOCOL:
- In early rounds: analyze the problem deeply. Think about what parameter regimes,
  solver configurations, or model choices could improve the primary metric.
  DO NOT propose a strategy yet — just reason.
- In the synthesis round: produce a concrete implementation plan.
- In the final round: produce a finalized proposal as JSON with these fields:
  - strategy: string — high-level approach
  - changes: list of strings — specific changes to make
  - expected_impact: string — why this should improve the score
  - technique_used: string or null — knowledge base technique name if used
  - code_outline: string — pseudocode of the key change
"""

_CRITIC_PROMPT = """\
You are a Critic in a structured debate for scientific ML research.

Your job: challenge the Proposer's reasoning. Point out flaws, gaps, or alternatives.
Be constructive but rigorous.

Output a JSON object with:
- flaws: list of strings — issues with the proposal
- suggestions: list of strings — alternative approaches or fixes
- feasibility: "feasible", "risky", or "infeasible"
- revised_strategy: string — refined strategy if needed (empty if proposal is good)
"""

_ENGINEER_PROMPT = """\
You are an Engineer for a scientific ML research system.

Your job: implement the proposed mutation by writing a complete, working experiment.py
file. You will receive the parent code and the finalized proposal (with critic feedback).

RULES:
- Output ONLY valid Python code for experiment.py
- Import from prepare.py exactly as the parent code does
- The code must define run_experiment() and call it in if __name__ == "__main__"
- Each result must call print_result() and log_result() from prepare
- Keep the code self-contained — no external files or new dependencies
- Do not modify prepare.py imports or the RESULT| format

Wrap your code in ```python ... ``` markers.
"""

_DEBUGGER_PROMPT = """\
You are a Debugger for a scientific ML research system.

You will receive experiment code that crashed, along with the stderr output.
Fix the code and return the complete corrected experiment.py.

Common issues:
- Import errors (wrong function names from prepare.py or qcccm)
- Shape mismatches in numpy/jax arrays
- Missing parameters in solver calls
- Off-by-one errors in loops

Wrap your corrected code in ```python ... ``` markers.
"""

_RESULT_ANALYST_PROMPT = """\
You are a Result Analyst for a scientific ML research system.

You will receive the stdout from an experiment run (containing RESULT| lines)
and the prior analysis report. Evaluate how this experiment performed.

Output a JSON object with:
- score: number — the primary metric value (best quantum_advantage from RESULT lines)
- comparison: string — how this compares to the previous best
- insights: string — what we learned (1-2 sentences)
- status: "ok", "crash", or "timeout"
"""

_SELECTOR_PROMPT = """\
You are a Selector in an ensemble voting system for scientific ML research.

You will receive summaries of the top solutions in the evolutionary tree.
Vote on which solutions should be mutated next.

Output a JSON object with:
- exploit_pick: string — node ID of the best solution to exploit further
- explore_picks: list of strings — 1-2 node IDs of promising or underexplored solutions
- reasoning: string — brief justification
"""


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, AgentConfig] = {
    "data_analyst": AgentConfig(
        role="data_analyst",
        model=HAIKU,
        system_prompt=_DATA_ANALYST_PROMPT,
        max_tokens=2048,
    ),
    "retriever": AgentConfig(
        role="retriever",
        model=HAIKU,
        system_prompt=_RETRIEVER_PROMPT,
        max_tokens=512,
    ),
    "proposer": AgentConfig(
        role="proposer",
        model=SONNET,
        system_prompt=_PROPOSER_PROMPT,
        max_tokens=4096,
        temperature=0.8,
    ),
    "critic": AgentConfig(
        role="critic",
        model=HAIKU,
        system_prompt=_CRITIC_PROMPT,
        max_tokens=2048,
    ),
    "engineer": AgentConfig(
        role="engineer",
        model=SONNET,
        system_prompt=_ENGINEER_PROMPT,
        max_tokens=8192,
        temperature=0.3,
    ),
    "debugger": AgentConfig(
        role="debugger",
        model=HAIKU,
        system_prompt=_DEBUGGER_PROMPT,
        max_tokens=8192,
        temperature=0.2,
    ),
    "result_analyst": AgentConfig(
        role="result_analyst",
        model=HAIKU,
        system_prompt=_RESULT_ANALYST_PROMPT,
        max_tokens=2048,
    ),
    "selector": AgentConfig(
        role="selector",
        model=HAIKU,
        system_prompt=_SELECTOR_PROMPT,
        max_tokens=1024,
    ),
}


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def call_agent(
    role: str,
    documents: dict[str, str],
    *,
    client: anthropic.Anthropic | None = None,
    cost_tracker: CostTracker | None = None,
    model_override: str | None = None,
) -> str:
    """Call an agent with assembled document context.

    Args:
        role: Agent role name from REGISTRY.
        documents: Named documents to include in the prompt.
        client: Anthropic client (created if not provided).
        cost_tracker: Optional cost tracker for token accounting.
        model_override: Override the agent's default model (for escalation).

    Returns:
        Raw text response from the agent.
    """
    config = REGISTRY[role]
    client = client or anthropic.Anthropic()
    model = model_override or config.model

    # Assemble user message from documents
    parts = []
    for name, content in documents.items():
        parts.append(f"<{name}>\n{content}\n</{name}>")
    user_message = "\n\n".join(parts)

    logger.info("Calling agent %s with model %s", role, model)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=config.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            break
        except (anthropic.APIStatusError,) as e:
            if getattr(e, "status_code", 0) == 529 and attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                logger.warning(
                    "API overloaded (529), retrying in %ds (attempt %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
            else:
                raise

    # Track costs
    if cost_tracker and response.usage:
        cost_tracker.add(
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    text = response.content[0].text if response.content else ""
    logger.debug("Agent %s response (%d chars)", role, len(text))
    return text


def parse_json_response(text: str, model_class: type[T]) -> T:
    """Extract JSON from agent response and validate with Pydantic model.

    Handles responses that wrap JSON in markdown code blocks.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (``` markers)
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)
    return model_class.model_validate(data)


def extract_code(text: str) -> str:
    """Extract Python code from agent response (from ```python blocks)."""
    lines = text.split("\n")
    in_block = False
    code_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```python"):
            in_block = True
            continue
        if line.strip() == "```" and in_block:
            in_block = False
            continue
        if in_block:
            code_lines.append(line)

    if not code_lines:
        # Fallback: if no code block found, try to use the whole response
        # (agent may have returned raw code)
        return text.strip()

    return "\n".join(code_lines)
