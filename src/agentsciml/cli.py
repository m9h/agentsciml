"""CLI entry point for AgenticSciML."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """AgenticSciML: Multi-agent evolutionary SciML discovery."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to project root (e.g. ~/dev/quantum-cognition)",
)
@click.option(
    "--budget", "-b",
    type=float,
    default=5.0,
    show_default=True,
    help="Maximum LLM API budget in USD",
)
@click.option(
    "--generations", "-g",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of evolutionary generations",
)
@click.option(
    "--knowledge", "-k",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to knowledge base YAML file",
)
@click.option(
    "--adapter", "-a",
    type=str,
    default="auto",
    show_default=True,
    help="Adapter: 'qcccm', 'dmipy', 'parameter_golf', 'meta', 'auto', or path to adapter.py",
)
@click.option(
    "--debate-rounds", "-d",
    type=int,
    default=4,
    show_default=True,
    help="Number of structured debate rounds per mutation",
)
def run(
    project: Path,
    budget: float,
    generations: int,
    knowledge: Path | None,
    adapter: str,
    debate_rounds: int,
) -> None:
    """Run the evolutionary multi-agent search on a project."""
    from .adapters.dmipy import DmipyAdapter
    from .adapters.meta import MetaSciMLAdapter
    from .adapters.parameter_golf import ParameterGolfAdapter
    from .adapters.qcccm import QCCCMAdapter
    from .orchestrator import Orchestrator

    BUILTIN_ADAPTERS = {
        "qcccm": QCCCMAdapter,
        "dmipy": DmipyAdapter,
        "parameter_golf": ParameterGolfAdapter,
        "meta": MetaSciMLAdapter,
    }

    # Select adapter
    if adapter in BUILTIN_ADAPTERS:
        project_adapter = BUILTIN_ADAPTERS[adapter](project)
    elif adapter != "auto" and Path(adapter).is_file():
        project_adapter = Orchestrator.load_adapter(adapter)
        project_adapter.project_root = project
    elif adapter == "auto":
        # Auto-detect from project path
        project_str = str(project).lower()
        if "quantum" in project_str or "qcccm" in project_str:
            project_adapter = QCCCMAdapter(project)
        elif "dmipy" in project_str:
            project_adapter = DmipyAdapter(project)
        elif "parameter" in project_str or "golf" in project_str:
            project_adapter = ParameterGolfAdapter(project)
        else:
            # Try loading adapter.py from project root
            adapter_file = project / "adapter.py"
            if adapter_file.exists():
                project_adapter = Orchestrator.load_adapter(str(adapter_file))
                project_adapter.project_root = project
            else:
                click.echo(
                    f"No adapter found for {project}. Use --adapter to specify.",
                    err=True,
                )
                sys.exit(1)
    else:
        click.echo(f"Adapter not found: {adapter}", err=True)
        sys.exit(1)

    click.echo(f"Project: {project}")
    click.echo(f"Adapter: {type(project_adapter).__name__}")
    click.echo(f"Budget: ${budget:.2f}")
    click.echo(f"Max generations: {generations}")
    click.echo(f"Debate rounds: {debate_rounds}")
    click.echo()

    orch = Orchestrator(
        project_adapter,
        budget_usd=budget,
        max_generations=generations,
        knowledge_file=knowledge,
        debate_rounds=debate_rounds,
    )

    best = orch.run()

    click.echo()
    click.echo("=" * 60)
    if best:
        click.echo(f"Best solution: {best.id}")
        click.echo(f"Score: {best.score:.6f}")
        click.echo(f"Description: {best.mutation_description}")
    else:
        click.echo("No successful solutions found.")

    click.echo()
    summary = orch.cost.summary()
    click.echo(f"Total API calls: {summary['total_calls']}")
    click.echo(f"Total cost: ${summary['estimated_cost_usd']:.4f}")
    click.echo(f"Tree: {orch.tree.summary()}")


@main.command()
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to project root",
)
def status(project: Path) -> None:
    """Show the current state of the solution tree."""
    tree_path = project / "autoresearch" / "tree.json"
    if not tree_path.exists():
        click.echo("No tree.json found — run `agentsciml run` first.")
        return

    from .tree import SolutionTree

    tree = SolutionTree(path=tree_path)
    summary = tree.summary()

    click.echo(f"Solution tree: {tree_path}")
    click.echo(f"Total nodes: {summary['total_nodes']}")
    click.echo(f"OK / Crashed: {summary['ok_nodes']} / {summary['crashed_nodes']}")
    click.echo(f"Generations: {summary['generations']}")
    click.echo(f"Best score: {summary['best_score']}")
    click.echo(f"Total LLM cost: ${summary['total_llm_cost']:.4f}")

    click.echo()
    click.echo("Top 5 solutions:")
    for i, node in enumerate(tree.top_k(5), 1):
        click.echo(
            f"  {i}. [{node.id}] score={node.score:.6f}"
            f" gen={node.generation} — {node.mutation_description}"
        )
