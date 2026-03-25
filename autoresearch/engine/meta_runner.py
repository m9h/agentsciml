import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
import os
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsciml.orchestrator import Orchestrator


def run_meta_experiment(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_data = yaml.safe_load(f)

    meta_config = config_data.get("config", {})
    
    # Target project
    qcccm_root = Path.home() / "dev" / "quantum-cognition"
    adapter = Orchestrator.load_adapter(str(qcccm_root / "adapter.py"))
    adapter.project_root = qcccm_root

    # Setup the inner orchestrator with the meta-parameters
    orch = Orchestrator(
        adapter=adapter,
        knowledge_file=qcccm_root / "autoresearch" / "knowledge.yaml",
        budget_usd=meta_config.get("budget_usd", 2.0),
        max_generations=meta_config.get("max_generations", 2),
        debate_rounds=meta_config.get("debate_rounds", 4),
        # Here we could also apply model overrides if we extend Orchestrator more
    )

    # Run the inner scientific loop
    best_node = orch.run()

    # Calculate efficiency
    best_advantage = best_node.score if best_node else 0.0
    total_cost = orch.cost.estimated_cost_usd
    efficiency = best_advantage / total_cost if total_cost > 0 else 0.0

    print(f"RESULT|advantage={best_advantage:.4f}|cost={total_cost:.4f}|efficiency={efficiency:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hypothesis", help="Path to the meta-hypothesis YAML")
    args = parser.parse_args()
    
    run_meta_experiment(args.hypothesis)
