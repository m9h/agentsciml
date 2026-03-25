"""Swarm Manager: orchestrates multiple AgenticSciML projects in parallel.
Handles GitHub synchronization and Slurm-based compute allocation.
"""

from __future__ import annotations

import asyncio
import logging
import yaml
import subprocess
import os
from pathlib import Path
from typing import List, Dict

from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

class SwarmRunner:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.projects = self.config.get("projects", [])
        self.meta = self.config.get("meta", {})

    def sync_repository(self, project_cfg: Dict):
        """Ensure the project repository is local and up-to-date."""
        path = Path(project_cfg["path"])
        url = project_cfg.get("repo_url")
        
        if not url:
            logger.warning(f"No repo_url for {project_cfg['name']}, skipping sync.")
            return

        if not path.exists():
            logger.info(f"📥 Cloning {project_cfg['name']} from {url}...")
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", url, str(path)], check=True)
        else:
            logger.info(f"🔄 Pulling latest changes for {project_cfg['name']}...")
            subprocess.run(["git", "-C", str(path), "stash"], check=True); subprocess.run(["git", "-C", str(path), "pull"], check=True)

    async def run_project(self, project_cfg: Dict):
        """Worker task for a single project swarm team."""
        name = project_cfg["name"]
        path = Path(project_cfg["path"])
        
        try:
            # 1. Sync from GitHub
            self.sync_repository(project_cfg)
            
            logger.info(f"🚀 Launching swarm team for {name}")
            
            # 2. Dynamically load adapter from project root
            adapter_path = path / "adapter.py"
            if not adapter_path.exists():
                raise FileNotFoundError(f"adapter.py not found in {path}")
                
            adapter = Orchestrator.load_adapter(str(adapter_path))
            adapter.project_root = path
            
            # 3. Initialize Orchestrator with project-specific Slurm config
            orch = Orchestrator(
                adapter=adapter,
                debate_rounds=self.meta.get("debate_rounds", 6),
                slurm_config=project_cfg.get("slurm"),
                knowledge_file=path / "autoresearch" / "knowledge.yaml"
            )
            
            # 4. Run the evolutionary loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, orch.run)
            
            logger.info(f"✅ Swarm team for {name} finished successfully.")
            
        except Exception as e:
            logger.error(f"❌ Swarm team for {name} failed: {e}")

    async def run_all(self):
        """Orchestrate the entire swarm."""
        tasks = [self.run_project(p) for p in self.projects]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="swarm.yaml")
    args = parser.parse_args()
    
    runner = SwarmRunner(args.config)
    asyncio.run(runner.run_all())
