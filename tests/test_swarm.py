"""Tests for SwarmRunner — config loading, repo sync, and project dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agentsciml.swarm import SwarmRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def swarm_config(tmp_path: Path) -> Path:
    """Write a minimal swarm.yaml and return its path."""
    config = {
        "projects": [
            {
                "name": "proj_a",
                "path": str(tmp_path / "proj_a"),
                "repo_url": "https://github.com/test/proj_a.git",
                "slurm": {"partition": "gpu", "gres": "gpu:1"},
            },
            {
                "name": "proj_b",
                "path": str(tmp_path / "proj_b"),
                "repo_url": "https://github.com/test/proj_b.git",
            },
        ],
        "meta": {
            "debate_rounds": 6,
            "max_concurrent_slurm_jobs": 4,
        },
    }
    cfg_path = tmp_path / "swarm.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path


@pytest.fixture
def no_url_config(tmp_path: Path) -> Path:
    """Config with a project missing repo_url."""
    config = {
        "projects": [
            {"name": "local_only", "path": str(tmp_path / "local_only")},
        ],
        "meta": {},
    }
    cfg_path = tmp_path / "swarm_no_url.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------

class TestSwarmConfig:

    @pytest.mark.unit
    def test_loads_projects(self, swarm_config):
        runner = SwarmRunner(str(swarm_config))
        assert len(runner.projects) == 2
        assert runner.projects[0]["name"] == "proj_a"
        assert runner.projects[1]["name"] == "proj_b"

    @pytest.mark.unit
    def test_loads_meta(self, swarm_config):
        runner = SwarmRunner(str(swarm_config))
        assert runner.meta["debate_rounds"] == 6
        assert runner.meta["max_concurrent_slurm_jobs"] == 4

    @pytest.mark.unit
    def test_empty_config_defaults(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text(yaml.dump({}))
        runner = SwarmRunner(str(cfg_path))
        assert runner.projects == []
        assert runner.meta == {}


# ---------------------------------------------------------------------------
# Repository sync tests
# ---------------------------------------------------------------------------

class TestSyncRepository:

    @pytest.mark.unit
    @patch("agentsciml.swarm.subprocess.run")
    def test_clones_when_path_missing(self, mock_run, swarm_config, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        runner = SwarmRunner(str(swarm_config))
        project_cfg = runner.projects[0]

        # Ensure project path does NOT exist
        proj_path = Path(project_cfg["path"])
        assert not proj_path.exists()

        runner.sync_repository(project_cfg)

        # Should have called git clone
        clone_call = mock_run.call_args_list[0]
        assert "clone" in clone_call[0][0]
        assert project_cfg["repo_url"] in clone_call[0][0]

    @pytest.mark.unit
    @patch("agentsciml.swarm.subprocess.run")
    def test_pulls_when_path_exists(self, mock_run, swarm_config, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        runner = SwarmRunner(str(swarm_config))
        project_cfg = runner.projects[0]

        # Create the project directory
        proj_path = Path(project_cfg["path"])
        proj_path.mkdir(parents=True)

        runner.sync_repository(project_cfg)

        # Should have called stash + pull (two calls)
        assert mock_run.call_count == 2
        stash_call = mock_run.call_args_list[0]
        pull_call = mock_run.call_args_list[1]
        assert "stash" in stash_call[0][0]
        assert "pull" in pull_call[0][0]

    @pytest.mark.unit
    def test_skips_sync_without_url(self, no_url_config):
        runner = SwarmRunner(str(no_url_config))
        # Should not raise — just logs a warning and returns
        runner.sync_repository(runner.projects[0])


# ---------------------------------------------------------------------------
# run_project tests
# ---------------------------------------------------------------------------

class TestRunProject:

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("agentsciml.swarm.Orchestrator")
    @patch("agentsciml.swarm.subprocess.run")
    async def test_run_project_missing_adapter_logs_error(
        self, mock_run, mock_orch_cls, swarm_config, tmp_path, caplog
    ):
        """run_project logs an error if adapter.py doesn't exist."""
        mock_run.return_value = MagicMock(returncode=0)
        runner = SwarmRunner(str(swarm_config))
        project_cfg = runner.projects[0]

        # Create the project dir but no adapter.py
        proj_path = Path(project_cfg["path"])
        proj_path.mkdir(parents=True)

        import logging
        with caplog.at_level(logging.ERROR):
            await runner.run_project(project_cfg)

        assert "failed" in caplog.text.lower() or "adapter.py not found" in caplog.text
