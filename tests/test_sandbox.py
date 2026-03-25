"""Tests for sandbox utilities — parsing, file writing, and Slurm path.

Pure-unit tests: no subprocess execution. Subprocess calls are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentsciml.sandbox import (
    ExecutionResult,
    best_score_from_results,
    parse_result_lines,
    run_experiment,
    run_slurm_remote,
)


def test_parse_result_lines():
    lines = [
        "RESULT|SK|complete|N=10|T=0.500|Gamma=1.000|method=pimc|E=-3.07|E_exact=-3.07|advantage=0.042|q_EA=0.05|time=2.80s|desc",
        "RESULT|SK|complete|N=10|T=0.300|Gamma=0.700|method=pimc|E=-3.07|E_exact=-3.07|advantage=0.015|q_EA=0.03|time=2.50s|desc2",
    ]
    parsed = parse_result_lines(lines)
    assert len(parsed) == 2
    assert parsed[0]["N"] == "10"
    assert parsed[0]["advantage"] == "0.042"
    assert parsed[1]["advantage"] == "0.015"


def test_best_score_from_results():
    parsed = [
        {"advantage": "0.042", "method": "pimc"},
        {"advantage": "0.015", "method": "pimc"},
        {"advantage": "-0.001", "method": "pimc"},
    ]
    assert best_score_from_results(parsed, "advantage") == 0.042


def test_best_score_empty():
    assert best_score_from_results([], "advantage") == float("-inf")


def test_best_score_missing_key():
    parsed = [{"method": "pimc"}]
    assert best_score_from_results(parsed, "advantage") == float("-inf")


def test_parse_result_lines_positional():
    """Fields without = are stored as field_N."""
    lines = ["RESULT|SK|complete|some_value"]
    parsed = parse_result_lines(lines)
    assert parsed[0]["field_0"] == "SK"
    assert parsed[0]["field_1"] == "complete"
    assert parsed[0]["field_2"] == "some_value"


# ---------------------------------------------------------------------------
# Multi-file write tests (run_experiment file dispatch)
# ---------------------------------------------------------------------------

class TestRunExperimentFileWrite:
    """Test that run_experiment correctly writes files for all input types."""

    @patch("agentsciml.sandbox.run_local")
    def test_plain_string_writes_experiment_file(self, mock_run_local, tmp_path):
        mock_run_local.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 1.0, ["RESULT|x=1"], "ok"
        )
        code = "print('RESULT|x=1')"
        run_experiment(code, tmp_path)

        written = (tmp_path / "autoresearch" / "experiment.py").read_text()
        assert written == code
        mock_run_local.assert_called_once()

    @patch("agentsciml.sandbox.run_local")
    def test_dict_writes_multiple_files(self, mock_run_local, tmp_path):
        mock_run_local.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 1.0, ["RESULT|x=1"], "ok"
        )
        files = {
            "autoresearch/experiment.py": "print('hello')",
            "autoresearch/utils.py": "def helper(): pass",
        }
        run_experiment(files, tmp_path)

        assert (tmp_path / "autoresearch" / "experiment.py").read_text() == "print('hello')"
        assert (tmp_path / "autoresearch" / "utils.py").read_text() == "def helper(): pass"

    @patch("agentsciml.sandbox.run_local")
    def test_json_string_dict_writes_files(self, mock_run_local, tmp_path):
        mock_run_local.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 1.0, ["RESULT|x=1"], "ok"
        )
        files = {"autoresearch/experiment.py": "code_here"}
        run_experiment(json.dumps(files), tmp_path)

        assert (tmp_path / "autoresearch" / "experiment.py").read_text() == "code_here"

    @patch("agentsciml.sandbox.run_local")
    def test_json_string_non_dict_falls_back(self, mock_run_local, tmp_path):
        """A JSON string that parses to a non-dict (e.g. a list) falls back to plain string."""
        mock_run_local.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 1.0, ["RESULT|x=1"], "ok"
        )
        code = json.dumps([1, 2, 3])
        run_experiment(code, tmp_path)

        written = (tmp_path / "autoresearch" / "experiment.py").read_text()
        assert written == code

    @patch("agentsciml.sandbox.run_local")
    def test_creates_nested_dirs(self, mock_run_local, tmp_path):
        mock_run_local.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 1.0, ["RESULT|x=1"], "ok"
        )
        files = {"deep/nested/dir/file.py": "content"}
        run_experiment(files, tmp_path)

        assert (tmp_path / "deep" / "nested" / "dir" / "file.py").read_text() == "content"

    @patch("agentsciml.sandbox.run_local")
    def test_hypothesis_yaml_changes_entry_point(self, mock_run_local, tmp_path):
        """When workspace/hypothesis.yaml is in files, entry point switches to engine/runner.py."""
        mock_run_local.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 1.0, ["RESULT|x=1"], "ok"
        )
        files = {
            "workspace/hypothesis.yaml": "name: test",
            "autoresearch/experiment.py": "code",
        }
        run_experiment(files, tmp_path)

        # Should call run_local with the engine runner path
        call_args = mock_run_local.call_args
        entry_point = call_args[0][0]
        assert "engine/runner.py" in str(entry_point)

    @patch("agentsciml.sandbox.run_slurm_remote")
    def test_slurm_config_dispatches_to_slurm(self, mock_slurm, tmp_path):
        mock_slurm.return_value = ExecutionResult(
            "RESULT|x=1\n", "", 0, 5.0, ["RESULT|x=1"], "ok"
        )
        run_experiment("code", tmp_path, slurm_config={"partition": "gpu"})
        mock_slurm.assert_called_once()


# ---------------------------------------------------------------------------
# Slurm remote tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestRunSlurmRemote:

    @patch("agentsciml.sandbox.time.sleep")
    @patch("agentsciml.sandbox.subprocess.run")
    @patch("agentsciml.sandbox.os.path.exists", return_value=False)
    def test_sbatch_script_generation(self, mock_exists, mock_run, mock_sleep, tmp_path):
        """Verifies the sbatch script is written with correct Slurm directives."""
        # scp returns ok
        scp_result = MagicMock(returncode=0, stdout="", stderr="")
        # sbatch returns job id
        sbatch_result = MagicMock(returncode=0, stdout="Submitted batch job 12345", stderr="")
        # squeue shows job gone immediately
        squeue_result = MagicMock(returncode=0, stdout="JOBID", stderr="")

        mock_run.side_effect = [scp_result, sbatch_result, squeue_result]

        config = {"partition": "a100", "gres": "gpu:2", "cpus": 16, "mem": "64G", "time": "01:00:00"}
        run_slurm_remote("autoresearch/experiment.py", tmp_path, config, timeout=600)

        script = (tmp_path / "slurm_submit.sh").read_text()
        assert "#SBATCH --partition=a100" in script
        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH --cpus-per-task=16" in script
        assert "#SBATCH --mem=64G" in script
        assert "#SBATCH --time=01:00:00" in script

    @patch("agentsciml.sandbox.subprocess.run")
    @patch("agentsciml.sandbox.os.path.exists", return_value=False)
    def test_sbatch_failure_returns_crash(self, mock_exists, mock_run, tmp_path):
        scp_result = MagicMock(returncode=0, stdout="", stderr="")
        sbatch_result = MagicMock(returncode=1, stdout="", stderr="Permission denied")
        mock_run.side_effect = [scp_result, sbatch_result]

        result = run_slurm_remote("experiment.py", tmp_path, {}, timeout=60)
        assert result.status == "crash"
        assert "Permission denied" in result.stderr

    @patch("agentsciml.sandbox.subprocess.run")
    @patch("agentsciml.sandbox.os.path.exists", return_value=False)
    def test_unparseable_job_id_returns_crash(self, mock_exists, mock_run, tmp_path):
        scp_result = MagicMock(returncode=0, stdout="", stderr="")
        sbatch_result = MagicMock(returncode=0, stdout="Something unexpected", stderr="")
        mock_run.side_effect = [scp_result, sbatch_result]

        result = run_slurm_remote("experiment.py", tmp_path, {}, timeout=60)
        assert result.status == "crash"
        assert "Failed to parse job ID" in result.stderr
