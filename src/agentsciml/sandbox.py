"""Sandbox: execute generated experiment code locally or via remote Slurm.

Standard mode: subprocess locally.
Slurm mode: sbatch to a cluster via SSH, wait for results.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import json
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 1200 

@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    wall_time: float
    result_lines: list[str]
    status: str 

def run_experiment(
    files_or_code: str | dict[str, str],
    project_root: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    experiment_file: str = "autoresearch/experiment.py",
    slurm_config: dict | None = None,
) -> ExecutionResult:
    """Executes the experiment. Choice of local or Slurm based on slurm_config."""
    
    # 1. Write files to project root
    if isinstance(files_or_code, str):
        try:
            files = json.loads(files_or_code)
            if not isinstance(files, dict):
                files = {experiment_file: files_or_code}
        except Exception:
            files = {experiment_file: files_or_code}
    else:
        files = files_or_code

    for fname, fcontent in files.items():
        out_path = project_root / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(fcontent)
        logger.info("Wrote file to %s (%d bytes)", out_path, len(fcontent))

    # Determine entry point
    entry_point = "autoresearch/experiment.py"
    if "workspace/hypothesis.yaml" in files:
        entry_point = "autoresearch/engine/runner.py"

    if slurm_config:
        return run_slurm_remote(entry_point, project_root, slurm_config, timeout)
    else:
        full_entry_path = project_root / entry_point
        return run_local(full_entry_path, project_root, timeout)

def run_local(entry_point: Path, project_root: Path, timeout: int) -> ExecutionResult:
    """Standard local execution."""
    t0 = time.time()
    status = "ok"
    try:
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        proc = subprocess.run(
            ["uv", "run", "--no-sync", "python", str(entry_point)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        wall_time = time.time() - t0
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        if returncode != 0:
            status = "crash"
    except subprocess.TimeoutExpired:
        wall_time = time.time() - t0
        returncode = -1
        stdout = ""
        stderr = f"Timeout after {timeout}s"
        status = "timeout"

    result_lines = [l for l in stdout.split("\n") if l.startswith("RESULT|")]
    if status == "ok" and not result_lines:
        status = "crash"
    return ExecutionResult(stdout, stderr, returncode, wall_time, result_lines, status)

def run_slurm_remote(entry_point: str, project_root: Path, config: dict, timeout: int) -> ExecutionResult:
    """Run via remote Slurm on DGX Spark via SSH."""
    t0 = time.time()
    
    # DGX Spark SSH details (from Makefile)
    dgx_host = "gx10-dgx-spark.local"
    dgx_user = "mhough"
    dgx_key = "/Users/mhough/Library/Application Support/NVIDIA/Sync/config/nvsync.key"
    
    # Check if the key exists locally (might be different path on Linux vs MacOS)
    if not os.path.exists(dgx_key):
        # Fallback for Linux environment
        dgx_key = os.path.expanduser("~/.ssh/id_ed25519")
        
    ssh_prefix = ["ssh", "-o", "ConnectTimeout=10", "-i", dgx_key, f"{dgx_user}@{dgx_host}"]
    
    job_name = f"agentsciml_{project_root.name}"
    out_file = project_root / f"slurm_{int(t0)}.out"
    
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={config.get('partition', 'gpu')}
#SBATCH --gres={config.get('gres', 'gpu:1')}
#SBATCH --cpus-per-task={config.get('cpus', 8)}
#SBATCH --mem={config.get('mem', '32G')}
#SBATCH --time={config.get('time', '00:30:00')}
#SBATCH --output={out_file}

cd {project_root}
uv run --no-sync python {entry_point}
"""
    script_path = project_root / "slurm_submit.sh"
    script_path.write_text(sbatch_script)
    
    # 1. Sync script to DGX (Basic assumption: /home/mhough/dev is shared or mirrors)
    # 2. Submit via SSH
    logger.info("Submitting job to DGX Spark via SSH...")
    res = subprocess.run(ssh_prefix + [f"sbatch {script_path}"], capture_output=True, text=True)
    
    if res.returncode != 0:
        logger.error("Failed to submit Slurm job: %s", res.stderr)
        return ExecutionResult("", res.stderr, res.returncode, 0, [], "crash")
    
    match = re.search(r"Submitted batch job (\d+)", res.stdout)
    if not match:
        return ExecutionResult("", "Failed to parse job ID", -1, 0, [], "crash")
    
    job_id = match.group(1)
    logger.info("Remote Slurm job %s submitted for %s", job_id, project_root.name)
    
    # Wait loop via remote squeue
    status = "ok"
    while True:
        check = subprocess.run(ssh_prefix + [f"squeue -j {job_id}"], capture_output=True, text=True)
        if job_id not in check.stdout:
            break 
        if time.time() - t0 > timeout:
            subprocess.run(ssh_prefix + [f"scancel {job_id}"])
            status = "timeout"
            break
        time.sleep(20)
        
    # Read the output file
    stdout = ""
    for _ in range(5):
        if out_file.exists():
            stdout = out_file.read_text()
            if "RESULT|" in stdout:
                break
        time.sleep(5)
        
    result_lines = [l for l in stdout.split("\n") if l.startswith("RESULT|")]
    return ExecutionResult(stdout, "", 0, time.time()-t0, result_lines, status)

def parse_result_lines(lines: list[str]) -> list[dict[str, str]]:
    parsed = []
    for line in lines:
        fields = line.split("|")[1:]
        record: dict[str, str] = {}
        for i, f in enumerate(fields):
            if "=" in f:
                key, val = f.split("=", 1)
                record[key.strip()] = val.strip()
            else:
                record[f"field_{i}"] = f.strip()
        parsed.append(record)
    return parsed

def best_score_from_results(parsed: list[dict[str, str]], metric: str = "advantage") -> float:
    values = []
    for record in parsed:
        if metric in record:
            try:
                values.append(float(record[metric]))
            except ValueError:
                continue
    return max(values) if values else float("-inf")
