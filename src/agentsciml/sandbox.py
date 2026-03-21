"""Sandbox: execute generated experiment code as a subprocess.

Writes experiment.py, runs it via uv, parses RESULT| lines from stdout.
No Docker needed — this runs the user's own code on their own hardware.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600  # 10 minutes, matching prepare.py


@dataclass
class ExecutionResult:
    """Result of running an experiment in the sandbox."""

    stdout: str
    stderr: str
    returncode: int
    wall_time: float
    result_lines: list[str]
    status: str  # "ok", "crash", "timeout"


def run_experiment(
    code: str,
    project_root: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    experiment_file: str = "autoresearch/experiment.py",
) -> ExecutionResult:
    """Write experiment code and execute it.

    Args:
        code: Complete experiment.py content.
        project_root: Path to the project root (e.g. ~/dev/quantum-cognition).
        timeout: Max seconds before killing the process.
        experiment_file: Relative path to experiment.py within project.

    Returns:
        ExecutionResult with stdout, stderr, parsed RESULT lines, and status.
    """
    exp_path = project_root / experiment_file
    exp_path.write_text(code)
    logger.info("Wrote experiment to %s (%d bytes)", exp_path, len(code))

    t0 = time.time()
    status = "ok"

    try:
        proc = subprocess.run(
            ["uv", "run", "python", str(exp_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        wall_time = time.time() - t0
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr

        if returncode != 0:
            status = "crash"
            logger.warning(
                "Experiment crashed (rc=%d): %s", returncode, stderr[:200]
            )

    except subprocess.TimeoutExpired:
        wall_time = time.time() - t0
        returncode = -1
        stdout = ""
        stderr = f"Timeout after {timeout}s"
        status = "timeout"
        logger.warning("Experiment timed out after %ds", timeout)

    # Parse RESULT| lines
    result_lines = [
        line for line in stdout.split("\n") if line.startswith("RESULT|")
    ]

    if status == "ok" and not result_lines:
        status = "crash"
        logger.warning("No RESULT| lines in stdout — treating as crash")

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        wall_time=wall_time,
        result_lines=result_lines,
        status=status,
    )


def parse_result_lines(lines: list[str]) -> list[dict[str, str]]:
    """Parse RESULT| lines into dicts.

    Format: RESULT|model|topology|N=val|T=val|...
    Returns list of dicts with key=value pairs.
    """
    parsed = []
    for line in lines:
        fields = line.split("|")[1:]  # skip the "RESULT" prefix
        record: dict[str, str] = {}
        for i, f in enumerate(fields):
            if "=" in f:
                key, val = f.split("=", 1)
                record[key.strip()] = val.strip()
            else:
                record[f"field_{i}"] = f.strip()
        parsed.append(record)
    return parsed


def best_score_from_results(
    parsed: list[dict[str, str]],
    metric: str = "advantage",
) -> float:
    """Extract the best (max) value of a metric from parsed results."""
    values = []
    for record in parsed:
        if metric in record:
            try:
                values.append(float(record[metric]))
            except ValueError:
                continue
    return max(values) if values else float("-inf")
