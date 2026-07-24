# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_gpu_crown_backward_regression.py"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh_gpu_crown_backward_baselines.py"


def write_policy(path: Path) -> None:
    write_custom_policy(
        path,
        [
            {
                "name": "cpu_path",
                "case": "acasxu_like",
                "phase": "cpu_production",
                "expected_status": "measured",
                "expected_parameter_count": 13305,
                "expected_estimated_cpu_peak_bytes": 40000,
                "expected_cpu_dense_budget_bytes": 2147483648,
                "baseline_seconds": 0.002,
                "max_regression_ratio": 5.0,
                "max_seconds": 0.05,
            },
            {
                "name": "gpu_guard",
                "case": "soundnessbench_exact_like",
                "phase": "cpu_production",
                "expected_status": "skipped",
                "expected_parameter_count": 1740696,
                "expected_estimated_cpu_peak_bytes": 154618822656,
                "expected_cpu_dense_budget_bytes": 2147483648,
                "expected_detail_substring": "dense_peak_exceeds_budget",
            },
            {
                "name": "gpu_path",
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "expected_status": "measured",
                "expected_parameter_count": 7410996,
                "expected_estimated_cpu_peak_bytes": 52613349376,
                "expected_cpu_dense_budget_bytes": 2147483648,
                "baseline_seconds": 4.0,
                "max_regression_ratio": 2.0,
                "max_seconds": 9.0,
            },
        ],
    )


def write_custom_policy(path: Path, checks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "suite": "gpu_crown_backward",
                "checks": checks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_candidate(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "phase",
                "seconds",
                "parameter_count",
                "estimated_cpu_peak_bytes",
                "cpu_dense_budget_bytes",
                "status",
                "detail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_checker(
    tmp_path: Path,
    candidates: list[Path],
    policy: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "--policy",
        str(policy),
        "--output",
        str(tmp_path / "reports" / "benchmarks" / "gpu_crown_backward_regression_latest.json"),
    ]
    if extra_args:
        cmd.extend(extra_args)
    for candidate in candidates:
        cmd.extend(["--candidate", str(candidate)])
    return run_checker_raw(tmp_path, cmd)


def run_checker_raw(
    tmp_path: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def run_refresh(
    tmp_path: Path,
    candidates: list[Path],
    policy: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ["--policy", str(policy)]
    if extra_args:
        cmd.extend(extra_args)
    for candidate in candidates:
        cmd.extend(["--candidate", str(candidate)])
    return run_refresh_raw(tmp_path, cmd)


def run_refresh_raw(
    tmp_path: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT), *args],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
