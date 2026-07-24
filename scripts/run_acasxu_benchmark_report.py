#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Run ACAS-Xu benchmark and persist a timestamped report.

This script wraps:
  ny bench --benchmark acasxu --json
and stores the JSON summary in reports/benchmarks with metadata. It also
updates metrics/benchmarks for pulse integration.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_NY_PATH = Path("target/release/ny")
REPORTS_DIR = Path("reports/benchmarks")
METRICS_DIR = Path("metrics/benchmarks")


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_commit() -> str:
    result = _run(["git", "rev-parse", "HEAD"])
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_metrics(summary: dict[str, Any], report_path: Path) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "benchmark": summary.get("benchmark", "acasxu"),
        "benchmark_year": summary.get("benchmark_year"),
        "verified": summary.get("verified"),
        "total": summary.get("total"),
        "pass_rate": summary.get("pass_rate"),
        "avg_time_ms": summary.get("avg_time_ms"),
        "timeout_count": summary.get("timeout_count"),
        "error_count": summary.get("error_count"),
        "commit": summary.get("commit", ""),
        "report_path": str(report_path),
        "recorded_at": summary.get("report_generated_at"),
    }

    latest_path = METRICS_DIR / "latest.json"
    latest_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    history_path = METRICS_DIR / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ACAS-Xu benchmark and save report")
    parser.add_argument("--ny", default=str(DEFAULT_NY_PATH), help="Path to ny binary")
    parser.add_argument("--year", type=int, default=2021, help="VNN-COMP year for ACAS-Xu")
    parser.add_argument("--timeout", type=int, help="Per-problem timeout (seconds)")
    parser.add_argument("--include-results", type=str, default="true", help="Include per-instance results")
    parser.add_argument("--model-filter", type=str, help="Filter to specific ACAS-Xu model")
    parser.add_argument("--property-filter", type=str, help="Filter to specific ACAS-Xu property")
    parser.add_argument("--branching", type=str, help="Branching heuristic")
    parser.add_argument("--max-domains", type=int, help="Maximum number of domains")
    args = parser.parse_args()

    ny_path = Path(args.ny)
    if not ny_path.exists():
        print(f"ny binary not found at {ny_path}. Build with cargo build -p ny-cli --release.")
        return 1

    cmd = [
        str(ny_path),
        "bench",
        "--benchmark",
        "acasxu",
        "--json",
        "--year",
        str(args.year),
        "--include-results",
        str(args.include_results).lower(),
    ]
    if args.timeout is not None:
        cmd += ["--timeout", str(args.timeout)]
    if args.model_filter:
        cmd += ["--model-filter", args.model_filter]
    if args.property_filter:
        cmd += ["--property-filter", args.property_filter]
    if args.branching:
        cmd += ["--branching", args.branching]
    if args.max_domains is not None:
        cmd += ["--max-domains", str(args.max_domains)]

    started_at = _iso_now()
    result = _run(cmd)
    completed_at = _iso_now()

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return result.returncode

    # ny CLI may output multiple JSON objects; parse only the first
    stdout = result.stdout.strip()
    try:
        # Try parsing as single JSON first
        summary = json.loads(stdout)
    except json.JSONDecodeError:
        # If that fails, try to find the first complete JSON object
        # Look for closing brace at top level
        brace_depth = 0
        json_end = -1
        for i, char in enumerate(stdout):
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = i + 1
                    break
        if json_end > 0:
            try:
                summary = json.loads(stdout[:json_end])
            except json.JSONDecodeError:
                print("Benchmark output was not valid JSON:")
                print(result.stdout)
                return 1
        else:
            print("Benchmark output was not valid JSON:")
            print(result.stdout)
            return 1

    summary = dict(summary)
    summary.update(
        {
            "report_generated_at": completed_at,
            "report_started_at": started_at,
            "report_host": socket.gethostname(),
            "report_command": " ".join(cmd),
            "report_commit": _git_commit(),
        }
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"acasxu_{args.year}_{_timestamp()}.json"
    summary["report_path"] = str(report_path)

    _write_json(report_path, summary)
    _write_json(REPORTS_DIR / "acasxu_latest.json", summary)
    _write_metrics(summary, report_path)

    print(f"Wrote benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
