#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Benchmark ny on VNN-COMP 2025 new categories.

Categories: lsnc_relu, soundnessbench, relusplitter.

Usage:
    python3 scripts/benchmark_vnncomp2025_new.py --category lsnc_relu --sample 10
    python3 scripts/benchmark_vnncomp2025_new.py --all --sample 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

NY_BINARY = Path(__file__).parent.parent / "target" / "release" / "ny"
BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks" / "vnncomp2025" / "benchmarks"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "benchmarks"


@dataclass
class Result:
    """Single verification result."""

    model: str
    property: str
    timeout: int
    method: str
    result: str
    elapsed: float
    domains: int
    error: str


def _build_cmd(model_path: Path, vnnlib_path: Path, timeout: int, method: str) -> list[str]:
    """Build ny command line for the given method."""
    if method == "beta":
        return [
            str(NY_BINARY), "beta-crown", str(model_path),
            "--property", str(vnnlib_path), "--timeout", str(timeout),
            "--branching", "input", "--pgd-attack",
            "--pgd-restarts", "1000", "--max-domains", "50000", "--json",
        ]
    return [
        str(NY_BINARY), "verify", str(model_path),
        "--property", str(vnnlib_path), "--method", method,
        "--timeout", str(timeout), "--json",
    ]


def _parse_json_from_output(text: str) -> dict | None:
    """Extract first JSON object from ny output text.

    Handles both single-line and multi-line (pretty-printed) JSON.
    """
    # Try single-line first (fast path)
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Multi-line: find matching braces
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalize_status(raw: str) -> str:
    """Normalize verification status string."""
    s = raw.lower()
    if s == "safe":
        return "verified"
    if s == "violated":
        return "falsified"
    return s


def _make_error_result(model_path: Path, vnnlib_path: Path, timeout: int,
                       method: str, elapsed: float, error: str) -> Result:
    return Result(model=model_path.name, property=vnnlib_path.name,
                  timeout=timeout, method=method, result="error",
                  elapsed=elapsed, domains=0, error=error)


def run_instance(model_path: Path, vnnlib_path: Path, timeout: int,
                 method: str = "beta") -> Result:
    """Run ny on a single instance."""
    start = time.time()
    cmd = _build_cmd(model_path, vnnlib_path, timeout, method)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return Result(model=model_path.name, property=vnnlib_path.name,
                      timeout=timeout, method=method, result="timeout",
                      elapsed=timeout + 30, domains=0, error="process_timeout")
    except Exception as e:
        return _make_error_result(model_path, vnnlib_path, timeout, method,
                                 time.time() - start, str(e)[:200])

    data = _parse_json_from_output(proc.stdout) or _parse_json_from_output(proc.stderr)
    if data:
        status = _normalize_status(data.get("property_status", data.get("status", "unknown")))
        return Result(model=model_path.name, property=vnnlib_path.name,
                      timeout=timeout, method=method, result=status,
                      elapsed=elapsed, domains=data.get("domains_explored", 0) or 0,
                      error="")

    output_text = (proc.stdout + proc.stderr)[:200]
    return _make_error_result(model_path, vnnlib_path, timeout, method,
                              elapsed, output_text or f"exit_code={proc.returncode}")


def _resolve_path(cat_dir: Path, name: str, subdir: str) -> Path:
    """Resolve a model/property path, trying subdir if direct path missing."""
    direct = cat_dir / name
    if direct.exists():
        return direct
    return cat_dir / subdir / name.split("/")[-1]


def load_instances(category: str) -> list[tuple[Path, Path, int]]:
    """Load instances from instances.csv for a category."""
    cat_dir = BENCHMARKS_DIR / category
    instances_csv = cat_dir / "instances.csv"
    if not instances_csv.exists():
        log.warning("No instances.csv for %s", category)
        return []

    instances = []
    with open(instances_csv) as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0].startswith("#") or row[0] == "network":
                continue
            try:
                timeout = int(float(row[2]))
            except (ValueError, IndexError):
                timeout = 60
            model_path = _resolve_path(cat_dir, row[0], "onnx")
            prop_path = _resolve_path(cat_dir, row[1], "vnnlib")
            if model_path.exists() and prop_path.exists():
                instances.append((model_path, prop_path, timeout))
    return instances


def _sample_instances(instances: list, sample_size: int) -> list:
    """Sample evenly across the instance list."""
    if sample_size >= len(instances):
        return instances
    step = len(instances) / sample_size
    return [instances[int(i * step)] for i in range(sample_size)]


def benchmark_category(category: str, sample_size: int, method: str = "beta") -> list[Result]:
    """Run benchmark on a category, sampling instances."""
    instances = load_instances(category)
    if not instances:
        log.info("No instances found for %s", category)
        return []

    total = len(instances)
    sampled = _sample_instances(instances, sample_size)
    log.info("Benchmarking: %s (%d/%d instances, method=%s)", category, len(sampled), total, method)

    results = []
    for i, (model_path, prop_path, timeout) in enumerate(sampled):
        effective_timeout = min(timeout, 60)
        log.info("  [%d/%d] %s (timeout=%ds)...", i + 1, len(sampled), prop_path.name, effective_timeout)
        result = run_instance(model_path, prop_path, effective_timeout, method=method)
        log.info("    %s (%.1fs, domains=%d)", result.result, result.elapsed, result.domains)
        results.append(result)

    _log_summary(category, results, total)
    return results


def _log_summary(category: str, results: list[Result], total_instances: int) -> None:
    """Log benchmark summary statistics."""
    n = len(results)
    counts = {s: sum(1 for r in results if r.result == s) for s in ("verified", "falsified", "unknown", "timeout", "error")}
    total_time = sum(r.elapsed for r in results)
    log.info("--- %s Summary ---", category)
    log.info("  Verified: %d/%d (%.0f%%)", counts["verified"], n, counts["verified"] / n * 100)
    log.info("  Falsified: %d/%d, Unknown: %d/%d, Timeout: %d/%d, Error: %d/%d",
             counts["falsified"], n, counts["unknown"], n, counts["timeout"], n, counts["error"], n)
    log.info("  Total time: %.1fs", total_time)
    log.info("  Projected full (%d): ~%.0f verified, ~%.0f falsified",
             total_instances, counts["verified"] * total_instances / n,
             counts["falsified"] * total_instances / n)


def save_results(category: str, results: list[Result]) -> Path:
    """Save results to CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"{category}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "property", "timeout", "method", "result", "elapsed", "domains"])
        for r in results:
            writer.writerow([r.model, r.property, r.timeout, r.method, r.result, f"{r.elapsed:.2f}", r.domains])
    log.info("Results saved to: %s", csv_path)
    return csv_path


def main() -> int:
    """Run VNN-COMP 2025 new category benchmarks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", type=str, help="Category to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all new categories")
    parser.add_argument("--sample", type=int, default=10, help="Number of instances to sample")
    parser.add_argument("--method", type=str, default="beta", help="Method: crown, alpha, beta")
    args = parser.parse_args()

    if not NY_BINARY.exists():
        log.error("Ny binary not found: %s", NY_BINARY)
        return 1

    categories: list[str] = []
    if args.all:
        categories = ["lsnc_relu", "soundnessbench", "relusplitter"]
    elif args.category:
        categories = [args.category]
    else:
        parser.print_help()
        return 1

    for category in categories:
        results = benchmark_category(category, args.sample, method=args.method)
        if results:
            save_results(category, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
