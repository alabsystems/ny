# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Shared helpers for benchmark tests.

These are intentionally *not* in `conftest.py` so test modules can import them
without risking collisions with other `conftest.py` files in the repo.

VNN-COMP 2021-2025 benchmark suite.
https://github.com/VNN-COMP/vnncomp2025_benchmarks
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    # Constants
    "BENCHMARK_DIR",
    "NY_BINARY",
    "VNNCOMP_DIR",
    "VNNCOMP_YEARS",
    "ACASXU_TEST_CASES",
    "BENCHMARKS_BY_YEAR",
    # Classes
    "VerificationResult",
    # Functions
    "run_ny_verify",
    "get_acasxu_test_cases",
    "get_benchmark_dir",
    "get_benchmark_instances",
    "run_benchmark_suite",
    "require_benchmark_items",
    "require_benchmark_path",
]


def _list_benchmark_files(directory: Path, pattern: str) -> list[Path]:
    """List benchmark files while ignoring transient generated artifacts."""

    return sorted(
        path
        for path in directory.glob(pattern)
        if not path.name.startswith("ny-onnx-shape-")
    )


def require_benchmark_path(path: Path | None, message: str) -> Path:
    """Skip the current test when an optional benchmark path is unavailable."""

    import pytest

    if path is None or not path.exists():
        pytest.skip(message)
    return path


def require_benchmark_items(items: list, message: str) -> list:
    """Skip the current test when an optional benchmark file set is empty."""

    import pytest

    if not items:
        pytest.skip(message)
    return items

# Benchmark directories
BENCHMARK_DIR = Path(__file__).parent
NY_BINARY = BENCHMARK_DIR.parent / "target" / "release" / "ny"

# VNN-COMP directories by year
# https://github.com/VNN-COMP/vnncomp2025_benchmarks
VNNCOMP_YEARS: dict[int, Path] = {
    2021: BENCHMARK_DIR / "vnncomp2021" / "benchmarks",
    2023: BENCHMARK_DIR / "vnncomp2023" / "benchmarks",
    2024: BENCHMARK_DIR / "vnncomp2024" / "benchmarks",
    2025: BENCHMARK_DIR / "vnncomp2025" / "benchmarks",
}

# Legacy alias for backwards compatibility
VNNCOMP_DIR = VNNCOMP_YEARS[2021]


@dataclass
class VerificationResult:
    """Result of a single verification task."""

    network: str
    property: str
    status: str  # "verified", "falsified", "unknown", "timeout", "error"
    time_seconds: float
    bounds: list | None = None
    domains_explored: int | None = None
    domains_verified: int | None = None
    cuts_generated: int | None = None
    max_depth_reached: int | None = None
    error_message: str | None = None


def run_ny_verify(
    network_path: Path,
    vnnlib_path: Path,
    timeout: int = 10,
    method: str = "crown",
    beta_flags: list[str] | None = None,
    beta_branching: str | None = None,
) -> VerificationResult:
    """Run ny verification on a single (network, property) pair.

    Returns VerificationResult with status, time, and bounds.
    """
    start = time.time()

    try:
        # For beta method, use beta-crown subcommand which has branch-and-bound
        if method == "beta":
            branching = beta_branching or "input"
            cmd = [
                str(NY_BINARY),
                "beta-crown",
                str(network_path),
                "--property",
                str(vnnlib_path),
                "--timeout",
                str(timeout),
                "--branching",
                branching,
                "--pgd-attack",  # Enable PGD attack for counterexample finding
                "--pgd-restarts",
                "5000",  # High restarts for hard cases
                "--max-domains",
                "50000",  # More domains for input splitting
                "--json",
            ]
            if beta_flags:
                cmd.extend(beta_flags)
        else:
            cmd = [
                str(NY_BINARY),
                "verify",
                str(network_path),
                "--property",
                str(vnnlib_path),
                "--method",
                method,
                "--timeout",
                str(timeout),  # Timeout in seconds
                "--json",
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # Allow some buffer
        )

        elapsed = time.time() - start

        # The ny binary uses exit code 0 for verified, exit code 2 for
        # unknown/not-verified. Parse JSON output regardless of exit code
        # since the binary always produces valid JSON with --json flag.
        try:
            output = json.loads(result.stdout)
            # Use property_status for actual verification result
            # (status = process status, property_status = verification result)
            status = output.get(
                "property_status", output.get("status", "unknown")
            ).lower()
            # Normalize status values: "safe" -> "verified", "violated" -> "falsified"
            if status == "safe":
                status = "verified"
            elif status == "violated":
                status = "falsified"
            bounds = output.get("output_bounds", output.get("bounds"))
            domains_explored = output.get("domains_explored")
            domains_verified = output.get("domains_verified")
            cuts_generated = output.get("cuts_generated")
            max_depth_reached = output.get("max_depth_reached")
            return VerificationResult(
                network=network_path.name,
                property=vnnlib_path.name,
                status=status,
                time_seconds=elapsed,
                bounds=bounds,
                domains_explored=domains_explored,
                domains_verified=domains_verified,
                cuts_generated=cuts_generated,
                max_depth_reached=max_depth_reached,
            )
        except json.JSONDecodeError:
            # Try to parse non-JSON output
            stdout = result.stdout.lower()
            if "verified" in stdout:
                status = "verified"
            elif "falsified" in stdout or "violated" in stdout:
                status = "falsified"
            elif result.returncode != 0:
                return VerificationResult(
                    network=network_path.name,
                    property=vnnlib_path.name,
                    status="error",
                    time_seconds=elapsed,
                    error_message=result.stderr[:500],
                )
            else:
                status = "unknown"
            return VerificationResult(
                network=network_path.name,
                property=vnnlib_path.name,
                status=status,
                time_seconds=elapsed,
            )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            network=network_path.name,
            property=vnnlib_path.name,
            status="timeout",
            time_seconds=timeout,
        )
    except Exception as e:
        return VerificationResult(
            network=network_path.name,
            property=vnnlib_path.name,
            status="error",
            time_seconds=time.time() - start,
            error_message=str(e),
        )


def get_acasxu_test_cases() -> list[tuple[Path, Path]]:
    """Generate (network, property) pairs for ACAS-Xu benchmark."""
    acasxu_dir = VNNCOMP_DIR / "acasxu"
    if not acasxu_dir.exists():
        return []

    networks = _list_benchmark_files(acasxu_dir, "*.onnx")
    properties = _list_benchmark_files(acasxu_dir, "*.vnnlib")

    # The CSV file maps which properties apply to which networks
    # For simplicity, we'll test all combinations (some may be N/A)
    return [(network, prop) for network in networks for prop in properties]


ACASXU_TEST_CASES = get_acasxu_test_cases()


def get_benchmark_dir(
    year: int, benchmark: str, benchmark_root: Path | None = None
) -> Path | None:
    """Get a benchmark directory, optionally below an explicit corpus root."""
    if benchmark_root is None:
        if year not in VNNCOMP_YEARS:
            return None
        base = VNNCOMP_YEARS[year]
    else:
        base = Path(benchmark_root)
    if not base.exists():
        return None

    # Handle year suffixes (e.g., acasxu_2023 in 2024)
    candidates = [
        base / benchmark,
        base / f"{benchmark}_{year}",
        base / f"{benchmark}_{year - 1}",  # Sometimes uses previous year
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_benchmark_instances(
    year: int, benchmark: str, benchmark_root: Path | None = None
) -> list[tuple[Path, Path, int]]:
    """Get (network, property, timeout) tuples from instances.csv.

    Returns list of (network_path, property_path, timeout_seconds).
    """
    bench_dir = get_benchmark_dir(year, benchmark, benchmark_root)
    if not bench_dir:
        return []

    instances_file = bench_dir / "instances.csv"
    if not instances_file.exists():
        # Fallback: look for acasxu_instances.csv etc.
        instances_file = bench_dir / f"{benchmark}_instances.csv"

    if not instances_file.exists():
        return []

    instances: list[tuple[Path, Path, int]] = []
    with open(instances_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].startswith("#") or row[0] == "network":
                continue  # Skip comments and headers

            network_name = row[0]
            prop_name = row[1]
            try:
                timeout = int(float(row[2])) if len(row) > 2 else 60
            except (ValueError, IndexError):
                timeout = 60

            # Find files - may be in onnx/ subdir or root
            # Also handle .gz files by trying without .gz extension
            network_candidates = [
                bench_dir / network_name,
                bench_dir / "onnx" / network_name,
            ]
            # If network ends with .gz, also try without .gz
            if network_name.endswith(".gz"):
                network_nogz = network_name[:-3]
                network_candidates.extend(
                    [
                        bench_dir / network_nogz,
                        bench_dir / "onnx" / network_nogz,
                    ]
                )

            prop_candidates = [
                bench_dir / prop_name,
                bench_dir / "vnnlib" / prop_name,
            ]

            network_path = next((p for p in network_candidates if p.exists()), None)
            prop_path = next((p for p in prop_candidates if p.exists()), None)

            if network_path and prop_path:
                instances.append((network_path, prop_path, timeout))

    return instances


def run_benchmark_suite(
    year: int,
    benchmark: str,
    method: str = "crown",
    timeout_override: int | None = None,
) -> dict:
    """Run all instances in a benchmark and return aggregate statistics.

    Returns dict with verified/falsified/unknown/timeout/error counts and times.
    """
    instances = get_benchmark_instances(year, benchmark)

    results: dict = {
        "year": year,
        "benchmark": benchmark,
        "method": method,
        "total": 0,
        "verified": 0,
        "falsified": 0,
        "unknown": 0,
        "timeout": 0,
        "error": 0,
        "total_time": 0.0,
        "instances": [],
    }
    domain_count = 0
    domains_explored_total = 0
    domains_verified_total = 0
    cuts_generated_total = 0
    max_depth_max = 0

    for network_path, prop_path, default_timeout in instances:
        timeout = timeout_override or default_timeout
        result = run_ny_verify(
            network_path, prop_path, timeout=timeout, method=method
        )

        results["total"] += 1
        results["total_time"] += result.time_seconds
        results[result.status] += 1
        if result.domains_explored is not None:
            domain_count += 1
            domains_explored_total += result.domains_explored
            if result.domains_verified is not None:
                domains_verified_total += result.domains_verified
            if result.cuts_generated is not None:
                cuts_generated_total += result.cuts_generated
            if result.max_depth_reached is not None:
                max_depth_max = max(max_depth_max, result.max_depth_reached)
        results["instances"].append(
            {
                "network": result.network,
                "property": result.property,
                "status": result.status,
                "time": result.time_seconds,
                "domains_explored": result.domains_explored,
                "domains_verified": result.domains_verified,
                "cuts_generated": result.cuts_generated,
                "max_depth_reached": result.max_depth_reached,
            }
        )

    if results["total"] > 0:
        results["verified_rate"] = results["verified"] / results["total"] * 100
        results["avg_time"] = results["total_time"] / results["total"]
    else:
        results["verified_rate"] = 0.0
        results["avg_time"] = 0.0
    if domain_count > 0:
        results["domains_explored_total"] = domains_explored_total
        results["domains_explored_avg"] = domains_explored_total / domain_count
        results["domains_verified_total"] = domains_verified_total
        results["cuts_generated_total"] = cuts_generated_total
        results["max_depth_max"] = max_depth_max

    return results


# Available benchmarks by year
BENCHMARKS_BY_YEAR = {
    2021: ["acasxu", "mnistfc", "cifar10_resnet", "cifar2020", "nn4sys", "oval21"],
    2023: [
        "acasxu",
        "vit",
        "vggnet16",
        "nn4sys",
        "cgan",
        "yolo",
        "traffic_signs_recognition",
    ],
    2024: [
        "acasxu_2023",
        "vit_2023",
        "vggnet16_2023",
        "cifar100",
        "cora",
        "safenlp",
        "tinyimagenet",
    ],
    2025: [
        "acasxu_2023",
        "vit_2023",
        "nn4sys",
        "soundnessbench",
        "malbeware",
        "sat_relu",
        "cersyve",
        "lsnc_relu",
        "relusplitter",
    ],
}
