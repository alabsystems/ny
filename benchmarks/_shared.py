# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Shared helpers for benchmark tests.

These are intentionally *not* in `conftest.py` so test modules can import them
without risking collisions with other `conftest.py` files in the repo.

VNN-COMP 2021-2026 benchmark suite.
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
    """Require a benchmark path for an explicitly requested diagnostic."""
    if path is None or not path.exists():
        raise AssertionError(
            f"{message}. Install the external corpus with "
            "benchmarks/download_benchmarks.sh before running this diagnostic."
        )
    return path


def require_benchmark_items(items: list, message: str) -> list:
    """Require benchmark items for an explicitly requested diagnostic."""
    if not items:
        raise AssertionError(
            f"{message}. Install the external corpus with "
            "benchmarks/download_benchmarks.sh before running this diagnostic."
        )
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
    2026: BENCHMARK_DIR / "vnncomp2026" / "benchmarks",
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


_STATUS_ALIASES = {
    "safe": "verified",
    "verified": "verified",
    "violated": "falsified",
    "falsified": "falsified",
    "unknown": "unknown",
    "potential_violation": "unknown",
    "potential violation": "unknown",
    "timeout": "timeout",
}
_STATUS_EXIT_CODES = {
    "verified": 0,
    "falsified": 1,
    "unknown": 2,
    "timeout": 3,
}


def _canonical_status(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _STATUS_ALIASES.get(value.strip().lower())


def _text_status(stdout: str) -> str | None:
    """Extract an exact status field from human-readable ny output."""
    statuses = set()
    for line in stdout.splitlines():
        normalized = line.strip().lower()
        for prefix in ("status:", "property status:"):
            if normalized.startswith(prefix):
                status = _canonical_status(normalized.removeprefix(prefix).strip())
                if status is not None:
                    statuses.add(status)
                break
    if len(statuses) == 1:
        return statuses.pop()
    return None


def _error_result(
    network_path: Path,
    vnnlib_path: Path,
    elapsed: float,
    result: subprocess.CompletedProcess[str],
    reason: str,
) -> VerificationResult:
    detail = result.stderr.strip()
    message = f"{reason} (exit code {result.returncode})"
    if detail:
        message = f"{message}: {detail[:500]}"
    return VerificationResult(
        network=network_path.name,
        property=vnnlib_path.name,
        status="error",
        time_seconds=elapsed,
        error_message=message,
    )


def _validate_process_status(
    network_path: Path,
    vnnlib_path: Path,
    elapsed: float,
    result: subprocess.CompletedProcess[str],
    status: str | None,
) -> VerificationResult | None:
    if status is None:
        return _error_result(
            network_path,
            vnnlib_path,
            elapsed,
            result,
            "ny output did not contain a recognized verification status",
        )
    expected_exit_code = _STATUS_EXIT_CODES[status]
    if result.returncode != expected_exit_code:
        return _error_result(
            network_path,
            vnnlib_path,
            elapsed,
            result,
            (
                f"ny reported {status!r}, which requires exit code "
                f"{expected_exit_code}"
            ),
        )
    return None


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

        try:
            output = json.loads(result.stdout)
            if not isinstance(output, dict):
                return _error_result(
                    network_path,
                    vnnlib_path,
                    elapsed,
                    result,
                    "ny JSON output was not an object",
                )
            # Use property_status for actual verification result
            # (status = process status, property_status = verification result)
            status = _canonical_status(
                output.get("property_status", output.get("status"))
            )
            invalid = _validate_process_status(
                network_path, vnnlib_path, elapsed, result, status
            )
            if invalid is not None:
                return invalid
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
            # Be strict if an older binary ignored --json: only accept an
            # explicit status field, and require it to agree with the exit code.
            status = _text_status(result.stdout)
            invalid = _validate_process_status(
                network_path, vnnlib_path, elapsed, result, status
            )
            if invalid is not None:
                return invalid
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
    year: int,
    benchmark: str,
    benchmark_root: Path | None = None,
    *,
    preserve_source_rows: bool = False,
) -> list[tuple[Path, Path, int]]:
    """Get (network, property, timeout) tuples from instances.csv.

    Returns list of (network_path, property_path, timeout_seconds).  By
    default, rows whose materialized inputs are unavailable are omitted for
    compatibility with the benchmark diagnostics.  With
    ``preserve_source_rows=True``, every parsed data row is retained in source
    order, using its expected path when an input is absent.  Index-based
    harnesses must use that mode so an index always denotes the corresponding
    unfiltered ``instances.csv`` row rather than a compacted local subset.
    """
    bench_dir = get_benchmark_dir(year, benchmark, benchmark_root)
    if not bench_dir:
        return []

    instances_file = bench_dir / "instances.csv"
    if not instances_file.exists():
        # Fallback: look for acasxu_instances.csv etc.
        instances_file = bench_dir / f"{benchmark}_instances.csv"

    if year >= 2026 and not instances_file.exists():
        # VNN-COMP 2026 publishes category payloads below version directories
        # (`category/1.0`, optionally alongside `2.0`) while the benchmark root
        # still resolves to `category`. Prefer the regular 1.0 manifest, then
        # any other version directory in deterministic lexical order. This is
        # deliberately based on an actual instances file rather than mere
        # directory existence, so legacy unversioned layouts are unchanged.
        version_dirs = sorted(
            (path for path in bench_dir.iterdir() if path.is_dir()),
            key=lambda path: (path.name != "1.0", path.name),
        )
        for version_dir in version_dirs:
            for candidate in (
                version_dir / "instances.csv",
                version_dir / f"{benchmark}_instances.csv",
            ):
                if candidate.is_file():
                    bench_dir = version_dir
                    instances_file = candidate
                    break
            if instances_file.exists():
                break

    if not instances_file.exists():
        return []

    instances: list[tuple[Path, Path, int]] = []
    with open(instances_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#") or row[0] == "network":
                continue  # Skip comments and headers
            if len(row) < 2:
                if preserve_source_rows:
                    raise ValueError(
                        f"{instances_file} line {reader.line_num} is not a complete "
                        "instances.csv data row"
                    )
                continue

            network_name = row[0]
            prop_name = row[1]
            try:
                raw_timeout = float(row[2]) if len(row) > 2 else 60.0
                if not raw_timeout.is_integer() or raw_timeout <= 0:
                    raise ValueError
                timeout = int(raw_timeout)
            except (ValueError, IndexError, OverflowError) as error:
                if preserve_source_rows:
                    raise ValueError(
                        f"{instances_file} line {reader.line_num} has an invalid timeout"
                    ) from error
                timeout = 60

            # Resolve to the logical decompressed input even when the manifest
            # itself names an adjacent `.onnx.gz`/`.vnnlib.gz` archive.  The
            # bounded runner stages archives; returning an existing archive as
            # though it were a solver-ready ONNX/VNNLIB file would pass gzip
            # bytes directly to NY.
            logical_network_name = (
                network_name[:-3]
                if network_name.endswith(".onnx.gz")
                else network_name
            )
            logical_prop_name = (
                prop_name[:-3]
                if prop_name.endswith(".vnnlib.gz")
                else prop_name
            )
            logical_network_path = Path(logical_network_name)
            logical_prop_path = Path(logical_prop_name)
            invalid_logical_paths = (
                logical_network_path.is_absolute()
                or ".." in logical_network_path.parts
                or logical_network_path.suffix != ".onnx"
                or logical_prop_path.is_absolute()
                or ".." in logical_prop_path.parts
                or logical_prop_path.suffix != ".vnnlib"
            )
            if invalid_logical_paths:
                if preserve_source_rows:
                    raise ValueError(
                        f"{instances_file} line {reader.line_num} has an invalid "
                        "logical ONNX/VNNLIB path"
                    )
                continue
            network_candidates = [
                bench_dir / logical_network_name,
                bench_dir / "onnx" / logical_network_name,
            ]
            prop_candidates = [
                bench_dir / logical_prop_name,
                bench_dir / "vnnlib" / logical_prop_name,
            ]

            # Official 2026 corpora commonly retain only an adjacent `.gz`
            # archive while `instances.csv` names the logical decompressed
            # path. Return that logical path only to staging-aware callers;
            # legacy callers keep omitting archive-only rows rather than
            # receiving a nonexistent path or gzip bytes as solver input.
            network_path = next(
                (
                    path
                    for path in network_candidates
                    if path.is_file()
                    or (
                        preserve_source_rows
                        and Path(f"{path}.gz").is_file()
                    )
                ),
                None,
            )
            prop_path = next(
                (
                    path
                    for path in prop_candidates
                    if path.is_file()
                    or (
                        preserve_source_rows
                        and Path(f"{path}.gz").is_file()
                    )
                ),
                None,
            )

            if network_path and prop_path:
                instances.append((network_path, prop_path, timeout))
            elif preserve_source_rows:
                instances.append(
                    (
                        network_path or network_candidates[0],
                        prop_path or prop_candidates[0],
                        timeout,
                    )
                )

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
