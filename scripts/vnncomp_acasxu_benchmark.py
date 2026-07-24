#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
VNN-COMP ACAS-Xu Benchmark for ny.

This script runs ny on the full ACAS-Xu benchmark suite:
- 45 networks (5 x 9 grid: prev_advisory x tau)
- 10 properties (VNNLIB format)

This establishes our baseline metrics for comparison with α,β-CROWN.

Usage:
    python scripts/vnncomp_acasxu_benchmark.py [--timeout 60] [--method crown]
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
NNET_DIR = PROJECT_ROOT / "research/repos/Marabou/resources/nnet/acasxu"
VNNLIB_DIR = PROJECT_ROOT / "research/repos/nnenum/examples/acasxu/data"
NY_BIN = PROJECT_ROOT / "target/release/ny"
REPORTS_DIR = PROJECT_ROOT / "reports/main"

# ACAS-Xu network mapping: property -> applicable networks
# From VNN-COMP specifications
PROPERTY_NETWORKS = {
    1: [(1, j) for j in range(1, 10)],  # All 1_x networks
    2: list((i, j) for i in range(1, 6) for j in range(1, 10) if not (i == 1 and j == 9)),  # All except 1_9
    3: [(1, j) for j in range(1, 8)] + [(2, j) for j in range(1, 10)],  # 1_1-1_7, 2_1-2_9
    4: [(1, j) for j in range(1, 8)] + [(2, j) for j in range(1, 10)],  # Same as prop 3
    5: [(1, 1)],  # Only network 1_1
    6: [(1, 1)],  # Only network 1_1
    7: [(1, 9)],  # Only network 1_9
    8: [(2, 9)],  # Only network 2_9
    9: [(3, 3)],  # Only network 3_3
    10: [(4, 5)],  # Only network 4_5
}


@dataclass
class VerificationResult:
    network: str
    property_id: int
    status: str  # "verified", "unknown", "violated", "error", "timeout"
    property_status: str  # Status of the specific property
    time_s: float
    bound_width: float
    error: str = ""


def get_network_path(i: int, j: int) -> Path:
    """Get path to ACAS-Xu network i_j."""
    return NNET_DIR / f"ACASXU_experimental_v2a_{i}_{j}.nnet"


def get_property_path(prop_id: int) -> Path:
    """Get path to VNNLIB property file."""
    return VNNLIB_DIR / f"prop_{prop_id}.vnnlib"


def run_verification(
    network_path: Path, property_path: Path, method: str, timeout: int
) -> VerificationResult:
    """Run ny verification on a network with a property."""
    network_name = network_path.stem
    prop_id = int(property_path.stem.split("_")[1])

    cmd = [
        str(NY_BIN),
        "verify",
        str(network_path),
        "--property", str(property_path),
        "--method", method,
        "--timeout", str(timeout),
        "--json"
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        elapsed = time.time() - start

        if result.returncode != 0:
            return VerificationResult(
                network=network_name,
                property_id=prop_id,
                status="error",
                property_status="error",
                time_s=elapsed,
                bound_width=float("inf"),
                error=result.stderr[:200]
            )

        output = json.loads(result.stdout)

        # Calculate bound width
        bound_width = 0.0
        if "output_bounds" in output:
            for b in output["output_bounds"]:
                width = b["upper"] - b["lower"]
                bound_width = max(bound_width, width)

        return VerificationResult(
            network=network_name,
            property_id=prop_id,
            status=output.get("status", "unknown"),
            property_status=output.get("property_status", "unknown"),
            time_s=elapsed,
            bound_width=bound_width
        )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            network=network_name,
            property_id=prop_id,
            status="timeout",
            property_status="timeout",
            time_s=timeout,
            bound_width=float("inf")
        )
    except Exception as e:
        return VerificationResult(
            network=network_name,
            property_id=prop_id,
            status="error",
            property_status="error",
            time_s=time.time() - start,
            bound_width=float("inf"),
            error=str(e)
        )


def run_benchmark(method: str, timeout: int, verbose: bool) -> list:
    """Run full VNN-COMP ACAS-Xu benchmark."""
    results = []

    # Validate paths
    if not NNET_DIR.exists():
        print(f"Error: Network directory not found: {NNET_DIR}")
        sys.exit(1)
    if not VNNLIB_DIR.exists():
        print(f"Error: VNNLIB directory not found: {VNNLIB_DIR}")
        sys.exit(1)
    if not NY_BIN.exists():
        print(f"Error: ny binary not found: {NY_BIN}")
        print("Run: cargo build --release -p ny-cli")
        sys.exit(1)

    # Count total verification instances
    total_instances = sum(len(networks) for networks in PROPERTY_NETWORKS.values())

    print("VNN-COMP ACAS-Xu Benchmark")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Timeout: {timeout}s")
    print(f"Total instances: {total_instances}")
    print(f"Networks: {NNET_DIR}")
    print(f"Properties: {VNNLIB_DIR}")
    print()

    # Run all property/network combinations
    instance_num = 0
    for prop_id, networks in PROPERTY_NETWORKS.items():
        prop_path = get_property_path(prop_id)
        if not prop_path.exists():
            print(f"Warning: Property file not found: {prop_path}")
            continue

        for i, j in networks:
            instance_num += 1
            network_path = get_network_path(i, j)

            if not network_path.exists():
                print(f"Warning: Network not found: {network_path}")
                continue

            print(f"[{instance_num}/{total_instances}] {network_path.stem} + prop_{prop_id}... ", end="", flush=True)

            result = run_verification(network_path, prop_path, method, timeout)
            results.append(result)

            # Status indicator
            status_char = {
                "verified": "V",  # Property verified
                "unknown": "?",   # Inconclusive
                "violated": "X",  # Property violated (counterexample found)
                "timeout": "T",
                "error": "E"
            }.get(result.property_status, "?")

            print(f"{status_char} ({result.time_s:.2f}s, width={result.bound_width:.1f})")

            if verbose and result.error:
                print(f"    Error: {result.error[:100]}")

    return results


def generate_report(results: list, method: str, timeout: int) -> dict:
    """Generate summary report from results."""

    # Count statuses
    verified = sum(1 for r in results if r.property_status == "verified")
    unknown = sum(1 for r in results if r.property_status == "unknown")
    violated = sum(1 for r in results if r.property_status == "violated")
    timeouts = sum(1 for r in results if r.property_status == "timeout")
    errors = sum(1 for r in results if r.property_status == "error")

    total = len(results)
    total_time = sum(r.time_s for r in results)

    # Bounds analysis (exclude errors/timeouts)
    valid_bounds = [r.bound_width for r in results if r.bound_width < float("inf")]
    avg_bound_width = sum(valid_bounds) / len(valid_bounds) if valid_bounds else float("inf")
    max_bound_width = max(valid_bounds) if valid_bounds else float("inf")

    # Per-property breakdown
    property_stats = {}
    for prop_id in PROPERTY_NETWORKS.keys():
        prop_results = [r for r in results if r.property_id == prop_id]
        if prop_results:
            property_stats[f"prop_{prop_id}"] = {
                "total": len(prop_results),
                "verified": sum(1 for r in prop_results if r.property_status == "verified"),
                "unknown": sum(1 for r in prop_results if r.property_status == "unknown"),
                "verified_rate": sum(1 for r in prop_results if r.property_status == "verified") / len(prop_results)
            }

    report = {
        "benchmark": "VNN-COMP ACAS-Xu",
        "method": method,
        "timeout_s": timeout,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_instances": total,
            "verified": verified,
            "unknown": unknown,
            "violated": violated,
            "timeouts": timeouts,
            "errors": errors,
            "verified_rate": verified / total if total > 0 else 0,
            "total_time_s": total_time,
            "avg_time_s": total_time / total if total > 0 else 0,
            "avg_bound_width": avg_bound_width,
            "max_bound_width": max_bound_width
        },
        "property_breakdown": property_stats,
        "results": [
            {
                "network": r.network,
                "property": r.property_id,
                "property_status": r.property_status,
                "time_s": r.time_s,
                "bound_width": r.bound_width if r.bound_width < float("inf") else None,
                "error": r.error if r.error else None
            }
            for r in results
        ]
    }

    return report


def print_summary(report: dict):
    """Print summary statistics."""
    summary = report["summary"]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total instances: {summary['total_instances']}")
    print(f"Verified:        {summary['verified']} ({summary['verified_rate']*100:.1f}%)")
    print(f"Unknown:         {summary['unknown']}")
    print(f"Violated:        {summary['violated']}")
    print(f"Timeouts:        {summary['timeouts']}")
    print(f"Errors:          {summary['errors']}")
    print()
    print(f"Total time:      {summary['total_time_s']:.1f}s")
    print(f"Avg time:        {summary['avg_time_s']:.2f}s")
    print(f"Avg bound width: {summary['avg_bound_width']:.1f}")
    print(f"Max bound width: {summary['max_bound_width']:.1f}")
    print()

    # Per-property table
    print("Per-property verified rate:")
    print("-" * 40)
    for prop_name, stats in report["property_breakdown"].items():
        rate = stats["verified_rate"] * 100
        print(f"  {prop_name}: {stats['verified']}/{stats['total']} ({rate:.1f}%)")
    print()

    # Comparison target
    print("VNN-COMP reference (α,β-CROWN):")
    print("  Verified rate: ~95%")
    print("  Avg time: <1s per instance")
    print()


def main():
    parser = argparse.ArgumentParser(description="VNN-COMP ACAS-Xu benchmark")
    parser.add_argument("--method", choices=["ibp", "crown", "alpha"], default="crown",
                        help="Verification method")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per instance in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(args.method, args.timeout, args.verbose)

    # Generate report
    report = generate_report(results, args.method, args.timeout)

    # Print summary
    print_summary(report)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"vnncomp_acasxu_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    # Also save latest as fixed name for easy comparison
    latest_path = REPORTS_DIR / f"vnncomp_acasxu_{args.method}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
