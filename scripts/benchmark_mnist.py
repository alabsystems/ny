#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Benchmark MNIST model verification with ny.

Runs IBP, CROWN, α-CROWN, and β-CROWN on the MNIST MLP benchmark.
Compares bound tightness and timing across methods.

Usage:
    python scripts/benchmark_mnist.py [--method METHOD] [--verbose]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

MODEL_PATH = "tests/models/mnist_mlp_2x50.onnx"
PROPERTY_PATH = "tests/models/mnist_robustness_eps0.020_label0.vnnlib"
OUTPUT_DIM = 10


def parse_ny_result(result, method, elapsed, *, require_bounds):
    """Validate ny JSON, semantic exit code, and any claimed bounds."""
    failure = {
        "method": method,
        "status": "ERROR",
        "bounds": [],
        "time_ms": elapsed * 1000,
        "success": False,
        "error": result.stderr.strip() or result.stdout.strip(),
    }
    if not math.isfinite(elapsed) or elapsed < 0:
        failure["error"] = f"invalid elapsed time: {elapsed}"
        return failure
    stdout = result.stdout.strip()
    json_start = stdout.find("{")
    if json_start < 0:
        return failure
    try:
        data = json.loads(stdout[json_start:])
    except json.JSONDecodeError:
        return failure

    aliases = {
        "safe": "verified",
        "verified": "verified",
        "violated": "falsified",
        "falsified": "falsified",
        "unknown": "unknown",
        "potential_violation": "unknown",
        "timeout": "timeout",
    }
    raw_status = data.get("property_status", data.get("status"))
    status = aliases.get(str(raw_status).lower())
    expected_codes = {"verified": 0, "falsified": 1, "unknown": 2, "timeout": 3}
    if status is None or result.returncode != expected_codes[status]:
        failure["error"] = (
            f"invalid status/exit contract: status={raw_status!r}, "
            f"exit={result.returncode}"
        )
        return failure

    bounds = data.get("output_bounds", [])
    if require_bounds:
        soundness = data.get("soundness")
        if not isinstance(soundness, dict) or soundness.get("mode") != "sound":
            failure["error"] = f"ny did not report sound verification: {soundness!r}"
            return failure
        expected_actual_methods = {
            "ibp": "ibp",
            "crown": "crown",
            "alpha": "alphacrown",
        }
        actual_method = (
            str(data.get("actual_method", ""))
            .lower()
            .replace("-", "")
            .replace("_", "")
        )
        if (
            data.get("method") != method
            or actual_method != expected_actual_methods.get(method)
        ):
            failure["error"] = (
                f"ny method mismatch: requested={method!r}, "
                f"reported={data.get('method')!r}, "
                f"actual={data.get('actual_method')!r}"
            )
            return failure
        if not isinstance(bounds, list) or len(bounds) != OUTPUT_DIM:
            failure["error"] = (
                f"ny JSON output shape mismatch: expected {OUTPUT_DIM}, "
                f"got {len(bounds) if isinstance(bounds, list) else 'invalid'}"
            )
            return failure
        try:
            valid_bounds = all(
                math.isfinite(float(bound["lower"]))
                and math.isfinite(float(bound["upper"]))
                and float(bound["lower"]) <= float(bound["upper"])
                for bound in bounds
            )
        except (KeyError, TypeError, ValueError):
            valid_bounds = False
        if not valid_bounds:
            failure["error"] = "ny JSON contained malformed or non-finite bounds"
            return failure
    else:
        integer_metrics = [
            data.get("domains_explored"),
            data.get("domains_verified"),
            data.get("max_depth_reached"),
            data.get("cuts_generated"),
        ]
        scalar_metrics = [
            data.get("epsilon"),
            data.get("threshold"),
            data.get("time_elapsed_s"),
        ]
        width = data.get("output_bound_width")
        metrics_valid = (
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in integer_metrics
            )
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0
                for value in scalar_metrics
            )
            and (
                width is None
                or (
                    isinstance(width, (int, float))
                    and not isinstance(width, bool)
                    and math.isfinite(float(width))
                    and float(width) >= 0
                )
            )
        )
        if not metrics_valid:
            failure["error"] = "beta-crown JSON contained invalid benchmark metrics"
            return failure

    return {
        "method": method,
        "status": status.upper(),
        "bounds": bounds,
        "domains_explored": data.get("domains_explored", 0),
        "time_ms": elapsed * 1000,
        "success": True,
    }


def run_ny_verify(model_path, property_path, method, verbose=False):
    """Run ny verify command and parse results."""
    cmd = [
        "cargo", "run", "--release", "-p", "ny-cli", "--",
        "verify", model_path,
        "--property", property_path,
        "--method", method,
        "--json",
        "--strict",
        "--require-sound",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if verbose:
        print(f"Command: {' '.join(cmd)}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr}")

    return parse_ny_result(result, method, elapsed, require_bounds=True)


def run_ny_beta_crown(model_path, epsilon, verbose=False):
    """Run ny beta-crown command and parse results."""
    cmd = [
        "cargo", "run", "--release", "-p", "ny-cli", "--",
        "beta-crown", model_path,
        "--epsilon", str(epsilon),
        "--threshold", "0.0",
        "--max-domains", "1000",  # Limit for benchmarking
        "--timeout", "30",
        "--json",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if verbose:
        print(f"Command: {' '.join(cmd)}")

    return parse_ny_result(
        result, "beta-crown", elapsed, require_bounds=False
    )


def compute_bound_width(bounds):
    """Compute total output bound width."""
    total = 0.0
    for b in bounds:
        if isinstance(b, dict):
            total += b.get("upper", 0) - b.get("lower", 0)
        elif isinstance(b, (list, tuple)) and len(b) == 2:
            total += b[1] - b[0]
    return total


def main():
    parser = argparse.ArgumentParser(description="Benchmark MNIST verification")
    parser.add_argument("--method", choices=["ibp", "crown", "alpha", "beta-crown", "all"],
                       default="all", help="Verification method")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Run: python scripts/generate_mnist_benchmark.py")
        sys.exit(1)

    if not os.path.exists(PROPERTY_PATH):
        print(f"Property not found: {PROPERTY_PATH}")
        print("Run: python scripts/generate_mnist_benchmark.py")
        sys.exit(1)

    print("=" * 70)
    print("MNIST MLP Verification Benchmark")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print("Architecture: 784 -> 50 -> 50 -> 10 (100 ReLUs)")
    print(f"Property: {PROPERTY_PATH}")
    print("Epsilon: 0.02")
    print()

    results = []
    methods = ["ibp", "crown", "alpha"] if args.method == "all" else [args.method]

    # Run verification methods
    for method in methods:
        if method == "beta-crown":
            continue  # Handle separately
        print(f"Running {method.upper()}...", end=" ", flush=True)
        result = run_ny_verify(MODEL_PATH, PROPERTY_PATH, method, args.verbose)
        results.append(result)

        width = compute_bound_width(result.get("bounds", []))
        outcome = "done" if result["success"] else "FAILED"
        print(
            f"{outcome} in {result['time_ms']:.1f}ms, "
            f"status={result['status']}, width={width:.2f}"
        )

    # Run beta-CROWN
    if args.method in ["all", "beta-crown"]:
        print("Running BETA-CROWN...", end=" ", flush=True)
        result = run_ny_beta_crown(MODEL_PATH, 0.02, args.verbose)
        results.append(result)
        outcome = "done" if result["success"] else "FAILED"
        print(
            f"{outcome} in {result['time_ms']:.1f}ms, "
            f"status={result['status']}, "
            f"domains={result.get('domains_explored', 0)}"
        )

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(f"{'Method':<15} {'Status':<12} {'Time (ms)':<12} {'Bound Width':<12}")
    print("-" * 55)

    for r in results:
        width = compute_bound_width(r.get("bounds", [])) if r.get("bounds") else "-"
        width_str = f"{width:.2f}" if isinstance(width, float) else width
        print(f"{r['method']:<15} {r['status']:<12} {r['time_ms']:<12.1f} {width_str:<12}")

    print()

    # CROWN vs IBP improvement
    ibp_result = next((r for r in results if r["method"] == "ibp"), None)
    crown_result = next((r for r in results if r["method"] == "crown"), None)

    if ibp_result and crown_result:
        ibp_width = compute_bound_width(ibp_result.get("bounds", []))
        crown_width = compute_bound_width(crown_result.get("bounds", []))
        if ibp_width > 0:
            improvement = (ibp_width - crown_width) / ibp_width * 100
            print(f"CROWN vs IBP: {improvement:.1f}% tighter bounds")
    print()

    # Save results
    if args.output:
        output_data = {
            "model": MODEL_PATH,
            "property": PROPERTY_PATH,
            "epsilon": 0.02,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")
    failed = [result for result in results if not result["success"]]
    if failed:
        for result in failed:
            print(f"ERROR: {result['method']}: {result.get('error', 'invalid result')}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
