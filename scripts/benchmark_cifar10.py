#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Benchmark CIFAR-10 model verification with ny.

Compares IBP, CROWN, and α-CROWN on the CIFAR-10 MLP benchmark.
CIFAR-10 has 3072 input dimensions (4x MNIST), stress-testing verification.

Usage:
    python scripts/benchmark_cifar10.py [--method METHOD] [--verbose]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

MODEL_PATH = "tests/models/cifar10_mlp_2x100.onnx"
PROPERTY_PATH = "tests/models/cifar10_robustness_eps0.020_label0.vnnlib"
OUTPUT_DIM = 10


def parse_ny_result(result, method, elapsed):
    """Validate ny JSON, semantic exit code, and output bounds."""
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

    bounds = data.get("output_bounds")
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
    if data.get("method") != method or actual_method != expected_actual_methods.get(method):
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

    return {
        "method": method,
        "status": status.upper(),
        "bounds": bounds,
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

    return parse_ny_result(result, method, elapsed)


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
    parser = argparse.ArgumentParser(description="Benchmark CIFAR-10 verification")
    parser.add_argument("--method", choices=["ibp", "crown", "alpha", "all"],
                       default="all", help="Verification method")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Run: python scripts/generate_cifar10_benchmark.py")
        sys.exit(1)

    if not os.path.exists(PROPERTY_PATH):
        print(f"Property not found: {PROPERTY_PATH}")
        print("Run: python scripts/generate_cifar10_benchmark.py")
        sys.exit(1)

    print("=" * 70)
    print("CIFAR-10 MLP Verification Benchmark")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print("Architecture: 3072 -> 100 -> 100 -> 10 (200 ReLUs)")
    print(f"Property: {PROPERTY_PATH}")
    print("Epsilon: 0.02")
    print("Input dimensions: 3072 (4x MNIST's 784)")
    print()

    results = []
    methods = ["ibp", "crown", "alpha"] if args.method == "all" else [args.method]

    # Run verification methods
    for method in methods:
        print(f"Running {method.upper()}...", end=" ", flush=True)
        result = run_ny_verify(MODEL_PATH, PROPERTY_PATH, method, args.verbose)
        results.append(result)

        width = compute_bound_width(result.get("bounds", []))
        outcome = "done" if result["success"] else "FAILED"
        print(
            f"{outcome} in {result['time_ms']:.1f}ms, "
            f"status={result['status']}, width={width:.2f}"
        )

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(f"{'Method':<15} {'Status':<12} {'Time (ms)':<12} {'Bound Width':<15} {'vs IBP':<12}")
    print("-" * 70)

    ibp_width = None
    for r in results:
        width = compute_bound_width(r.get("bounds", []))
        if r["method"] == "ibp":
            ibp_width = width

        improvement = ""
        if ibp_width and r["method"] != "ibp":
            pct = (ibp_width - width) / ibp_width * 100
            improvement = f"{pct:.1f}% tighter"

        print(f"{r['method']:<15} {r['status']:<12} {r['time_ms']:<12.1f} {width:<15.2f} {improvement:<12}")

    print()

    # CROWN vs IBP improvement
    crown_result = next((r for r in results if r["method"] == "crown"), None)
    alpha_result = next((r for r in results if r["method"] == "alpha"), None)

    if ibp_width and crown_result:
        crown_width = compute_bound_width(crown_result.get("bounds", []))
        improvement = (ibp_width - crown_width) / ibp_width * 100
        print(f"CROWN vs IBP: {improvement:.1f}% tighter bounds")

    if crown_result and alpha_result:
        crown_width = compute_bound_width(crown_result.get("bounds", []))
        alpha_width = compute_bound_width(alpha_result.get("bounds", []))
        if crown_width > 0:
            improvement = (crown_width - alpha_width) / crown_width * 100
            print(f"α-CROWN vs CROWN: {improvement:.1f}% tighter bounds")

    print()

    # Save results
    if args.output:
        output_data = {
            "model": MODEL_PATH,
            "property": PROPERTY_PATH,
            "epsilon": 0.02,
            "input_dims": 3072,
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
