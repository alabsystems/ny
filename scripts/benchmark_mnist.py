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
import os
import re
import subprocess
import sys
import time

MODEL_PATH = "tests/models/mnist_mlp_2x50.onnx"
PROPERTY_PATH = "tests/models/mnist_robustness_eps0.020_label0.vnnlib"

def run_ny_verify(model_path, property_path, method, verbose=False):
    """Run ny verify command and parse results."""
    cmd = [
        "cargo", "run", "--release", "-p", "ny-cli", "--",
        "verify", model_path,
        "--property", property_path,
        "--method", method,
        "--json"
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if verbose:
        print(f"Command: {' '.join(cmd)}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr}")

    # Parse JSON output
    try:
        # Extract JSON from output (may have build messages before it)
        output = result.stdout
        json_start = output.find('{')
        if json_start >= 0:
            json_str = output[json_start:]
            data = json.loads(json_str)
            # Convert status to uppercase for display
            prop_status = data.get("property_status", "unknown").upper()
            return {
                "method": method,
                "status": prop_status,
                "bounds": data.get("output_bounds", []),
                "time_ms": elapsed * 1000,
                "success": True
            }
    except json.JSONDecodeError as e:
        if verbose:
            print(f"JSON parse error: {e}")
            print(f"Output: {result.stdout[:500]}")

    # Fallback: parse text output
    return {
        "method": method,
        "status": "PARSE_ERROR",
        "bounds": [],
        "time_ms": elapsed * 1000,
        "success": False,
        "raw_output": result.stdout[-500:] if result.stdout else result.stderr[-500:]
    }


def run_ny_beta_crown(model_path, epsilon, verbose=False):
    """Run ny beta-crown command and parse results."""
    cmd = [
        "cargo", "run", "--release", "-p", "ny-cli", "--",
        "beta-crown", model_path,
        "--epsilon", str(epsilon),
        "--threshold", "0.0",
        "--max-domains", "1000",  # Limit for benchmarking
        "--timeout", "30"
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if verbose:
        print(f"Command: {' '.join(cmd)}")

    output = result.stdout + result.stderr

    # Parse status
    status = "UNKNOWN"
    domains = 0
    if "VERIFIED" in output:
        status = "VERIFIED"
    elif "FALSIFIED" in output:
        status = "FALSIFIED"

    # Parse domains explored
    match = re.search(r"Domains explored: (\d+)", output)
    if match:
        domains = int(match.group(1))

    return {
        "method": "beta-crown",
        "status": status,
        "domains_explored": domains,
        "time_ms": elapsed * 1000,
        "success": True
    }


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
        print(f"done in {result['time_ms']:.1f}ms, status={result['status']}, width={width:.2f}")

    # Run beta-CROWN
    if args.method in ["all", "beta-crown"]:
        print("Running BETA-CROWN...", end=" ", flush=True)
        result = run_ny_beta_crown(MODEL_PATH, 0.02, args.verbose)
        results.append(result)
        print(f"done in {result['time_ms']:.1f}ms, status={result['status']}, domains={result.get('domains_explored', 0)}")

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


if __name__ == "__main__":
    main()
