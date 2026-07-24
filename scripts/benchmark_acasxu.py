#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
ACAS-Xu Verification Benchmark for ny.

This script benchmarks ny on the ACAS-Xu collision avoidance networks,
a standard benchmark from VNN-COMP (Verification of Neural Networks Competition).

ACAS-Xu network structure:
- 5 inputs: ρ (distance), θ (angle to intruder), ψ (heading), v_own, v_int
- 6 hidden layers of 50 neurons each
- 5 outputs: COC (clear of conflict), weak left, weak right, strong left, strong right

Usage:
    python scripts/benchmark_acasxu.py [--epsilon 0.01] [--method ibp|crown|beta]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ACAS-Xu models directory
MODELS_DIR = Path(__file__).parent.parent / "research/repos/Marabou/resources/nnet/acasxu"
NY_BIN = Path(__file__).parent.parent / "target/release/ny"


def run_verification(model_path: Path, epsilon: float, method: str) -> dict:
    """Run ny verification on a model."""
    if method == "beta":
        cmd = [
            str(NY_BIN),
            "beta-crown",
            str(model_path),
            "--epsilon", str(epsilon),
            "--threshold", "0.0",
            "--max-domains", "1000",
            "--timeout", "30",
            "--json"
        ]
    else:
        cmd = [
            str(NY_BIN),
            "verify",
            str(model_path),
            "--epsilon", str(epsilon),
            "--method", method,
            "--json"
        ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr,
                "time_s": elapsed
            }

        output = json.loads(result.stdout)
        output["time_s"] = elapsed
        return output
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "time_s": 60.0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time_s": time.time() - start
        }


def get_bounds_width(output: dict) -> float:
    """Get maximum output bound width from verification result."""
    if "output_bounds" not in output:
        return float("inf")

    bounds = output["output_bounds"]
    max_width = 0
    for b in bounds:
        width = b["upper"] - b["lower"]
        max_width = max(max_width, width)
    return max_width


def main():
    parser = argparse.ArgumentParser(description="ACAS-Xu verification benchmark")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Perturbation epsilon")
    parser.add_argument("--method", choices=["ibp", "crown", "alpha", "beta"], default="crown",
                        help="Verification method")
    parser.add_argument("--num-models", type=int, default=5, help="Number of models to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check if ny binary exists
    if not NY_BIN.exists():
        print(f"Error: ny binary not found at {NY_BIN}")
        print("Run: cargo build --release -p ny-cli")
        sys.exit(1)

    # Check if models directory exists
    if not MODELS_DIR.exists():
        # Try local test model
        local_model = Path(__file__).parent.parent / "tests/models/acasxu_1_1.nnet"
        if local_model.exists():
            models = [local_model]
        else:
            print(f"Error: ACAS-Xu models not found at {MODELS_DIR}")
            print("Copy ACAS-Xu .nnet files to tests/models/")
            sys.exit(1)
    else:
        # Get first N models
        models = sorted(MODELS_DIR.glob("ACASXU_experimental_v2a_*.nnet"))[:args.num_models]

    if not models:
        print("No ACAS-Xu models found!")
        sys.exit(1)

    print("ACAS-Xu Verification Benchmark")
    print("==============================")
    print(f"Method: {args.method}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Models: {len(models)}")
    print()

    results = []
    total_time = 0

    print(f"{'Model':<30} {'Status':<15} {'Time (s)':<10} {'Bound Width':<15}")
    print("-" * 70)

    for model_path in models:
        model_name = model_path.name

        result = run_verification(model_path, args.epsilon, args.method)
        results.append({
            "model": model_name,
            **result
        })

        status = result.get("status", "unknown")
        time_s = result.get("time_s", 0)
        total_time += time_s

        bound_width = get_bounds_width(result)
        bound_str = f"{bound_width:.4f}" if bound_width < float("inf") else "N/A"

        print(f"{model_name:<30} {status:<15} {time_s:<10.3f} {bound_str:<15}")

        if args.verbose and "error" in result:
            print(f"  Error: {result['error'][:100]}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {'':<15} {total_time:<10.3f}")
    print()

    # Summary statistics
    verified = sum(1 for r in results if r.get("status") == "verified")
    errors = sum(1 for r in results if r.get("status") == "error")
    timeouts = sum(1 for r in results if r.get("status") == "timeout")

    print("Summary:")
    print(f"  Verified: {verified}/{len(results)}")
    print(f"  Errors: {errors}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Average time: {total_time/len(results):.3f}s")

    # Save results
    report_dir = Path(__file__).parent.parent / "reports" / "main"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"acasxu_benchmark_{args.method}.json"

    with open(report_path, "w") as f:
        json.dump({
            "method": args.method,
            "epsilon": args.epsilon,
            "total_time_s": total_time,
            "num_models": len(results),
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    main()
