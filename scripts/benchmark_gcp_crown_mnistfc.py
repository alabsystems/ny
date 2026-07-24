#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Benchmark GCP-CROWN effectiveness on MNIST FC.

Compares verification rate with and without cutting planes.
"""

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def run_verification(model: str, prop: str, timeout: int, enable_cuts: bool) -> dict:
    """Run ny beta-crown and parse the result."""
    cmd = [
        "./target/release/ny", "beta-crown",
        model,
        "--property", prop,
        "--timeout", str(timeout),
        "--json"
    ]
    if enable_cuts:
        cmd.append("--enable-cuts")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30  # Extra buffer
        )

        # Parse JSON output (may be multi-line pretty-printed)
        stdout = result.stdout.strip()

        # Try to parse as JSON first
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

        # Fallback: look for JSON object in output (may have other text before)
        if '{' in stdout:
            json_start = stdout.find('{')
            json_end = stdout.rfind('}') + 1
            if json_end > json_start:
                try:
                    return json.loads(stdout[json_start:json_end])
                except json.JSONDecodeError:
                    pass

        # Fallback: parse text output
        if "VERIFIED" in stdout.upper():
            return {"status": "verified", "domains_explored": 0}
        if "VIOLATED" in stdout.upper():
            return {"status": "falsified", "domains_explored": 0}
        return {"status": "unknown", "domains_explored": 0}

    except subprocess.TimeoutExpired:
        return {"status": "Timeout", "domains_explored": 0}
    except Exception as e:
        return {"status": f"Error: {e}", "domains_explored": 0}


def main():
    benchmark_dir = Path("benchmarks/vnncomp2021/benchmarks/mnistfc")
    models = [
        "mnist-net_256x2.onnx",
        # "mnist-net_256x4.onnx",  # Skip for now
    ]

    timeout = 30  # seconds per instance

    # Collect instances
    instances = []
    for model in models:
        model_path = benchmark_dir / model
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        # Find all properties for this model
        for prop_file in sorted(benchmark_dir.glob("prop_*.vnnlib")):
            instances.append((str(model_path), str(prop_file)))

    print(f"Found {len(instances)} instances")
    print(f"Timeout: {timeout}s per instance")
    print()

    # Run benchmark
    results = {"without_cuts": defaultdict(int), "with_cuts": defaultdict(int)}
    details = []

    for i, (model, prop) in enumerate(instances):  # All instances
        prop_name = Path(prop).stem
        print(f"[{i+1}/{len(instances)}] {prop_name}...", end=" ", flush=True)

        # Without cuts
        t0 = time.time()
        result_no_cuts = run_verification(model, prop, timeout, enable_cuts=False)
        t_no_cuts = time.time() - t0
        status_no_cuts = result_no_cuts.get("status", "Unknown")

        # With cuts
        t0 = time.time()
        result_with_cuts = run_verification(model, prop, timeout, enable_cuts=True)
        t_with_cuts = time.time() - t0
        status_with_cuts = result_with_cuts.get("status", "Unknown")

        results["without_cuts"][status_no_cuts] += 1
        results["with_cuts"][status_with_cuts] += 1

        details.append({
            "property": prop_name,
            "without_cuts": {"status": status_no_cuts, "time": t_no_cuts},
            "with_cuts": {"status": status_with_cuts, "time": t_with_cuts}
        })

        # Print compact result
        def short_status(s):
            if s == "Verified" or s == "verified":
                return "VER"
            if s == "Falsified" or s == "falsified":
                return "FAL"
            if s == "Unknown" or s == "unknown":
                return "UNK"
            return s[:3].upper()

        print(f"no_cuts={short_status(status_no_cuts)} ({t_no_cuts:.1f}s), "
              f"cuts={short_status(status_with_cuts)} ({t_with_cuts:.1f}s)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nWithout GCP-CROWN:")
    for status, count in sorted(results["without_cuts"].items()):
        print(f"  {status}: {count}")

    print("\nWith GCP-CROWN:")
    for status, count in sorted(results["with_cuts"].items()):
        print(f"  {status}: {count}")

    # Calculate improvement (handle various case variations)
    def count_verified(d):
        return sum(v for k, v in d.items() if k.lower() == "verified")
    def count_falsified(d):
        return sum(v for k, v in d.items() if k.lower() in ("falsified", "violated"))
    def count_unknown(d):
        return sum(v for k, v in d.items() if k.lower() == "unknown")

    verified_no_cuts = count_verified(results["without_cuts"])
    verified_with_cuts = count_verified(results["with_cuts"])
    total = sum(results["without_cuts"].values())

    print("\nVerification rate:")
    print(f"  Without cuts: {verified_no_cuts}/{total} ({100*verified_no_cuts/total:.1f}%)")
    print(f"  With cuts:    {verified_with_cuts}/{total} ({100*verified_with_cuts/total:.1f}%)")

    if verified_with_cuts > verified_no_cuts:
        improvement = verified_with_cuts - verified_no_cuts
        print(f"\n  GCP-CROWN improves verification by {improvement} instances ({100*improvement/total:.1f}%)")
    elif verified_with_cuts == verified_no_cuts:
        print("\n  No change with GCP-CROWN (may need more iterations)")
    else:
        print("\n  Warning: GCP-CROWN reduced verification rate (unexpected)")


if __name__ == "__main__":
    main()
