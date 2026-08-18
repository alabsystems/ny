#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Audit the quarantined legacy GCP-CROWN surface on MNIST FC.

The cut-enabled arm is expected to return a configuration error until
proof-derived cuts are folded through backward CROWN. It is retained as a
research/quarantine regression, not as a scored verification benchmark.
"""

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

QUARANTINE_MARKER = "cut proof authority is quarantined"


def run_verification(model: str, prop: str, timeout: int, enable_cuts: bool) -> dict:
    """Run ny beta-crown; cut requests must fail at configuration ingress."""
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

        if enable_cuts:
            diagnostic = f"{result.stdout}\n{result.stderr}".lower()
            if result.returncode == 0 or QUARANTINE_MARKER not in diagnostic:
                raise RuntimeError(
                    "cut-authority quarantine regression: --enable-cuts did not "
                    f"fail with the expected marker (returncode={result.returncode}, "
                    f"stdout={result.stdout!r}, stderr={result.stderr!r})"
                )
            return {
                "status": "quarantined",
                "domains_explored": 0,
                "returncode": result.returncode,
            }

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
        if enable_cuts:
            raise RuntimeError(
                "--enable-cuts reached execution/timeout instead of failing at "
                "the cut-authority configuration quarantine"
            )
        return {"status": "Timeout", "domains_explored": 0}
    except Exception as e:
        if enable_cuts:
            raise
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
    results = {"without_cuts": defaultdict(int), "cut_request": defaultdict(int)}
    details = []

    for i, (model, prop) in enumerate(instances):  # All instances
        prop_name = Path(prop).stem
        print(f"[{i+1}/{len(instances)}] {prop_name}...", end=" ", flush=True)

        # Without cuts
        t0 = time.time()
        result_no_cuts = run_verification(model, prop, timeout, enable_cuts=False)
        t_no_cuts = time.time() - t0
        status_no_cuts = result_no_cuts.get("status", "Unknown")

        # A cut request must be rejected before any proof execution.
        t0 = time.time()
        result_with_cuts = run_verification(model, prop, timeout, enable_cuts=True)
        t_with_cuts = time.time() - t0
        status_with_cuts = result_with_cuts.get("status", "Unknown")

        results["without_cuts"][status_no_cuts] += 1
        results["cut_request"][status_with_cuts] += 1

        details.append({
            "property": prop_name,
            "without_cuts": {"status": status_no_cuts, "time": t_no_cuts},
            "cut_request": {"status": status_with_cuts, "time": t_with_cuts}
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
              f"cut_request={short_status(status_with_cuts)} ({t_with_cuts:.1f}s)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nWithout GCP-CROWN:")
    for status, count in sorted(results["without_cuts"].items()):
        print(f"  {status}: {count}")

    print("\nQuarantined --enable-cuts request:")
    for status, count in sorted(results["cut_request"].items()):
        print(f"  {status}: {count}")

    # Report only the real, cut-dark verification arm. The rejected request is
    # a configuration regression and must never enter a verification-rate
    # comparison.
    def count_verified(d):
        return sum(v for k, v in d.items() if k.lower() == "verified")

    verified_no_cuts = count_verified(results["without_cuts"])
    total = sum(results["without_cuts"].values())
    quarantined = results["cut_request"].get("quarantined", 0)
    if quarantined != total:
        raise RuntimeError(
            f"expected all {total} cut requests to be quarantined, got {quarantined}"
        )

    print("\nVerification rate:")
    print(f"  Without cuts: {verified_no_cuts}/{total} ({100*verified_no_cuts/total:.1f}%)")
    print(
        f"\nCut-authority quarantine: PASSED ({quarantined}/{total} requests "
        "rejected before proof execution)"
    )


if __name__ == "__main__":
    main()
