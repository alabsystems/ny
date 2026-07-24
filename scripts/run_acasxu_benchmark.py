#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Full ACAS-Xu benchmark script for ny.

Runs all 186 instances from VNN-COMP 2021 and reports detailed statistics.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Paths
NY_BINARY = Path(__file__).parent.parent / "target" / "release" / "ny"
ACASXU_DIR = Path(__file__).parent.parent / "benchmarks" / "vnncomp2021" / "benchmarks" / "acasxu"
INSTANCES_CSV = ACASXU_DIR / "acasxu_instances.csv"


@dataclass
class Result:
    network: str
    prop: str
    status: str
    time_sec: float
    cuts_generated: int = 0
    domains_explored: int = 0
    domains_verified: int = 0
    max_depth_reached: int = 0
    error: Optional[str] = None


def _normalize_status(status: str) -> str:
    status = status.lower().strip()
    if status in {"safe", "verified"}:
        return "verified"
    if status in {"violated", "falsified", "unsafe"}:
        return "falsified"
    if status in {"timeout"}:
        return "timeout"
    if status in {"error"}:
        return "error"
    return "unknown"


def run_instance(
    network_path: Path,
    prop_path: Path,
    *,
    timeout: int,
    max_domains: int,
    branching: str,
    pgd_attack: bool,
    pgd_restarts: int,
    proactive_cuts: bool,
    max_proactive_cuts: int,
) -> Result:
    """Run ny beta-crown on a single instance."""
    start = time.time()
    network_name = network_path.name
    prop_name = prop_path.name

    cmd = [
        str(NY_BINARY),
        "beta-crown",
        str(network_path),
        "--property", str(prop_path),
        "--timeout", str(timeout),
        "--branching", branching,
        "--max-domains",
        str(max_domains),
        "--json",
    ]

    if pgd_attack:
        cmd.extend(["--pgd-attack", "--pgd-restarts", str(pgd_restarts)])

    if proactive_cuts:
        cmd.extend(["--proactive-cuts", "--max-proactive-cuts", str(max_proactive_cuts)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                status_raw = output.get("property_status", output.get("status", "unknown"))
                return Result(
                    network=network_name,
                    prop=prop_name,
                    status=_normalize_status(str(status_raw)),
                    time_sec=elapsed,
                    cuts_generated=int(output.get("cuts_generated", 0) or 0),
                    domains_explored=int(output.get("domains_explored", 0) or 0),
                    domains_verified=int(output.get("domains_verified", 0) or 0),
                    max_depth_reached=int(output.get("max_depth_reached", 0) or 0),
                )
            except json.JSONDecodeError:
                stdout_lower = result.stdout.lower()
                if "verified" in stdout_lower or "safe" in stdout_lower:
                    status = "verified"
                elif "falsified" in stdout_lower or "violated" in stdout_lower:
                    status = "falsified"
                else:
                    status = "unknown"
                return Result(network_name, prop_name, status, elapsed, error="invalid_json")
        else:
            return Result(network_name, prop_name, "error", elapsed, error=result.stderr[:200])
    except subprocess.TimeoutExpired:
        return Result(network_name, prop_name, "timeout", timeout)
    except Exception as e:
        return Result(network_name, prop_name, "error", time.time() - start, error=str(e))


def load_instances() -> list[tuple[Path, Path, int]]:
    """Load instances from CSV."""
    instances = []
    with open(INSTANCES_CSV) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or row[0].startswith("#"):
                continue
            network = ACASXU_DIR / row[0]
            prop = ACASXU_DIR / row[1]
            timeout = int(row[2]) if len(row) > 2 else 30
            if network.exists() and prop.exists():
                instances.append((network, prop, timeout))
    return instances


def _save_report(report: dict, *, tag: str, latest_tag: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent.parent / "reports" / "main" / f"acasxu_beta_crown_{tag}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    latest_path = output_path.parent / f"acasxu_beta_crown_{latest_tag}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="ACAS-Xu VNN-COMP beta-crown benchmark")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per instance")
    parser.add_argument("--use-csv-timeouts", action="store_true", help="Use per-instance timeouts from CSV")
    parser.add_argument("--max-instances", type=int, default=0, help="Limit to first N instances (0 = all)")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (ignored with --compare-proactive-cuts)")
    parser.add_argument("--max-domains", type=int, default=50000, help="Max domains for beta-crown")
    parser.add_argument(
        "--branching",
        type=str,
        default="width",
        choices=["width", "input", "sequential", "babsr", "fsb"],
        help="Branching strategy (default: width, which uses ReLU splitting)",
    )
    parser.add_argument("--no-pgd-attack", action="store_true", help="Disable PGD counterexample search")
    parser.add_argument("--pgd-restarts", type=int, default=5000, help="PGD restarts (if enabled)")
    parser.add_argument("--proactive-cuts", action="store_true", help="Enable proactive cuts (BICCOS-lite)")
    parser.add_argument("--max-proactive-cuts", type=int, default=100, help="Max proactive cuts to generate")
    parser.add_argument(
        "--compare-proactive-cuts",
        action="store_true",
        help="Run each instance with and without proactive cuts and compare outcomes",
    )
    args = parser.parse_args()
    pgd_attack = not args.no_pgd_attack

    instances = load_instances()
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    if not NY_BINARY.exists():
        print(f"Error: ny binary not found at {NY_BINARY}")
        print("Run: cargo build --release -p ny-cli")
        return 2

    print(f"Running {len(instances)} ACAS-Xu instances...")
    print("=" * 70)

    results: list[Result] = []
    counts = {"verified": 0, "falsified": 0, "unknown": 0, "timeout": 0, "error": 0}
    total_time = 0.0
    total_cuts = 0

    start_all = time.time()

    def run_one(network: Path, prop: Path, timeout_csv: int, *, proactive: bool) -> Result:
        return run_instance(
            network,
            prop,
            timeout=timeout_csv if args.use_csv_timeouts else args.timeout,
            max_domains=args.max_domains,
            branching=args.branching,
            pgd_attack=pgd_attack,
            pgd_restarts=args.pgd_restarts,
            proactive_cuts=proactive,
            max_proactive_cuts=args.max_proactive_cuts,
        )

    def record_result(completed: int, network: Path, prop: Path, result: Result) -> None:
        nonlocal total_time, total_cuts
        counts[result.status] = counts.get(result.status, 0) + 1
        total_time += result.time_sec
        total_cuts += result.cuts_generated

        net_id = network.stem.replace("ACASXU_run2a_", "").replace("_batch_2000", "")
        prop_id = prop.stem.replace("prop_", "")

        if result.status != "verified":
            print(
                f"[{completed:3}/{len(instances)}] {net_id} x {prop_id}: {result.status:10} "
                f"({result.time_sec:.2f}s, cuts={result.cuts_generated}, domains={result.domains_explored})"
            )

        if completed % 50 == 0:
            resolved = counts["verified"] + counts["falsified"]
            print(
                f"--- Progress: {completed}/{len(instances)} | Resolved: {resolved} | Verified: {counts['verified']} "
                f"| Falsified: {counts['falsified']} | Unknown: {counts['unknown']} ---"
            )

    compare_rows: list[dict] = []
    mismatches: list[dict] = []

    if args.compare_proactive_cuts:
        base_counts = {"verified": 0, "falsified": 0, "unknown": 0, "timeout": 0, "error": 0}
        base_total_time = 0.0
        base_total_cuts = 0

        for i, (network, prop, timeout_csv) in enumerate(instances, 1):
            base = run_one(network, prop, timeout_csv, proactive=False)
            pro = run_one(network, prop, timeout_csv, proactive=True)

            results.append(pro)
            record_result(i, network, prop, pro)

            base_counts[base.status] = base_counts.get(base.status, 0) + 1
            base_total_time += base.time_sec
            base_total_cuts += base.cuts_generated

            compare_rows.append(
                {
                    "network": base.network,
                    "property": base.prop,
                    "baseline": base.__dict__,
                    "proactive": pro.__dict__,
                }
            )
            if base.status != pro.status:
                mismatches.append(
                    {
                        "network": base.network,
                        "property": base.prop,
                        "baseline_status": base.status,
                        "proactive_status": pro.status,
                    }
                )
    elif args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_meta: dict[object, tuple[Path, Path]] = {}
            for network, prop, timeout_csv in instances:
                future = executor.submit(run_one, network, prop, timeout_csv, proactive=args.proactive_cuts)
                future_to_meta[future] = (network, prop)

            completed = 0
            for future in as_completed(future_to_meta):
                network, prop = future_to_meta[future]
                result = future.result()
                completed += 1
                results.append(result)
                record_result(completed, network, prop, result)
    else:
        for i, (network, prop, timeout_csv) in enumerate(instances, 1):
            result = run_one(network, prop, timeout_csv, proactive=args.proactive_cuts)
            results.append(result)
            record_result(i, network, prop, result)

    elapsed_all = time.time() - start_all

    # Final report
    print("\n" + "=" * 70)
    print("ACAS-Xu Benchmark Results")
    print("=" * 70)

    resolved = counts["verified"] + counts["falsified"]
    resolution_rate = resolved / len(instances) * 100 if instances else 0
    verified_rate = counts["verified"] / len(instances) * 100 if instances else 0

    print(f"\nTotal instances: {len(instances)}")
    print(f"  Verified:   {counts['verified']:4} ({counts['verified']/len(instances)*100:.1f}%)")
    print(f"  Falsified:  {counts['falsified']:4} ({counts['falsified']/len(instances)*100:.1f}%)")
    print(f"  Unknown:    {counts['unknown']:4} ({counts['unknown']/len(instances)*100:.1f}%)")
    print(f"  Timeout:    {counts['timeout']:4} ({counts['timeout']/len(instances)*100:.1f}%)")
    print(f"  Error:      {counts['error']:4} ({counts['error']/len(instances)*100:.1f}%)")
    print()
    print(f"Resolution rate: {resolution_rate:.1f}% (target: >95%)")
    print(f"Verified rate:   {verified_rate:.1f}%")
    print(f"Total time:      {elapsed_all:.1f}s")
    print(f"Avg time:        {total_time/len(instances):.2f}s per instance")
    print(f"Avg cuts:        {total_cuts/len(instances):.1f} per instance")

    # List unknown/timeout instances
    if counts["unknown"] > 0:
        print("\nUnknown instances:")
        for r in results:
            if r.status == "unknown":
                net_id = r.network.replace("ACASXU_run2a_", "").replace("_batch_2000.onnx", "")
                prop_id = r.prop.replace("prop_", "").replace(".vnnlib", "")
                print(f"  {net_id} x {prop_id}")

    if counts["falsified"] > 0:
        print("\nFalsified instances (counterexample found):")
        for r in results:
            if r.status == "falsified":
                net_id = r.network.replace("ACASXU_run2a_", "").replace("_batch_2000.onnx", "")
                prop_id = r.prop.replace("prop_", "").replace(".vnnlib", "")
                print(f"  {net_id} x {prop_id}")

    # Save results to JSON
    output = {
        "total": len(instances),
        "verified": counts["verified"],
        "falsified": counts["falsified"],
        "unknown": counts["unknown"],
        "timeout": counts["timeout"],
        "error": counts["error"],
        "resolution_rate": resolution_rate,
        "verified_rate": verified_rate,
        "total_time_sec": elapsed_all,
        "avg_time_sec": total_time / len(instances) if instances else 0,
        "avg_cuts_generated": total_cuts / len(instances) if instances else 0,
        "config": {
            "timeout_s": args.timeout,
            "max_domains": args.max_domains,
            "branching": args.branching,
            "pgd_attack": pgd_attack,
            "pgd_restarts": args.pgd_restarts,
            "proactive_cuts": args.proactive_cuts or args.compare_proactive_cuts,
            "max_proactive_cuts": args.max_proactive_cuts,
            "compare_proactive_cuts": args.compare_proactive_cuts,
            "max_instances": args.max_instances,
        },
        "instances": [
            {
                "network": r.network,
                "property": r.prop,
                "status": r.status,
                "time": r.time_sec,
                "cuts_generated": r.cuts_generated,
                "domains_explored": r.domains_explored,
                "domains_verified": r.domains_verified,
                "max_depth_reached": r.max_depth_reached,
                "error": r.error,
            }
            for r in results
        ]
    }

    base_tag = "compare" if args.compare_proactive_cuts else ("proactive" if args.proactive_cuts else "baseline")
    tag = base_tag
    if args.max_instances and args.max_instances > 0:
        tag = f"{tag}_n{args.max_instances}"

    if args.compare_proactive_cuts:
        output["baseline_summary"] = {
            "verified": base_counts["verified"],
            "falsified": base_counts["falsified"],
            "unknown": base_counts["unknown"],
            "timeout": base_counts["timeout"],
            "error": base_counts["error"],
            "total_time_sec": base_total_time,
            "avg_time_sec": base_total_time / len(instances) if instances else 0,
            "avg_cuts_generated": base_total_cuts / len(instances) if instances else 0,
        }
        output["compare"] = compare_rows
        output["mismatches"] = mismatches

    output_path = _save_report(output, tag=tag, latest_tag=base_tag)
    print(f"\nResults saved to: {output_path}")

    # Return exit code based on target
    if args.compare_proactive_cuts and mismatches:
        print(f"\n✗ MISMATCH: {len(mismatches)} instances changed status with proactive cuts")
        return 1
    if args.max_instances and args.max_instances > 0:
        return 0
    return 0 if resolution_rate >= 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())
