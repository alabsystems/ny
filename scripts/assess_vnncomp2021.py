#!/usr/bin/env python
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Quick VNN-COMP 2021 assessment for ny.

Runs a sample of instances from each benchmark to get baseline metrics.
"""

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

NY = Path(__file__).parent.parent / "target" / "release" / "ny"
BENCHMARKS = Path(__file__).parent.parent / "benchmarks" / "vnncomp2021" / "benchmarks"
DEFAULT_BENCHMARKS = [
    "acasxu",
    "mnistfc",
    "cifar10_resnet",
    "eran",
    "marabou-cifar10",
    "nn4sys",
    "oval21",
    "verivital",
]
SKIP_DONE_BENCHMARKS = DEFAULT_BENCHMARKS[:3]

@dataclass
class Result:
    network: str
    property: str
    status: str
    time_s: float
    domains: int = 0

def run_instance(
    network: Path,
    prop: Path,
    timeout: int,
    method: str = "beta",
    enable_cuts: bool = True,
    branching: str = "width",
    pgd_attack: bool = False,
    pgd_restarts: int = 100,
    beta_iterations: int = 10,
    lr_beta: float = 0.1,
    proactive_cuts: bool = False,
    relaxed_clip: bool = False,
    relaxed_clip_iterations: int = 1,
    clip_interm_domain: bool = False,
    clip_interm_topk: int = 3,
) -> Result:
    """Run a single verification instance."""
    start = time.time()

    if method == "beta":
        cmd = [
            str(NY), "beta-crown", str(network),
            "--property", str(prop),
            "--timeout", str(timeout),
            "--branching", branching,
            "--enable-cuts" if enable_cuts else "--no-cuts",
            "--max-domains", "10000",
            "--beta-iterations", str(beta_iterations),
            "--lr-beta", str(lr_beta),
            "--json",
        ]
        if pgd_attack:
            cmd.extend(["--pgd-attack", "--pgd-restarts", str(pgd_restarts)])
        if proactive_cuts:
            cmd.append("--proactive-cuts")
        if relaxed_clip:
            cmd.append("--relaxed-clip")
            cmd.extend(["--relaxed-clip-iterations", str(relaxed_clip_iterations)])
        if clip_interm_domain:
            cmd.append("--clip-interm-domain")
            cmd.extend(["--clip-interm-topk", str(clip_interm_topk)])
    else:
        cmd = [
            str(NY), "verify", str(network),
            "--property", str(prop),
            "--method", method,
            "--timeout", str(timeout),
            "--json",
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        elapsed = time.time() - start

        if result.returncode == 0:
            try:
                out = json.loads(result.stdout)
                status = out.get("property_status", out.get("status", "unknown")).lower()
                if status == "safe":
                    status = "verified"
                elif status == "violated":
                    status = "falsified"
                domains = out.get("domains_explored", 0)
                return Result(network.name, prop.name, status, elapsed, domains)
            except json.JSONDecodeError:
                stdout = result.stdout.lower()
                if "verified" in stdout:
                    return Result(network.name, prop.name, "verified", elapsed)
                if "falsified" in stdout or "violated" in stdout:
                    return Result(network.name, prop.name, "falsified", elapsed)
                return Result(network.name, prop.name, "unknown", elapsed)
        else:
            return Result(network.name, prop.name, "error", time.time() - start)

    except subprocess.TimeoutExpired:
        return Result(network.name, prop.name, "timeout", timeout)
    except Exception:
        return Result(network.name, prop.name, "error", time.time() - start)

def load_instances(benchmark: str, sample_size: int = None) -> List[Tuple[Path, Path, int]]:
    """Load instances from a benchmark."""
    bench_dir = BENCHMARKS / benchmark
    if not bench_dir.exists():
        return []

    # Find instances file
    instances_file = None
    for pattern in [f"{benchmark}_instances.csv", "instances.csv"]:
        path = bench_dir / pattern
        if path.exists():
            instances_file = path
            break

    if not instances_file:
        return []

    instances = []
    with open(instances_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or row[0].startswith("#") or row[0] == "network":
                continue

            network = bench_dir / row[0]
            prop = bench_dir / row[1]
            try:
                timeout = int(float(row[2])) if len(row) > 2 else 60
            except (ValueError, IndexError):
                timeout = 60

            # Handle .gz files - try without .gz extension if gz doesn't exist
            if str(network).endswith(".gz"):
                network_nogz = Path(str(network)[:-3])  # strip .gz
                if network_nogz.exists():
                    network = network_nogz

            if network.exists() and prop.exists():
                instances.append((network, prop, timeout))

    if sample_size and len(instances) > sample_size:
        # Take evenly spaced samples
        step = len(instances) // sample_size
        instances = [instances[i * step] for i in range(sample_size)]

    return instances

# Per-benchmark parameter profiles based on α,β-CROWN configurations
BENCHMARK_PROFILES = {
    "acasxu": {
        "branching": "input",
        "beta_iterations": 10,
        "lr_beta": 0.1,
    },
    "mnistfc": {
        "branching": "width",
        "beta_iterations": 20,  # α,β-CROWN uses 20
        "lr_beta": 0.03,        # α,β-CROWN uses 0.03 for MNIST
        "proactive_cuts": True, # Essential for 80% vs 40% verification rate
    },
    "cifar10_resnet": {
        "branching": "relu",
        "beta_iterations": 10,
        "lr_beta": 0.1,
    },
    "marabou-cifar10": {
        "branching": "width",
        "beta_iterations": 50,  # α,β-CROWN uses 50
        "lr_beta": 0.5,         # α,β-CROWN uses 0.5 for CIFAR
    },
    "eran": {
        "branching": "width",
        "beta_iterations": 10,
        "lr_beta": 0.1,
    },
    "nn4sys": {
        "branching": "width",
        "beta_iterations": 10,
        "lr_beta": 0.1,
    },
    "oval21": {
        "branching": "width",
        "beta_iterations": 10,
        "lr_beta": 0.1,
    },
    "verivital": {
        "branching": "width",
        "beta_iterations": 10,
        "lr_beta": 0.1,
    },
}

def assess_benchmark(
    benchmark: str,
    sample_size: int = 20,
    timeout: int = 30,
    method: str = "beta",
    enable_cuts: bool = True,
    pgd_mode: str = "auto",
    pgd_restarts: int = 100,
    use_tuned_params: bool = False,
    relaxed_clip: bool = False,
    relaxed_clip_iterations: int = 1,
    clip_interm_domain: bool = False,
    clip_interm_topk: int = 3,
) -> dict:
    """Assess a single benchmark."""
    instances = load_instances(benchmark, sample_size)

    if not instances:
        return {"benchmark": benchmark, "total": 0, "error": "no instances found"}

    results = {"benchmark": benchmark, "total": 0, "verified": 0, "falsified": 0,
               "unknown": 0, "timeout": 0, "error": 0, "times": [], "instances": []}

    # Get per-benchmark parameters
    profile = BENCHMARK_PROFILES.get(benchmark, {})
    branching = profile.get("branching", "width")

    # Use tuned parameters if enabled, otherwise defaults
    if use_tuned_params:
        beta_iterations = profile.get("beta_iterations", 10)
        lr_beta = profile.get("lr_beta", 0.1)
        proactive_cuts = profile.get("proactive_cuts", False)
    else:
        beta_iterations = 10
        lr_beta = 0.1
        proactive_cuts = False

    pgd_attack = pgd_mode == "on" or (pgd_mode == "auto" and benchmark == "acasxu")

    for network, prop, default_timeout in instances:
        result = run_instance(
            network,
            prop,
            min(timeout, default_timeout),
            method=method,
            enable_cuts=enable_cuts,
            branching=branching,
            pgd_attack=pgd_attack,
            pgd_restarts=pgd_restarts,
            beta_iterations=beta_iterations,
            lr_beta=lr_beta,
            proactive_cuts=proactive_cuts,
            relaxed_clip=relaxed_clip,
            relaxed_clip_iterations=relaxed_clip_iterations,
            clip_interm_domain=clip_interm_domain,
            clip_interm_topk=clip_interm_topk,
        )
        results["total"] += 1
        results[result.status] += 1
        results["times"].append(result.time_s)
        results["instances"].append({
            "network": result.network,
            "property": result.property,
            "status": result.status,
            "time": result.time_s,
            "domains": result.domains,
        })

        # Progress indicator
        print(f"  {benchmark}: {result.network[:30]:<30} {result.property[:20]:<20} "
              f"{result.status:<10} {result.time_s:.2f}s")

    if results["total"] > 0:
        results["verified_rate"] = results["verified"] / results["total"] * 100
        results["avg_time"] = sum(results["times"]) / len(results["times"])
        results["avg_domains"] = sum(r["domains"] for r in results["instances"]) / results["total"]
    else:
        results["verified_rate"] = 0
        results["avg_time"] = 0
        results["avg_domains"] = 0

    return results

def write_markdown_summary(
    all_results: List[dict],
    output_path: Path,
    sample_size: int,
    timeout: int,
    method: str,
    enable_cuts: bool,
    pgd_mode: str,
    pgd_restarts: int,
    skipped_benchmarks: List[str],
    relaxed_clip: bool,
    relaxed_clip_iterations: int,
    clip_interm_domain: bool,
    clip_interm_topk: int,
):
    rows = []
    total_verified = 0
    total_instances = 0

    for r in all_results:
        if r.get("total", 0) <= 0:
            continue
        rows.append(
            (
                r["benchmark"],
                f'{r.get("verified", 0)}/{r.get("total", 0)}',
                f'{r.get("verified_rate", 0.0):.1f}%',
                f'{r.get("avg_time", 0.0):.2f}s',
                f'{r.get("avg_domains", 0.0):.1f}',
                str(r.get("timeout", 0)),
                str(r.get("error", 0)),
            )
        )
        total_verified += r.get("verified", 0)
        total_instances += r.get("total", 0)

    overall_rate = (total_verified / total_instances * 100.0) if total_instances > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# VNN-COMP 2021 Assessment (Sample)\n\n")
        f.write(f"- sample_size: {sample_size}\n")
        f.write(f"- timeout_s: {timeout}\n")
        f.write(f"- method: {method}\n")
        f.write(f"- cuts: {'enabled' if enable_cuts else 'disabled'}\n\n")
        f.write(f"- relaxed_clip: {relaxed_clip}\n")
        if relaxed_clip:
            f.write(f"- relaxed_clip_iterations: {relaxed_clip_iterations}\n")
        f.write(f"- clip_interm_domain: {clip_interm_domain}\n")
        if clip_interm_domain:
            f.write(f"- clip_interm_topk: {clip_interm_topk}\n")
        f.write("\n")
        f.write(f"- pgd: {pgd_mode}\n")
        if pgd_mode != "off":
            f.write(f"- pgd_restarts: {pgd_restarts}\n")
        if skipped_benchmarks:
            f.write(f"- skipped_benchmarks: {', '.join(skipped_benchmarks)}\n")
        f.write("\n")
        f.write("| Benchmark | Verified | Rate | Avg time | Avg domains | Timeout | Error |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for bench, verified, rate, avg_time, avg_domains, timeouts, errors in rows:
            f.write(
                f"| {bench} | {verified} | {rate} | {avg_time} | {avg_domains} | {timeouts} | {errors} |\n"
            )
        f.write("\n")
        f.write(f"Overall: {total_verified}/{total_instances} verified ({overall_rate:.1f}%)\n")

def main():
    parser = argparse.ArgumentParser(description="Quick VNN-COMP 2021 assessment (sampled)")
    parser.add_argument("sample_size", nargs="?", type=int, default=10, help="Instances per benchmark to run")
    parser.add_argument("timeout", nargs="?", type=int, default=30, help="Timeout per instance (seconds)")
    parser.add_argument("--method", default="beta", help="Verification method (default: beta)")
    parser.add_argument("--no-cuts", action="store_true", help="Disable GCP-CROWN cuts (beta-crown only)")
    parser.add_argument(
        "--pgd",
        choices=["auto", "on", "off"],
        default="auto",
        help="PGD attack mode for beta-crown (default: auto; currently enables only for acasxu)",
    )
    parser.add_argument("--pgd-restarts", type=int, default=1000, help="PGD restarts when PGD is enabled")
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help=(
            "Resume a partial run by skipping already completed benchmarks "
            "(acasxu/mnistfc/cifar10_resnet)."
        ),
    )
    parser.add_argument("--tuned", action="store_true", help="Use per-benchmark tuned parameters (lr_beta, beta_iterations)")
    parser.add_argument(
        "--relaxed-clip",
        action="store_true",
        help="Enable relaxed clipping (Clip-and-Verify) for input-split domains",
    )
    parser.add_argument(
        "--relaxed-clip-iterations",
        type=int,
        default=1,
        help="Relaxed clipping iterations per split (default: 1)",
    )
    parser.add_argument(
        "--clip-interm-domain",
        action="store_true",
        help="Enable intermediate domain clipping (clip_interm_domain)",
    )
    parser.add_argument(
        "--clip-interm-topk",
        type=int,
        default=3,
        help="Objective neurons per layer to tighten with clip_interm_domain (default: 3)",
    )
    args = parser.parse_args()

    benchmarks = list(DEFAULT_BENCHMARKS)
    skipped_benchmarks: List[str] = []

    # If --skip-done is passed, skip first 3 benchmarks (already completed)
    if args.skip_done:
        skipped_benchmarks = list(SKIP_DONE_BENCHMARKS)
        benchmarks = benchmarks[len(SKIP_DONE_BENCHMARKS):]

    enable_cuts = not args.no_cuts
    use_tuned_params = args.tuned

    print(f"VNN-COMP 2021 Assessment (sample={args.sample_size}, timeout={args.timeout}s)")
    print(f"Method: {args.method}, cuts: {'enabled' if enable_cuts else 'disabled'}")
    print(f"Tuned parameters: {'enabled' if use_tuned_params else 'disabled'}")
    print(
        "Relaxed clip: {} (iters={}), clip_interm_domain: {} (topk={})".format(
            args.relaxed_clip,
            args.relaxed_clip_iterations,
            args.clip_interm_domain,
            args.clip_interm_topk,
        )
    )
    print(f"PGD: {args.pgd} (restarts={args.pgd_restarts})")
    if skipped_benchmarks:
        print(f"Skipping benchmarks: {', '.join(skipped_benchmarks)}")
    print("=" * 80)

    all_results = []

    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()}")
        print("-" * 40)
        results = assess_benchmark(
            benchmark,
            sample_size=args.sample_size,
            timeout=args.timeout,
            method=args.method,
            enable_cuts=enable_cuts,
            pgd_mode=args.pgd,
            pgd_restarts=args.pgd_restarts,
            use_tuned_params=use_tuned_params,
            relaxed_clip=args.relaxed_clip,
            relaxed_clip_iterations=args.relaxed_clip_iterations,
            clip_interm_domain=args.clip_interm_domain,
            clip_interm_topk=args.clip_interm_topk,
        )
        all_results.append(results)

        if results["total"] > 0:
            print(f"\nSummary: {results['verified']}/{results['total']} verified "
                  f"({results['verified_rate']:.1f}%), avg {results['avg_time']:.2f}s")

    # Final summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(
        f"{'Benchmark':<20} {'Verified':<12} {'Rate':<10} {'Avg Time':<10} "
        f"{'Avg Domains':<12} {'Timeout':<8}"
    )
    print("-" * 80)

    total_verified = 0
    total_instances = 0

    for r in all_results:
        if r["total"] > 0:
            print(f"{r['benchmark']:<20} {r['verified']}/{r['total']:<10} "
                  f"{r['verified_rate']:.1f}%{'':>5} {r['avg_time']:.2f}s{'':>5} "
                  f"{r['avg_domains']:.1f}{'':>6} {r['timeout']}")
            total_verified += r["verified"]
            total_instances += r["total"]

    if total_instances > 0:
        overall_rate = total_verified / total_instances * 100
        print("-" * 80)
        print(f"{'TOTAL':<20} {total_verified}/{total_instances:<10} {overall_rate:.1f}%")

    # Save detailed results
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    out_dir = Path(__file__).parent.parent / "reports" / "main"
    output_path = out_dir / f"vnncomp2021_assessment_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    md_path = out_dir / f"vnncomp2021_assessment_{timestamp}.md"
    write_markdown_summary(
        all_results,
        md_path,
        sample_size=args.sample_size,
        timeout=args.timeout,
        method=args.method,
        enable_cuts=enable_cuts,
        pgd_mode=args.pgd,
        pgd_restarts=args.pgd_restarts,
        skipped_benchmarks=skipped_benchmarks,
        relaxed_clip=args.relaxed_clip,
        relaxed_clip_iterations=args.relaxed_clip_iterations,
        clip_interm_domain=args.clip_interm_domain,
        clip_interm_topk=args.clip_interm_topk,
    )
    print(f"Markdown summary saved to: {md_path}")

if __name__ == "__main__":
    main()
