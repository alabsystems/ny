#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
CIFAR10 ResNet benchmark runner for the supported cut-dark beta-CROWN lane.

Tests ny beta-crown on VNN-COMP 2021 cifar10_resnet.

Certificate-bearing cut authority is quarantined. This runner defaults to the
supported no-cuts lane; `--enable-cuts` is retained only as an explicit research
request and will be rejected until a certified fold replaces the legacy path.
"""

import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks/vnncomp2021/benchmarks/cifar10_resnet"
NY_BINARY = Path(__file__).parent.parent / "target/release/ny"


@dataclass
class Result:
    model: str
    property: str
    status: str
    time_seconds: float
    domains: int


def run_instance(
    model: Path,
    prop: Path,
    timeout: int,
    enable_cuts: bool,
    *,
    max_domains: Optional[int] = None,
    branching: str = "relu",
    batch_size: Optional[int] = None,
    backend: Optional[str] = None,
    alpha_iterations: Optional[int] = None,
    no_alpha: bool = False,
    alpha_gradient_method: Optional[str] = None,
    crown_ibp_intermediates: bool = False,
    beta_iterations: Optional[int] = None,
    proactive_cuts: bool = False,
    max_proactive_cuts: Optional[int] = None,
    enable_near_miss_cuts: bool = False,
    near_miss_margin: Optional[float] = None,
    pgd_attack: bool = False,
) -> Result:
    """Run a single benchmark instance."""
    cmd = [
        str(NY_BINARY),
        "beta-crown",
        str(model),
        "--property", str(prop),
        "--timeout", str(timeout),
        "--branching",
        branching,
    ]

    cmd.append("--enable-cuts" if enable_cuts else "--no-cuts")

    if max_domains is not None:
        cmd.extend(["--max-domains", str(max_domains)])

    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])

    if backend is not None:
        cmd.extend(["--backend", backend])

    if no_alpha:
        cmd.append("--no-alpha")
    elif alpha_iterations is not None:
        cmd.extend(["--alpha-iterations", str(alpha_iterations)])

    if alpha_gradient_method is not None:
        cmd.extend(["--alpha-gradient-method", alpha_gradient_method])

    if crown_ibp_intermediates:
        cmd.append("--crown-ibp-intermediates")

    if beta_iterations is not None:
        cmd.extend(["--beta-iterations", str(beta_iterations)])

    if proactive_cuts:
        cmd.append("--proactive-cuts")
        if max_proactive_cuts is not None:
            cmd.extend(["--max-proactive-cuts", str(max_proactive_cuts)])

    if enable_near_miss_cuts:
        cmd.append("--enable-near-miss-cuts")
        if near_miss_margin is not None:
            cmd.extend(["--near-miss-margin", str(near_miss_margin)])

    if pgd_attack:
        cmd.append("--pgd-attack")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # Buffer beyond internal timeout; deep networks can overshoot due to long iterations.
            timeout=timeout + max(30, timeout // 2),
        )
        elapsed = time.time() - start

        # Parse output
        output = result.stdout + result.stderr

        # Extract status (case-insensitive matching)
        # Look for the final "--- Result ---" section status first
        output_lower = output.lower()

        # Check for "Status: VERIFIED/UNKNOWN/FALSIFIED" in the result section
        if "status: verified" in output_lower:
            status = "verified"
        elif "status: falsified" in output_lower or "potentialviolation" in output_lower:
            status = "falsified"
        elif "reason: timeout" in output_lower:
            status = "timeout"
        elif "status: unknown" in output_lower:
            status = "unknown"
        elif "verification timed out" in output_lower:
            status = "timeout"
        else:
            status = "error"

        # Extract domains explored
        domains = 0
        for line in output.split('\n'):
            if "Domains explored:" in line:
                try:
                    domains = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        return Result(
            model=model.name,
            property=prop.name,
            status=status,
            time_seconds=elapsed,
            domains=domains
        )

    except subprocess.TimeoutExpired:
        return Result(
            model=model.name,
            property=prop.name,
            status="timeout",
            time_seconds=timeout,
            domains=0
        )
    except Exception:
        return Result(
            model=model.name,
            property=prop.name,
            status="error",
            time_seconds=0,
            domains=0
        )


def load_instances(max_instances: int = None) -> List[Tuple[Path, Path]]:
    """Load benchmark instances from CSV."""
    instances = []
    csv_path = BENCHMARK_DIR / "cifar10_resnet_instances.csv"

    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                model_path = BENCHMARK_DIR / row[0]
                prop_path = BENCHMARK_DIR / row[1]
                if model_path.exists() and prop_path.exists():
                    instances.append((model_path, prop_path))

    if max_instances:
        instances = instances[:max_instances]

    return instances


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CIFAR10 ResNet benchmark")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per instance (seconds). Default: 60s for 2b, 180s for 4b")
    parser.add_argument("--max-instances", type=int, default=10, help="Max instances to run")
    parser.add_argument("--max-domains", type=int, default=None, help="Maximum number of BaB domains to explore (default: ny's default)")
    cuts_group = parser.add_mutually_exclusive_group()
    cuts_group.add_argument(
        "--enable-cuts",
        action="store_true",
        help="Request quarantined research cuts (verification will reject the request)",
    )
    cuts_group.add_argument("--no-cuts", action="store_true", help="Disable cuts (default)")
    parser.add_argument("--model", choices=["2b", "4b", "both"], default="2b", help="Which model to test")
    parser.add_argument("--branching", type=str, default="relu", help="Branching mode (default: relu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for domain processing (default: ny's default)")
    parser.add_argument("--backend", type=str, choices=["cpu", "wgpu"], default=None, help="Compute backend (default: ny's default)")
    parser.add_argument("--alpha-iterations", type=int, default=None, help="α-CROWN iterations (default: ny's default)")
    parser.add_argument("--no-alpha", action="store_true", help="Disable α-CROWN (use IBP bounds)")
    parser.add_argument("--alpha-gradient-method", type=str, default=None, choices=["spsa", "fd", "analytic", "analytic-chain"], help="Gradient method for α-CROWN (default: spsa)")
    parser.add_argument("--crown-ibp-intermediates", action="store_true", help="Use CROWN-IBP for intermediate bounds (tighter but slower)")
    parser.add_argument("--beta-iterations", type=int, default=None, help="β-CROWN optimization iterations per domain (default: 20)")
    parser.add_argument(
        "--proactive-cuts",
        action="store_true",
        help="Request quarantined proactive cuts (verification will reject the request)",
    )
    parser.add_argument("--max-proactive-cuts", type=int, default=None, help="Maximum proactive cuts to generate")
    parser.add_argument(
        "--enable-near-miss-cuts",
        action="store_true",
        help="Request quarantined near-miss cuts (verification will reject the request)",
    )
    parser.add_argument("--near-miss-margin", type=float, default=None, help="Margin for near-miss cut generation")
    parser.add_argument("--pgd-attack", action="store_true", help="Enable PGD attack (debugging)")
    args = parser.parse_args()

    enable_cuts = args.enable_cuts and not args.no_cuts

    # Model-specific default timeouts based on α-CROWN initialization time:
    # - 2b: ~5s init, 60s total is sufficient
    # - 4b: ~52s init, 180s needed for meaningful BaB exploration
    if args.timeout is None:
        if args.model == "4b":
            timeout = 180  # 4b α-CROWN init takes ~52s
        else:
            timeout = 60   # 2b α-CROWN init takes ~5s
    else:
        timeout = args.timeout

    print("CIFAR10 ResNet Benchmark")
    print(f"{'='*60}")
    print(f"Timeout: {timeout}s")
    print(f"Cuts: {'enabled' if enable_cuts else 'disabled'}")
    print(f"Branching: {args.branching}")
    if args.backend is not None:
        print(f"Backend: {args.backend}")
    if args.batch_size is not None:
        print(f"Batch size: {args.batch_size}")
    if args.max_domains is not None:
        print(f"Max domains: {args.max_domains}")
    print(f"Alpha gradient method: {args.alpha_gradient_method or 'spsa (default)'}")
    print(f"CROWN-IBP intermediates: {args.crown_ibp_intermediates}")
    print(f"Model: {args.model}")
    print()

    # Load instances
    all_instances = load_instances()

    # Filter by model
    if args.model == "2b":
        instances = [i for i in all_instances if "2b" in i[0].name]
    elif args.model == "4b":
        instances = [i for i in all_instances if "4b" in i[0].name]
    else:
        instances = all_instances

    instances = instances[:args.max_instances]

    print(f"Running {len(instances)} instances...")
    print()

    results = []
    for i, (model, prop) in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {model.name} + {prop.name}", end=" ... ", flush=True)
        result = run_instance(
            model,
            prop,
            timeout,
            enable_cuts,
            max_domains=args.max_domains,
            branching=args.branching,
            batch_size=args.batch_size,
            backend=args.backend,
            alpha_iterations=args.alpha_iterations,
            no_alpha=args.no_alpha,
            alpha_gradient_method=args.alpha_gradient_method,
            crown_ibp_intermediates=args.crown_ibp_intermediates,
            beta_iterations=args.beta_iterations,
            proactive_cuts=args.proactive_cuts,
            max_proactive_cuts=args.max_proactive_cuts,
            enable_near_miss_cuts=args.enable_near_miss_cuts,
            near_miss_margin=args.near_miss_margin,
            pgd_attack=args.pgd_attack,
        )
        results.append(result)
        print(f"{result.status} ({result.time_seconds:.1f}s, {result.domains} domains)")

    # Summary
    print()
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    verified = sum(1 for r in results if r.status == "verified")
    falsified = sum(1 for r in results if r.status == "falsified")
    unknown = sum(1 for r in results if r.status == "unknown")
    timeout = sum(1 for r in results if r.status == "timeout")
    error = sum(1 for r in results if r.status == "error")
    total = len(results)

    print(f"Total: {total}")
    print(f"Verified: {verified} ({100*verified/total:.1f}%)")
    print(f"Falsified: {falsified} ({100*falsified/total:.1f}%)")
    print(f"Unknown: {unknown} ({100*unknown/total:.1f}%)")
    print(f"Timeout: {timeout} ({100*timeout/total:.1f}%)")
    print(f"Error: {error} ({100*error/total:.1f}%)")
    print(f"Average time: {sum(r.time_seconds for r in results)/total:.1f}s")
    print(f"Average domains: {sum(r.domains for r in results)//total}")


if __name__ == "__main__":
    main()
