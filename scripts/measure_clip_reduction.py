#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Measure Clip-and-Verify subproblem reduction on VNN-COMP benchmarks.

This script runs beta-crown twice per instance: baseline and with clipping
flags enabled. It compares domains explored to estimate subproblem reduction.

Examples:
  python scripts/measure_clip_reduction.py --year 2021 --benchmark acasxu --limit 10
  python scripts/measure_clip_reduction.py --branching input --relaxed-clip
  python scripts/measure_clip_reduction.py --branching relu --clip-interm-domain
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks._shared import get_benchmark_instances, run_ny_verify


def build_clip_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.relaxed_clip:
        flags.extend(["--relaxed-clip", "--relaxed-clip-iterations", str(args.relaxed_clip_iterations)])
    if args.clip_interm_domain:
        flags.extend(["--clip-interm-domain", "--clip-interm-topk", str(args.clip_interm_topk)])
    return flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Clip-and-Verify subproblem reduction on VNN-COMP benchmarks."
    )
    parser.add_argument("--year", type=int, default=2021, help="VNN-COMP year (default: 2021)")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="acasxu",
        help="Benchmark name (default: acasxu)",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per instance (seconds)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of instances (0 = all)",
    )
    parser.add_argument(
        "--branching",
        type=str,
        default="input",
        help="Branching heuristic for beta-crown (default: input)",
    )
    parser.add_argument(
        "--relaxed-clip",
        action="store_true",
        help="Enable relaxed clipping (input split only)",
    )
    parser.add_argument(
        "--relaxed-clip-iterations",
        type=int,
        default=1,
        help="Relaxed clipping iterations (default: 1)",
    )
    parser.add_argument(
        "--clip-interm-domain",
        action="store_true",
        help="Enable intermediate domain clipping (ReLU split only)",
    )
    parser.add_argument(
        "--clip-interm-topk",
        type=int,
        default=3,
        help="Top-k objective neurons for clip-intern-domain (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path (default: reports/main/clip_reduction_<ts>.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.clip_interm_domain and args.branching == "input":
        print(
            "Warning: --clip-interm-domain typically requires --branching relu; "
            "input splitting will likely ignore clip-interm-domain.",
            file=sys.stderr,
        )

    instances = get_benchmark_instances(args.year, args.benchmark)
    if not instances:
        print("No benchmark instances found.", file=sys.stderr)
        return 1

    if args.limit > 0:
        instances = instances[: args.limit]

    clip_flags = build_clip_flags(args)

    results = []
    reductions = []

    print(
        f"Running {len(instances)} instances (year={args.year}, benchmark={args.benchmark}, "
        f"branching={args.branching}, timeout={args.timeout}s)"
    )

    for idx, (network_path, prop_path, default_timeout) in enumerate(instances, start=1):
        timeout = args.timeout or default_timeout
        print(f"[{idx}/{len(instances)}] {network_path.name} + {prop_path.name}")

        baseline = run_ny_verify(
            network_path,
            prop_path,
            timeout=timeout,
            method="beta",
            beta_branching=args.branching,
        )

        clipped = run_ny_verify(
            network_path,
            prop_path,
            timeout=timeout,
            method="beta",
            beta_flags=clip_flags,
            beta_branching=args.branching,
        )

        reduction = None
        if baseline.domains_explored and clipped.domains_explored is not None:
            if baseline.domains_explored > 0:
                reduction = (baseline.domains_explored - clipped.domains_explored) / baseline.domains_explored
                reductions.append(reduction)

        results.append(
            {
                "network": baseline.network,
                "property": baseline.property,
                "baseline": asdict(baseline),
                "clipped": asdict(clipped),
                "reduction": reduction,
            }
        )

        print(
            f"  baseline domains={baseline.domains_explored} status={baseline.status} | "
            f"clipped domains={clipped.domains_explored} status={clipped.status} | "
            f"reduction={reduction if reduction is not None else 'n/a'}"
        )

    summary = {
        "year": args.year,
        "benchmark": args.benchmark,
        "branching": args.branching,
        "timeout": args.timeout,
        "clip_flags": clip_flags,
        "instances": len(instances),
        "measured": len(reductions),
        "avg_reduction": statistics.mean(reductions) if reductions else None,
        "median_reduction": statistics.median(reductions) if reductions else None,
        "max_reduction": max(reductions) if reductions else None,
        "min_reduction": min(reductions) if reductions else None,
    }

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = Path("reports/main") / f"clip_reduction_{args.year}_{args.benchmark}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nSaved report to {output_path}")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
