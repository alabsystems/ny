#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Compare ACAS-Xu benchmark summaries.

This script compares two JSON summaries produced by:
  ny bench --benchmark acasxu --json > summary.json

It reports aggregate deltas (pass rate, average time, timeouts, errors) and
optionally enumerates per-instance status changes when results are present.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Summary:
    name: str
    year: int
    total: int
    verified: int
    falsified: int
    unknown: int
    timeout_count: int
    error_count: int
    pass_rate: float
    avg_time_ms: int
    total_time_ms: int
    timeout_seconds: int
    commit: str
    results: List[Dict[str, Any]]


def load_summary(path: str) -> Summary:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return Summary(
        name=data.get("benchmark", "acasxu"),
        year=int(data.get("benchmark_year", 0)),
        total=int(data.get("total", 0)),
        verified=int(data.get("verified", 0)),
        falsified=int(data.get("falsified", 0)),
        unknown=int(data.get("unknown", 0)),
        timeout_count=int(data.get("timeout_count", 0)),
        error_count=int(data.get("error_count", 0)),
        pass_rate=float(data.get("pass_rate", 0.0)),
        avg_time_ms=int(data.get("avg_time_ms", 0)),
        total_time_ms=int(data.get("total_time_ms", 0)),
        timeout_seconds=int(data.get("timeout_seconds", 0)),
        commit=str(data.get("commit", "")),
        results=list(data.get("results", [])),
    )


def percent(delta: float) -> str:
    return f"{delta:+.2f}%"


def ms(delta: int) -> str:
    return f"{delta:+d} ms"


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def index_results(results: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in results:
        model = str(item.get("model", ""))
        prop = str(item.get("property", ""))
        if model and prop:
            indexed[(model, prop)] = item
    return indexed


def format_status(status: str) -> str:
    return status.lower()


def compare_instances(
    baseline: Summary,
    candidate: Summary,
    max_diffs: int,
) -> Tuple[Counter, List[str]]:
    baseline_index = index_results(baseline.results)
    candidate_index = index_results(candidate.results)

    all_keys = sorted(set(baseline_index.keys()) | set(candidate_index.keys()))
    changes: List[str] = []
    counts: Counter = Counter()

    for key in all_keys:
        base = baseline_index.get(key)
        cand = candidate_index.get(key)
        if not base or not cand:
            counts["missing"] += 1
            continue

        base_status = format_status(str(base.get("status", "")))
        cand_status = format_status(str(cand.get("status", "")))
        if base_status == cand_status:
            continue

        counts["changed"] += 1
        if base_status == "verified" and cand_status != "verified":
            counts["regressed_verified"] += 1
        elif base_status != "verified" and cand_status == "verified":
            counts["improved_verified"] += 1
        elif cand_status == "falsified":
            counts["became_falsified"] += 1

        if len(changes) < max_diffs:
            changes.append(f"{key[0]} / {key[1]}: {base_status} -> {cand_status}")

    return counts, changes


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ACAS-Xu benchmark summaries")
    parser.add_argument("baseline", help="Baseline JSON summary file")
    parser.add_argument("candidate", help="Candidate JSON summary file")
    parser.add_argument("--max-diffs", type=int, default=20, help="Max per-instance diffs to print")
    parser.add_argument("--json", action="store_true", help="Emit comparison JSON")
    args = parser.parse_args()

    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)

    pass_rate_delta = (candidate.pass_rate - baseline.pass_rate) * 100.0
    avg_time_delta = candidate.avg_time_ms - baseline.avg_time_ms

    report = {
        "baseline": {
            "benchmark": baseline.name,
            "year": baseline.year,
            "commit": baseline.commit,
            "total": baseline.total,
            "pass_rate": baseline.pass_rate,
            "avg_time_ms": baseline.avg_time_ms,
            "timeouts": baseline.timeout_count,
            "errors": baseline.error_count,
        },
        "candidate": {
            "benchmark": candidate.name,
            "year": candidate.year,
            "commit": candidate.commit,
            "total": candidate.total,
            "pass_rate": candidate.pass_rate,
            "avg_time_ms": candidate.avg_time_ms,
            "timeouts": candidate.timeout_count,
            "errors": candidate.error_count,
        },
        "delta": {
            "pass_rate_pct": pass_rate_delta,
            "avg_time_ms": avg_time_delta,
            "timeout_count": candidate.timeout_count - baseline.timeout_count,
            "error_count": candidate.error_count - baseline.error_count,
        },
    }

    instance_counts: Optional[Counter] = None
    instance_changes: List[str] = []
    if baseline.results and candidate.results:
        instance_counts, instance_changes = compare_instances(
            baseline, candidate, args.max_diffs
        )
        report["instance_changes"] = dict(instance_counts)
        report["instance_examples"] = instance_changes

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("ACAS-Xu Benchmark Comparison")
    print("=" * 40)
    print(f"Baseline:  {args.baseline} (commit {baseline.commit}, {baseline.year})")
    print(f"Candidate: {args.candidate} (commit {candidate.commit}, {candidate.year})")
    print()
    print("Aggregate Summary")
    print(f"  Pass rate: {baseline.pass_rate:.3f} -> {candidate.pass_rate:.3f} ({percent(pass_rate_delta)})")
    print(f"  Avg time:  {baseline.avg_time_ms} ms -> {candidate.avg_time_ms} ms ({ms(avg_time_delta)})")
    print(f"  Timeouts:  {baseline.timeout_count} -> {candidate.timeout_count} ({candidate.timeout_count - baseline.timeout_count:+d})")
    print(f"  Errors:    {baseline.error_count} -> {candidate.error_count} ({candidate.error_count - baseline.error_count:+d})")

    if instance_counts is not None:
        print()
        print("Instance Changes")
        print(f"  Changed:   {instance_counts.get('changed', 0)}")
        print(f"  Improved:  {instance_counts.get('improved_verified', 0)}")
        print(f"  Regressed: {instance_counts.get('regressed_verified', 0)}")
        print(f"  Falsified: {instance_counts.get('became_falsified', 0)}")
        if instance_counts.get("missing"):
            print(f"  Missing:   {instance_counts['missing']}")
        if instance_changes:
            print()
            print("Example Changes")
            for line in instance_changes:
                print(f"  {line}")
    else:
        print()
        print("Instance Changes")
        print("  No per-instance results found in one or both summaries.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
