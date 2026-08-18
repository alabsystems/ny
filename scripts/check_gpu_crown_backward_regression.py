#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Check GPU CROWN benchmark CSVs against checked-in regression thresholds.

Consumes the CSV emitted by:
`cargo run -p ny-gpu --release --example measure_crown_backward_workloads`

The default policy covers one small CPU sanity path, the oversized
soundnessbench CPU skip guard, and both the graph-collection plus direct warm
GPU timings for the metaroom and soundnessbench representative workloads.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from gpu_crown_backward_regression_lib import (
    evaluate_check,
    load_candidate_rows,
    load_policy,
    write_json,
)


DEFAULT_POLICY = Path("configs/benchmark_regressions/gpu_crown_backward.json")
DEFAULT_OUTPUT = Path("reports/benchmarks/gpu_crown_backward_regression_latest.json")
LOG = logging.getLogger(__name__)
UTC = timezone.utc


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description=(
            "Check GPU CROWN benchmark CSVs against the CPU guard, graph-engine, "
            "and direct warm-GPU regression thresholds"
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="CSV produced by measure_crown_backward_workloads",
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY),
        help=f"Threshold policy JSON (default: {DEFAULT_POLICY})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check thresholds without writing a JSON report",
    )
    args = parser.parse_args()

    candidate_paths = [Path(candidate) for candidate in args.candidate]
    policy_path = Path(args.policy)
    output_path = Path(args.output)

    policy_payload, checks = load_policy(policy_path)
    observed_rows = load_candidate_rows(candidate_paths)
    check_results = [evaluate_check(spec, observed_rows) for spec in checks]
    regression = any(item["regression"] for item in check_results)

    report = {
        "generated_at": _iso_now(),
        "suite": policy_payload.get("suite", "gpu_crown_backward"),
        "candidates": [str(path) for path in candidate_paths],
        "policy": str(policy_path),
        "regression": regression,
        "checks": check_results,
    }

    if not args.check_only:
        write_json(output_path, report)

    if regression:
        failed = [
            f"{item['name']} ({', '.join(item['reasons'])})"
            for item in check_results
            if item["regression"]
        ]
        LOG.error("GPU CROWN regression detected: %s", "; ".join(failed))
        return 1

    LOG.info("No GPU CROWN regression detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
