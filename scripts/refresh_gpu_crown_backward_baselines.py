#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Refresh GPU CROWN regression policy baselines from benchmark CSVs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gpu_crown_backward_regression_lib import (
    load_candidate_rows,
    load_policy,
    refresh_policy_payload,
    write_json,
)


DEFAULT_POLICY = Path("configs/benchmark_regressions/gpu_crown_backward.json")
LOG = logging.getLogger(__name__)


def _policy_root(policy_path: Path) -> Path:
    resolved = policy_path.resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    return resolved.parent


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description=(
            "Refresh GPU CROWN regression source_artifact pins and measured "
            "baseline_seconds from validated benchmark CSVs"
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
        help=f"Existing policy JSON to refresh (default: {DEFAULT_POLICY})",
    )
    parser.add_argument(
        "--output",
        help="Optional destination for the refreshed policy (default: overwrite --policy)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the refresh inputs without writing the updated policy",
    )
    args = parser.parse_args()

    candidate_paths = [Path(candidate) for candidate in args.candidate]
    policy_path = Path(args.policy)
    output_path = policy_path if args.output is None else Path(args.output)

    policy_payload, checks = load_policy(policy_path)
    observed_rows = load_candidate_rows(candidate_paths)
    refreshed_payload, refresh_results = refresh_policy_payload(
        policy_payload,
        checks,
        observed_rows,
        root=_policy_root(policy_path),
    )

    failures = [item for item in refresh_results if item["reasons"]]
    if failures:
        summary = "; ".join(
            f"{item['name']} ({', '.join(item['reasons'])})" for item in failures
        )
        LOG.error("GPU CROWN baseline refresh rejected: %s", summary)
        return 1

    if not args.check_only:
        write_json(output_path, refreshed_payload)

    if args.check_only:
        LOG.info("GPU CROWN baseline refresh inputs validated.")
    else:
        LOG.info(
            "Refreshed %d GPU CROWN policy checks at %s.",
            len(refresh_results),
            output_path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
