#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark-report renderer for backend_benchmark_row_v1 methodology lanes.

Supports two lane families:
- compare-backends (vnncomp_compare_backends, avoice_kokoro_backend_delta)
- profile (metaroom_host_profile)

CLI:
    python3 scripts/render_backend_benchmark_report.py \
        --metadata <metadata.json> \
        --csv <artifact.csv> [--csv <artifact2.csv> ...] \
        --output <report.md>
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from _benchmark_report_helpers import (
    COMPARE_LANES,
    PROFILE_LANES,
    render_profile_report,
    render_report,
    validate_metadata,
    validate_profile_metadata,
    validate_profile_rows,
    validate_rows,
)

logger = logging.getLogger(__name__)


def load_csv_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    """Load and concatenate CSV rows from one or more files."""
    rows: list[dict[str, str]] = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def _detect_lane_family(rows: list[dict[str, str]]) -> str | None:
    """Detect lane family from loaded CSV rows. Returns 'compare', 'profile', or None."""
    if not rows:
        return None
    lanes = {r.get("lane", "") for r in rows}
    if lanes & COMPARE_LANES:
        return "compare"
    if lanes & PROFILE_LANES:
        return "profile"
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Render a benchmark report from CSV and metadata (compare or profile mode)."
    )
    parser.add_argument(
        "--metadata", type=Path, required=True,
        help="Path to benchmark_report_metadata_v1 JSON sidecar",
    )
    parser.add_argument(
        "--csv", type=Path, action="append", required=True, dest="csv_paths",
        help="Path to backend_benchmark_row_v1 CSV (may be repeated)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    if args.metadata.suffix != ".json":
        logger.error("metadata must be a .json file, got %s", args.metadata)
        return 1

    try:
        with open(args.metadata, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("failed to load metadata: %s", e)
        return 1

    rows = load_csv_rows(args.csv_paths)
    family = _detect_lane_family(rows)

    if family == "profile":
        meta_errors = validate_profile_metadata(meta, args.output)
        if meta_errors:
            logger.error("metadata validation failed:")
            for err in meta_errors:
                logger.error("  - %s", err)
            return 1
        row_errors = validate_profile_rows(rows, args.output)
        if row_errors:
            logger.error("CSV row validation failed:")
            for err in row_errors:
                logger.error("  - %s", err)
            return 1
        report = render_profile_report(meta, rows)
    else:
        meta_errors = validate_metadata(meta, args.output)
        if meta_errors:
            logger.error("metadata validation failed:")
            for err in meta_errors:
                logger.error("  - %s", err)
            return 1
        row_errors = validate_rows(rows)
        if row_errors:
            logger.error("CSV row validation failed:")
            for err in row_errors:
                logger.error("  - %s", err)
            return 1
        report = render_report(meta, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
