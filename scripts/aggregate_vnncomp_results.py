#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate per-category VNN-COMP benchmark CSVs into one JSON summary.

Usage:
    python3 scripts/aggregate_vnncomp_results.py \
        --output reports/benchmarks/vnncomp_summary_TIMESTAMP.json \
        --year 2025 --commit abc123 --version v0.1.0 --wall-time 1234.5 \
        --skipped '{"cctsdb_yolo_2023": "YOLO detection head not supported (ScatterND op)"}' \
        --category-csv dist_shift_2023=reports/benchmarks/dist_shift_release_20260306.csv \
        reports/benchmarks/malbeware_*.csv reports/benchmarks/cersyve_*.csv

With --publish-metrics, also writes stable canonical artifacts:
    reports/benchmarks/vnncomp_latest.json
    metrics/benchmarks/vnncomp_latest.json
    metrics/benchmarks/vnncomp_history.jsonl

Publication is guarded: only full tracker-year runs with no failed categories
produce canonical artifacts (see design doc for #3582).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_result(raw_result: str) -> str:
    """Map benchmark CSV result aliases into the aggregator's canonical set."""
    result = raw_result.strip().lower()
    if result == "falsified":
        return "violated"
    if result == "timeout_ext":
        return "timeout"
    return result


def _infer_category_from_filename(csv_path: str) -> str:
    basename = os.path.basename(csv_path)
    return basename.rsplit("_", 2)[0]


def _row_elapsed_seconds(row: dict[str, str]) -> float:
    raw = row.get("wall_seconds") or row.get("elapsed") or "0"
    return float(raw)


def _row_result(row: dict[str, str]) -> str:
    return _normalize_result(row.get("status") or row.get("result") or "error")


def _schema_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (row.get("schema_version") or "").strip() == "backend_benchmark_row_v1"
    ]


def _schema_row_category(schema_rows: list[dict[str, str]]) -> str | None:
    row_categories = {
        (row.get("category") or "").strip()
        for row in schema_rows
        if (row.get("category") or "").strip()
    }
    if len(row_categories) == 1:
        return next(iter(row_categories))
    return None


def _benchmark_rows_for_aggregation(
    csv_path: str,
    rows: list[dict[str, str]],
    category_name: str | None,
    filename_category: str,
) -> tuple[list[dict[str, str]], str | None]:
    schema_rows = _schema_rows(rows)
    if not schema_rows:
        return rows, category_name or filename_category

    benchmark_rows = [
        row for row in schema_rows if (row.get("lane") or "").strip() == "vnncomp_single_backend"
    ]
    category_rows = benchmark_rows or schema_rows
    resolved_category = category_name or _schema_row_category(category_rows) or filename_category
    skipped_rows = len(schema_rows) - len(benchmark_rows)
    if skipped_rows > 0:
        logger.warning(
            "Skipping %s non-breadth backend_benchmark_row_v1 rows from %s",
            skipped_rows,
            csv_path,
        )
    if not benchmark_rows:
        return [], None
    return benchmark_rows, resolved_category


def parse_category_csv(csv_path: str, *, category_name: str | None = None) -> dict | None:
    """Parse a per-category benchmark CSV into aggregate statistics.

    Args:
        csv_path: Path to the CSV file.
        category_name: Explicit category name override.  When ``None``
            (the default) the category is inferred from the filename
            via ``basename.rsplit("_", 2)[0]``.
    """
    filename_category = _infer_category_from_filename(csv_path)

    total = 0
    verified = 0
    falsified = 0
    unknown = 0
    timeout = 0
    error = 0
    total_elapsed = 0.0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    benchmark_rows, category_name = _benchmark_rows_for_aggregation(
        csv_path,
        rows,
        category_name,
        filename_category,
    )
    if category_name is None:
        return None

    for row in benchmark_rows:
        total += 1
        result = _row_result(row)
        elapsed = _row_elapsed_seconds(row)
        total_elapsed += elapsed

        if result == "verified":
            verified += 1
        elif result == "violated":
            falsified += 1
        elif result == "timeout":
            timeout += 1
        elif result == "unknown":
            unknown += 1
        else:
            error += 1

    score = verified + falsified
    solve_rate = (score / total * 100) if total > 0 else 0.0

    # Detect preset config
    preset = None
    for year_suffix in ["25", "24", "23"]:
        preset_path = f"configs/vnncomp{year_suffix}/{category_name}.yaml"
        if os.path.isfile(preset_path):
            preset = preset_path
            break

    return {
        "category": category_name,
        "total": total,
        "verified": verified,
        "falsified": falsified,
        "unknown": unknown,
        "timeout": timeout,
        "error": error,
        "score": score,
        "solve_rate": round(solve_rate, 1),
        "wall_time_seconds": round(total_elapsed, 1),
        "preset": preset,
        "csv_report": csv_path,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def publish_metrics(summary: dict, report_path: str) -> None:
    """Write canonical vnncomp_latest.json and append to vnncomp_history.jsonl.

    Only called when publication_scope == "canonical".
    """
    reports_latest = Path("reports/benchmarks/vnncomp_latest.json")
    metrics_latest = Path("metrics/benchmarks/vnncomp_latest.json")
    history_path = Path("metrics/benchmarks/vnncomp_history.jsonl")

    _write_json(reports_latest, summary)
    _write_json(metrics_latest, summary)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")

    logger.info("Published canonical metrics: %s, %s", reports_latest, metrics_latest)


def _aggregate_categories(
    csv_files: list[str],
    category_csv_map: dict[str, str] | None = None,
) -> tuple[dict, int, int]:
    """Aggregate category CSVs into a categories dict.

    Returns ``(categories, total_instances, total_score)``.
    """
    categories: dict[str, dict] = {}
    total_instances = 0
    total_score = 0

    for csv_path in csv_files:
        if not os.path.isfile(csv_path):
            logger.warning("CSV not found: %s", csv_path)
            continue
        stats = parse_category_csv(csv_path)
        if stats is None:
            continue
        category = stats["category"]
        categories[category] = stats
        total_instances += stats["total"]
        total_score += stats["score"]

    for cat_name, csv_path in (category_csv_map or {}).items():
        if not os.path.isfile(csv_path):
            logger.warning("CSV not found for %s: %s", cat_name, csv_path)
            continue
        stats = parse_category_csv(csv_path, category_name=cat_name)
        if stats is None:
            continue
        category = stats["category"]
        if category in categories:
            old = categories[category]
            total_instances -= old["total"]
            total_score -= old["score"]
        categories[category] = stats
        total_instances += stats["total"]
        total_score += stats["score"]

    return categories, total_instances, total_score


def build_summary(
    csv_files: list[str],
    *,
    year: int,
    commit: str,
    version: str,
    wall_time: float,
    skipped: dict,
    failed: dict,
    run_scope: str,
    report_path: str,
    tracker_year: int,
    category_csv_map: dict[str, str] | None = None,
) -> dict:
    """Build the aggregate summary dict from category CSVs.

    ``category_csv_map`` provides explicit ``{category_name: csv_path}``
    overrides that take precedence over the filename-inferred category name.
    """
    categories, total_instances, total_score = _aggregate_categories(
        csv_files, category_csv_map,
    )
    overall_solve_rate = (total_score / total_instances * 100) if total_instances > 0 else 0.0
    is_canonical = run_scope == "full" and year == tracker_year and not failed

    return {
        "benchmark": "vnncomp",
        "report_kind": "breadth",
        "benchmark_year": year,
        "run_scope": run_scope,
        "publication_scope": "canonical" if is_canonical else "timestamp_only",
        "commit": commit,
        "ny_version": version,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report_path": report_path,
        "wall_time_seconds": wall_time,
        "total_instances": total_instances,
        "total_score": total_score,
        "overall_solve_rate": round(overall_solve_rate, 1),
        "categories_attempted": len(categories),
        "categories_skipped": sorted(skipped.keys()),
        "failed": failed,
        "categories": categories,
        "skipped": skipped,
    }


def _parse_category_csv_pairs(
    pairs: list[str], parser: argparse.ArgumentParser,
) -> dict[str, str]:
    """Parse ``NAME=PATH`` pairs from ``--category-csv`` arguments."""
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            parser.error(f"--category-csv requires NAME=PATH format, got: {pair}")
        name, path = pair.split("=", 1)
        result[name] = path
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate VNN-COMP benchmark CSVs")
    parser.add_argument("csv_files", nargs="*", help="Per-category CSV report files")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--year", type=int, default=2025, help="Benchmark year")
    parser.add_argument("--commit", default="unknown", help="Git commit hash")
    parser.add_argument("--version", default="dev", help="Ny version string")
    parser.add_argument("--wall-time", type=float, default=0, help="Overall wall time")
    parser.add_argument("--skipped", default="{}", help="JSON dict of skipped categories")
    parser.add_argument("--failed", default="{}", help="JSON dict of failed categories")
    parser.add_argument("--run-scope", default="full", choices=["full", "partial"],
                        help="Whether this was a full or partial breadth run")
    parser.add_argument("--tracker-year", type=int, default=2025,
                        help="The canonical tracker year for publication")
    parser.add_argument("--publish-metrics", action="store_true",
                        help="Write canonical vnncomp_latest.json and history.jsonl")
    parser.add_argument(
        "--category-csv", action="append", default=[], metavar="NAME=PATH",
        help="Explicit category-to-CSV mapping (repeatable). Example: "
             "--category-csv dist_shift_2023=reports/benchmarks/dist_shift_release.csv",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    skipped = json.loads(args.skipped)
    failed = json.loads(args.failed)
    category_csv_map = _parse_category_csv_pairs(args.category_csv, parser)

    summary = build_summary(
        args.csv_files, year=args.year, commit=args.commit, version=args.version,
        wall_time=args.wall_time, skipped=skipped, failed=failed,
        run_scope=args.run_scope, report_path=args.output,
        tracker_year=args.tracker_year, category_csv_map=category_csv_map,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("Wrote summary to %s", args.output)

    if args.publish_metrics:
        if summary["publication_scope"] != "canonical":
            logger.info(
                "Skipping publication: scope=%s (run_scope=%s, year=%d, tracker_year=%d, failed=%d)",
                summary["publication_scope"], summary["run_scope"],
                summary["benchmark_year"], args.tracker_year, len(failed),
            )
        else:
            publish_metrics(summary, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
