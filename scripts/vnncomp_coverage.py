#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
# Author Email: Andrew Yates <andrewyates.name@gmail.com>

"""Audit VNN-COMP benchmark category availability.

Scans benchmarks/vnncompYYYY/benchmarks for categories and counts model/property files.
This does not verify correctness; it only reports dataset presence.

Usage:
    python scripts/vnncomp_coverage.py
    python scripts/vnncomp_coverage.py --year 2021 --json --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

MODEL_EXTS = [".onnx", ".nnet", ".pt", ".pth", ".safetensors", ".pb", ".mlmodel", ".tflite"]
PROPERTY_EXTS = [".vnnlib"]


@dataclass
class CategoryCoverage:
    year: int
    category: str
    path: str
    model_count: int
    property_count: int
    instance_csv_count: int
    model_exts: dict[str, int]
    status: str


@dataclass
class YearCoverage:
    year: int
    benchmark_dir: str
    categories: list[CategoryCoverage]
    status: str


def find_benchmark_roots(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("vnncomp20*/benchmarks") if p.is_dir())


def count_files(dir_path: Path, exts: list[str]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    for ext in exts:
        matches = list(dir_path.rglob(f"*{ext}"))
        count = len(matches)
        if count:
            counts[ext] = count
            total += count
    return total, counts


def count_instances_csv(dir_path: Path) -> int:
    return sum(1 for _ in dir_path.rglob("instances.csv")) + sum(
        1 for _ in dir_path.rglob("acasxu_instances.csv")
    )


def classify_status(model_count: int, property_count: int, instance_csv_count: int) -> str:
    if model_count > 0 and property_count == 0 and instance_csv_count == 0:
        return "auxiliary-assets"
    if model_count == 0 and property_count == 0:
        return "missing-models-properties"
    if model_count == 0:
        return "missing-models"
    if property_count == 0:
        return "missing-properties"
    return "available"


def collect_year(year_dir: Path) -> YearCoverage:
    year = int(year_dir.parent.name.replace("vnncomp", ""))
    categories: list[CategoryCoverage] = []
    for category_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
        model_count, model_exts = count_files(category_dir, MODEL_EXTS)
        property_count, _ = count_files(category_dir, PROPERTY_EXTS)
        instance_csv_count = count_instances_csv(category_dir)
        status = classify_status(model_count, property_count, instance_csv_count)
        categories.append(
            CategoryCoverage(
                year=year,
                category=category_dir.name,
                path=str(category_dir),
                model_count=model_count,
                property_count=property_count,
                instance_csv_count=instance_csv_count,
                model_exts=model_exts,
                status=status,
            )
        )
    status = "available" if categories else "missing"
    return YearCoverage(
        year=year,
        benchmark_dir=str(year_dir),
        categories=categories,
        status=status,
    )


def render_table(years: list[YearCoverage]) -> str:
    rows = []
    header = [
        "Year",
        "Category",
        "Models",
        "Props",
        "Instances.csv",
        "Status",
    ]
    rows.append(header)
    for year in years:
        if not year.categories:
            rows.append([str(year.year), "(none)", "0", "0", "0", year.status])
            continue
        for category in year.categories:
            rows.append(
                [
                    str(category.year),
                    category.category,
                    str(category.model_count),
                    str(category.property_count),
                    str(category.instance_csv_count),
                    category.status,
                ]
            )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for idx, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="VNN-COMP year to include (repeatable). Default: all present years.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmarks"),
        help="Benchmarks root directory (default: benchmarks).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any category is missing models or properties.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"Benchmarks root not found: {root}", file=sys.stderr)
        return 2

    year_dirs = find_benchmark_roots(root)
    if args.year:
        year_set = {int(y) for y in args.year}
        year_dirs = [p for p in year_dirs if int(p.parent.name.replace("vnncomp", "")) in year_set]
    if not year_dirs:
        print("No benchmark directories found.", file=sys.stderr)
        return 2

    years = [collect_year(p) for p in year_dirs]

    if args.json:
        payload = [
            {
                **asdict(year),
                "categories": [asdict(cat) for cat in year.categories],
            }
            for year in years
        ]
        if args.pretty:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload))
    else:
        print(render_table(years))

    if args.strict:
        missing = [
            cat
            for year in years
            for cat in year.categories
            if cat.status not in {"available", "auxiliary-assets"}
        ]
        if missing:
            print(f"\nMissing categories: {len(missing)}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
