#!/usr/bin/env python3
"""Exact free-dimension census for NN4SYS 2048-dual VNNLIB properties.

The generated NN4SYS files put one top-level `(and ...)` disjunct on each
line.  Every input has an explicit lower and upper decimal bound.  This tool
compares those decimals exactly (via ``Decimal``), validates the per-clause
bound pairs, and reports how many genuinely ranged inputs each disjunct has.

It is deliberately a property-shape diagnostic: it does not load a model and
has no verification/verdict authority.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, TextIO


FILE_RE = re.compile(r"^cardinality_1_(\d+)_2048_dual\.vnnlib$")
LOWER_RE = re.compile(r"\(>= X_(\d+) ([^()\s]+)\)")
UPPER_RE = re.compile(r"\(<= X_(\d+) ([^()\s]+)\)")


@dataclass(frozen=True)
class PropertyCensus:
    property_clauses: int
    observed_clauses: int
    free_dimension_histogram: Counter[int]


def _decimal(value: str, *, path: Path, clause: int, axis: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"{path}: clause {clause}: X_{axis} has invalid decimal {value!r}"
        ) from error
    if not parsed.is_finite():
        raise ValueError(
            f"{path}: clause {clause}: X_{axis} bound must be finite, got {value!r}"
        )
    return parsed


def census_property(path: Path) -> PropertyCensus:
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected NN4SYS dual property filename: {path.name}")
    expected = int(match.group(1))
    histogram: Counter[int] = Counter()
    observed = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.lstrip().startswith("(and "):
                continue
            observed += 1
            lower_pairs = LOWER_RE.findall(line)
            upper_pairs = UPPER_RE.findall(line)
            lower_raw = {int(axis): value for axis, value in lower_pairs}
            upper_raw = {int(axis): value for axis, value in upper_pairs}
            if len(lower_raw) != len(lower_pairs) or len(upper_raw) != len(upper_pairs):
                raise ValueError(
                    f"{path}:{line_number}: duplicate lower/upper input bound"
                )
            if lower_raw.keys() != upper_raw.keys():
                missing_upper = sorted(lower_raw.keys() - upper_raw.keys())
                missing_lower = sorted(upper_raw.keys() - lower_raw.keys())
                raise ValueError(
                    f"{path}:{line_number}: mismatched input bounds; "
                    f"missing_upper={missing_upper}, missing_lower={missing_lower}"
                )
            if not lower_raw:
                raise ValueError(f"{path}:{line_number}: clause has no input bounds")

            free_dimensions = 0
            for axis in sorted(lower_raw):
                lower = _decimal(
                    lower_raw[axis], path=path, clause=observed, axis=axis
                )
                upper = _decimal(
                    upper_raw[axis], path=path, clause=observed, axis=axis
                )
                if lower > upper:
                    raise ValueError(
                        f"{path}:{line_number}: X_{axis} lower {lower} > upper {upper}"
                    )
                free_dimensions += lower != upper
            histogram[free_dimensions] += 1

    if observed != expected:
        raise ValueError(
            f"{path}: filename declares {expected} clauses, observed {observed}"
        )
    return PropertyCensus(expected, observed, histogram)


def census_directory(directory: Path, *, open_only: bool) -> list[PropertyCensus]:
    properties: list[tuple[int, Path]] = []
    for path in directory.glob("cardinality_1_*_2048_dual.vnnlib"):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        clauses = int(match.group(1))
        if open_only and clauses == 1:
            continue
        properties.append((clauses, path))
    properties.sort()
    if not properties:
        raise ValueError(f"{directory}: no 2048-dual VNNLIB properties found")
    return [census_property(path) for _, path in properties]


CSV_COLUMNS = [
    "property_clauses",
    "observed_clauses",
    "free_dim_0",
    "free_dim_1",
    "free_dim_2",
    "free_dim_3",
    "free_dim_4",
    "free_dim_5",
    "free_dim_gt5",
    "one_dim_percent",
]


def _row(census: PropertyCensus) -> dict[str, object]:
    hist = census.free_dimension_histogram
    return {
        "property_clauses": census.property_clauses,
        "observed_clauses": census.observed_clauses,
        "free_dim_0": hist[0],
        "free_dim_1": hist[1],
        "free_dim_2": hist[2],
        "free_dim_3": hist[3],
        "free_dim_4": hist[4],
        "free_dim_5": hist[5],
        "free_dim_gt5": sum(count for dims, count in hist.items() if dims > 5),
        "one_dim_percent": f"{100.0 * hist[1] / census.observed_clauses:.6f}",
    }


def write_csv(censuses: Iterable[PropertyCensus], destination: TextIO) -> None:
    censuses = list(censuses)
    writer = csv.DictWriter(destination, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for census in censuses:
        writer.writerow(_row(census))

    total_hist: Counter[int] = Counter()
    total_clauses = 0
    for census in censuses:
        total_clauses += census.observed_clauses
        total_hist.update(census.free_dimension_histogram)
    total = PropertyCensus(total_clauses, total_clauses, total_hist)
    row = _row(total)
    row["property_clauses"] = "TOTAL"
    writer.writerow(row)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "vnnlib_dir",
        type=Path,
        help="directory containing cardinality_1_*_2048_dual.vnnlib",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="exclude the already-solved cardinality_1_1 property",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write CSV here instead of stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        censuses = census_directory(args.vnnlib_dir, open_only=args.open_only)
        if args.output is None:
            write_csv(censuses, sys.stdout)
        else:
            with args.output.open("w", encoding="utf-8", newline="") as destination:
                write_csv(censuses, destination)
    except (OSError, ValueError) as error:
        print(f"nn4sys dual census: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
