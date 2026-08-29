#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the measured-2026 coverage census from committed result banks.

Top-level ``reports/measured-2026/*.csv`` files are candidate complete-family
banks; partial and invalid sweeps live below ``partials_20260812_NOT_BANKED``.
Every candidate must exactly match the ordered official instances and budgets
pinned in ``bank_manifest.json`` before it can appear as complete. The manifest
also binds the reviewed bank bytes, so a verdict/runtime edit cannot silently
become the new index of record. The README used to repeat that inventory by hand
and silently fell behind both a new bank and a replacement sweep. This helper
makes that small, factual part generated while leaving interpretation and
measurement caveats hand-authored.

Usage::

    python3 scripts/update_measured_2026_index.py
    python3 scripts/update_measured_2026_index.py --check

``--check`` writes nothing and exits nonzero when the committed census is
stale, making it suitable for the repository drift gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "measured-2026"
README = REPORT_DIR / "README.md"
PARTIAL_DIR = REPORT_DIR / "partials_20260812_NOT_BANKED"
BANK_MANIFEST = REPORT_DIR / "bank_manifest.json"

START_MARKER = "<!-- BEGIN GENERATED: measured-2026 coverage census -->"
END_MARKER = "<!-- END GENERATED: measured-2026 coverage census -->"
PROTOCOL_VERDICTS = ("sat", "unsat", "unknown", "timeout")
MANIFEST_SCHEMA = "ny_measured_2026_bank_manifest_v1"
SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CoverageError(ValueError):
    """A result bank cannot support the coverage claim being generated."""


@dataclass(frozen=True)
class OfficialInstance:
    onnx: str
    vnnlib: str
    budget_seconds: Decimal


@dataclass(frozen=True)
class BankContract:
    category: str
    instances_csv: str
    instances_csv_sha256: str
    bank_csv_sha256: str
    instances: tuple[OfficialInstance, ...]


@dataclass(frozen=True)
class BankManifest:
    source_repository: str
    source_commit: str
    categories: frozenset[str]
    contracts: dict[str, BankContract]


@dataclass(frozen=True)
class Bank:
    category: str
    path: Path
    rows: int
    verdicts: tuple[tuple[str, int], ...]

    @property
    def decided(self) -> int:
        counts = dict(self.verdicts)
        return counts.get("sat", 0) + counts.get("unsat", 0)


@dataclass(frozen=True)
class Coverage:
    banks: tuple[Bank, ...]
    partials: tuple[tuple[str, tuple[Path, ...]], ...]
    untouched: tuple[str, ...]
    total_categories: int


def _require_object_keys(
    value: dict[str, object], expected: set[str], label: str
) -> None:
    found = set(value)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise CoverageError(f"{label} has invalid fields ({'; '.join(details)})")


def _parse_budget(raw: object, label: str) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise CoverageError(f"{label} must be a positive decimal number")
    text = str(raw)
    if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", text) is None:
        raise CoverageError(f"{label} must be a canonical positive decimal number")
    try:
        budget = Decimal(text)
    except InvalidOperation as error:  # pragma: no cover - syntax checked above
        raise CoverageError(f"{label} is not a decimal number") from error
    if not budget.is_finite() or budget <= 0:
        raise CoverageError(f"{label} must be greater than zero")
    return budget


def load_bank_manifest(path: Path) -> BankManifest:
    """Load the checkout-independent official-row contracts for complete banks."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageError(f"cannot read bank manifest {path}: {error}") from error
    if not isinstance(raw, dict):
        raise CoverageError(f"{path}: bank manifest must be a JSON object")
    _require_object_keys(
        raw,
        {
            "schema",
            "source_repository",
            "source_commit",
            "scored_categories",
            "banks",
        },
        str(path),
    )
    if raw["schema"] != MANIFEST_SCHEMA:
        raise CoverageError(
            f"{path}: unsupported bank manifest schema {raw['schema']!r}"
        )
    source_repository = raw["source_repository"]
    source_commit = raw["source_commit"]
    if not isinstance(source_repository, str) or not source_repository:
        raise CoverageError(f"{path}: source_repository must be a non-empty string")
    if not isinstance(source_commit, str) or GIT_COMMIT.fullmatch(source_commit) is None:
        raise CoverageError(f"{path}: source_commit must be a lowercase full Git hash")

    category_list = raw["scored_categories"]
    if (
        not isinstance(category_list, list)
        or not category_list
        or not all(
            isinstance(category, str) and SAFE_CATEGORY.fullmatch(category) is not None
            for category in category_list
        )
    ):
        raise CoverageError(f"{path}: scored_categories must contain safe category names")
    if category_list != sorted(category_list) or len(category_list) != len(set(category_list)):
        raise CoverageError(
            f"{path}: scored_categories must be sorted and contain no duplicates"
        )
    categories = frozenset(category_list)

    bank_values = raw["banks"]
    if not isinstance(bank_values, dict):
        raise CoverageError(f"{path}: banks must be a JSON object")
    contracts: dict[str, BankContract] = {}
    for category, value in bank_values.items():
        label = f"{path}: bank {category!r}"
        if category not in categories:
            raise CoverageError(f"{label} is outside scored_categories")
        if not isinstance(value, dict):
            raise CoverageError(f"{label} must be a JSON object")
        _require_object_keys(
            value,
            {
                "instances_csv",
                "instances_csv_sha256",
                "bank_csv_sha256",
                "instances",
            },
            label,
        )
        instances_csv = value["instances_csv"]
        if (
            not isinstance(instances_csv, str)
            or not instances_csv
            or Path(instances_csv).is_absolute()
            or ".." in Path(instances_csv).parts
        ):
            raise CoverageError(f"{label} has an unsafe instances_csv path")
        instances_csv_sha256 = value["instances_csv_sha256"]
        if (
            not isinstance(instances_csv_sha256, str)
            or SHA256.fullmatch(instances_csv_sha256) is None
        ):
            raise CoverageError(f"{label} has an invalid instances_csv_sha256")
        bank_csv_sha256 = value["bank_csv_sha256"]
        if (
            not isinstance(bank_csv_sha256, str)
            or SHA256.fullmatch(bank_csv_sha256) is None
        ):
            raise CoverageError(f"{label} has an invalid bank_csv_sha256")
        instance_values = value["instances"]
        if not isinstance(instance_values, list) or not instance_values:
            raise CoverageError(f"{label} must pin at least one official instance")
        instances = []
        for index, instance in enumerate(instance_values, start=1):
            instance_label = f"{label} instance {index}"
            if (
                not isinstance(instance, list)
                or len(instance) != 3
                or not isinstance(instance[0], str)
                or not instance[0]
                or not isinstance(instance[1], str)
                or not instance[1]
            ):
                raise CoverageError(
                    f"{instance_label} must be [onnx, vnnlib, budget_seconds]"
                )
            instances.append(
                OfficialInstance(
                    onnx=instance[0],
                    vnnlib=instance[1],
                    budget_seconds=_parse_budget(instance[2], f"{instance_label} budget"),
                )
            )
        contracts[category] = BankContract(
            category=category,
            instances_csv=instances_csv,
            instances_csv_sha256=instances_csv_sha256,
            bank_csv_sha256=bank_csv_sha256,
            instances=tuple(instances),
        )
    return BankManifest(
        source_repository=source_repository,
        source_commit=source_commit,
        categories=categories,
        contracts=contracts,
    )


def _load_bank(path: Path, contract: BankContract) -> Bank:
    category = path.stem
    counts: Counter[str] = Counter()
    rows = 0

    try:
        bank_bytes = path.read_bytes()
    except OSError as error:
        raise CoverageError(f"cannot read result bank {path}: {error}") from error
    bank_sha256 = hashlib.sha256(bank_bytes).hexdigest()
    try:
        bank_text = bank_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CoverageError(f"{path}: result bank is not UTF-8: {error}") from error

    try:
        with io.StringIO(bank_text, newline="") as source:
            for line_number, row in enumerate(csv.reader(source), start=1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                cells = [cell.strip() for cell in row]
                normalized = [cell.lower() for cell in cells]
                verdict_positions = [
                    index
                    for index, cell in enumerate(normalized)
                    if cell in PROTOCOL_VERDICTS
                ]

                # Some producers include a conventional header while the
                # current banks do not.  Accept exactly one leading header.
                if (
                    line_number == 1
                    and not verdict_positions
                    and {"verdict", "result", "status"}.intersection(normalized)
                ):
                    continue
                if len(verdict_positions) != 1:
                    raise CoverageError(
                        f"{path}:{line_number}: expected exactly one protocol "
                        f"verdict ({', '.join(PROTOCOL_VERDICTS)}), found "
                        f"{[normalized[index] for index in verdict_positions] or 'none'}"
                    )
                if len(cells) < 3 or cells[0] != category:
                    raise CoverageError(
                        f"{path}:{line_number}: category "
                        f"{cells[0] if cells else '<missing>'!r} does not "
                        f"match bank filename {category!r}"
                    )
                if rows >= len(contract.instances):
                    raise CoverageError(
                        f"{path}:{line_number}: bank has more than the pinned "
                        f"{len(contract.instances)} official rows"
                    )
                expected = contract.instances[rows]
                if (cells[1], cells[2]) != (expected.onnx, expected.vnnlib):
                    raise CoverageError(
                        f"{path}:{line_number}: instance occurrence {rows + 1} does "
                        "not match the pinned instances.csv row; expected "
                        f"{(expected.onnx, expected.vnnlib)!r}, found "
                        f"{(cells[1], cells[2])!r}"
                    )
                verdict_position = verdict_positions[0]
                if verdict_position + 1 >= len(cells):
                    raise CoverageError(
                        f"{path}:{line_number}: protocol verdict has no elapsed-seconds field"
                    )
                elapsed_raw = cells[verdict_position + 1]
                try:
                    elapsed = Decimal(elapsed_raw)
                except InvalidOperation as error:
                    raise CoverageError(
                        f"{path}:{line_number}: elapsed seconds {elapsed_raw!r} "
                        "is not numeric"
                    ) from error
                if not elapsed.is_finite() or elapsed < 0:
                    raise CoverageError(
                        f"{path}:{line_number}: elapsed seconds must be finite and nonnegative"
                    )
                if elapsed > expected.budget_seconds:
                    raise CoverageError(
                        f"{path}:{line_number}: elapsed {elapsed} s exceeds official "
                        f"budget {expected.budget_seconds} s for pinned instance "
                        f"occurrence {rows + 1}"
                    )
                counts[normalized[verdict_position]] += 1
                rows += 1
    except csv.Error as error:
        raise CoverageError(f"{path}: invalid CSV: {error}") from error

    if rows != len(contract.instances):
        raise CoverageError(
            f"{path}: complete bank has {rows} rows; pinned instances.csv has "
            f"{len(contract.instances)} occurrences"
        )
    if bank_sha256 != contract.bank_csv_sha256:
        raise CoverageError(
            f"{path}: bank CSV sha256 {bank_sha256} does not match reviewed pin "
            f"{contract.bank_csv_sha256}; review the result change before regenerating"
        )
    return Bank(
        category=category,
        path=path,
        rows=rows,
        verdicts=tuple((token, counts[token]) for token in PROTOCOL_VERDICTS if counts[token]),
    )


def collect_coverage(
    report_dir: Path = REPORT_DIR,
    manifest: BankManifest | None = None,
) -> Coverage:
    """Classify every scored category from complete and partial bank files."""
    if manifest is None:
        manifest = load_bank_manifest(report_dir / BANK_MANIFEST.name)
    categories = manifest.categories
    bank_paths = sorted(report_dir.glob("*.csv"))
    missing_contracts = sorted(
        path.stem for path in bank_paths if path.stem not in manifest.contracts
    )
    if missing_contracts:
        raise CoverageError(
            "complete banks lack a pinned official instances.csv contract: "
            + ", ".join(missing_contracts)
        )
    banks = tuple(
        _load_bank(path, manifest.contracts[path.stem]) for path in bank_paths
    )
    banked = {bank.category for bank in banks}
    if len(banked) != len(banks):  # pragma: no cover - stems make this defensive
        raise CoverageError("duplicate complete bank category")

    unknown_banks = sorted(banked - categories)
    if unknown_banks:
        raise CoverageError(
            "complete banks name categories outside the 2026 census: "
            + ", ".join(unknown_banks)
        )

    partial_paths: dict[str, list[Path]] = defaultdict(list)
    partial_dir = report_dir / PARTIAL_DIR.name
    for path in sorted(partial_dir.glob("*_PARTIAL*.csv")):
        category, separator, _suffix = path.stem.partition("_PARTIAL")
        if not separator or not category:
            raise CoverageError(f"cannot derive partial category from {path}")
        if category not in categories:
            raise CoverageError(
                f"partial artifact {path} names category outside the 2026 census"
            )
        if category not in banked:
            partial_paths[category].append(path)

    partials = tuple(
        (category, tuple(paths)) for category, paths in sorted(partial_paths.items())
    )
    partial_categories = set(partial_paths)
    untouched = tuple(sorted(categories - banked - partial_categories))
    if len(banked) + len(partial_categories) + len(untouched) != len(categories):
        raise CoverageError("coverage census is not a disjoint category partition")

    return Coverage(
        banks=banks,
        partials=partials,
        untouched=untouched,
        total_categories=len(categories),
    )


def _format_verdicts(bank: Bank) -> str:
    return ", ".join(f"{count} {token}" for token, count in bank.verdicts)


def render_census(coverage: Coverage, report_dir: Path = REPORT_DIR) -> str:
    """Render the generated Markdown block."""
    lines = [
        START_MARKER,
        "<!-- Validated against `bank_manifest.json`; do not edit by hand. Run "
        "`python3 scripts/update_measured_2026_index.py`. -->",
        "",
        "### Banked — measured on the 2026 corpus",
        "",
        "| family | rows | protocol result | decided rows |",
        "|---|---:|---|---:|",
    ]
    for bank in coverage.banks:
        percent = 100.0 * bank.decided / bank.rows
        lines.append(
            f"| `{bank.category}` | {bank.rows} | {_format_verdicts(bank)} | "
            f"{bank.decided} / {bank.rows} ({percent:.1f}%) |"
        )
    lines.extend(
        [
            "",
            f"Complete banks cover **{len(coverage.banks)} of "
            f"{coverage.total_categories}** scored categories.",
            "",
            "### Measured but not banked — partial or invalid artifacts",
            "",
            "| family | retained artifacts |",
            "|---|---|",
        ]
    )
    for category, paths in coverage.partials:
        links = ", ".join(
            f"[`{path.name}`]({path.relative_to(report_dir).as_posix()})" for path in paths
        )
        lines.append(f"| `{category}` | {links} |")
    lines.extend(
        [
            "",
            f"Non-superseded partial artifacts cover **{len(coverage.partials)}** "
            "additional categories. They are not scores.",
            "",
            f"### Untouched on the 2026 corpus — {len(coverage.untouched)} categories",
            "",
            ", ".join(f"`{category}`" for category in coverage.untouched),
            "",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def update_readme(original: str, census: str) -> str:
    """Replace exactly one generated census block in ``original``."""
    if original.count(START_MARKER) != 1 or original.count(END_MARKER) != 1:
        raise CoverageError(
            "README must contain exactly one measured-2026 coverage marker pair"
        )
    start = original.index(START_MARKER)
    end = original.index(END_MARKER, start) + len(END_MARKER)
    if end <= start:  # pragma: no cover - index ordering is otherwise explicit
        raise CoverageError("README coverage markers are out of order")
    return original[:start] + census + original[end:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero without writing when the committed census is stale",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="complete-bank directory (default: reports/measured-2026)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        help="README to update (default: <report-dir>/README.md)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="pinned bank manifest (default: <report-dir>/bank_manifest.json)",
    )
    return parser.parse_args(argv)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = args.report_dir
    readme = args.readme or report_dir / README.name
    manifest_path = args.manifest or report_dir / BANK_MANIFEST.name
    try:
        manifest = load_bank_manifest(manifest_path)
        coverage = collect_coverage(report_dir, manifest)
        original = readme.read_text(encoding="utf-8")
        updated = update_readme(original, render_census(coverage, report_dir))
    except (CoverageError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if updated == original:
        print(
            f"OK: {_display_path(readme)} matches {len(coverage.banks)} "
            "complete result banks"
        )
        return 0
    if args.check:
        print(
            f"STALE: {_display_path(readme)} does not match the complete "
            "result banks; run scripts/update_measured_2026_index.py",
            file=sys.stderr,
        )
        return 1

    with readme.open("w", encoding="utf-8", newline="\n") as output:
        output.write(updated)
    print(
        f"updated {_display_path(readme)} from "
        f"{len(coverage.banks)} complete result banks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
