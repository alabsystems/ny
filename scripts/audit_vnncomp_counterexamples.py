#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""
Replay-classify VNN-COMP counterexample disagreements.

Usage:
  python3 scripts/audit_vnncomp_counterexamples.py \\
    --ny-csv reports/benchmarks/acasxu_2023_*.csv \\
    --reference-csv reports/benchmarks/reference/acasxu_2023_alpha_beta_crown.csv \\
    --ny-binary ./target/release/ny \\
    --output-json reports/benchmarks/acasxu_classifier_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from scripts.vnnlib_parser import (
        VnnlibParseError,
        evaluate_output_property,
        parse_vnnlib_output_property,
    )
except ImportError:
    from vnnlib_parser import (  # type: ignore[no-redef]
        VnnlibParseError,
        evaluate_output_property,
        parse_vnnlib_output_property,
    )

log = logging.getLogger(__name__)

CLASSIFICATIONS = (
    "solver_hallucination",
    "property_eval_bug",
    "reference_or_asset_conflict",
    "replay_failed",
)


# -- CSV helpers --

def _strip_model_key(path: str) -> str:
    name = os.path.basename(path)
    for suffix in (".onnx.gz", ".onnx"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _strip_property_key(path: str) -> str:
    name = os.path.basename(path)
    return name[:-7] if name.endswith(".vnnlib") else name


def _normalize_result(result: str) -> str:
    mapping = {
        "unsat": "verified", "verified": "verified", "holds": "verified",
        "sat": "violated", "violated": "violated", "falsified": "violated",
    }
    return mapping.get(result.strip().lower(), "unknown")


@dataclass
class NyRow:
    model_key: str
    property_key: str
    status: str
    model_path: str
    property_path: str
    preset_path: str


def parse_ny_csv(path: Path) -> list[NyRow]:
    """Parse a ny benchmark CSV (backend_benchmark_row_v1 format)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "schema_version":
            raise ValueError(f"expected backend_benchmark_row_v1 header, got: {header[:3]}")
        for row in reader:
            if len(row) < 13 or row[1] != "vnncomp_single_backend":
                continue
            rows.append(NyRow(
                model_key=_strip_model_key(row[7]),
                property_key=_strip_property_key(row[8]),
                status=_normalize_result(row[12]),
                model_path=row[7], property_path=row[8], preset_path=row[9],
            ))
    return rows


def parse_reference_csv(path: Path) -> dict[str, str]:
    """Parse reference CSV → {model_key|property_key: normalized_result}."""
    results = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not (header and header[0] == "model" and len(header) == 3):
            raise ValueError(f"unsupported reference CSV format: {header[:3]}")
        for row in reader:
            if len(row) < 3:
                continue
            key = f"{_strip_model_key(row[0])}|{_strip_property_key(row[1])}"
            results[key] = _normalize_result(row[2])
    return results


# -- Classification types --

@dataclass
class ClassificationResult:
    model_key: str
    property_key: str
    ny_status: str
    ref_status: str
    classification: str
    detail: str = ""
    counterexample_input: Optional[list[float]] = None
    counterexample_output: Optional[list[float]] = None
    replayed_output: Optional[list[float]] = None
    constraint_satisfied: Optional[bool] = None


@dataclass
class ClassifierReport:
    generated_at: str
    ny_csv: str
    reference_csv: str
    total_critical_mismatches: int
    classified_rows: list[ClassificationResult] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)


# -- Ny rerun --

def _rerun_ny(
    ny_binary: str, model_path: str, property_path: str,
    preset_path: str, timeout_seconds: int = 120,
) -> Optional[dict]:
    """Rerun ny with --json, return parsed JSON or None."""
    cmd = [ny_binary, "beta-crown", model_path, "--property", property_path, "--json"]
    if preset_path and os.path.exists(preset_path):
        cmd.extend(["--preset", preset_path])
    cmd.extend(["--timeout", str(timeout_seconds)])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds + 30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("ny rerun failed: %s", exc)
        return None
    if result.returncode not in (0, 1, 10, 20):
        return None
    stdout = result.stdout.strip()
    start, end = stdout.find("{"), stdout.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None


# -- ONNX replay --

def _replay_onnx(model_path: str, input_array: list[float]) -> Optional[list[float]]:
    """Replay input through ONNX ReferenceEvaluator, return flat output list."""
    try:
        import numpy as np
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError as exc:
        log.warning("missing ONNX dependency: %s", exc)
        return None
    try:
        model = onnx.load(model_path)
        init_names = {i.name for i in model.graph.initializer}
        data_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
        if not data_inputs:
            log.warning("no data inputs found in %s", model_path)
            return None
        input_info = data_inputs[0]
        shape = [d.dim_value if d.dim_value > 0 else 1
                 for d in input_info.type.tensor_type.shape.dim]
        input_np = np.array(input_array, dtype=np.float32).reshape(shape)
        evaluator = ReferenceEvaluator(model)
        results = evaluator.run(None, {input_info.name: input_np})
        return results[0].flatten().tolist()
    except Exception as exc:
        log.warning("ONNX replay error: %s", exc)
        return None


# -- Row classification --

def _extract_counterexample(ny_json: dict) -> Optional[tuple[list[float], list[float]]]:
    """Extract (input, output) from ny JSON, or None."""
    if _normalize_result(ny_json.get("status", "")) != "violated":
        return None
    cex = ny_json.get("counterexample")
    if not cex or "input" not in cex or "output" not in cex:
        return None
    return cex["input"], cex["output"]


def _check_replay_divergence(
    cex_output: list[float], replayed: list[float],
) -> Optional[str]:
    """Return divergence detail string, or None if outputs are close."""
    import numpy as np
    ny_out = np.array(cex_output, dtype=np.float64)
    replay_out = np.array(replayed, dtype=np.float64)
    if not np.allclose(ny_out, replay_out, atol=1e-4, rtol=1e-3):
        max_diff = float(np.max(np.abs(ny_out - replay_out)))
        return f"replay diverges from ny output: max_diff={max_diff:.6e}"
    return None


def classify_row(
    row: NyRow, ny_binary: str, timeout_seconds: int = 120,
) -> ClassificationResult:
    """Classify a single ny=violated / ref=verified mismatch."""
    result = ClassificationResult(
        model_key=row.model_key, property_key=row.property_key,
        ny_status=row.status, ref_status="verified", classification="replay_failed",
    )
    log.info("Rerunning: %s / %s", row.model_key, row.property_key)
    ny_json = _rerun_ny(
        ny_binary, row.model_path, row.property_path,
        row.preset_path, timeout_seconds,
    )
    if ny_json is None:
        result.detail = "ny rerun produced no parseable JSON output"
        return result

    cex_pair = _extract_counterexample(ny_json)
    if cex_pair is None:
        result.detail = f"ny rerun status={ny_json.get('status', '?')}, no counterexample"
        return result
    cex_input, cex_output = cex_pair
    result.counterexample_input = cex_input
    result.counterexample_output = cex_output

    replayed = _replay_onnx(row.model_path, cex_input)
    if replayed is None:
        result.detail = "ONNX ReferenceEvaluator replay failed"
        return result
    result.replayed_output = replayed

    divergence = _check_replay_divergence(cex_output, replayed)
    if divergence is not None:
        result.classification = "solver_hallucination"
        result.detail = divergence
        return result

    try:
        prop_text = Path(row.property_path).read_text(encoding="utf-8")
        prop = parse_vnnlib_output_property(prop_text)
    except (VnnlibParseError, OSError) as exc:
        result.detail = f"VNN-LIB parse error: {exc}"
        return result

    satisfied = evaluate_output_property(prop, replayed)
    result.constraint_satisfied = satisfied
    if satisfied:
        result.classification = "reference_or_asset_conflict"
        result.detail = (
            "witness replays AND satisfies output constraints; "
            "reference says verified but counterexample is real"
        )
    else:
        result.classification = "property_eval_bug"
        result.detail = "witness replays but output constraints not satisfied"
    return result


# -- Report building --

def build_report(
    ny_csv_path: Path, reference_csv_path: Path, ny_binary: str,
    row_filter: Optional[str] = None, timeout_seconds: int = 120,
) -> ClassifierReport:
    """Build classifier report for all critical mismatches."""
    ny_rows = parse_ny_csv(ny_csv_path)
    ref_results = parse_reference_csv(reference_csv_path)

    critical_rows = [
        row for row in ny_rows
        if row.status == "violated"
        and ref_results.get(f"{row.model_key}|{row.property_key}", "unknown") == "verified"
        and (row_filter is None or row_filter in f"{row.model_key}|{row.property_key}")
    ]
    report = ClassifierReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ny_csv=str(ny_csv_path), reference_csv=str(reference_csv_path),
        total_critical_mismatches=len(critical_rows),
    )
    log.info("Found %d critical mismatch(es) to classify", len(critical_rows))
    for i, row in enumerate(critical_rows, 1):
        log.info("[%d/%d] %s / %s", i, len(critical_rows), row.model_key, row.property_key)
        report.classified_rows.append(classify_row(row, ny_binary, timeout_seconds))
    for label in CLASSIFICATIONS:
        report.summary[label] = sum(1 for r in report.classified_rows if r.classification == label)
    return report


def format_text_report(report: ClassifierReport) -> str:
    """Format a human-readable text report."""
    lines = [
        "=== VNN-COMP Counterexample Replay Classifier ===",
        f"Generated: {report.generated_at}",
        f"Ny CSV: {report.ny_csv}",
        f"Reference CSV: {report.reference_csv}",
        f"Total critical mismatches: {report.total_critical_mismatches}",
        "", "--- Classification Summary ---",
    ]
    for label in CLASSIFICATIONS:
        lines.append(f"  {label}: {report.summary.get(label, 0)}")
    lines.append("")
    lines.append("--- Per-row Results ---")
    for r in report.classified_rows:
        lines.append(f"  {r.model_key} / {r.property_key}: {r.classification} — {r.detail}")
    lines.append("")
    return "\n".join(lines)


# -- CLI --

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay-classify VNN-COMP counterexample disagreements",
    )
    parser.add_argument("--ny-csv", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--ny-binary", required=True)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--row-filter", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = _parse_args(argv)

    if not args.ny_csv.exists():
        log.error("ny CSV not found: %s", args.ny_csv)
        return 2
    if not args.reference_csv.exists():
        log.error("reference CSV not found: %s", args.reference_csv)
        return 2

    report = build_report(
        ny_csv_path=args.ny_csv, reference_csv_path=args.reference_csv,
        ny_binary=args.ny_binary, row_filter=args.row_filter,
        timeout_seconds=args.timeout,
    )
    sys.stdout.write(format_text_report(report))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8",
        )
        log.info("Classifier artifact: %s", args.output_json)

    return 1 if report.summary.get("replay_failed", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
