#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Merge benchmark metrics into pulse metrics/latest.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnncomp_current_surface import DEFAULT_REPORTS_DIR, load_current_latest

PULSE_FILE_NAMES = ("latest.json", "latest_partial.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge benchmark metrics into pulse snapshots")
    parser.add_argument("--pulse-dir", default="metrics", help="Pulse metrics directory")
    parser.add_argument(
        "--benchmarks-dir",
        default="metrics/benchmarks",
        help="Benchmark metrics directory",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Reports directory containing optional skip-reason overrides",
    )
    parser.add_argument(
        "--skip-reason-overrides",
        default=None,
        help="Optional JSON file with current skip-reason overrides",
    )
    return parser


def _resolve_overrides_path(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.reports_dir)
    if args.skip_reason_overrides:
        return Path(args.skip_reason_overrides)
    return reports_dir / "vnncomp_skip_reason_overrides.json"


def _load_bench_payload(bench_dir: Path) -> dict | None:
    bench_latest = bench_dir / "latest.json"
    if not bench_latest.exists():
        return None
    return json.loads(bench_latest.read_text(encoding="utf-8"))


def _iter_pulse_paths(pulse_dir: Path) -> list[Path]:
    return [pulse_dir / name for name in PULSE_FILE_NAMES if (pulse_dir / name).exists()]


def _write_pulse_snapshot(
    pulse_path: Path,
    bench_data: dict | None,
    vnncomp_data: dict | None,
) -> None:
    pulse_data = json.loads(pulse_path.read_text(encoding="utf-8"))
    if bench_data is not None:
        pulse_data["benchmarks"] = bench_data
    if vnncomp_data is not None:
        pulse_data["vnncomp_benchmarks"] = vnncomp_data
    pulse_path.write_text(json.dumps(pulse_data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    pulse_dir = Path(args.pulse_dir)
    bench_dir = Path(args.benchmarks_dir)
    overrides_path = _resolve_overrides_path(args)

    pulse_paths = _iter_pulse_paths(pulse_dir)
    if not pulse_paths:
        return 0

    bench_data = _load_bench_payload(bench_dir)
    vnncomp_data = load_current_latest(metrics_dir=bench_dir, overrides_path=overrides_path)

    for pulse_path in pulse_paths:
        _write_pulse_snapshot(pulse_path, bench_data, vnncomp_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
