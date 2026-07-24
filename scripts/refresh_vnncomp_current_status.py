#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Refresh the VNN-COMP current-status surface from published metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from vnncomp_current_surface import add_current_surface_args


def _run(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], check=True)


def _build_parser() -> argparse.ArgumentParser:
    return add_current_surface_args(
        argparse.ArgumentParser(
        description="Refresh dashboard, monoculture, and pulse from VNN-COMP published metrics"
        ),
        metrics_help="Published metrics directory",
        reports_help="Reports directory",
    )


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    metrics_dir = Path(args.metrics_dir)
    reports_dir = Path(args.reports_dir)
    overrides_path = (
        Path(args.skip_reason_overrides)
        if args.skip_reason_overrides
        else reports_dir / "vnncomp_skip_reason_overrides.json"
    )
    return metrics_dir, reports_dir, overrides_path


def _current_surface_args(metrics_dir: Path, reports_dir: Path, overrides_path: Path) -> list[str]:
    return [
        "--metrics-dir",
        str(metrics_dir),
        "--reports-dir",
        str(reports_dir),
        "--skip-reason-overrides",
        str(overrides_path),
    ]


def _merge_args(metrics_dir: Path, reports_dir: Path, overrides_path: Path) -> list[str]:
    return [
        "--benchmarks-dir",
        str(metrics_dir),
        "--pulse-dir",
        str(metrics_dir.parent),
        "--reports-dir",
        str(reports_dir),
        "--skip-reason-overrides",
        str(overrides_path),
    ]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    metrics_dir, reports_dir, overrides_path = _resolve_paths(args)
    latest_path = metrics_dir / "vnncomp_latest.json"
    if not latest_path.exists():
        sys.stdout.write(f"No published VNN-COMP metrics found at {latest_path}. Nothing to refresh.\n")
        return 0

    script_dir = Path(__file__).resolve().parent
    common_args = _current_surface_args(metrics_dir, reports_dir, overrides_path)
    _run(script_dir / "vnncomp_dashboard.py", *common_args)
    _run(script_dir / "vnncomp_monoculture.py", *common_args)
    _run(script_dir / "merge_benchmark_metrics.py", *_merge_args(metrics_dir, reports_dir, overrides_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
