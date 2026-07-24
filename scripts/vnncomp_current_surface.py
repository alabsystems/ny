#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Shared helpers for the VNN-COMP current-status surface."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_REPORTS_DIR = Path("reports/benchmarks")
DEFAULT_METRICS_DIR = Path("metrics/benchmarks")
PSEUDO_BENCHMARK_CATEGORIES = frozenset({"test"})


def add_current_surface_args(
    parser: argparse.ArgumentParser,
    *,
    metrics_help: str,
    reports_help: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--metrics-dir", default=str(DEFAULT_METRICS_DIR), help=metrics_help)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help=reports_help)
    parser.add_argument(
        "--skip-reason-overrides",
        default=None,
        help="Optional JSON file with current skip-reason overrides",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_skip_reason_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected skip-reason overrides object at {path}")

    overrides: dict[str, str] = {}
    for category, reason in payload.items():
        if not isinstance(category, str) or not isinstance(reason, str):
            raise ValueError(f"Expected string skip-reason override entries at {path}")
        overrides[category] = reason
    return overrides


def normalize_current_latest(latest: dict[str, Any]) -> dict[str, Any]:
    """Drop non-benchmark placeholder categories from current-surface payloads."""
    skipped = latest.get("skipped")
    categories_skipped = latest.get("categories_skipped")
    has_pseudo_skipped = isinstance(skipped, dict) and any(
        category in PSEUDO_BENCHMARK_CATEGORIES for category in skipped
    )
    has_pseudo_categories = isinstance(categories_skipped, list) and any(
        category in PSEUDO_BENCHMARK_CATEGORIES for category in categories_skipped
    )
    if not has_pseudo_skipped and not has_pseudo_categories:
        return latest

    snapshot = deepcopy(latest)
    if isinstance(skipped, dict):
        snapshot["skipped"] = {
            category: reason
            for category, reason in skipped.items()
            if category not in PSEUDO_BENCHMARK_CATEGORIES
        }
    if isinstance(categories_skipped, list):
        snapshot["categories_skipped"] = [
            category
            for category in categories_skipped
            if category not in PSEUDO_BENCHMARK_CATEGORIES
        ]
    return snapshot


def apply_skip_reason_overrides(
    latest: dict[str, Any],
    overrides: dict[str, str],
) -> dict[str, Any]:
    if not overrides:
        return latest

    snapshot = deepcopy(latest)
    skipped = snapshot.get("skipped")
    if not isinstance(skipped, dict):
        return snapshot

    for category, reason in overrides.items():
        if category in skipped:
            skipped[category] = reason

    return snapshot


def load_current_latest(
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    overrides_path: Path | None = None,
) -> dict[str, Any] | None:
    latest_path = metrics_dir / "vnncomp_latest.json"
    if not latest_path.exists():
        return None

    latest = normalize_current_latest(load_json(latest_path))
    if overrides_path is None:
        return latest

    overrides = load_skip_reason_overrides(overrides_path)
    return normalize_current_latest(apply_skip_reason_overrides(latest, overrides))


def load_current_dashboard(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    overrides_path: Path | None = None,
) -> dict[str, Any] | None:
    dashboard_path = reports_dir / "vnncomp_dashboard.json"
    if dashboard_path.exists():
        dashboard = load_json(dashboard_path)
        latest = dashboard.get("latest")
        if isinstance(latest, dict):
            dashboard = dict(dashboard)
            dashboard["latest"] = normalize_current_latest(latest)
            if overrides_path is None:
                return dashboard
            dashboard["latest"] = apply_skip_reason_overrides(
                dashboard["latest"],
                load_skip_reason_overrides(overrides_path),
            )
            dashboard["latest"] = normalize_current_latest(dashboard["latest"])
            return dashboard

    latest = load_current_latest(metrics_dir=metrics_dir, overrides_path=overrides_path)
    if latest is None:
        return None

    return {
        "generated_at": "",
        "latest": latest,
    }
