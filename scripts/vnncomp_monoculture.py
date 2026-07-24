#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""VNN-COMP monoculture tracker renderer.

Consumes reports/benchmarks/vnncomp_dashboard.json and derives a
deterministic competitive/partial/blocked classification for each
tracked category. Writes:

- reports/benchmarks/vnncomp_monoculture.json
- reports/benchmarks/vnncomp_monoculture.md

Design: designs/2026-03-11-issue-2569-vnncomp-monoculture-tracker.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vnncomp_current_surface import (
    DEFAULT_REPORTS_DIR,
    add_current_surface_args,
    load_current_dashboard,
)

UTC = timezone.utc

DEFAULT_DASHBOARD_PATH = DEFAULT_REPORTS_DIR / "vnncomp_dashboard.json"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORTS_DIR / "vnncomp_monoculture.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORTS_DIR / "vnncomp_monoculture.md"

COMPETITIVE_SOLVE_RATE = 80.0


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _classify(score: int, solve_rate: float) -> str:
    """Classify a category as competitive/partial/blocked.

    Rules from design doc section 3:
    - competitive: score > 0 and solve_rate >= 80.0
    - partial: score > 0 and solve_rate < 80.0
    - blocked: score == 0
    """
    if score > 0 and solve_rate >= COMPETITIVE_SOLVE_RATE:
        return "competitive"
    if score > 0:
        return "partial"
    return "blocked"


def _sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    """Sort key: status order, then descending solve_rate, descending score, name."""
    status_order = {"competitive": 0, "partial": 1, "blocked": 2}
    return (
        status_order.get(row["status"], 3),
        -row["solve_rate"],
        -row["score"],
        row["category"],
    )


def _build_rows(latest: dict[str, Any]) -> list[dict[str, Any]]:
    """Build and classify category rows from dashboard latest snapshot."""
    categories_data = latest.get("categories", {})
    skipped_data = latest.get("skipped", {})
    rows: list[dict[str, Any]] = []

    for name, stats in categories_data.items():
        score = int(stats.get("score", 0))
        solve_rate = float(stats.get("solve_rate", 0.0))
        rows.append({
            "category": name,
            "status": _classify(score, solve_rate),
            "score": score,
            "total": int(stats.get("total", 0)),
            "solve_rate": solve_rate,
            "verified": int(stats.get("verified", 0)),
            "falsified": int(stats.get("falsified", 0)),
            "timeout": int(stats.get("timeout", 0)),
            "unknown": int(stats.get("unknown", 0)),
            "error": int(stats.get("error", 0)),
            "skip_reason": None,
            "measurement_kind": "attempted",
        })

    for name, reason in skipped_data.items():
        rows.append({
            "category": name, "status": "blocked", "score": 0, "total": 0,
            "solve_rate": 0.0, "verified": 0, "falsified": 0, "timeout": 0,
            "unknown": 0, "error": 0, "skip_reason": reason,
            "measurement_kind": "skipped",
        })

    rows.sort(key=_sort_key)
    return rows


def build_tracker(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Build the monoculture tracker from a dashboard payload."""
    latest = dashboard["latest"]
    rows = _build_rows(latest)

    competitive_count = sum(1 for r in rows if r["status"] == "competitive")
    partial_count = sum(1 for r in rows if r["status"] == "partial")
    blocked_count = sum(1 for r in rows if r["status"] == "blocked")
    categories_with_score = sum(1 for r in rows if r["score"] > 0)

    non_acas = [r for r in rows if not r["category"].startswith("acasxu")]
    non_acas_with_score = sum(1 for r in non_acas if r["score"] > 0)
    non_acas_solved = sum(r["score"] for r in non_acas if r["score"] > 0)
    monoculture_status = "active" if non_acas_with_score == 0 else "cleared"

    summary = {
        "total_score": int(latest.get("total_score", 0)),
        "total_instances": int(latest.get("total_instances", 0)),
        "overall_solve_rate": float(latest.get("overall_solve_rate", 0.0)),
        "categories_tracked": len(rows),
        "categories_attempted": int(latest.get("categories_attempted", 0)),
        "categories_skipped": len(latest.get("skipped", {})),
        "categories_with_score": categories_with_score,
        "competitive_count": competitive_count,
        "partial_count": partial_count,
        "blocked_count": blocked_count,
        "non_acas_categories_with_score": non_acas_with_score,
        "non_acas_solved_instances": non_acas_solved,
        "monoculture_status": monoculture_status,
        "commit": str(latest.get("commit", "")),
        "recorded_at": str(latest.get("recorded_at", "")),
    }

    return {
        "generated_at": _iso_now(),
        "dashboard_generated_at": str(dashboard.get("generated_at", "")),
        "competitive_solve_rate_threshold": COMPETITIVE_SOLVE_RATE,
        "summary": summary,
        "categories": rows,
    }


def render_markdown(tracker: dict[str, Any]) -> str:
    """Render the monoculture tracker as markdown."""
    s = tracker["summary"]
    lines: list[str] = []

    lines.append("# VNN-COMP Monoculture Tracker")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Score:** {s['total_score']} / {s['total_instances']}")
    lines.append(f"- **Overall solve rate:** {s['overall_solve_rate']:.1f}%")
    lines.append(
        f"- **Categories:** {s['categories_tracked']} tracked"
        f" ({s['categories_attempted']} attempted, {s['categories_skipped']} skipped)"
    )
    lines.append(
        f"- **Classification:** {s['competitive_count']} competitive,"
        f" {s['partial_count']} partial, {s['blocked_count']} blocked"
    )
    lines.append(f"- **Categories with score:** {s['categories_with_score']}")
    lines.append(f"- **Non-ACAS categories with score:** {s['non_acas_categories_with_score']}")
    lines.append(f"- **Non-ACAS solved instances:** {s['non_acas_solved_instances']}")
    lines.append(f"- **Monoculture status:** {s['monoculture_status']}")
    lines.append(f"- **Commit:** {s['commit']}")
    lines.append(f"- **Recorded at:** {s['recorded_at']}")
    lines.append("")

    lines.append("## Category Classification")
    lines.append("")

    # Table header
    lines.append(
        "| Category | Status | Solved | Total | Solve Rate"
        " | Verified | Falsified | Timeout | Unknown | Error | Skip Reason |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---:"
        " | ---: | ---: | ---: | ---: | ---: | --- |"
    )

    for row in tracker["categories"]:
        reason = row["skip_reason"] or ""
        lines.append(
            f"| {row['category']} | {row['status']}"
            f" | {row['score']} | {row['total']} | {row['solve_rate']:.1f}%"
            f" | {row['verified']} | {row['falsified']}"
            f" | {row['timeout']} | {row['unknown']} | {row['error']}"
            f" | {reason} |"
        )

    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Derived from `reports/benchmarks/vnncomp_dashboard.json`")
    lines.append(f"- Classification threshold: {COMPETITIVE_SOLVE_RATE:.1f}%")
    lines.append("- Skipped categories follow the current dashboard skip reasons, including overrides when present")
    lines.append("")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    return add_current_surface_args(
        argparse.ArgumentParser(
            description="Render the VNN-COMP monoculture tracker from the current dashboard"
        ),
        metrics_help="Published metrics directory",
        reports_help="Reports directory",
    )


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    reports_dir = Path(args.reports_dir)
    output_json = reports_dir / DEFAULT_OUTPUT_JSON.name
    output_md = reports_dir / DEFAULT_OUTPUT_MD.name
    overrides_path = (
        Path(args.skip_reason_overrides)
        if args.skip_reason_overrides
        else reports_dir / "vnncomp_skip_reason_overrides.json"
    )
    return reports_dir, output_json, output_md, overrides_path


def _load_dashboard_payload(
    reports_dir: Path,
    metrics_dir: Path,
    overrides_path: Path,
) -> dict[str, Any] | None:
    dashboard_path = reports_dir / DEFAULT_DASHBOARD_PATH.name
    dashboard = load_current_dashboard(
        reports_dir=reports_dir,
        metrics_dir=metrics_dir,
        overrides_path=overrides_path,
    )
    if dashboard is not None:
        return dashboard

    if dashboard_path.exists():
        existing = json.loads(dashboard_path.read_text(encoding="utf-8"))
        if "latest" not in existing:
            sys.stdout.write("Dashboard has no 'latest' key. Nothing to do.\n")
            return None

    sys.stdout.write("No VNN-COMP dashboard or published metrics found. Nothing to do.\n")
    return None


def _write_tracker_outputs(output_json: Path, output_md: Path, tracker: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(tracker, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(tracker), encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    reports_dir, output_json, output_md, overrides_path = _resolve_paths(args)
    dashboard = _load_dashboard_payload(reports_dir, Path(args.metrics_dir), overrides_path)
    if dashboard is None:
        return 0

    tracker = build_tracker(dashboard)
    _write_tracker_outputs(output_json, output_md, tracker)

    s = tracker["summary"]
    sys.stdout.write(
        f"Monoculture tracker: {s['categories_tracked']} categories"
        f" ({s['competitive_count']} competitive,"
        f" {s['partial_count']} partial,"
        f" {s['blocked_count']} blocked)."
        f" Monoculture: {s['monoculture_status']}.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
