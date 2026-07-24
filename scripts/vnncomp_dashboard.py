#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Render a current VNN-COMP dashboard from published breadth metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vnncomp_current_surface import (
    add_current_surface_args,
    load_current_latest,
)

UTC = timezone.utc

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _history_sort_key(entry: dict[str, Any]) -> tuple[datetime, str]:
    recorded_at = str(entry.get("recorded_at") or "")
    parsed = _parse_iso_timestamp(recorded_at) or datetime.min.replace(tzinfo=UTC)
    return parsed, recorded_at


def _normalize_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "recorded_at": entry.get("recorded_at", ""),
        "total_score": entry.get("total_score"),
        "total_instances": entry.get("total_instances"),
        "overall_solve_rate": entry.get("overall_solve_rate"),
        "categories_attempted": entry.get("categories_attempted"),
        "commit": entry.get("commit", ""),
    }


def _short_commit(value: Any) -> str:
    commit = str(value or "")
    if not commit:
        return "-"
    return commit[:8]


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.1f}"
    return str(value) if value not in (None, "") else "-"


def _format_rate(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}%"
    return "-"


def _category_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int, str]:
    name, stats = item
    solve_rate = float(stats.get("solve_rate") or 0.0)
    score = int(stats.get("score") or 0)
    return (-solve_rate, -score, name)


def _load_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []

    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(_normalize_history_entry(json.loads(line)))
    entries.sort(key=_history_sort_key)
    return entries


def _headline_lines(latest: dict[str, Any]) -> list[str]:
    skipped = latest.get("skipped", {})
    return [
        "# VNN-COMP Breadth Dashboard",
        "",
        "## Headline Summary",
        f"- Benchmark year: {latest.get('benchmark_year', '-')}",
        f"- Solved: {_format_number(latest.get('total_score'))}/{_format_number(latest.get('total_instances'))}",
        f"- Overall solve rate: {_format_rate(latest.get('overall_solve_rate'))}",
        f"- Categories attempted: {_format_number(latest.get('categories_attempted'))}",
        f"- Skipped categories: {len(skipped)}",
        f"- Commit: {_short_commit(latest.get('commit'))}",
        f"- Recorded at: {latest.get('recorded_at', '-')}",
        "",
    ]


def _category_table_lines(categories: dict[str, Any]) -> list[str]:
    lines = [
        "## Current Categories",
        "",
        "| Category | Solved | Total | Solve Rate | Verified | Falsified | Timeout | Unknown | Error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    sorted_categories = sorted(categories.items(), key=_category_sort_key)
    if sorted_categories:
        for name, stats in sorted_categories:
            lines.append(
                "| {name} | {solved} | {total} | {solve_rate} | {verified} | {falsified} | {timeout} | {unknown} | {error} |".format(
                    name=name,
                    solved=_format_number(stats.get("score")),
                    total=_format_number(stats.get("total")),
                    solve_rate=_format_rate(stats.get("solve_rate")),
                    verified=_format_number(stats.get("verified")),
                    falsified=_format_number(stats.get("falsified")),
                    timeout=_format_number(stats.get("timeout")),
                    unknown=_format_number(stats.get("unknown")),
                    error=_format_number(stats.get("error")),
                )
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    lines.append("")
    return lines


def _skipped_table_lines(skipped: dict[str, Any]) -> list[str]:
    lines = [
        "## Skipped Categories",
        "",
        "| Category | Reason |",
        "| --- | --- |",
    ]

    if skipped:
        for name in sorted(skipped):
            lines.append(f"| {name} | {skipped[name]} |")
    else:
        lines.append("| - | - |")

    lines.append("")
    return lines


def _history_table_lines(history: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Recent History",
        "",
        "| Timestamp | Solved / Total | Solve Rate | Categories Attempted | Commit |",
        "| --- | --- | --- | --- | --- |",
    ]

    if history:
        for entry in reversed(history):
            lines.append(
                "| {timestamp} | {solved}/{total} | {solve_rate} | {categories} | {commit} |".format(
                    timestamp=entry.get("recorded_at") or "-",
                    solved=_format_number(entry.get("total_score")),
                    total=_format_number(entry.get("total_instances")),
                    solve_rate=_format_rate(entry.get("overall_solve_rate")),
                    categories=_format_number(entry.get("categories_attempted")),
                    commit=_short_commit(entry.get("commit")),
                )
            )
    else:
        lines.append("| - | - | - | - | - |")

    return lines


def _build_markdown(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    lines = _headline_lines(latest)
    lines.extend(_category_table_lines(latest.get("categories", {})))
    lines.extend(_skipped_table_lines(latest.get("skipped", {})))
    lines.extend(_history_table_lines(history))
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    return add_current_surface_args(
        argparse.ArgumentParser(
            description="Render the current VNN-COMP dashboard from published metrics"
        ),
        metrics_help="Directory with published metrics",
        reports_help="Directory for dashboard outputs",
    )


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    metrics_dir = Path(args.metrics_dir)
    reports_dir = Path(args.reports_dir)
    history_path = metrics_dir / "vnncomp_history.jsonl"
    overrides_path = (
        Path(args.skip_reason_overrides)
        if args.skip_reason_overrides
        else reports_dir / "vnncomp_skip_reason_overrides.json"
    )
    return metrics_dir, reports_dir, history_path, overrides_path


def _write_dashboard_outputs(
    metrics_dir: Path,
    reports_dir: Path,
    latest: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    generated_at = _now_iso()
    dashboard = {
        "generated_at": generated_at,
        "latest": latest,
        "history": history,
    }
    trend = {
        "generated_at": generated_at,
        "entries": history,
    }

    _write_json(reports_dir / "vnncomp_dashboard.json", dashboard)
    _write_text(reports_dir / "vnncomp_dashboard.md", _build_markdown(latest, history))
    _write_json(metrics_dir / "vnncomp_trend.json", trend)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    metrics_dir, reports_dir, history_path, overrides_path = _resolve_paths(args)
    latest_path = metrics_dir / "vnncomp_latest.json"

    if not latest_path.exists():
        logger.info("No published VNN-COMP metrics found at %s; nothing to render.", latest_path)
        return 0

    latest = load_current_latest(metrics_dir=metrics_dir, overrides_path=overrides_path)
    if latest is None:
        logger.info("No published VNN-COMP metrics found at %s; nothing to render.", latest_path)
        return 0
    history = _load_history(history_path)
    _write_dashboard_outputs(metrics_dir, reports_dir, latest, history)

    logger.info(
        "Wrote VNN-COMP dashboard with "
        "%d categories and %d history entries.",
        len(latest.get("categories", {})),
        len(history),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    raise SystemExit(main())
