# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.vnncomp_current_surface as current_surface


REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SCRIPT = REPO_ROOT / "scripts" / "vnncomp_dashboard.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_history(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry) for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_latest_snapshot() -> dict:
    return {
        "benchmark": "vnncomp",
        "report_kind": "breadth",
        "benchmark_year": 2025,
        "run_scope": "full",
        "publication_scope": "canonical",
        "commit": "abcdef1234567890",
        "ny_version": "v0.1.0-test",
        "recorded_at": "2026-03-11T12:00:00Z",
        "report_path": "reports/benchmarks/vnncomp_summary_20260311_120000.json",
        "total_instances": 5,
        "total_score": 4,
        "overall_solve_rate": 80.0,
        "categories_attempted": 2,
        "categories_skipped": ["cctsdb_yolo_2023"],
        "failed": {},
        "categories": {
            "malbeware": {
                "total": 3,
                "verified": 1,
                "falsified": 1,
                "unknown": 0,
                "timeout": 1,
                "error": 0,
                "score": 2,
                "solve_rate": 66.7,
            },
            "sat_relu": {
                "total": 2,
                "verified": 2,
                "falsified": 0,
                "unknown": 0,
                "timeout": 0,
                "error": 0,
                "score": 2,
                "solve_rate": 100.0,
            },
        },
        "skipped": {
            "cctsdb_yolo_2023": "YOLO detection head not supported (ScatterND op)",
        },
    }


def _sample_history() -> list[dict]:
    return [
        {
            "recorded_at": "2026-03-10T10:00:00Z",
            "total_score": 3,
            "total_instances": 5,
            "overall_solve_rate": 60.0,
            "categories_attempted": 2,
            "commit": "11111111deadbeef",
        },
        {
            "recorded_at": "2026-03-11T12:00:00Z",
            "total_score": 4,
            "total_instances": 5,
            "overall_solve_rate": 80.0,
            "categories_attempted": 2,
            "commit": "abcdef1234567890",
        },
    ]


def _run_dashboard(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DASHBOARD_SCRIPT)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _assert_dashboard_outputs(reports_dir: Path, metrics_dir: Path) -> None:
    dashboard_json = reports_dir / "vnncomp_dashboard.json"
    dashboard_md = reports_dir / "vnncomp_dashboard.md"
    trend_json = metrics_dir / "vnncomp_trend.json"
    assert dashboard_json.exists(), "expected vnncomp_dashboard.json to be written"
    assert dashboard_md.exists(), "expected vnncomp_dashboard.md to be written"
    assert trend_json.exists(), "expected vnncomp_trend.json to be written"

    dashboard = json.loads(dashboard_json.read_text(encoding="utf-8"))
    assert dashboard["latest"]["total_score"] == 4, (
        f"expected latest total_score=4, got {dashboard['latest']['total_score']!r}"
    )
    assert [entry["commit"] for entry in dashboard["history"]] == [
        "11111111deadbeef",
        "abcdef1234567890",
    ], f"unexpected history ordering: {dashboard['history']!r}"

    markdown = dashboard_md.read_text(encoding="utf-8")
    assert "- Solved: 4/5" in markdown, f"headline missing solved summary: {markdown}"
    assert "- Overall solve rate: 80.0%" in markdown, (
        f"headline missing solve rate: {markdown}"
    )
    sat_row = markdown.index("| sat_relu | 2 | 2 | 100.0% | 2 | 0 | 0 | 0 | 0 |")
    mal_row = markdown.index("| malbeware | 2 | 3 | 66.7% | 1 | 1 | 1 | 0 | 0 |")
    assert sat_row < mal_row, "category rows should sort by solve rate descending"
    assert "YOLO detection head not supported (ScatterND op)" in markdown, (
        f"skipped reason missing from markdown: {markdown}"
    )
    assert "| 2026-03-11T12:00:00Z | 4/5 | 80.0% | 2 | abcdef12 |" in markdown, (
        f"recent history row missing from markdown: {markdown}"
    )


def test_vnncomp_dashboard_generation_from_published_metrics(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics" / "benchmarks"
    reports_dir = tmp_path / "reports" / "benchmarks"

    _write_json(metrics_dir / "vnncomp_latest.json", _sample_latest_snapshot())
    _write_history(metrics_dir / "vnncomp_history.jsonl", _sample_history())

    result = _run_dashboard(tmp_path)

    assert result.returncode == 0, f"dashboard failed: {result.stderr}"
    _assert_dashboard_outputs(reports_dir, metrics_dir)


def test_vnncomp_dashboard_missing_latest_is_noop(tmp_path: Path) -> None:
    result = _run_dashboard(tmp_path)

    assert result.returncode == 0, f"dashboard should no-op cleanly: {result.stderr}"
    assert "No published VNN-COMP metrics found" in result.stdout, (
        f"expected no-data message, got: {result.stdout!r}"
    )
    assert not (tmp_path / "reports" / "benchmarks" / "vnncomp_dashboard.json").exists(), (
        "dashboard JSON should not be created when latest metrics are absent"
    )
    assert not (tmp_path / "reports" / "benchmarks" / "vnncomp_dashboard.md").exists(), (
        "dashboard markdown should not be created when latest metrics are absent"
    )
    assert not (tmp_path / "metrics" / "benchmarks" / "vnncomp_trend.json").exists(), (
        "trend JSON should not be created when latest metrics are absent"
    )


def test_vnncomp_dashboard_trend_matches_history_input(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    _write_json(
        metrics_dir / "vnncomp_latest.json",
        {
            "benchmark_year": 2025,
            "total_instances": 10,
            "total_score": 6,
            "overall_solve_rate": 60.0,
            "categories_attempted": 3,
            "commit": "feedfacecafebeef",
            "recorded_at": "2026-03-11T15:00:00Z",
            "categories": {},
            "skipped": {},
        },
    )
    history = [
        {
            "recorded_at": "2026-03-10T15:00:00Z",
            "total_score": 5,
            "total_instances": 10,
            "overall_solve_rate": 50.0,
            "categories_attempted": 3,
            "commit": "11111111aaaaaaa1",
            "extra": "ignored",
        },
        {
            "recorded_at": "2026-03-11T15:00:00Z",
            "total_score": 6,
            "total_instances": 10,
            "overall_solve_rate": 60.0,
            "categories_attempted": 3,
            "commit": "feedfacecafebeef",
        },
    ]
    _write_history(metrics_dir / "vnncomp_history.jsonl", history)

    result = _run_dashboard(tmp_path)

    assert result.returncode == 0, f"dashboard failed unexpectedly: {result.stderr}"
    trend = json.loads(
        (metrics_dir / "vnncomp_trend.json").read_text(encoding="utf-8")
    )
    assert trend["entries"] == [
        {
            "recorded_at": "2026-03-10T15:00:00Z",
            "total_score": 5,
            "total_instances": 10,
            "overall_solve_rate": 50.0,
            "categories_attempted": 3,
            "commit": "11111111aaaaaaa1",
        },
        {
            "recorded_at": "2026-03-11T15:00:00Z",
            "total_score": 6,
            "total_instances": 10,
            "overall_solve_rate": 60.0,
            "categories_attempted": 3,
            "commit": "feedfacecafebeef",
        },
    ], f"trend entries should mirror normalized history rows, got: {trend['entries']!r}"


def test_vnncomp_dashboard_filters_pseudo_skipped_categories(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics" / "benchmarks"
    reports_dir = tmp_path / "reports" / "benchmarks"
    latest = _sample_latest_snapshot()
    latest["categories_skipped"] = ["cctsdb_yolo_2023", "test"]
    latest["skipped"]["test"] = "test category - not a real benchmark"

    _write_json(metrics_dir / "vnncomp_latest.json", latest)
    _write_history(metrics_dir / "vnncomp_history.jsonl", _sample_history())

    result = _run_dashboard(tmp_path)

    assert result.returncode == 0, f"dashboard failed unexpectedly: {result.stderr}"
    dashboard = json.loads((reports_dir / "vnncomp_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["latest"]["categories_skipped"] == ["cctsdb_yolo_2023"], (
        f"pseudo skipped category should be removed from categories_skipped: {dashboard}"
    )
    assert "test" not in dashboard["latest"]["skipped"], (
        f"pseudo skipped category should be removed from skipped map: {dashboard}"
    )
    markdown = (reports_dir / "vnncomp_dashboard.md").read_text(encoding="utf-8")
    assert "- Skipped categories: 1" in markdown, (
        f"dashboard headline should count only real skipped categories: {markdown}"
    )
    assert "| test |" not in markdown, (
        f"dashboard markdown should not render pseudo benchmark rows: {markdown}"
    )


def test_normalize_current_latest_filters_pseudo_skipped_category() -> None:
    latest = _sample_latest_snapshot()
    latest["categories_skipped"] = ["cctsdb_yolo_2023", "test"]
    latest["skipped"]["test"] = "test category - not a real benchmark"

    normalized = current_surface.normalize_current_latest(latest)

    assert normalized["categories_skipped"] == ["cctsdb_yolo_2023"], (
        f"expected pseudo skipped category to be removed from categories_skipped: {normalized}"
    )
    assert "test" not in normalized["skipped"], (
        f"expected pseudo skipped category to be removed from skipped map: {normalized}"
    )


def test_load_current_dashboard_normalizes_pseudo_categories_before_overrides(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "reports" / "benchmarks"
    latest = _sample_latest_snapshot()
    latest["categories_skipped"] = ["cctsdb_yolo_2023", "test"]
    latest["skipped"]["test"] = "test category - not a real benchmark"
    _write_json(
        reports_dir / "vnncomp_dashboard.json",
        {
            "generated_at": "2026-03-11T12:30:00Z",
            "latest": latest,
        },
    )
    overrides_path = reports_dir / "vnncomp_skip_reason_overrides.json"
    override_reason = "override reason remains the active blocker"
    _write_json(overrides_path, {"cctsdb_yolo_2023": override_reason})

    dashboard = current_surface.load_current_dashboard(
        reports_dir=reports_dir,
        metrics_dir=tmp_path / "metrics" / "benchmarks",
        overrides_path=overrides_path,
    )

    assert dashboard is not None, "existing dashboard payload should load"
    assert dashboard["latest"]["categories_skipped"] == ["cctsdb_yolo_2023"], (
        "dashboard reuse path should drop pseudo categories before returning latest"
    )
    assert dashboard["latest"]["skipped"] == {
        "cctsdb_yolo_2023": override_reason,
    }, "dashboard reuse path should keep real categories and apply overrides"
