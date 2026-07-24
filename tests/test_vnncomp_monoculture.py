#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Tests for scripts/vnncomp_monoculture.py monoculture tracker renderer.

Design: designs/2026-03-11-issue-2569-vnncomp-monoculture-tracker.md
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path("scripts/vnncomp_monoculture.py")

# Minimal dashboard fixture with mixed category statuses.
FIXTURE_DASHBOARD = {
    "generated_at": "2026-03-13T08:00:00Z",
    "latest": {
        "benchmark": "vnncomp",
        "commit": "abc1234",
        "recorded_at": "2026-03-13T07:00:00Z",
        "total_instances": 500,
        "total_score": 300,
        "overall_solve_rate": 60.0,
        "categories_attempted": 4,
        "categories": {
            "sat_relu": {
                "total": 100,
                "verified": 50,
                "falsified": 50,
                "unknown": 0,
                "timeout": 0,
                "error": 0,
                "score": 100,
                "solve_rate": 100.0,
            },
            "malbeware": {
                "total": 150,
                "verified": 125,
                "falsified": 4,
                "unknown": 21,
                "timeout": 0,
                "error": 0,
                "score": 129,
                "solve_rate": 86.0,
            },
            "relusplitter": {
                "total": 220,
                "verified": 74,
                "falsified": 8,
                "unknown": 66,
                "timeout": 72,
                "error": 0,
                "score": 82,
                "solve_rate": 37.3,
            },
            "lsnc_relu": {
                "total": 30,
                "verified": 0,
                "falsified": 0,
                "unknown": 30,
                "timeout": 0,
                "error": 0,
                "score": 0,
                "solve_rate": 0.0,
            },
        },
        "skipped": {
            "cctsdb_yolo_2023": "YOLO detection head not supported (ScatterND op)",
            "traffic_signs_recognition_2023": "CPU CROWN backward timeout on 43-class disjunction",
        },
    },
}


def _run_script(dashboard_dir: Path) -> subprocess.CompletedProcess:
    """Run the monoculture script in a temporary directory."""
    return subprocess.run(
        [sys.executable, str(SCRIPT.resolve())],
        capture_output=True,
        text=True,
        cwd=str(dashboard_dir),
    )


class TestMonocultureGeneration:
    """Test case 1: Classification generation from dashboard."""

    def test_both_outputs_written(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)
        (reports_dir / "vnncomp_dashboard.json").write_text(
            json.dumps(FIXTURE_DASHBOARD), encoding="utf-8"
        )

        result = _run_script(tmp_path)
        assert result.returncode == 0, f"script failed: {result.stderr}"

        assert (reports_dir / "vnncomp_monoculture.json").exists(), "JSON output missing"
        assert (reports_dir / "vnncomp_monoculture.md").exists(), "markdown output missing"

    def test_json_summary_counts(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)
        (reports_dir / "vnncomp_dashboard.json").write_text(
            json.dumps(FIXTURE_DASHBOARD), encoding="utf-8"
        )

        _run_script(tmp_path)
        tracker = json.loads(
            (reports_dir / "vnncomp_monoculture.json").read_text(encoding="utf-8")
        )

        s = tracker["summary"]
        # sat_relu (100%) and malbeware (86%) are competitive
        assert s["competitive_count"] == 2, f"expected 2 competitive, got {s['competitive_count']}"
        # relusplitter (37.3%) is partial
        assert s["partial_count"] == 1, f"expected 1 partial, got {s['partial_count']}"
        # lsnc_relu (0 score) + cctsdb_yolo_2023 (skipped) + traffic_signs_recognition_2023 (skipped) are blocked
        assert s["blocked_count"] == 3, f"expected 3 blocked, got {s['blocked_count']}"
        assert s["categories_tracked"] == 6, f"expected 6 tracked, got {s['categories_tracked']}"
        assert s["categories_with_score"] == 3, f"expected 3 with score, got {s['categories_with_score']}"
        assert s["monoculture_status"] == "cleared", f"expected cleared, got {s['monoculture_status']}"
        assert s["non_acas_categories_with_score"] == 3, f"expected 3 non-ACAS, got {s['non_acas_categories_with_score']}"

    def test_markdown_headline(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)
        (reports_dir / "vnncomp_dashboard.json").write_text(
            json.dumps(FIXTURE_DASHBOARD), encoding="utf-8"
        )

        _run_script(tmp_path)
        md = (reports_dir / "vnncomp_monoculture.md").read_text(encoding="utf-8")

        assert "Monoculture status:** cleared" in md, "monoculture status missing from markdown"
        assert "Non-ACAS categories with score:** 3" in md, "non-ACAS count missing from markdown"
        assert "Non-ACAS solved instances:**" in md, "non-ACAS solved line missing from markdown"

    def test_category_sort_order(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)
        (reports_dir / "vnncomp_dashboard.json").write_text(
            json.dumps(FIXTURE_DASHBOARD), encoding="utf-8"
        )

        _run_script(tmp_path)
        tracker = json.loads(
            (reports_dir / "vnncomp_monoculture.json").read_text(encoding="utf-8")
        )

        statuses = [r["status"] for r in tracker["categories"]]
        # competitive first, then partial, then blocked
        competitive_indices = [i for i, s in enumerate(statuses) if s == "competitive"]
        partial_indices = [i for i, s in enumerate(statuses) if s == "partial"]
        blocked_indices = [i for i, s in enumerate(statuses) if s == "blocked"]

        if competitive_indices and partial_indices:
            assert max(competitive_indices) < min(partial_indices), \
                f"competitive after partial: {statuses}"
        if partial_indices and blocked_indices:
            assert max(partial_indices) < min(blocked_indices), \
                f"partial after blocked: {statuses}"


class TestMissingInput:
    """Test case 2: Missing dashboard is a no-op."""

    def test_missing_dashboard_exits_zero(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)
        # No dashboard file

        result = _run_script(tmp_path)
        assert result.returncode == 0, f"expected exit 0, got {result.returncode}"
        assert "Nothing to do" in result.stdout, f"expected 'Nothing to do' in stdout: {result.stdout}"

    def test_missing_dashboard_no_outputs(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)

        _run_script(tmp_path)
        assert not (reports_dir / "vnncomp_monoculture.json").exists(), "JSON created without dashboard"
        assert not (reports_dir / "vnncomp_monoculture.md").exists(), "markdown created without dashboard"


class TestSkippedVsAttemptedZero:
    """Test case 3: Skipped vs attempted-zero-score behavior."""

    def test_both_classify_as_blocked(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)

        dashboard = {
            "generated_at": "2026-03-13T08:00:00Z",
            "latest": {
                "commit": "test123",
                "recorded_at": "2026-03-13T07:00:00Z",
                "total_instances": 50,
                "total_score": 0,
                "overall_solve_rate": 0.0,
                "categories_attempted": 1,
                "categories": {
                    "zero_score_cat": {
                        "total": 30,
                        "verified": 0,
                        "falsified": 0,
                        "unknown": 30,
                        "timeout": 0,
                        "error": 0,
                        "score": 0,
                        "solve_rate": 0.0,
                    },
                },
                "skipped": {
                    "skipped_cat": "no model support yet",
                },
            },
        }

        (reports_dir / "vnncomp_dashboard.json").write_text(
            json.dumps(dashboard), encoding="utf-8"
        )

        _run_script(tmp_path)
        tracker = json.loads(
            (reports_dir / "vnncomp_monoculture.json").read_text(encoding="utf-8")
        )

        by_name = {r["category"]: r for r in tracker["categories"]}

        assert by_name["zero_score_cat"]["status"] == "blocked", \
            f"zero-score category should be blocked, got {by_name['zero_score_cat']['status']}"
        assert by_name["skipped_cat"]["status"] == "blocked", \
            f"skipped category should be blocked, got {by_name['skipped_cat']['status']}"

    def test_only_skipped_has_reason(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports" / "benchmarks"
        reports_dir.mkdir(parents=True)

        dashboard = {
            "generated_at": "2026-03-13T08:00:00Z",
            "latest": {
                "commit": "test123",
                "recorded_at": "2026-03-13T07:00:00Z",
                "total_instances": 50,
                "total_score": 0,
                "overall_solve_rate": 0.0,
                "categories_attempted": 1,
                "categories": {
                    "zero_score_cat": {
                        "total": 30,
                        "verified": 0,
                        "falsified": 0,
                        "unknown": 30,
                        "timeout": 0,
                        "error": 0,
                        "score": 0,
                        "solve_rate": 0.0,
                    },
                },
                "skipped": {
                    "skipped_cat": "no model support yet",
                },
            },
        }

        (reports_dir / "vnncomp_dashboard.json").write_text(
            json.dumps(dashboard), encoding="utf-8"
        )

        _run_script(tmp_path)
        tracker = json.loads(
            (reports_dir / "vnncomp_monoculture.json").read_text(encoding="utf-8")
        )

        by_name = {r["category"]: r for r in tracker["categories"]}

        assert by_name["zero_score_cat"]["skip_reason"] is None, \
            f"attempted category should have no skip_reason, got {by_name['zero_score_cat']['skip_reason']}"
        assert by_name["skipped_cat"]["skip_reason"] == "no model support yet", \
            f"skipped category reason mismatch: {by_name['skipped_cat']['skip_reason']}"


def test_monoculture_filters_pseudo_skipped_categories_from_dashboard(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports" / "benchmarks"
    reports_dir.mkdir(parents=True)

    dashboard = json.loads(json.dumps(FIXTURE_DASHBOARD))
    dashboard["latest"]["categories_skipped"] = [
        "cctsdb_yolo_2023",
        "traffic_signs_recognition_2023",
        "test",
    ]
    dashboard["latest"]["skipped"]["test"] = "test category - not a real benchmark"
    (reports_dir / "vnncomp_dashboard.json").write_text(json.dumps(dashboard), encoding="utf-8")

    result = _run_script(tmp_path)

    assert result.returncode == 0, f"script failed unexpectedly: {result.stderr}"
    tracker = json.loads(
        (reports_dir / "vnncomp_monoculture.json").read_text(encoding="utf-8")
    )
    assert tracker["summary"]["categories_skipped"] == 2, (
        f"pseudo skipped category should not count toward skipped summary: {tracker}"
    )
    assert tracker["summary"]["categories_tracked"] == 6, (
        f"pseudo skipped category should not create an extra tracked row: {tracker}"
    )
    categories = {row["category"] for row in tracker["categories"]}
    assert "test" not in categories, (
        f"pseudo benchmark row should be absent from monoculture output: {tracker}"
    )
