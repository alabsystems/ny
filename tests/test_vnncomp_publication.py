# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Fixture-based tests for VNN-COMP breadth metrics publication pipeline.

Tests aggregator publication, pulse merge, and publication guards per
designs/2026-03-11-issue-2569-vnncomp-breadth-metrics-publication.md
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATOR = REPO_ROOT / "scripts" / "aggregate_vnncomp_results.py"
MERGE_SCRIPT = REPO_ROOT / "scripts" / "merge_benchmark_metrics.py"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh_vnncomp_current_status.py"
BACKEND_V1_FIELDNAMES = [
    "schema_version",
    "lane",
    "subject_kind",
    "subject_id",
    "comparison_key",
    "category",
    "workload",
    "model_path",
    "property_path",
    "preset_path",
    "backend",
    "timeout_seconds",
    "status",
    "actual_method",
    "wall_seconds",
    "domains_explored",
    "output_width_sum",
    "profile_artifact_path",
    "notes",
]


def _write_category_csv(directory: Path, category: str, rows: list[dict]) -> Path:
    """Write a minimal benchmark CSV fixture for a category."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{category}_20260311_120000.csv"
    header = "model,property,timeout,result,elapsed,domains\n"
    lines = []
    for row in rows:
        lines.append(
            f"{row.get('model', 'model.onnx')},"
            f"{row.get('property', 'prop.vnnlib')},"
            f"{row.get('timeout', '2')},"
            f"{row['result']},"
            f"{row.get('elapsed', '1.0')},"
            f"{row.get('domains', '0')}"
        )
    csv_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _write_backend_v1_category_csv(
    directory: Path,
    filename_category: str,
    row_category: str,
    rows: list[dict],
    *,
    lane: str = "vnncomp_single_backend",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{filename_category}_20260311_120000.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BACKEND_V1_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_backend_v1_rows(row_category, rows, lane=lane))
    return csv_path


def _backend_v1_rows(
    row_category: str,
    rows: list[dict],
    *,
    lane: str = "vnncomp_single_backend",
) -> list[dict[str, str]]:
    encoded_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        encoded_rows.append(
            {
                "schema_version": "backend_benchmark_row_v1",
                "lane": lane,
                "subject_kind": "vnncomp_instance",
                "subject_id": row.get(
                    "subject_id",
                    f"{row_category}::model_{index}.onnx::prop_{index}.vnnlib",
                ),
                "comparison_key": row.get(
                    "comparison_key",
                    f"{row_category}::model_{index}.onnx::prop_{index}.vnnlib",
                ),
                "category": row_category,
                "workload": "",
                "model_path": row.get(
                    "model_path",
                    f"benchmarks/vnncomp2025/benchmarks/{row_category}/onnx/model_{index}.onnx",
                ),
                "property_path": row.get(
                    "property_path",
                    f"benchmarks/vnncomp2025/benchmarks/{row_category}/vnnlib/prop_{index}.vnnlib",
                ),
                "preset_path": row.get("preset_path", f"configs/vnncomp25/{row_category}.yaml"),
                "backend": row.get("backend", "cpu"),
                "timeout_seconds": row.get("timeout_seconds", "2"),
                "status": row["status"],
                "actual_method": row.get("actual_method", ""),
                "wall_seconds": row.get("wall_seconds", "1.0"),
                "domains_explored": row.get("domains_explored", "0"),
                "output_width_sum": row.get("output_width_sum", ""),
                "profile_artifact_path": row.get("profile_artifact_path", ""),
                "notes": row.get("notes", ""),
            }
        )
    return encoded_rows


def _append_backend_v1_category_rows(
    csv_path: Path,
    row_category: str,
    rows: list[dict],
    *,
    lane: str = "vnncomp_single_backend",
) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BACKEND_V1_FIELDNAMES)
        writer.writerows(_backend_v1_rows(row_category, rows, lane=lane))


def _run_aggregator(
    tmp_path: Path,
    csv_files: list[Path],
    *,
    publish: bool = False,
    run_scope: str = "full",
    year: int = 2025,
    tracker_year: int = 2025,
    failed: dict | None = None,
) -> tuple[Path, dict]:
    """Run aggregate_vnncomp_results.py and return (output_path, parsed_summary)."""
    output = tmp_path / "reports" / "benchmarks" / "vnncomp_summary_test.json"
    cmd = [
        sys.executable, str(AGGREGATOR),
        "--output", str(output),
        "--year", str(year),
        "--commit", "abc123",
        "--version", "v0.1.0-test",
        "--wall-time", "10.0",
        "--skipped", json.dumps({"cctsdb_yolo_2023": "ScatterND op"}),
        "--failed", json.dumps(failed or {}),
        "--run-scope", run_scope,
        "--tracker-year", str(tracker_year),
    ]
    if publish:
        cmd.append("--publish-metrics")
    cmd.extend(str(f) for f in csv_files)

    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert output.exists(), f"Summary not written. stderr: {result.stderr}"
    summary = json.loads(output.read_text(encoding="utf-8"))
    return output, summary


def test_aggregator_help_imports_on_supported_python() -> None:
    """The publication CLI must import on the repository's Python 3.9 floor."""
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: aggregate_vnncomp_results.py" in result.stdout


def _run_refresh(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )


def _seed_refresh_chain_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics"
    bench_dir = metrics_dir / "benchmarks"

    csv1 = _write_category_csv(report_dir, "sat_relu", [
        {"result": "verified"},
    ])
    _run_aggregator(tmp_path, [csv1], publish=True)

    overrides = {
        "cctsdb_yolo_2023": "current override: operator support remains the active blocker"
    }
    (report_dir / "vnncomp_skip_reason_overrides.json").write_text(
        json.dumps(overrides, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "latest.json").write_text(
        json.dumps({"existing": "preserved"}) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "latest_partial.json").write_text(
        json.dumps({"partial": "preserved"}) + "\n",
        encoding="utf-8",
    )
    return report_dir, metrics_dir, bench_dir, overrides


# --- Test 1: Aggregator publication (full tracker-year, no failures) ---

def test_aggregator_publishes_canonical_on_full_tracker_year(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "sat_relu", [
        {"result": "verified", "elapsed": "0.5"},
        {"result": "verified", "elapsed": "0.3"},
    ])
    csv2 = _write_category_csv(report_dir, "malbeware", [
        {"result": "verified", "elapsed": "1.0"},
        {"result": "violated", "elapsed": "0.8"},
        {"result": "timeout", "elapsed": "10.0"},
    ])

    _, summary = _run_aggregator(tmp_path, [csv1, csv2], publish=True)

    assert summary["run_scope"] == "full", f"Expected run_scope=full, got {summary['run_scope']!r}"
    assert summary["publication_scope"] == "canonical", (
        f"Expected publication_scope=canonical, got {summary['publication_scope']!r}"
    )
    assert summary["benchmark_year"] == 2025, (
        f"Expected benchmark_year=2025, got {summary['benchmark_year']!r}"
    )
    assert summary["total_instances"] == 5, (
        f"Expected 5 total instances (2+3), got {summary['total_instances']}"
    )
    assert summary["total_score"] == 4, (
        f"Expected score=4 (2 verified + 1 verified + 1 violated), got {summary['total_score']}"
    )
    assert summary["categories_attempted"] == 2, (
        f"Expected 2 categories, got {summary['categories_attempted']}"
    )
    assert "sat_relu" in summary["categories"], "sat_relu missing from categories"
    assert "malbeware" in summary["categories"], "malbeware missing from categories"

    # Canonical files should exist
    assert (report_dir / "vnncomp_latest.json").exists(), (
        "reports/benchmarks/vnncomp_latest.json not created"
    )
    assert (metrics_dir / "vnncomp_latest.json").exists(), (
        "metrics/benchmarks/vnncomp_latest.json not created"
    )
    assert (metrics_dir / "vnncomp_history.jsonl").exists(), (
        "metrics/benchmarks/vnncomp_history.jsonl not created"
    )

    # History should have exactly one line
    history_lines = (
        (metrics_dir / "vnncomp_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    )
    assert len(history_lines) == 1, f"Expected 1 history entry, got {len(history_lines)}"
    history_entry = json.loads(history_lines[0])
    assert history_entry["total_score"] == 4, (
        f"Expected history total_score=4, got {history_entry['total_score']}"
    )


def test_aggregator_accepts_backend_benchmark_row_v1_single_backend_rows(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"

    csv1 = _write_backend_v1_category_csv(
        report_dir,
        "synthetic_compare_backends",
        "sat_relu",
        [
            {"status": "verified", "wall_seconds": "0.5"},
            {"status": "violated", "wall_seconds": "0.7"},
            {"status": "timeout", "wall_seconds": "2.0"},
        ],
    )

    _, summary = _run_aggregator(tmp_path, [csv1], publish=False)

    assert "sat_relu" in summary["categories"], summary
    assert "synthetic_compare_backends" not in summary["categories"], summary
    sat_relu = summary["categories"]["sat_relu"]
    assert sat_relu["total"] == 3, sat_relu
    assert sat_relu["verified"] == 1, sat_relu
    assert sat_relu["falsified"] == 1, sat_relu
    assert sat_relu["timeout"] == 1, sat_relu


def test_aggregator_skips_non_breadth_backend_benchmark_rows(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"

    csv1 = _write_backend_v1_category_csv(
        report_dir,
        "synthetic_compare_backends",
        "sat_relu",
        [
            {"status": "verified", "backend": "cpu", "wall_seconds": "0.5"},
            {"status": "verified", "backend": "wgpu", "wall_seconds": "0.2"},
        ],
        lane="vnncomp_compare_backends",
    )

    _, summary = _run_aggregator(tmp_path, [csv1], publish=False)

    assert summary["categories"] == {}, summary
    assert summary["categories_attempted"] == 0, summary
    assert summary["total_instances"] == 0, summary
    assert summary["total_score"] == 0, summary


def test_aggregator_prefers_single_backend_category_for_mixed_lane_csv(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"

    csv1 = _write_backend_v1_category_csv(
        report_dir,
        "synthetic_compare_backends",
        "sat_relu",
        [
            {"status": "verified", "wall_seconds": "0.5"},
        ],
    )
    _append_backend_v1_category_rows(
        csv1,
        "malbeware",
        [
            {
                "status": "verified",
                "backend": "wgpu",
                "wall_seconds": "0.2",
                "subject_id": "malbeware::model.onnx::prop.vnnlib",
                "comparison_key": "malbeware::model.onnx::prop.vnnlib",
            },
        ],
        lane="vnncomp_compare_backends",
    )

    _, summary = _run_aggregator(tmp_path, [csv1], publish=False)

    assert summary["categories_attempted"] == 1, summary
    assert "sat_relu" in summary["categories"], summary
    assert "synthetic_compare_backends" not in summary["categories"], summary
    assert "malbeware" not in summary["categories"], summary


def test_aggregator_accepts_backend_v1_rows_with_quoted_fields(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"

    csv1 = _write_backend_v1_category_csv(
        report_dir,
        "quoted_fields",
        "sat_relu",
        [
            {
                "status": "verified",
                "wall_seconds": "0.5",
                "model_path": "benchmarks/vnncomp2025/benchmarks/sat_relu/onnx/model,0.onnx",
                "property_path": "benchmarks/vnncomp2025/benchmarks/sat_relu/vnnlib/prop,0.vnnlib",
            },
        ],
    )

    _, summary = _run_aggregator(tmp_path, [csv1], publish=False)

    sat_relu = summary["categories"]["sat_relu"]
    assert sat_relu["total"] == 1, sat_relu
    assert sat_relu["verified"] == 1, sat_relu
    assert sat_relu["wall_time_seconds"] == 0.5, sat_relu


# --- Test 2: Pulse merge ---

def test_pulse_merge_adds_vnncomp_benchmarks(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    bench_dir = metrics_dir / "benchmarks"
    bench_dir.mkdir(parents=True)

    # Existing pulse data
    pulse = {"some_key": "preserved", "benchmarks": {"should": "stay"}}
    (metrics_dir / "latest.json").write_text(json.dumps(pulse) + "\n", encoding="utf-8")
    (metrics_dir / "latest_partial.json").write_text(json.dumps({"partial": True}) + "\n", encoding="utf-8")

    # Existing ACAS-Xu benchmark
    acas = {"benchmark": "acasxu", "verified": 45}
    (bench_dir / "latest.json").write_text(json.dumps(acas) + "\n", encoding="utf-8")

    # VNN-COMP latest
    vnncomp = {"benchmark": "vnncomp", "total_score": 100}
    (bench_dir / "vnncomp_latest.json").write_text(
        json.dumps(vnncomp) + "\n", encoding="utf-8"
    )

    subprocess.run(
        [sys.executable, str(MERGE_SCRIPT)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    merged = json.loads((metrics_dir / "latest.json").read_text(encoding="utf-8"))
    assert merged["some_key"] == "preserved", "Unrelated pulse keys should be preserved"
    assert merged["benchmarks"] == acas, "ACAS-Xu benchmarks should be preserved exactly"
    assert merged["vnncomp_benchmarks"] == vnncomp, (
        "VNN-COMP should be under vnncomp_benchmarks"
    )

    merged_partial = json.loads((metrics_dir / "latest_partial.json").read_text(encoding="utf-8"))
    assert merged_partial["partial"] is True, "latest_partial.json should preserve unrelated keys"
    assert merged_partial["benchmarks"] == acas, "ACAS-Xu benchmarks should also flow into latest_partial.json"
    assert merged_partial["vnncomp_benchmarks"] == vnncomp, (
        "VNN-COMP should also be merged into latest_partial.json"
    )


def test_pulse_merge_without_vnncomp_leaves_acas_only(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    bench_dir = metrics_dir / "benchmarks"
    bench_dir.mkdir(parents=True)

    pulse = {"existing": True}
    (metrics_dir / "latest.json").write_text(json.dumps(pulse) + "\n", encoding="utf-8")

    acas = {"benchmark": "acasxu"}
    (bench_dir / "latest.json").write_text(json.dumps(acas) + "\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(MERGE_SCRIPT)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    merged = json.loads((metrics_dir / "latest.json").read_text(encoding="utf-8"))
    assert merged["existing"] is True, f"Expected existing=True, got {merged.get('existing')!r}"
    assert merged["benchmarks"] == acas, (
        f"Expected ACAS data preserved, got {merged.get('benchmarks')!r}"
    )
    assert "vnncomp_benchmarks" not in merged, (
        "vnncomp_benchmarks should not be present without vnncomp_latest.json"
    )


# --- Test 3: No-publication mode ---

def test_no_publish_flag_produces_no_metrics_side_effects(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "sat_relu", [
        {"result": "verified"},
    ])

    _, summary = _run_aggregator(tmp_path, [csv1], publish=False)

    assert summary["run_scope"] == "full", (
        f"Expected run_scope=full, got {summary['run_scope']!r}"
    )
    # No canonical files should be created when --publish-metrics is not passed
    assert not (report_dir / "vnncomp_latest.json").exists(), (
        "vnncomp_latest.json should not exist without --publish-metrics"
    )
    assert not (metrics_dir / "vnncomp_latest.json").exists(), (
        "metrics vnncomp_latest.json should not exist without --publish-metrics"
    )
    assert not (metrics_dir / "vnncomp_history.jsonl").exists(), (
        "vnncomp_history.jsonl should not exist without --publish-metrics"
    )


# --- Test 4: Partial-run guard ---

def test_partial_run_does_not_publish_even_with_flag(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "sat_relu", [
        {"result": "verified"},
    ])

    _, summary = _run_aggregator(tmp_path, [csv1], publish=True, run_scope="partial")

    assert summary["run_scope"] == "partial", (
        f"Expected run_scope=partial, got {summary['run_scope']!r}"
    )
    assert summary["publication_scope"] == "timestamp_only", (
        f"Expected publication_scope=timestamp_only for partial run, got {summary['publication_scope']!r}"
    )
    assert not (report_dir / "vnncomp_latest.json").exists(), (
        "vnncomp_latest.json should not be created for partial runs"
    )
    assert not (metrics_dir / "vnncomp_latest.json").exists(), (
        "metrics vnncomp_latest.json should not be created for partial runs"
    )
    assert not (metrics_dir / "vnncomp_history.jsonl").exists(), (
        "vnncomp_history.jsonl should not be created for partial runs"
    )


# --- Test 5: Non-tracker-year full-run guard ---

def test_non_tracker_year_full_run_does_not_publish(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "acasxu_2023", [
        {"result": "verified"},
    ])

    _, summary = _run_aggregator(
        tmp_path, [csv1], publish=True,
        year=2024, tracker_year=2025,
    )

    assert summary["benchmark_year"] == 2024, (
        f"Expected benchmark_year=2024, got {summary['benchmark_year']!r}"
    )
    assert summary["publication_scope"] == "timestamp_only", (
        f"Expected timestamp_only for non-tracker year, got {summary['publication_scope']!r}"
    )
    assert not (report_dir / "vnncomp_latest.json").exists(), (
        "vnncomp_latest.json should not be created for non-tracker year"
    )
    assert not (metrics_dir / "vnncomp_latest.json").exists(), (
        "metrics vnncomp_latest.json should not be created for non-tracker year"
    )


# --- Test 6: Failed categories prevent publication ---

def test_failed_categories_prevent_canonical_publication(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "sat_relu", [
        {"result": "verified"},
    ])

    _, summary = _run_aggregator(
        tmp_path, [csv1], publish=True,
        failed={"metaroom_2023": {"exit_code": 1, "reason": "validation disagreement"}},
    )

    assert summary["publication_scope"] == "timestamp_only", (
        f"Expected timestamp_only with failed categories, got {summary['publication_scope']!r}"
    )
    assert "metaroom_2023" in summary["failed"], (
        f"Expected metaroom_2023 in failed map, got {summary['failed']!r}"
    )
    assert not (report_dir / "vnncomp_latest.json").exists(), (
        "vnncomp_latest.json should not be created when categories failed"
    )
    assert not (metrics_dir / "vnncomp_latest.json").exists(), (
        "metrics vnncomp_latest.json should not be created when categories failed"
    )


# --- Test 7: History appends on repeated canonical runs ---

def test_history_appends_on_repeated_canonical_runs(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"
    metrics_dir = tmp_path / "metrics" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "sat_relu", [
        {"result": "verified"},
    ])

    # First canonical run
    _run_aggregator(tmp_path, [csv1], publish=True)

    # Second canonical run
    _run_aggregator(tmp_path, [csv1], publish=True)

    history_lines = (
        (metrics_dir / "vnncomp_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    )
    assert len(history_lines) == 2, f"Expected 2 history entries, got {len(history_lines)}"


def test_aggregator_normalizes_current_benchmark_result_aliases(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks"

    csv1 = _write_category_csv(report_dir, "cersyve", [
        {"result": "falsified", "elapsed": "0.2"},
        {"result": "timeout_ext", "elapsed": "45.0"},
        {"result": "verified", "elapsed": "1.0"},
    ])

    _, summary = _run_aggregator(tmp_path, [csv1], publish=False)
    cersyve = summary["categories"]["cersyve"]

    assert cersyve["verified"] == 1, (
        f"expected verified alias count preserved, got {cersyve['verified']}"
    )
    assert cersyve["falsified"] == 1, (
        f"expected falsified alias to count as violated/falsified, got {cersyve['falsified']}"
    )
    assert cersyve["timeout"] == 1, (
        f"expected timeout_ext alias to count as timeout, got {cersyve['timeout']}"
    )
    assert cersyve["error"] == 0, (
        f"expected alias statuses to avoid falling into error bucket, got {cersyve['error']}"
    )
    assert cersyve["score"] == 2, (
        f"expected verified+falsified score of 2, got {cersyve['score']}"
    )


def test_refresh_chain_updates_dashboard_monoculture_and_pulse(tmp_path: Path) -> None:
    report_dir, metrics_dir, bench_dir, overrides = _seed_refresh_chain_fixture(tmp_path)
    _run_refresh(tmp_path)

    dashboard_json = report_dir / "vnncomp_dashboard.json"
    dashboard_md = report_dir / "vnncomp_dashboard.md"
    monoculture_json = report_dir / "vnncomp_monoculture.json"
    monoculture_md = report_dir / "vnncomp_monoculture.md"
    assert dashboard_json.exists(), "dashboard JSON should be created by refresh chain"
    assert dashboard_md.exists(), "dashboard markdown should be created by refresh chain"
    assert monoculture_json.exists(), "monoculture JSON should be created by refresh chain"
    assert monoculture_md.exists(), "monoculture markdown should be created by refresh chain"

    dashboard = json.loads(dashboard_json.read_text(encoding="utf-8"))
    assert dashboard["latest"]["skipped"]["cctsdb_yolo_2023"] == overrides["cctsdb_yolo_2023"], (
        "dashboard should use skip-reason overrides for the current surface"
    )

    merged_latest = json.loads((metrics_dir / "latest.json").read_text(encoding="utf-8"))
    merged_partial = json.loads((metrics_dir / "latest_partial.json").read_text(encoding="utf-8"))
    assert merged_latest["existing"] == "preserved", "latest.json should preserve unrelated pulse data"
    assert merged_partial["partial"] == "preserved", (
        "latest_partial.json should preserve unrelated pulse data"
    )
    assert merged_latest["vnncomp_benchmarks"]["skipped"]["cctsdb_yolo_2023"] == overrides["cctsdb_yolo_2023"], (
        "pulse latest.json should reflect the current override surface"
    )
    assert merged_partial["vnncomp_benchmarks"]["skipped"]["cctsdb_yolo_2023"] == overrides["cctsdb_yolo_2023"], (
        "pulse latest_partial.json should reflect the current override surface"
    )
    assert merged_latest["vnncomp_benchmarks"]["total_score"] == 1, (
        "refresh chain should preserve the published numeric results"
    )
    assert merged_partial["vnncomp_benchmarks"]["total_score"] == 1, (
        "latest_partial.json should preserve the published numeric results"
    )
    assert (bench_dir / "vnncomp_trend.json").exists(), (
        "dashboard refresh should also write the trend artifact"
    )


def test_refresh_chain_filters_pseudo_categories_from_dashboard_monoculture_and_pulse(
    tmp_path: Path,
) -> None:
    report_dir, metrics_dir, bench_dir, overrides = _seed_refresh_chain_fixture(tmp_path)
    latest_path = bench_dir / "vnncomp_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["categories_skipped"].append("test")
    latest["skipped"]["test"] = "test category - not a real benchmark"
    latest_path.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")

    _run_refresh(tmp_path)

    dashboard = json.loads((report_dir / "vnncomp_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["latest"]["categories_skipped"] == ["cctsdb_yolo_2023"], (
        "refresh chain should scrub pseudo categories before rendering dashboard outputs"
    )
    assert dashboard["latest"]["skipped"] == {
        "cctsdb_yolo_2023": overrides["cctsdb_yolo_2023"],
    }, "refresh chain should preserve real skipped categories and their overrides"

    tracker = json.loads((report_dir / "vnncomp_monoculture.json").read_text(encoding="utf-8"))
    assert tracker["summary"]["categories_skipped"] == 1, (
        "pseudo skipped category should not inflate monoculture summary counts"
    )
    assert "test" not in {row["category"] for row in tracker["categories"]}, (
        "pseudo skipped category should not produce a monoculture row"
    )

    merged_latest = json.loads((metrics_dir / "latest.json").read_text(encoding="utf-8"))
    merged_partial = json.loads((metrics_dir / "latest_partial.json").read_text(encoding="utf-8"))
    assert merged_latest["vnncomp_benchmarks"]["categories_skipped"] == ["cctsdb_yolo_2023"], (
        "pulse latest.json should inherit the normalized current-surface payload"
    )
    assert merged_partial["vnncomp_benchmarks"]["categories_skipped"] == ["cctsdb_yolo_2023"], (
        "pulse latest_partial.json should inherit the normalized current-surface payload"
    )
    assert "test" not in merged_latest["vnncomp_benchmarks"]["skipped"], (
        "pulse latest.json should not re-publish pseudo skipped categories"
    )
    assert "test" not in merged_partial["vnncomp_benchmarks"]["skipped"], (
        "pulse latest_partial.json should not re-publish pseudo skipped categories"
    )


def test_refresh_chain_without_published_metrics_exits_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert "Nothing to refresh" in result.stdout, (
        f"expected no-data message, got {result.stdout!r}"
    )
    assert not (tmp_path / "reports" / "benchmarks" / "vnncomp_dashboard.json").exists(), (
        "dashboard JSON should not be created without published metrics"
    )
    assert not (tmp_path / "reports" / "benchmarks" / "vnncomp_monoculture.json").exists(), (
        "monoculture JSON should not be created without published metrics"
    )
