# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.test_support.gpu_crown_backward_regression_test_utils import (
    run_checker,
    run_checker_raw,
    write_candidate,
    write_policy,
)

# Keep a direct literal path so scoped-test indexing maps checker edits here.
_CHECKER_PATH = "scripts/check_gpu_crown_backward_regression.py"


def write_passing_candidate(candidate: Path) -> None:
    write_candidate_with_metaroom_seconds(candidate, "6.500000")


def write_regressing_candidate(candidate: Path) -> None:
    write_candidate_with_metaroom_seconds(candidate, "12.000000")


def write_candidate_with_metaroom_seconds(candidate: Path, seconds: str) -> None:
    write_candidate(
        candidate,
        [
            {
                "case": "acasxu_like",
                "phase": "cpu_production",
                "seconds": "0.010000",
                "parameter_count": "13305",
                "estimated_cpu_peak_bytes": "40000",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
            {
                "case": "soundnessbench_exact_like",
                "phase": "cpu_production",
                "seconds": "",
                "parameter_count": "1740696",
                "estimated_cpu_peak_bytes": "154618822656",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "skipped",
                "detail": "dense_peak_exceeds_budget",
            },
            {
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "seconds": seconds,
                "parameter_count": "7410996",
                "estimated_cpu_peak_bytes": "52613349376",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
        ],
    )


def report_path(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "benchmarks" / "gpu_crown_backward_regression_latest.json"


def preserve_report_timestamp(path: Path) -> tuple[str, int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["generated_at"] = "2026-03-01T00:00:00Z"
    preserved_content = json.dumps(report, indent=2) + "\n"
    path.write_text(preserved_content, encoding="utf-8")
    # Pin a sentinel mtime so rerun tests can prove whether the file was rewritten.
    os.utime(path, ns=(946684800000000000, 946684800000000000))
    return preserved_content, path.stat().st_mtime_ns


def test_gpu_crown_backward_regression_help_tracks_cli_contract(tmp_path: Path) -> None:
    result = run_checker_raw(tmp_path, ["--help"])
    normalized_stdout = " ".join(result.stdout.split())

    assert result.returncode == 0, f"--help should exit cleanly: {result.stderr}\n{result.stdout}"
    assert (
        "CPU guard, graph-engine, and direct warm-GPU regression thresholds"
        in normalized_stdout
    ), (
        f"expected current policy summary in help text, got {result.stdout!r}"
    )
    assert "measure_crown_backward_workloads" in normalized_stdout, (
        f"expected candidate help to reference the benchmark generator, got {result.stdout!r}"
    )


def test_gpu_crown_backward_regression_check_only_skips_json_report(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    latest_report_path = report_path(tmp_path)
    write_policy(policy)
    write_passing_candidate(candidate)

    result = run_checker(tmp_path, [candidate], policy, extra_args=["--check-only"])

    assert result.returncode == 0, (
        f"--check-only should keep clean candidates passing: {result.stderr}\n{result.stdout}"
    )
    assert latest_report_path.exists() is False, "--check-only should skip writing the JSON report"


def test_gpu_crown_backward_regression_preserves_report_on_timestamp_only_rerun(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    latest_report_path = report_path(tmp_path)
    write_policy(policy)
    write_passing_candidate(candidate)

    first_result = run_checker(tmp_path, [candidate], policy)

    assert first_result.returncode == 0, (
        f"initial report generation should succeed: {first_result.stderr}\n{first_result.stdout}"
    )

    preserved_content, preserved_mtime_ns = preserve_report_timestamp(latest_report_path)

    second_result = run_checker(tmp_path, [candidate], policy)

    assert second_result.returncode == 0, (
        f"timestamp-only rerun should still pass: {second_result.stderr}\n{second_result.stdout}"
    )
    assert latest_report_path.read_text(encoding="utf-8") == preserved_content, (
        "timestamp-only reruns should preserve the existing latest-report content"
    )
    assert latest_report_path.stat().st_mtime_ns == preserved_mtime_ns, (
        "timestamp-only reruns should skip rewriting the latest-report file"
    )


def test_gpu_crown_backward_regression_preserves_failing_report_on_timestamp_only_rerun(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    latest_report_path = report_path(tmp_path)
    write_policy(policy)
    write_regressing_candidate(candidate)

    first_result = run_checker(tmp_path, [candidate], policy)

    assert first_result.returncode == 1, (
        f"initial regression report should fail cleanly: {first_result.stderr}\n{first_result.stdout}"
    )

    preserved_content, preserved_mtime_ns = preserve_report_timestamp(latest_report_path)

    second_result = run_checker(tmp_path, [candidate], policy)

    assert second_result.returncode == 1, (
        f"timestamp-only failing rerun should preserve the regression outcome: "
        f"{second_result.stderr}\n{second_result.stdout}"
    )
    assert latest_report_path.read_text(encoding="utf-8") == preserved_content, (
        "timestamp-only failing reruns should preserve the existing latest-report content"
    )
    assert latest_report_path.stat().st_mtime_ns == preserved_mtime_ns, (
        "timestamp-only failing reruns should skip rewriting the latest-report file"
    )


def test_gpu_crown_backward_regression_rewrites_report_on_semantic_change(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    latest_report_path = report_path(tmp_path)
    write_policy(policy)
    write_passing_candidate(candidate)

    first_result = run_checker(tmp_path, [candidate], policy)

    assert first_result.returncode == 0, (
        f"initial report generation should succeed: {first_result.stderr}\n{first_result.stdout}"
    )

    _, preserved_mtime_ns = preserve_report_timestamp(latest_report_path)
    write_candidate_with_metaroom_seconds(candidate, "7.500000")

    second_result = run_checker(tmp_path, [candidate], policy)

    assert second_result.returncode == 0, (
        f"semantic-change rerun should still pass: {second_result.stderr}\n{second_result.stdout}"
    )
    second_report = json.loads(latest_report_path.read_text(encoding="utf-8"))
    assert second_report["generated_at"] != "2026-03-01T00:00:00Z", (
        "semantic payload changes should refresh generated_at"
    )
    assert latest_report_path.stat().st_mtime_ns != preserved_mtime_ns, (
        "semantic payload changes should rewrite the latest-report file"
    )
    graph_check = next(item for item in second_report["checks"] if item["name"] == "gpu_path")
    assert graph_check["observed_seconds"] == 7.5, (
        f"expected rewritten report to capture new timing evidence, got {graph_check!r}"
    )
