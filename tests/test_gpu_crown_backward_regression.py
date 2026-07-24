# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
from pathlib import Path

from scripts.test_support.gpu_crown_backward_regression_test_utils import (
    run_checker,
    write_candidate,
    write_custom_policy,
    write_policy,
)

# Keep a direct literal path so scoped-test indexing maps checker edits here.
_CHECKER_PATH = "scripts/check_gpu_crown_backward_regression.py"


def test_gpu_crown_backward_regression_accepts_candidate_within_thresholds(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_policy(policy)
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
                "seconds": "6.500000",
                "parameter_count": "7410996",
                "estimated_cpu_peak_bytes": "52613349376",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
        ],
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 0, f"checker failed unexpectedly: {result.stderr}\n{result.stdout}"
    assert "No GPU CROWN regression detected." in result.stdout, (
        f"expected success message, got stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    assert report["regression"] is False, f"expected no regression, got {report!r}"
    assert all(
        not item["regression"] for item in report["checks"]
    ), f"expected all checks clean, got {report['checks']!r}"


def test_gpu_crown_backward_regression_fails_when_seconds_exceed_threshold(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_policy(policy)
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
                "seconds": "12.000000",
                "parameter_count": "7410996",
                "estimated_cpu_peak_bytes": "52613349376",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
        ],
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, f"checker should fail on regression: {result.stdout}"
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["regression"] is True, f"expected gpu regression, got {gpu_check!r}"
    assert gpu_check["reasons"] == ["seconds_exceeded"], (
        f"expected seconds_exceeded, got {gpu_check['reasons']!r}"
    )


def test_gpu_crown_backward_regression_fails_when_expected_row_missing(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_policy(policy)
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
        ],
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, f"checker should fail on missing row: {result.stdout}"
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["regression"] is True, f"expected missing-row regression, got {gpu_check!r}"
    assert gpu_check["reasons"] == ["missing_row"], (
        f"expected missing_row, got {gpu_check['reasons']!r}"
    )


def test_gpu_crown_backward_regression_fails_on_status_mismatch(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_policy(policy)
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
                "status": "failed",
                "detail": "dense_peak_exceeds_budget",
            },
            {
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "seconds": "6.500000",
                "parameter_count": "7410996",
                "estimated_cpu_peak_bytes": "52613349376",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
        ],
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, f"checker should fail on status mismatch: {result.stdout}"
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    guard_check = next(item for item in report["checks"] if item["name"] == "gpu_guard")
    assert guard_check["regression"] is True, (
        f"expected status-mismatch regression, got {guard_check!r}"
    )
    assert guard_check["reasons"] == ["status_mismatch"], (
        f"expected status_mismatch, got {guard_check['reasons']!r}"
    )


def test_gpu_crown_backward_regression_fails_when_workload_metadata_changes(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_policy(policy)
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
                "seconds": "6.500000",
                "parameter_count": "7410997",
                "estimated_cpu_peak_bytes": "52613349377",
                "cpu_dense_budget_bytes": "2147483649",
                "status": "measured",
                "detail": "",
            },
        ],
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, (
        f"checker should fail when workload metadata changes: {result.stdout}"
    )
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["regression"] is True, (
        f"expected workload metadata regression, got {gpu_check!r}"
    )
    assert gpu_check["reasons"] == [
        "parameter_count_mismatch",
        "estimated_cpu_peak_bytes_mismatch",
        "cpu_dense_budget_bytes_mismatch",
    ], f"expected workload metadata mismatch reasons, got {gpu_check['reasons']!r}"


def test_gpu_crown_backward_regression_fails_when_wgpu_warm_exceeds_threshold(
    tmp_path: Path,
) -> None:
    """Exercises seconds_exceeded specifically for the wgpu_production_warm
    phase — the production check type with the thinnest regression margin
    (soundnessbench at 96.8% of limit as of 2026-03-15)."""
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_custom_policy(
        policy,
        [
            {
                "name": "wgpu_warm_check",
                "case": "soundnessbench_exact_like",
                "phase": "wgpu_production_warm",
                "expected_status": "measured",
                "expected_parameter_count": 1740696,
                "expected_estimated_cpu_peak_bytes": 154618822656,
                "expected_cpu_dense_budget_bytes": 2147483648,
                "baseline_seconds": 11.84,
                "max_regression_ratio": 1.6,
                "max_seconds": 20.0,
            },
        ],
    )
    write_candidate(
        candidate,
        [
            {
                "case": "soundnessbench_exact_like",
                "phase": "wgpu_production_warm",
                "seconds": "25.000000",
                "parameter_count": "1740696",
                "estimated_cpu_peak_bytes": "154618822656",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
        ],
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, (
        f"checker should fail on wgpu_warm regression: {result.stdout}"
    )
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    warm_check = next(
        item for item in report["checks"] if item["name"] == "wgpu_warm_check"
    )
    assert warm_check["regression"] is True, (
        f"expected wgpu_warm regression, got {warm_check!r}"
    )
    assert warm_check["reasons"] == ["seconds_exceeded"], (
        f"expected seconds_exceeded for wgpu_warm, got {warm_check['reasons']!r}"
    )
    assert warm_check["allowed_seconds"] == 11.84 * 1.6, (
        f"allowed should be baseline * ratio, got {warm_check['allowed_seconds']!r}"
    )


def test_gpu_crown_backward_regression_accepts_split_candidate_files(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate_a = tmp_path / "reports" / "benchmarks" / "candidate_a.csv"
    candidate_b = tmp_path / "reports" / "benchmarks" / "candidate_b.csv"
    write_policy(policy)
    write_candidate(
        candidate_a,
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
        ],
    )
    write_candidate(
        candidate_b,
        [
            {
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "seconds": "6.500000",
                "parameter_count": "7410996",
                "estimated_cpu_peak_bytes": "52613349376",
                "cpu_dense_budget_bytes": "2147483648",
                "status": "measured",
                "detail": "",
            },
        ],
    )

    result = run_checker(tmp_path, [candidate_a, candidate_b], policy)
    assert result.returncode == 0, f"checker failed unexpectedly: {result.stderr}\n{result.stdout}"
    report = json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )
    assert report["regression"] is False, f"expected no regression, got {report!r}"
    assert report["candidates"] == [str(candidate_a), str(candidate_b)], (
        f"expected split candidate list, got {report['candidates']!r}"
    )
