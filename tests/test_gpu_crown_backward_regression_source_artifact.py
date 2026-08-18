# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
from pathlib import Path

from scripts.test_support.gpu_crown_backward_regression_test_utils import (
    REPO_ROOT,
    run_checker,
    write_candidate,
    write_custom_policy,
)

# Keep a direct literal path so scoped-test indexing maps checker edits here.
_CHECKER_PATH = "scripts/check_gpu_crown_backward_regression.py"


def graph_engine_row(seconds: str) -> dict[str, str]:
    return {
        "case": "metaroom_6cnn_ry_like",
        "phase": "graph_crown_ibp_collection_engine",
        "seconds": seconds,
        "parameter_count": "7410996",
        "estimated_cpu_peak_bytes": "52613349376",
        "cpu_dense_budget_bytes": "2147483648",
        "status": "measured",
        "detail": "",
    }


def load_report(tmp_path: Path) -> dict[str, object]:
    report_path = (
        tmp_path
        / "reports"
        / "benchmarks"
        / "gpu_crown_backward_regression_latest.json"
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_gpu_crown_backward_regression_rejects_stale_split_artifacts_after_engine_only_pin(
    tmp_path: Path,
) -> None:
    policy = REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
    candidates = [
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_backward_timing_20260312_clean.csv",
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_ibp_graph_soundnessbench_v3_20260313.csv",
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_backward_timing_20260315_W3_3813_graph.csv",
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_ibp_graph_metaroom_v3_20260313.csv",
    ]

    result = run_checker(tmp_path, candidates, policy)

    assert result.returncode == 1, (
        "checked-in engine-only policy should reject the stale split candidate "
        f"bundle: {result.stderr}\n{result.stdout}"
    )
    assert "source_artifact_missing" in result.stdout, result.stdout
    report = load_report(tmp_path)
    assert report["regression"] is True, f"expected split bundle mismatch regression, got {report!r}"
    failing_checks = [item["name"] for item in report["checks"] if item["regression"] is True]
    assert failing_checks == [
        "acasxu_cpu_production",
        "soundnessbench_cpu_budget_guard",
        "soundnessbench_graph_engine",
        "soundnessbench_wgpu_warm",
        "metaroom_graph_engine",
        "metaroom_wgpu_warm",
    ], f"expected all checks to flag stale split artifacts, got {failing_checks!r}"
    for check in report["checks"]:
        assert check["reasons"] == ["source_artifact_missing"], (
            f"stale split artifact bundle should fail on source pin mismatch only: {check!r}"
        )


def test_gpu_crown_backward_regression_prefers_source_artifact_match(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate_a = tmp_path / "reports" / "benchmarks" / "candidate_a.csv"
    candidate_b = tmp_path / "reports" / "benchmarks" / "candidate_b.csv"
    write_custom_policy(
        policy,
        [
            {
                "name": "gpu_path",
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "expected_status": "measured",
                "baseline_seconds": 4.0,
                "max_regression_ratio": 2.0,
                "max_seconds": 9.0,
                "source_artifact": "reports/benchmarks/candidate_b.csv",
            },
        ],
    )
    write_candidate(candidate_a, [graph_engine_row("12.000000")])
    write_candidate(candidate_b, [graph_engine_row("6.500000")])

    result = run_checker(tmp_path, [candidate_a, candidate_b], policy)

    assert result.returncode == 0, (
        f"source_artifact match should disambiguate duplicates: {result.stderr}\n{result.stdout}"
    )
    report = load_report(tmp_path)
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["observed_candidate"] == str(candidate_b), (
        f"expected source_artifact to pick candidate_b, got {gpu_check!r}"
    )
    assert gpu_check["regression"] is False, (
        f"source-selected candidate should keep the check clean: {gpu_check!r}"
    )
    assert gpu_check["selection_mode"] == "source_artifact_match", (
        f"expected a direct source-artifact match, got {gpu_check!r}"
    )


def test_gpu_crown_backward_regression_soft_fallback_sole_candidate(
    tmp_path: Path,
) -> None:
    """When source_artifact doesn't match but there's exactly one candidate row
    for the (case, phase) pair, the checker uses that row with a warning instead
    of hard-failing.  This supports consolidated CSVs alongside the multi-CSV
    workflow."""
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate_a.csv"
    write_custom_policy(
        policy,
        [
            {
                "name": "gpu_path",
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "expected_status": "measured",
                "baseline_seconds": 4.0,
                "max_regression_ratio": 2.0,
                "max_seconds": 9.0,
                "source_artifact": "reports/benchmarks/candidate_b.csv",
            },
        ],
    )
    write_candidate(candidate, [graph_engine_row("6.500000")])

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 0, (
        f"sole candidate soft fallback should accept: {result.stderr}\n{result.stdout}"
    )
    assert "source_artifact" in result.stderr or "source_artifact" in result.stdout, (
        "expected a warning about source_artifact mismatch in output"
    )
    report = load_report(tmp_path)
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["regression"] is False, (
        f"sole-candidate soft fallback should not be a regression: {gpu_check!r}"
    )
    assert gpu_check["observed_candidate"] == str(candidate), (
        f"should resolve to the sole candidate: {gpu_check!r}"
    )
    assert gpu_check["selection_mode"] == "source_artifact_sole_candidate_fallback", (
        f"expected the report to record the sole-candidate fallback path: {gpu_check!r}"
    )


def test_gpu_crown_backward_regression_fails_on_conflicting_duplicates_without_source_match(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate_a = tmp_path / "reports" / "benchmarks" / "candidate_a.csv"
    candidate_b = tmp_path / "reports" / "benchmarks" / "candidate_b.csv"
    write_custom_policy(
        policy,
        [
            {
                "name": "gpu_path",
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "expected_status": "measured",
                "baseline_seconds": 4.0,
                "max_regression_ratio": 2.0,
                "max_seconds": 9.0,
                "source_artifact": "reports/benchmarks/nonexistent.csv",
            },
        ],
    )
    write_candidate(candidate_a, [graph_engine_row("6.500000")])
    write_candidate(candidate_b, [graph_engine_row("8.500000")])

    result = run_checker(tmp_path, [candidate_a, candidate_b], policy)

    assert result.returncode == 1, (
        "conflicting duplicates without a matching source_artifact must fail"
    )
    report = load_report(tmp_path)
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["regression"] is True, f"expected ambiguity regression, got {gpu_check!r}"
    assert gpu_check["reasons"] == ["source_artifact_missing"], (
        f"expected source_artifact_missing, got {gpu_check['reasons']!r}"
    )


def test_gpu_crown_backward_regression_soft_fallback_still_detects_regression(
    tmp_path: Path,
) -> None:
    """Soft fallback uses the sole candidate row but still detects timing
    regressions.  The source_artifact mismatch is forgiven; the exceeded
    threshold is not."""
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "consolidated.csv"
    write_custom_policy(
        policy,
        [
            {
                "name": "gpu_path",
                "case": "metaroom_6cnn_ry_like",
                "phase": "graph_crown_ibp_collection_engine",
                "expected_status": "measured",
                "baseline_seconds": 4.0,
                "max_regression_ratio": 2.0,
                "max_seconds": 9.0,
                "source_artifact": "reports/benchmarks/original_baseline.csv",
            },
        ],
    )
    write_candidate(candidate, [graph_engine_row("12.000000")])

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, (
        f"soft fallback must still detect timing regression: {result.stderr}\n{result.stdout}"
    )
    report = load_report(tmp_path)
    gpu_check = next(item for item in report["checks"] if item["name"] == "gpu_path")
    assert gpu_check["regression"] is True, (
        f"expected timing regression despite soft fallback: {gpu_check!r}"
    )
    assert "seconds_exceeded" in gpu_check["reasons"], (
        f"expected seconds_exceeded in reasons: {gpu_check['reasons']!r}"
    )
