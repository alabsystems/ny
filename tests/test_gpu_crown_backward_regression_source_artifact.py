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


def assert_resolves_to_artifact(
    observed_candidates: dict[str, str],
    check_name: str,
    suffix: str,
) -> None:
    assert observed_candidates[check_name].endswith(suffix), (
        f"{check_name} should resolve to its pinned source artifact: {observed_candidates!r}"
    )


def assert_parameter_count(
    report: dict[str, object],
    check_name: str,
    expected_parameter_count: int,
) -> None:
    check = next(item for item in report["checks"] if item["name"] == check_name)
    assert check["expected_parameter_count"] == expected_parameter_count, (
        f"checked-in policy should pin workload metadata for {check_name}: {check!r}"
    )
    assert check["observed_parameter_count"] == expected_parameter_count, (
        f"checked-in artifact should match the pinned workload metadata for {check_name}: {check!r}"
    )


def assert_observed_seconds(
    report: dict[str, object],
    check_name: str,
    expected_seconds: float,
) -> None:
    check = next(item for item in report["checks"] if item["name"] == check_name)
    assert check["observed_seconds"] == expected_seconds, (
        f"checked-in artifact should preserve the pinned timing evidence for {check_name}: {check!r}"
    )


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


def test_gpu_crown_backward_regression_accepts_checked_in_engine_only_artifact(
    tmp_path: Path,
) -> None:
    policy = REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
    candidate = (
        REPO_ROOT
        / "reports"
        / "benchmarks"
        / "gpu_crown_backward_engine_only_20260319.csv"
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 0, (
        "checked-in policy should accept the pinned engine-only artifact "
        f"without fallback warnings: {result.stderr}\n{result.stdout}"
    )
    assert "source_artifact" not in result.stdout, result.stdout
    report = load_report(tmp_path)
    observed_candidates = {
        item["name"]: item["observed_candidate"]
        for item in report["checks"]
        if item["observed_candidate"] is not None
    }
    assert report["regression"] is False, f"expected pinned engine-only artifact to pass, got {report!r}"
    for check_name in observed_candidates:
        assert_resolves_to_artifact(
            observed_candidates,
            check_name,
            "reports/benchmarks/gpu_crown_backward_engine_only_20260319.csv",
        )
    for check in report["checks"]:
        assert check["selection_mode"] == "source_artifact_match", (
            "checked-in engine-only artifact should resolve via a direct "
            f"source-artifact match: {check!r}"
        )
    assert_parameter_count(report, "soundnessbench_graph_engine", 1740696)
    assert_parameter_count(report, "metaroom_graph_engine", 7410996)
    assert_observed_seconds(report, "soundnessbench_graph_engine", 14.900598)
    assert_observed_seconds(report, "metaroom_graph_engine", 3.895002)


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


def test_gpu_crown_backward_regression_currenthead_consolidated_artifact_surfaces_real_regression(
    tmp_path: Path,
) -> None:
    """The checked-in consolidated currenthead artifact should exercise the
    sole-candidate fallback and still report the real metaroom timing
    regression instead of failing on source-artifact routing."""
    policy = REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
    candidate = (
        REPO_ROOT
        / "reports"
        / "benchmarks"
        / "gpu_crown_backward_timing_currenthead_20260314.csv"
    )

    result = run_checker(tmp_path, [candidate], policy)

    assert result.returncode == 1, (
        "the checked-in currenthead artifact should currently fail on the real "
        "metaroom graph-engine timing regression"
    )
    assert "metaroom_graph_engine: source_artifact" in result.stdout, (
        "the checked-in currenthead artifact should still exercise the "
        "sole-candidate soft fallback for metaroom_graph_engine"
    )
    report = load_report(tmp_path)
    metaroom_check = next(
        item for item in report["checks"] if item["name"] == "metaroom_graph_engine"
    )
    soundnessbench_check = next(
        item for item in report["checks"] if item["name"] == "soundnessbench_graph_engine"
    )
    failing_checks = [item["name"] for item in report["checks"] if item["regression"] is True]
    assert report["regression"] is True, f"expected currenthead report regression, got {report!r}"
    assert failing_checks == ["metaroom_graph_engine"], (
        "the checked-in currenthead artifact should only regress on "
        f"metaroom_graph_engine, got {failing_checks!r}"
    )
    assert metaroom_check["observed_candidate"].endswith(
        "reports/benchmarks/gpu_crown_backward_timing_currenthead_20260314.csv"
    ), f"expected consolidated currenthead artifact to be selected, got {metaroom_check!r}"
    assert metaroom_check["selection_mode"] == "source_artifact_sole_candidate_fallback", (
        "the currenthead consolidated artifact should record the soft fallback "
        f"selection path: {metaroom_check!r}"
    )
    assert metaroom_check["reasons"] == ["seconds_exceeded"], (
        f"expected real metaroom timing regression, got {metaroom_check['reasons']!r}"
    )
    assert metaroom_check["observed_seconds"] == 32.648467, (
        f"expected pinned metaroom timing evidence, got {metaroom_check!r}"
    )
    assert soundnessbench_check["regression"] is False, (
        f"soundnessbench currenthead timing should remain within policy: {soundnessbench_check!r}"
    )
