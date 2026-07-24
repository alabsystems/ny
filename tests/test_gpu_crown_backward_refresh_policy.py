# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
from pathlib import Path

from scripts.test_support.gpu_crown_backward_regression_test_utils import (
    REPO_ROOT,
    run_refresh,
    write_candidate,
    write_custom_policy,
    write_policy,
)

# Keep a direct literal path so scoped-test indexing maps refresher edits here.
_REFRESH_PATH = "scripts/refresh_gpu_crown_backward_baselines.py"
_HELPER_PATH = "scripts/gpu_crown_backward_regression_lib.py"


def load_policy(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gpu_crown_backward_refresh_policy_updates_baselines_and_source_artifacts(
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

    result = run_refresh(tmp_path, [candidate], policy)

    assert result.returncode == 0, f"refresh should succeed: {result.stderr}\n{result.stdout}"
    refreshed = load_policy(policy)
    checks = {item["name"]: item for item in refreshed["checks"]}
    assert checks["cpu_path"]["baseline_seconds"] == 0.01, checks["cpu_path"]
    assert checks["gpu_path"]["baseline_seconds"] == 6.5, checks["gpu_path"]
    assert checks["gpu_guard"]["source_artifact"] == "reports/benchmarks/candidate.csv", (
        f"refresh should pin source_artifact to the observed CSV: {checks['gpu_guard']!r}"
    )


def test_gpu_crown_backward_refresh_policy_rejects_status_mismatch(tmp_path: Path) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate = tmp_path / "reports" / "benchmarks" / "candidate.csv"
    write_policy(policy)
    original = policy.read_text(encoding="utf-8")
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

    result = run_refresh(tmp_path, [candidate], policy)

    assert result.returncode == 1, "refresh must reject status mismatches"
    assert "gpu_guard (status_mismatch)" in result.stdout, result.stdout
    assert policy.read_text(encoding="utf-8") == original, (
        "failed refresh should leave the policy file unchanged"
    )


def test_gpu_crown_backward_refresh_policy_prefers_source_artifact_match(
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
                "expected_parameter_count": 7410996,
                "expected_estimated_cpu_peak_bytes": 52613349376,
                "expected_cpu_dense_budget_bytes": 2147483648,
                "baseline_seconds": 4.0,
                "max_regression_ratio": 2.0,
                "max_seconds": 9.0,
                "source_artifact": "reports/benchmarks/candidate_b.csv",
            },
        ],
    )
    row = {
        "case": "metaroom_6cnn_ry_like",
        "phase": "graph_crown_ibp_collection_engine",
        "parameter_count": "7410996",
        "estimated_cpu_peak_bytes": "52613349376",
        "cpu_dense_budget_bytes": "2147483648",
        "status": "measured",
        "detail": "",
    }
    write_candidate(candidate_a, [{**row, "seconds": "8.500000"}])
    write_candidate(candidate_b, [{**row, "seconds": "6.500000"}])

    result = run_refresh(tmp_path, [candidate_a, candidate_b], policy)

    assert result.returncode == 0, f"refresh should honor source_artifact: {result.stdout}"
    refreshed = load_policy(policy)
    gpu_check = refreshed["checks"][0]
    assert gpu_check["baseline_seconds"] == 6.5, (
        f"refresh should use the candidate selected by source_artifact: {gpu_check!r}"
    )
    assert gpu_check["source_artifact"] == "reports/benchmarks/candidate_b.csv", gpu_check


def test_gpu_crown_backward_refresh_policy_rejects_stale_split_artifacts_after_engine_only_pin(
    tmp_path: Path,
) -> None:
    output = tmp_path / "configs" / "gpu_refreshed.json"
    policy = REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
    candidates = [
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_backward_timing_20260312_clean.csv",
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_ibp_graph_soundnessbench_v3_20260313.csv",
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_backward_timing_20260315_W3_3813_graph.csv",
        REPO_ROOT / "reports" / "benchmarks" / "gpu_crown_ibp_graph_metaroom_v3_20260313.csv",
    ]

    result = run_refresh(
        tmp_path,
        candidates,
        policy,
        extra_args=["--output", str(output)],
    )

    assert result.returncode == 1, (
        "stale split artifacts should no longer satisfy the checked-in engine-only "
        f"policy pins: {result.stderr}\n{result.stdout}"
    )
    assert "source_artifact_missing" in result.stdout, result.stdout
    assert not output.exists(), "rejected refresh should not write an updated policy"


def test_gpu_crown_backward_refresh_policy_accepts_checked_in_engine_only_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "configs" / "gpu_refreshed_engine_only.json"
    policy = REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
    candidate = (
        REPO_ROOT
        / "reports"
        / "benchmarks"
        / "gpu_crown_backward_engine_only_20260319.csv"
    )

    result = run_refresh(
        tmp_path,
        [candidate],
        policy,
        extra_args=["--output", str(output)],
    )

    assert result.returncode == 0, (
        "checked-in engine-only artifact should refresh cleanly without "
        f"source-artifact drift: {result.stderr}\n{result.stdout}"
    )
    assert "source_artifact" not in result.stdout, result.stdout
    refreshed = load_policy(output)
    checks = {item["name"]: item for item in refreshed["checks"]}
    for name, check in checks.items():
        assert check["source_artifact"] == (
            "reports/benchmarks/gpu_crown_backward_engine_only_20260319.csv"
        ), f"{name} should pin the checked-in engine-only artifact: {check!r}"
    assert output.read_text(encoding="utf-8") == policy.read_text(encoding="utf-8"), (
        "refreshing the pinned engine-only policy should be byte-stable for the "
        "checked-in JSON, including the non-ASCII variance note"
    )
    assert checks["metaroom_graph_engine"]["baseline_seconds"] == 3.895002, (
        f"engine-only refresh should preserve metaroom graph evidence: {checks['metaroom_graph_engine']!r}"
    )
    assert checks["soundnessbench_wgpu_warm"]["baseline_seconds"] == 16.602116, (
        "engine-only refresh should preserve the warm GPU soundnessbench timing: "
        f"{checks['soundnessbench_wgpu_warm']!r}"
    )


def test_gpu_crown_backward_refresh_policy_rejects_currenthead_consolidated_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "configs" / "gpu_refreshed_currenthead.json"
    policy = REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
    candidate = (
        REPO_ROOT
        / "reports"
        / "benchmarks"
        / "gpu_crown_backward_timing_currenthead_20260314.csv"
    )

    result = run_refresh(
        tmp_path,
        [candidate],
        policy,
        extra_args=["--output", str(output)],
    )

    assert result.returncode == 1, (
        "refresh should reject the currenthead consolidated artifact because it "
        "does not match the checked-in source_artifact pins"
    )
    assert "source_artifact_missing" in result.stdout, result.stdout
    assert not output.exists(), "rejected refresh should not rewrite the policy"
