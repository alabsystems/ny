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
)

# Keep a direct literal path so scoped-test indexing maps checker edits here.
_CHECKER_PATH = "scripts/check_gpu_crown_backward_regression.py"


def _shared_phase_policy(source_artifact: str | None = None) -> list[dict[str, object]]:
    check: dict[str, object] = {
        "name": "shared_phase",
        "case": "soundnessbench_exact_like",
        "phase": "wgpu_production_cold",
        "expected_status": "measured",
        "baseline_seconds": 10.0,
        "max_regression_ratio": 1.2,
        "max_seconds": 12.5,
    }
    if source_artifact is not None:
        check["source_artifact"] = source_artifact
    return [check]


def _shared_phase_row(seconds: str) -> dict[str, str]:
    return {
        "case": "soundnessbench_exact_like",
        "phase": "wgpu_production_cold",
        "seconds": seconds,
        "parameter_count": "1740696",
        "estimated_cpu_peak_bytes": "154618822656",
        "cpu_dense_budget_bytes": "2147483648",
        "status": "measured",
        "detail": "",
    }


def _load_report(tmp_path: Path) -> dict[str, object]:
    return json.loads(
        (
            tmp_path
            / "reports"
            / "benchmarks"
            / "gpu_crown_backward_regression_latest.json"
        ).read_text(encoding="utf-8")
    )


def test_gpu_crown_backward_regression_uses_source_artifact_to_resolve_duplicate_rows(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate_a = tmp_path / "reports" / "benchmarks" / "candidate_a.csv"
    candidate_b = tmp_path / "reports" / "benchmarks" / "candidate_b.csv"
    write_custom_policy(
        policy,
        _shared_phase_policy(source_artifact="reports/benchmarks/candidate_b.csv"),
    )
    write_candidate(candidate_a, [_shared_phase_row("18.000000")])
    write_candidate(candidate_b, [_shared_phase_row("11.500000")])

    result = run_checker(tmp_path, [candidate_a, candidate_b], policy)

    assert result.returncode == 0, f"checker should honor source_artifact: {result.stdout}"
    report = _load_report(tmp_path)
    shared = report["checks"][0]
    assert shared["regression"] is False, f"expected disambiguated check to pass, got {shared!r}"
    assert shared["observed_candidate"] == str(candidate_b), (
        f"expected source_artifact row to win, got {shared!r}"
    )


def test_gpu_crown_backward_regression_flags_ambiguous_duplicate_checked_rows(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "configs" / "gpu.json"
    candidate_a = tmp_path / "reports" / "benchmarks" / "candidate_a.csv"
    candidate_b = tmp_path / "reports" / "benchmarks" / "candidate_b.csv"
    write_custom_policy(policy, _shared_phase_policy())
    write_candidate(candidate_a, [_shared_phase_row("11.500000")])
    write_candidate(candidate_b, [_shared_phase_row("12.000000")])

    result = run_checker(tmp_path, [candidate_a, candidate_b], policy)

    assert result.returncode == 1, (
        f"checker should fail when duplicate checked rows disagree: {result.stdout}"
    )
    report = _load_report(tmp_path)
    shared = report["checks"][0]
    assert shared["regression"] is True, f"expected ambiguous row regression, got {shared!r}"
    assert shared["reasons"] == ["ambiguous_row"], (
        f"expected ambiguous_row, got {shared['reasons']!r}"
    )
