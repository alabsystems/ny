# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "benchmark_vnncomp_all.sh"
AGGREGATOR_SCRIPT = REPO_ROOT / "scripts" / "aggregate_vnncomp_results.py"


def _write_category_fixture(tmp_path: Path, category: str, *, year: int = 2025) -> None:
    category_dir = tmp_path / "benchmarks" / f"vnncomp{year}" / "benchmarks" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    (category_dir / "instances.csv").write_text(
        "model.onnx,prop.vnnlib,2\n",
        encoding="utf-8",
    )


def _install_runner_harness(
    tmp_path: Path,
    *,
    child_body: str,
    aggregator_body: str | None = None,
) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    runner_copy = scripts_dir / "benchmark_vnncomp_all.sh"
    aggregator_copy = scripts_dir / "aggregate_vnncomp_results.py"
    child_copy = scripts_dir / "benchmark_vnncomp.sh"

    shutil.copy2(RUNNER_SCRIPT, runner_copy)
    if aggregator_body is None:
        shutil.copy2(AGGREGATOR_SCRIPT, aggregator_copy)
    else:
        aggregator_copy.write_text(aggregator_body, encoding="utf-8")
    child_copy.write_text(child_body, encoding="utf-8")

    runner_copy.chmod(0o755)
    aggregator_copy.chmod(0o755)
    child_copy.chmod(0o755)
    return runner_copy


def _run_runner(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(tmp_path / "scripts" / "benchmark_vnncomp_all.sh"), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _load_summary(tmp_path: Path) -> dict:
    summaries = sorted((tmp_path / "reports" / "benchmarks").glob("vnncomp_summary_*.json"))
    assert len(summaries) == 1, f"expected exactly one summary file, got {summaries}"
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def test_benchmark_vnncomp_all_records_failed_category_without_reusing_stale_csv(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "stalecat")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            echo "synthetic failure" >&2
            exit 1
            """
        ),
    )

    report_dir = tmp_path / "reports" / "benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    stale_csv = report_dir / "stalecat_20260310_010203.csv"
    stale_csv.write_text(
        "model,property,timeout,result,elapsed,domains\n"
        "old.onnx,old.vnnlib,2,verified,0.1,7\n",
        encoding="utf-8",
    )

    result = _run_runner(tmp_path, "--categories", "stalecat")

    assert result.returncode == 0, f"runner failed unexpectedly: {result.stderr}"
    assert "Summary:" in result.stdout, f"expected summary output, got: {result.stdout}"
    summary = _load_summary(tmp_path)
    assert summary["categories"] == {}, f"stale CSV should not be aggregated: {summary}"
    assert summary["failed"] == {
        "stalecat": {"exit_code": 1, "reason": "non-zero exit code"}
    }, f"expected failed map for stalecat, got {summary['failed']!r}"
    assert summary["total_instances"] == 0, (
        f"expected 0 instances when only stale CSV existed, got {summary['total_instances']}"
    )
    assert summary["publication_scope"] == "timestamp_only", (
        f"expected timestamp_only summary, got {summary['publication_scope']!r}"
    )
    assert not (report_dir / "vnncomp_latest.json").exists(), (
        "canonical report latest should not be written for failed partial runs"
    )
    assert not (tmp_path / "metrics" / "benchmarks" / "vnncomp_latest.json").exists(), (
        "canonical metrics latest should not be written for failed partial runs"
    )


def test_benchmark_vnncomp_all_excludes_current_csv_from_validation_failure(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "validfail")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            category="$1"
            mkdir -p reports/benchmarks
            report="reports/benchmarks/${category}_20260311_120000.csv"
            printf 'model,property,timeout,result,elapsed,domains\n' > "$report"
            printf 'model.onnx,prop.vnnlib,2,verified,0.5,3\n' >> "$report"
            echo "Report: $report"
            exit 1
            """
        ),
    )

    result = _run_runner(tmp_path, "--categories", "validfail")

    assert result.returncode == 0, f"runner failed unexpectedly: {result.stderr}"
    summary = _load_summary(tmp_path)
    assert (tmp_path / "reports" / "benchmarks" / "validfail_20260311_120000.csv").exists(), (
        "expected synthetic current-run CSV to exist"
    )
    assert summary["categories"] == {}, (
        f"current-run CSV from a non-zero child exit must be excluded: {summary['categories']!r}"
    )
    assert summary["failed"] == {
        "validfail": {"exit_code": 1, "reason": "non-zero exit code"}
    }, f"expected failed map for validation failure, got {summary['failed']!r}"
    assert summary["publication_scope"] == "timestamp_only", (
        f"expected timestamp_only summary, got {summary['publication_scope']!r}"
    )


def test_benchmark_vnncomp_all_skips_publish_flag_when_full_run_has_failures(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "malbeware")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            exit 1
            """
        ),
        aggregator_body=textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            Path("aggregator_args.json").write_text(json.dumps(args), encoding="utf-8")

            output = Path(args[args.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_instances": 0,
                "categories_attempted": 0,
                "total_score": 0,
                "overall_solve_rate": 0.0,
                "publication_scope": "timestamp_only",
            }
            output.write_text(json.dumps(payload) + "\\n", encoding="utf-8")
            """
        ),
    )

    result = _run_runner(tmp_path)

    assert result.returncode == 0, f"runner failed unexpectedly: {result.stderr}"
    aggregator_args = json.loads((tmp_path / "aggregator_args.json").read_text(encoding="utf-8"))
    assert "--run-scope" in aggregator_args, f"expected run-scope arg, got {aggregator_args}"
    assert aggregator_args[aggregator_args.index("--run-scope") + 1] == "full", (
        f"expected full run-scope for default invocation, got {aggregator_args}"
    )
    assert "--publish-metrics" not in aggregator_args, (
        f"full runs with failed categories must not request canonical publish: {aggregator_args}"
    )


def test_benchmark_vnncomp_all_help_describes_runnable_default_set(
    tmp_path: Path,
) -> None:
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            exit 0
            """
        ),
    )

    result = _run_runner(tmp_path, "--help")

    assert result.returncode == 0, f"runner help failed unexpectedly: {result.stderr}"
    assert "default: current runnable set" in result.stdout, (
        f"help text should describe the narrowed default set, got: {result.stdout}"
    )
    assert "default: all supported" not in result.stdout, (
        f"stale help text still claims all supported categories, got: {result.stdout}"
    )


def test_benchmark_vnncomp_all_lists_runtime_limited_categories_as_skipped_by_default(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "vit_2023")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            exit 0
            """
        ),
    )

    result = _run_runner(tmp_path, "--dry-run")

    assert result.returncode == 0, f"runner failed unexpectedly: {result.stderr}"
    supported_section, skipped_section = result.stdout.split("=== Skipped Categories ===", 1)
    assert "vit_2023" not in supported_section, (
        f"vit_2023 should be excluded from the default runnable set, got: {result.stdout}"
    )
    assert "vit_2023" in skipped_section, (
        f"expected vit_2023 in skipped categories, got: {result.stdout}"
    )
    assert "cifar100_2024" in skipped_section, (
        f"expected cifar100_2024 in skipped categories, got: {result.stdout}"
    )
    assert "tinyimagenet_2024" in skipped_section, (
        f"expected tinyimagenet_2024 in skipped categories, got: {result.stdout}"
    )
    assert "yolo_2023" in skipped_section, (
        f"expected yolo_2023 in skipped categories, got: {result.stdout}"
    )
    assert "test category - not a real benchmark" not in skipped_section, (
        f"pseudo test category should not leak into skipped categories, got: {result.stdout}"
    )


def test_benchmark_vnncomp_all_allows_explicit_runtime_limited_probe(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "vit_2023")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            exit 0
            """
        ),
    )

    result = _run_runner(tmp_path, "--dry-run", "--categories", "vit_2023")

    assert result.returncode == 0, f"runner failed unexpectedly: {result.stderr}"
    supported_section, skipped_section = result.stdout.split("=== Skipped Categories ===", 1)
    assert "vit_2023" in supported_section, (
        f"explicit category selection should still allow vit_2023 probes, got: {result.stdout}"
    )
    assert "vit_2023" in skipped_section, (
        "skip metadata should remain visible so explicit probes still report the known blocker"
    )


def test_benchmark_vnncomp_all_keeps_current_head_measured_categories_supported(
    tmp_path: Path,
) -> None:
    for category in ("cora_2024", "linearizenn_2024", "lsnc_relu"):
        _write_category_fixture(tmp_path, category)

    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            exit 0
            """
        ),
    )

    result = _run_runner(tmp_path, "--dry-run")

    assert result.returncode == 0, f"runner failed unexpectedly: {result.stderr}"
    supported_section, skipped_section = result.stdout.split("=== Skipped Categories ===", 1)

    for category in ("cora_2024", "linearizenn_2024", "lsnc_relu"):
        supported_line = next(
            (line for line in supported_section.splitlines() if category in line),
            None,
        )
        assert supported_line is not None, (
            f"{category} should remain in the runnable default set, got: {result.stdout}"
        )
        assert "1 instances" in supported_line and "NOT FOUND" not in supported_line, (
            f"{category} should resolve to a real runnable fixture line, got: {supported_line!r}"
        )
        assert category not in skipped_section, (
            f"{category} should not regress back into skipped categories, got: {result.stdout}"
        )
