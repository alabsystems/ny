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


def _write_category_fixture(
    tmp_path: Path,
    category: str,
    *,
    year: int = 2025,
    version: str | None = None,
) -> None:
    category_dir = tmp_path / "benchmarks" / f"vnncomp{year}" / "benchmarks" / category
    if version is not None:
        category_dir /= version
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


def test_benchmark_vnncomp_all_records_successful_child_without_report_as_failed(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "missingreport")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            echo "child completed without publishing a report"
            exit 0
            """
        ),
    )

    result = _run_runner(tmp_path, "--categories", "missingreport")

    assert result.returncode == 0, (
        "a missing child report should be represented in the aggregate summary, "
        f"not abort the runner: {result.stderr}"
    )
    summary = _load_summary(tmp_path)
    assert summary["categories"] == {}, summary
    assert summary["failed"] == {
        "missingreport": {"exit_code": 0, "reason": "no report path in output"}
    }, summary


def test_benchmark_vnncomp_all_rejects_report_path_traversal(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "traversal")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            mkdir -p reports/benchmarks
            printf 'model,property,timeout,result,elapsed,domains\n' \
                > reports/escaped.csv
            echo "Report: reports/benchmarks/../escaped.csv"
            """
        ),
    )

    result = _run_runner(tmp_path, "--categories", "traversal")

    assert result.returncode == 0, result.stderr
    summary = _load_summary(tmp_path)
    assert summary["categories"] == {}, summary
    assert summary["failed"] == {
        "traversal": {
            "exit_code": 0,
            "reason": "report path outside reports/benchmarks/",
        }
    }, summary


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
    assert "vit_2023" not in skipped_section, (
        "an explicitly attempted category must not also be reported as skipped"
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
        assert "1 instances" in supported_line, (
            f"{category} should resolve to a real runnable fixture line, got: {supported_line!r}"
        )
        assert "NOT FOUND" not in supported_line, (
            f"{category} should resolve to a real runnable fixture line, got: {supported_line!r}"
        )
        assert category not in skipped_section, (
            f"{category} should not regress back into skipped categories, got: {result.stdout}"
        )


def test_benchmark_vnncomp_all_2026_defaults_partition_official_categories(
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

    result = _run_runner(tmp_path, "--year", "2026", "--dry-run")

    assert result.returncode == 0, f"2026 dry run failed unexpectedly: {result.stderr}"
    supported_section, skipped_section = result.stdout.split("=== Skipped Categories ===", 1)
    supported = {
        line.split()[0]
        for line in supported_section.splitlines()
        if line.startswith("  ") and len(line.split()) >= 2
    }
    skipped = {
        line.split()[0]
        for line in skipped_section.splitlines()
        if line.startswith("  ") and len(line.split()) >= 2
    }
    official = {
        "acasxu_2023",
        "adaptive_cruise_control_non_linear_2026",
        "cctsdb_yolo_2023",
        "cersyve",
        "cgan2026",
        "challenging_certified_training_2026",
        "cifar100_2024",
        "collins_aerospace_benchmark",
        "collins_rul_cnn_2022",
        "cora_2024",
        "dist_shift_2023",
        "isomorphic_acasxu_2026",
        "linearizenn_2024",
        "lsnc_relu",
        "malbeware",
        "metaroom_2023",
        "ml4acopf_2024",
        "monotonic_acasxu_2026",
        "nn4sys",
        "relusplitter_2026",
        "safenlp_2024",
        "sat_relu",
        "smart_turn_multimodal_2026",
        "soundnessbench_2026",
        "tinyimagenet_2024",
        "tllverifybench_2023",
        "traffic_signs_recognition_2023",
        "vggnet16_2022",
        "vit_2023",
        "yolo_2023",
    }

    assert supported.isdisjoint(skipped), (
        f"2026 categories cannot be both attempted and skipped: {supported & skipped}"
    )
    assert supported | skipped == official, (
        f"2026 default scope drifted: missing={official - supported - skipped}, "
        f"extra={(supported | skipped) - official}"
    )
    assert "cgan2026" in supported
    assert "relusplitter_2026" in supported
    assert "cgan_2023" not in supported
    assert "relusplitter" not in supported
    assert "soundnessbench_2026" in skipped
    assert "soundnessbench" not in skipped


def test_benchmark_vnncomp_all_2026_routes_version_root_preset_and_tracker(
    tmp_path: Path,
) -> None:
    category = "cgan2026"
    _write_category_fixture(tmp_path, category, year=2026, version="1.0")
    preset = tmp_path / "configs" / "vnncomp26" / f"{category}.yaml"
    preset.parent.mkdir(parents=True, exist_ok=True)
    preset.write_text("general: {}\n", encoding="utf-8")

    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            printf '%s\\n%s\\n%s\\n%s\\n' \\
              "$BENCH_ROOT" "$BENCH_DIR" "$PRESET_PATH_OVERRIDE" "${2:-}" > child_env.txt
            mkdir -p reports/benchmarks
            report="reports/benchmarks/${1}_20260729_120000.csv"
            printf 'model,property,timeout,result,elapsed,domains\\n' > "$report"
            printf 'model.onnx,prop.vnnlib,2,verified,0.5,3\\n' >> "$report"
            echo "Report: $report"
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
            output.write_text(json.dumps({
                "total_instances": 1,
                "categories_attempted": 1,
                "total_score": 1,
                "overall_solve_rate": 100.0,
                "publication_scope": "timestamp_only",
            }) + "\\n", encoding="utf-8")
            """
        ),
    )

    result = _run_runner(tmp_path, "--year", "2026", "--categories", category)

    assert result.returncode == 0, f"2026 routed run failed: {result.stderr}"
    child_env = (tmp_path / "child_env.txt").read_text(encoding="utf-8").splitlines()
    assert child_env == [
        "benchmarks/vnncomp2026/benchmarks",
        f"benchmarks/vnncomp2026/benchmarks/{category}/1.0",
        f"configs/vnncomp26/{category}.yaml",
        "--competition-wrapper",
    ]
    aggregator_args = json.loads((tmp_path / "aggregator_args.json").read_text(encoding="utf-8"))
    assert aggregator_args[aggregator_args.index("--year") + 1] == "2026"
    assert aggregator_args[aggregator_args.index("--tracker-year") + 1] == "2026"


def test_benchmark_vnncomp_all_2026_can_explicitly_retain_diagnostic_surface(
    tmp_path: Path,
) -> None:
    _install_runner_harness(
        tmp_path,
        child_body="#!/bin/sh\nexit 0\n",
    )

    result = _run_runner(
        tmp_path,
        "--year",
        "2026",
        "--diagnostic-beta-crown",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "Surface:    beta-crown diagnostic (not eligible for score claims)" in result.stdout
    assert "ny vnncomp competition wrapper" not in result.stdout


def test_benchmark_vnncomp_all_2026_uses_v2_for_v2_only_extended_category(
    tmp_path: Path,
) -> None:
    category = "isomorphic_acasxu_2026"
    _write_category_fixture(tmp_path, category, year=2026, version="2.0")
    _install_runner_harness(
        tmp_path,
        child_body=textwrap.dedent(
            """\
            #!/bin/sh
            exit 0
            """
        ),
    )

    result = _run_runner(
        tmp_path,
        "--year",
        "2026",
        "--dry-run",
        "--categories",
        category,
    )

    assert result.returncode == 0, f"2026 V2 dry run failed: {result.stderr}"
    supported_section, skipped_section = result.stdout.split("=== Skipped Categories ===", 1)
    category_line = next(
        line for line in supported_section.splitlines() if category in line
    )
    assert "1 instances" in category_line
    assert "NOT FOUND" not in category_line
    assert category not in skipped_section


def test_benchmark_vnncomp_all_2026_never_falls_back_to_unversioned_copy(
    tmp_path: Path,
) -> None:
    category = "isomorphic_acasxu_2026"
    _write_category_fixture(tmp_path, category, year=2026)
    _install_runner_harness(
        tmp_path,
        child_body="#!/bin/sh\nexit 0\n",
    )

    result = _run_runner(
        tmp_path,
        "--year",
        "2026",
        "--dry-run",
        "--categories",
        category,
    )

    assert result.returncode == 0, result.stderr
    category_line = next(line for line in result.stdout.splitlines() if category in line)
    assert "NOT FOUND" in category_line, (
        "a 2.0-only category must not use an unversioned local manifest"
    )
