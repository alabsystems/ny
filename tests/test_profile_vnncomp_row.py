# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import os
import subprocess
import textwrap
from pathlib import Path

from tests.test_benchmark_vnncomp_script import REPO_ROOT, _benchmark_root, _write_fake_ny


PROFILE_SCRIPT_PATH = REPO_ROOT / "scripts" / "profile_vnncomp_row.sh"
PROFILE_TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))


def _write_profile_fixture(
    tmp_path: Path,
    category: str,
    *,
    suite: str = "vnncomp2025",
) -> tuple[Path, Path, Path]:
    category_dir = _benchmark_root(tmp_path, suite) / category
    model_path = category_dir / "onnx" / "profile_model.onnx"
    property_path = category_dir / "vnnlib" / "profile_prop.vnnlib"
    preset_path = tmp_path / "configs" / "vnncomp25" / f"{category}.yaml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    property_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"\x08\x01\x12\x03foo")
    property_path.write_text("", encoding="utf-8")
    preset_path.write_text("dummy: preset\n", encoding="utf-8")
    return model_path, property_path, preset_path


def _run_profile_wrapper(
    tmp_path: Path,
    ny_path: Path,
    *,
    timeout_seconds: str,
    sample_early_seconds: str,
    sample_late_seconds: str,
    sample_duration: str = "1",
    notes: str | None = None,
    benchmark_suite: str | None = None,
    source_index: str | None = None,
    category: str = "metaroom_2023",
    fixture_category: str | None = None,
    fixture_suite: str = "vnncomp2025",
    load_output: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    model_path, property_path, preset_path = _write_profile_fixture(
        tmp_path,
        fixture_category or category,
        suite=fixture_suite,
    )
    report_path = tmp_path / "reports" / "benchmarks" / "issue-4291-metaroom-host-profile-current.md"
    output_path = tmp_path / "reports" / "benchmarks" / "metaroom_profile.csv"
    command = [
        "bash",
        str(PROFILE_SCRIPT_PATH),
        "--category",
        category,
        "--model",
        str(model_path.relative_to(tmp_path)),
        "--property",
        str(property_path.relative_to(tmp_path)),
        "--preset",
        str(preset_path.relative_to(tmp_path)),
        "--backend",
        "wgpu",
        "--timeout",
        timeout_seconds,
        "--report-path",
        str(report_path.relative_to(tmp_path)),
        "--output",
        str(output_path.relative_to(tmp_path)),
        "--sample-early-seconds",
        sample_early_seconds,
        "--sample-late-seconds",
        sample_late_seconds,
        "--sample-duration",
        sample_duration,
    ]
    if notes is not None:
        command.extend(["--notes", notes])
    if benchmark_suite is not None:
        command.extend(["--benchmark-suite", benchmark_suite])
    if source_index is not None:
        command.extend(["--source-index", source_index])

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={
            **os.environ,
            "NY_BIN": str(ny_path),
            "SAMPLE_STUB_TEXT": "Call graph:\n  sample target",
            "EXTERNAL_TIMEOUT_SLACK": "1",
            "WATCHDOG_TERM_GRACE": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if not load_output:
        return result, {}
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, rows
    return result, rows[0]


def test_profile_vnncomp_row_emits_normalized_profile_row(tmp_path: Path) -> None:
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VERIFIED\nActual method: beta-crown\nDomains explored: 11\n'
            sleep 0.3
            """
        ),
    )
    result, row = _run_profile_wrapper(
        tmp_path,
        ny_path,
        timeout_seconds="2",
        sample_early_seconds="0.05",
        sample_late_seconds="0.05",
        notes="bounded sample",
    )
    assert result.returncode == 0, f"profile wrapper failed: {result.stderr}"
    assert row["schema_version"] == "backend_benchmark_row_v1", row
    assert row["lane"] == "metaroom_host_profile", row
    assert row["subject_kind"] == "vnncomp_instance", row
    assert row["subject_id"] == row["comparison_key"], row
    assert row["subject_id"].startswith(
        "vnncomp2025::metaroom_2023::benchmarks/vnncomp2025/benchmarks/metaroom_2023/onnx/profile_model.onnx::"
    ), row
    assert row["backend"] == "wgpu", row
    assert row["status"] == "verified", row
    assert row["actual_method"] == "beta-crown", row
    assert row["domains_explored"] == "11", row
    assert row["model_path"].endswith("metaroom_2023/onnx/profile_model.onnx"), row
    assert row["property_path"].endswith("metaroom_2023/vnnlib/profile_prop.vnnlib"), row
    assert row["preset_path"] == "configs/vnncomp25/metaroom_2023.yaml", row
    assert row["profile_artifact_path"] == "reports/benchmarks/issue-4291-metaroom-host-profile-current.md", row
    assert row["notes"] == "samples=early,late; bounded sample", row
    assert f"early_sample={PROFILE_TMP_ROOT}/profile_vnncomp_row_" in result.stdout, (
        result.stdout
    )
    assert f"late_sample={PROFILE_TMP_ROOT}/profile_vnncomp_row_" in result.stdout, (
        result.stdout
    )


def test_profile_vnncomp_row_accepts_benchmark_suite_and_source_index_alignment(
    tmp_path: Path,
) -> None:
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VERIFIED\nActual method: beta-crown\nDomains explored: 11\n'
            sleep 0.1
            """
        ),
    )
    result, row = _run_profile_wrapper(
        tmp_path,
        ny_path,
        timeout_seconds="2",
        sample_early_seconds="0.01",
        sample_late_seconds="0.01",
        benchmark_suite="vnncomp2025",
        source_index="7",
    )

    assert result.returncode == 0, f"profile wrapper failed: {result.stderr}"
    assert (
        row["subject_id"]
        == "vnncomp2025::metaroom_2023::row=7::onnx/profile_model.onnx::vnnlib/profile_prop.vnnlib"
    ), row
    assert row["comparison_key"] == row["subject_id"], row


def test_profile_vnncomp_row_rejects_non_numeric_source_index(tmp_path: Path) -> None:
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VERIFIED\nActual method: beta-crown\nDomains explored: 11\n'
            sleep 0.1
            """
        ),
    )
    result, _ = _run_profile_wrapper(
        tmp_path,
        ny_path,
        timeout_seconds="2",
        sample_early_seconds="0.01",
        sample_late_seconds="0.01",
        benchmark_suite="vnncomp2025",
        source_index="row7",
        load_output=False,
    )

    assert result.returncode != 0, result
    assert "--source-index must be a positive integer" in result.stderr, result.stderr


def test_profile_vnncomp_row_rejects_unsafe_time_values_before_starting_solver(
    tmp_path: Path,
) -> None:
    started_marker = tmp_path / "solver-started"
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            touch "$(dirname "$0")/solver-started"
            printf 'Status: VERIFIED\nDomains explored: 1\n'
            """
        ),
    )

    cases = [
        (
            "nan",
            "1",
            "--timeout must be a positive integer",
        ),
        (
            "1.5",
            "1",
            "--timeout must be a positive integer",
        ),
        (
            "1",
            "inf",
            "--sample-duration must be a finite positive decimal number",
        ),
        (
            "1",
            "1'); __import__('os').system('false'); #",
            "--sample-duration must be a finite positive decimal number",
        ),
    ]
    for timeout_seconds, sample_duration, expected in cases:
        result, _ = _run_profile_wrapper(
            tmp_path,
            ny_path,
            timeout_seconds=timeout_seconds,
            sample_early_seconds="0",
            sample_late_seconds="0",
            sample_duration=sample_duration,
            load_output=False,
        )
        assert result.returncode != 0, result
        assert expected in result.stderr, result.stderr

    assert not started_marker.exists(), "invalid timing input must stop before solver startup"


def test_profile_vnncomp_row_rejects_exact_alignment_category_mismatch(
    tmp_path: Path,
) -> None:
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VERIFIED\nActual method: beta-crown\nDomains explored: 11\n'
            sleep 0.1
            """
        ),
    )
    result, _ = _run_profile_wrapper(
        tmp_path,
        ny_path,
        timeout_seconds="2",
        sample_early_seconds="0.01",
        sample_late_seconds="0.01",
        benchmark_suite="vnncomp2025",
        source_index="7",
        category="wrong_category",
        fixture_category="metaroom_2023",
        load_output=False,
    )

    assert result.returncode != 0, result
    assert "Exact row alignment requires --model to live under category" in result.stderr, (
        result.stderr
    )


def test_profile_vnncomp_row_records_early_only_sample_when_process_finishes_first(
    tmp_path: Path,
) -> None:
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: UNKNOWN\nDomains explored: 0\n'
            sleep 0.1
            """
        ),
    )
    result, row = _run_profile_wrapper(
        tmp_path,
        ny_path,
        timeout_seconds="2",
        sample_early_seconds="0.01",
        sample_late_seconds="0.2",
    )
    assert result.returncode == 0, f"profile wrapper failed: {result.stderr}"
    assert row["status"] == "unknown", row
    assert row["domains_explored"] == "0", row
    assert row["notes"] == "samples=early-only", row
    assert f"early_sample={PROFILE_TMP_ROOT}/profile_vnncomp_row_" in result.stdout, (
        result.stdout
    )
    assert "late_sample=\n" in result.stdout, result.stdout


def test_profile_vnncomp_row_writes_timeout_row_when_watchdog_terminates_ny(
    tmp_path: Path,
) -> None:
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            trap 'exit 143' TERM
            sleep 5
            """
        ),
    )
    result, row = _run_profile_wrapper(
        tmp_path,
        ny_path,
        timeout_seconds="1",
        sample_early_seconds="0.01",
        sample_late_seconds="0.01",
    )
    assert result.returncode == 0, f"profile wrapper failed: {result.stderr}"
    assert row["status"] == "timeout", row
    assert row["domains_explored"] == "0", row
    assert row["notes"] == "samples=early,late", row
