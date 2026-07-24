# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import scripts.vnncomp_dry_run as dry_run


REPO_ROOT = Path(__file__).resolve().parent.parent
DRY_RUN_SCRIPT = REPO_ROOT / "scripts" / "vnncomp_dry_run.py"


def _audit_summary(*categories: tuple[str, int]) -> dict:
    return {
        "year": 2025,
        "categories": [
            {"name": name, "instance_count": instance_count}
            for name, instance_count in categories
        ],
    }


def _benchmark_summary(
    *,
    run_scope: str = "full",
    attempted: list[str] | None = None,
    skipped: dict[str, str] | None = None,
    failed: dict[str, dict[str, object]] | None = None,
) -> dict:
    return {
        "run_scope": run_scope,
        "categories": {category: {"total": 1} for category in (attempted or [])},
        "skipped": skipped or {},
        "failed": failed or {},
    }


def _reference_manifest(*categories: str, include_benchmark_provenance: bool = False) -> dict:
    manifest: dict = {
        "categories": list(categories),
        "reference_files": {
            category: {
                "output_path": f"reports/benchmarks/reference/{category}_alpha_beta_crown.csv",
                "instance_count": 1,
            }
            for category in categories
        },
    }
    if include_benchmark_provenance:
        manifest["benchmark_repo_root"] = "/tmp/benchmarks"
        manifest["benchmark_commit"] = "abc123"
    return manifest


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)
    return path


def _install_cli_harness(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    dry_run_copy = scripts_dir / "vnncomp_dry_run.py"
    shutil.copy2(DRY_RUN_SCRIPT, dry_run_copy)

    fake_ny = _write_executable(
        tmp_path / "fake_ny.sh",
        """\
        #!/bin/sh
        cat <<'EOF'
        {
          "year": 2025,
          "categories": [
            {"name": "acasxu_2023", "instance_count": 1}
          ]
        }
        EOF
        """,
    )
    fake_sync = _write_executable(
        scripts_dir / "sync_vnncomp_reference_results.sh",
        """\
        #!/bin/sh
        mkdir -p reports/benchmarks/reference
        cat > reports/benchmarks/reference/manifest.json <<'EOF'
        {
          "categories": ["acasxu_2023"],
          "reference_files": {
            "acasxu_2023": {
              "output_path": "reports/benchmarks/reference/acasxu_2023_alpha_beta_crown.csv",
              "instance_count": 1
            }
          }
        }
        EOF
        """,
    )
    fake_breadth = _write_executable(
        scripts_dir / "benchmark_vnncomp_all.sh",
        """\
        #!/bin/sh
        mkdir -p reports/benchmarks
        cat > reports/benchmarks/vnncomp_summary_fake.json <<'EOF'
        {
          "run_scope": "full",
          "categories": {
            "acasxu_2023": {"total": 1}
          },
          "skipped": {},
          "failed": {}
        }
        EOF
        echo "Summary: reports/benchmarks/vnncomp_summary_fake.json"
        """,
    )
    return dry_run_copy, fake_ny, fake_sync, fake_breadth


def _run_dry_run_cli(
    tmp_path: Path,
    *,
    dry_run_copy: Path,
    fake_ny: Path,
    fake_sync: Path,
    fake_breadth: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(dry_run_copy),
            "--year",
            "2025",
            "--ny-bin",
            str(fake_ny),
            "--sync-script",
            str(fake_sync),
            "--breadth-script",
            str(fake_breadth),
            "--repo-root",
            str(tmp_path / "unused"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_build_dry_run_report_marks_clean_full_run_as_pass() -> None:
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1), ("malbeware", 2)),
        benchmark_summary=_benchmark_summary(attempted=["acasxu_2023", "malbeware"]),
        reference_manifest=_reference_manifest("acasxu_2023", "malbeware"),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )

    assert report["status"] == "pass", report
    assert report["passes_zero_false_verified"] is True, report
    assert report["passes_all_categories"] is True, report
    assert report["missing_reference_categories"] == [], report
    assert report["unaccounted_categories"] == [], report


def test_build_dry_run_report_fails_when_reference_is_missing() -> None:
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1)),
        benchmark_summary=_benchmark_summary(attempted=["acasxu_2023"]),
        reference_manifest=_reference_manifest(),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )

    assert report["status"] == "fail", report
    assert report["missing_reference_categories"] == ["acasxu_2023"], report
    assert report["passes_zero_false_verified"] is False, report


def test_build_dry_run_report_detects_audit_drift_on_full_runs() -> None:
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1), ("malbeware", 1)),
        benchmark_summary=_benchmark_summary(attempted=["acasxu_2023"]),
        reference_manifest=_reference_manifest("acasxu_2023"),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )

    assert report["status"] == "fail", report
    assert report["unaccounted_categories"] == ["malbeware"], report
    assert report["passes_all_categories"] is False, report


def test_build_dry_run_report_fails_when_category_run_failed() -> None:
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1)),
        benchmark_summary=_benchmark_summary(
            failed={"acasxu_2023": {"exit_code": 1, "reason": "validator disagreement"}},
        ),
        reference_manifest=_reference_manifest("acasxu_2023"),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )

    assert report["status"] == "fail", report
    assert report["failed_categories"] == {
        "acasxu_2023": {"exit_code": 1, "reason": "validator disagreement"},
    }, report
    assert report["passes_zero_false_verified"] is False, report


def test_build_dry_run_report_marks_skipped_categories_as_blocked() -> None:
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1), ("vit_2023", 1)),
        benchmark_summary=_benchmark_summary(
            attempted=["acasxu_2023"],
            skipped={"vit_2023": "runtime-limited", "test": "fixture category"},
        ),
        reference_manifest=_reference_manifest("acasxu_2023"),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )

    assert report["status"] == "blocked", report
    assert report["passes_zero_false_verified"] is True, report
    assert report["passes_all_categories"] is False, report


def test_build_dry_run_report_marks_partial_smoke_runs_as_blocked_without_drift() -> None:
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1), ("malbeware", 1)),
        benchmark_summary=_benchmark_summary(run_scope="partial", attempted=["acasxu_2023"]),
        reference_manifest=_reference_manifest("acasxu_2023"),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )

    assert report["status"] == "blocked", report
    assert report["unaccounted_categories"] == [], report
    assert report["passes_zero_false_verified"] is True, report
    assert report["passes_all_categories"] is False, report


def test_vnncomp_dry_run_cli_writes_timestamped_and_latest_artifacts_for_full_run(
    tmp_path: Path,
) -> None:
    dry_run_copy, fake_ny, fake_sync, fake_breadth = _install_cli_harness(tmp_path)
    result = _run_dry_run_cli(
        tmp_path,
        dry_run_copy=dry_run_copy,
        fake_ny=fake_ny,
        fake_sync=fake_sync,
        fake_breadth=fake_breadth,
    )

    assert result.returncode == 0, f"dry-run script failed: {result.stderr}\n{result.stdout}"
    dry_run_reports = sorted(
        path
        for path in (tmp_path / "reports" / "benchmarks").glob("vnncomp_dry_run_*.json")
        if "_support_" not in path.name and not path.name.endswith("_latest.json")
    )
    assert len(dry_run_reports) == 1, f"expected exactly one timestamped report, got {dry_run_reports!r}"

    payload = json.loads(dry_run_reports[0].read_text(encoding="utf-8"))
    assert payload["status"] == "pass", payload
    assert payload["passes_zero_false_verified"] is True, payload
    assert payload["passes_all_categories"] is True, payload
    assert (tmp_path / "reports" / "benchmarks" / "vnncomp_dry_run_latest.json").exists(), (
        "expected full dry-run to publish vnncomp_dry_run_latest.json",
    )
    assert (tmp_path / "reports" / "benchmarks" / "vnncomp_dry_run_latest.md").exists(), (
        "expected full dry-run to publish vnncomp_dry_run_latest.md",
    )


def test_build_dry_run_report_tolerates_benchmark_provenance_fields() -> None:
    """Manifest with benchmark_repo_root/benchmark_commit doesn't break dry-run."""
    report = dry_run.build_dry_run_report(
        audit_summary=_audit_summary(("acasxu_2023", 1)),
        benchmark_summary=_benchmark_summary(attempted=["acasxu_2023"]),
        reference_manifest=_reference_manifest(
            "acasxu_2023", include_benchmark_provenance=True,
        ),
        benchmark_year=2025,
        support_audit_path=Path("reports/benchmarks/support.json"),
        reference_manifest_path=Path("reports/benchmarks/reference/manifest.json"),
        benchmark_summary_path=Path("reports/benchmarks/summary.json"),
        recorded_timestamp="2026-03-18T12:00:00Z",
    )
    assert report["status"] == "pass", report
    assert report["passes_zero_false_verified"] is True, report
