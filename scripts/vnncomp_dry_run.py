#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Run the Phase 5A.5 VNN-COMP dry-run gate."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "benchmarks"
REFERENCE_DIR = REPORT_DIR / "reference"
SUMMARY_PATTERN = re.compile(r"^Summary:\s+(?P<path>.+)$", re.MULTILINE)
TRACKER_YEAR = int(os.environ.get("TRACKER_YEAR", "2025"))
logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def timestamp_token() -> str:
    """Return a filesystem-safe timestamp token."""
    return utc_now().strftime("%Y%m%d_%H%M%S")


def recorded_at() -> str:
    """Return the current UTC time in the report format."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def display_path(path: Path) -> str:
    """Render a path relative to the repo root when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(raw_path: str | Path) -> Path:
    """Resolve a repo-relative or absolute path."""
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture its output."""
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def require_success(process: subprocess.CompletedProcess[str], context: str) -> None:
    """Raise a helpful error when a subprocess fails."""
    if process.returncode == 0:
        return

    details = [f"{context} failed with exit code {process.returncode}."]
    if process.stdout.strip():
        details.append(f"stdout:\n{process.stdout.strip()}")
    if process.stderr.strip():
        details.append(f"stderr:\n{process.stderr.strip()}")
    raise RuntimeError("\n\n".join(details))


def parse_summary_path(output: str) -> Path:
    """Extract the breadth summary path from benchmark_vnncomp_all.sh output."""
    match = SUMMARY_PATTERN.search(output)
    if match is None:
        raise ValueError("breadth runner output did not contain a 'Summary: <path>' line")
    return resolve_repo_path(match.group("path").strip())


def run_support_audit(*, ny_bin: str, year: int, support_audit_path: Path) -> dict[str, Any]:
    """Capture `ny vnncomp-audit` JSON output on disk and in memory."""
    audit_process = run_command([ny_bin, "vnncomp-audit", "--year", str(year), "--json"])
    require_success(audit_process, "ny vnncomp-audit")
    support_audit_path.write_text(audit_process.stdout, encoding="utf-8")
    return json.loads(audit_process.stdout)


def sync_reference_manifest(*, repo_root: str, tool: str, year: int, sync_script: str) -> dict[str, Any]:
    """Refresh official references and load the manifest."""
    sync_process = run_command(
        [
            "bash",
            sync_script,
            "--repo-root",
            os.path.expanduser(repo_root),
            "--tool",
            tool,
            "--year",
            str(year),
        ],
    )
    require_success(sync_process, "reference sync")

    reference_manifest_path = REFERENCE_DIR / "manifest.json"
    if not reference_manifest_path.exists():
        raise RuntimeError(
            f"reference sync completed without writing {display_path(reference_manifest_path)}",
        )
    return load_json(reference_manifest_path)


def run_breadth_benchmark(
    *,
    breadth_script: str,
    year: int,
    categories: str | None,
    limit: int | None,
    start_from: str | None,
) -> tuple[dict[str, Any], Path]:
    """Run the existing breadth benchmark and load its summary."""
    breadth_command = ["bash", breadth_script, "--year", str(year)]
    if categories:
        breadth_command.extend(["--categories", categories])
    if limit is not None:
        breadth_command.extend(["--limit", str(limit)])
    if start_from:
        breadth_command.extend(["--start-from", start_from])

    breadth_process = run_command(breadth_command)
    require_success(breadth_process, "VNN-COMP breadth benchmark")
    benchmark_summary_path = parse_summary_path(breadth_process.stdout)
    return load_json(benchmark_summary_path), benchmark_summary_path


def discover_nonempty_audit_categories(audit_summary: Mapping[str, Any]) -> list[str]:
    """Return sorted non-empty category names from ny vnncomp-audit."""
    categories = []
    for category in audit_summary.get("categories", []):
        name = category.get("name")
        if not name or name == "test":
            continue
        if int(category.get("instance_count", 0) or 0) <= 0:
            continue
        categories.append(str(name))
    return sorted(set(categories))


def manifest_reference_categories(reference_manifest: Mapping[str, Any]) -> set[str]:
    """Return the set of categories covered by the reference manifest."""
    reference_files = reference_manifest.get("reference_files")
    if isinstance(reference_files, Mapping):
        return {str(category) for category in reference_files.keys()}
    categories = reference_manifest.get("categories", [])
    return {str(category) for category in categories}


def build_dry_run_report(
    *,
    audit_summary: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    benchmark_year: int,
    support_audit_path: Path,
    reference_manifest_path: Path,
    benchmark_summary_path: Path,
    recorded_timestamp: str,
) -> dict[str, Any]:
    """Build the canonical dry-run gate report."""
    run_scope = str(benchmark_summary.get("run_scope", "full"))
    attempted_categories = sorted(benchmark_summary.get("categories", {}).keys())
    skipped_categories = dict(benchmark_summary.get("skipped", {}))
    failed_categories = dict(benchmark_summary.get("failed", {}))
    reference_categories = manifest_reference_categories(reference_manifest)

    missing_reference_categories = sorted(
        category for category in attempted_categories if category not in reference_categories
    )

    unaccounted_categories: list[str] = []
    if run_scope == "full":
        accounted_categories = (
            set(attempted_categories)
            | set(skipped_categories.keys())
            | set(failed_categories.keys())
        )
        unaccounted_categories = sorted(
            category
            for category in discover_nonempty_audit_categories(audit_summary)
            if category not in accounted_categories
        )

    non_test_skipped = {
        category: reason
        for category, reason in skipped_categories.items()
        if category != "test"
    }
    passes_zero_false_verified = (
        not failed_categories
        and not missing_reference_categories
        and not unaccounted_categories
    )
    passes_all_categories = (
        run_scope == "full"
        and not non_test_skipped
        and not unaccounted_categories
    )

    if failed_categories or missing_reference_categories or unaccounted_categories:
        status = "fail"
    elif passes_all_categories:
        status = "pass"
    else:
        status = "blocked"

    return {
        "benchmark": "vnncomp",
        "report_kind": "dry_run_gate",
        "benchmark_year": benchmark_year,
        "run_scope": run_scope,
        "status": status,
        "recorded_at": recorded_timestamp,
        "support_audit_path": display_path(support_audit_path),
        "reference_manifest_path": display_path(reference_manifest_path),
        "benchmark_summary_path": display_path(benchmark_summary_path),
        "attempted_categories": attempted_categories,
        "skipped_categories": skipped_categories,
        "failed_categories": failed_categories,
        "missing_reference_categories": missing_reference_categories,
        "unaccounted_categories": unaccounted_categories,
        "passes_zero_false_verified": passes_zero_false_verified,
        "passes_all_categories": passes_all_categories,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the human-readable dry-run report."""
    lines = [
        f"# VNN-COMP Dry Run Gate: {str(report['status']).upper()}",
        "",
        f"- Benchmark year: {report['benchmark_year']}",
        f"- Run scope: {report['run_scope']}",
        f"- Recorded at: {report['recorded_at']}",
        f"- Zero false VERIFIED: {'yes' if report['passes_zero_false_verified'] else 'no'}",
        f"- All categories covered: {'yes' if report['passes_all_categories'] else 'no'}",
        "",
        "## Attempted Categories",
    ]

    attempted = list(report.get("attempted_categories", []))
    if attempted:
        lines.extend(f"- `{category}`" for category in attempted)
    else:
        lines.append("- none")

    lines.extend(["", "## Skipped / Blocked Categories"])
    skipped = dict(report.get("skipped_categories", {}))
    if skipped:
        lines.extend(
            f"- `{category}`: {reason}"
            for category, reason in sorted(skipped.items())
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Failure Buckets", "", "### Missing Reference Coverage"])
    missing_references = list(report.get("missing_reference_categories", []))
    if missing_references:
        lines.extend(f"- `{category}`" for category in missing_references)
    else:
        lines.append("- none")

    lines.extend(["", "### Category Failures"])
    failed_categories = dict(report.get("failed_categories", {}))
    if failed_categories:
        for category, details in sorted(failed_categories.items()):
            exit_code = details.get("exit_code", "unknown")
            reason = details.get("reason", "unknown")
            lines.append(f"- `{category}`: exit {exit_code} ({reason})")
    else:
        lines.append("- none")

    lines.extend(["", "### Audit Drift"])
    unaccounted = list(report.get("unaccounted_categories", []))
    if unaccounted:
        lines.extend(f"- `{category}`" for category in unaccounted)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Source Artifacts",
            f"- Support audit: `{report['support_audit_path']}`",
            f"- Reference manifest: `{report['reference_manifest_path']}`",
            f"- Benchmark summary: `{report['benchmark_summary_path']}`",
            "",
        ],
    )
    return "\n".join(lines)


def write_report_artifacts(
    *,
    report: Mapping[str, Any],
    stamp: str,
    run_scope: str,
    year: int,
) -> tuple[Path, Path]:
    """Persist timestamped and canonical dry-run artifacts."""
    dry_run_json_path = REPORT_DIR / f"vnncomp_dry_run_{stamp}.json"
    dry_run_markdown_path = REPORT_DIR / f"vnncomp_dry_run_{stamp}.md"
    dry_run_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dry_run_markdown_path.write_text(render_markdown(report), encoding="utf-8")

    if run_scope == "full" and year == TRACKER_YEAR:
        shutil.copyfile(dry_run_json_path, REPORT_DIR / "vnncomp_dry_run_latest.json")
        shutil.copyfile(dry_run_markdown_path, REPORT_DIR / "vnncomp_dry_run_latest.md")

    return dry_run_json_path, dry_run_markdown_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the VNN-COMP dry-run gate")
    parser.add_argument("--year", type=int, default=2025, help="Benchmark year")
    parser.add_argument("--categories", help="Optional category subset for smoke runs")
    parser.add_argument("--limit", type=int, help="Optional per-category instance limit")
    parser.add_argument("--start-from", help="Optional starting category")
    parser.add_argument(
        "--ny-bin",
        default=os.environ.get("NY_BIN", "./target/release/ny"),
        help="Path to the ny binary (default: ./target/release/ny)",
    )
    parser.add_argument(
        "--repo-root",
        default="~/vnncomp2025_results-ref",
        help="Path to the vnncomp results checkout used for reference sync",
    )
    parser.add_argument(
        "--tool",
        default="alpha_beta_crown",
        help="Tool lane to sync from the reference checkout",
    )
    parser.add_argument(
        "--sync-script",
        default="scripts/sync_vnncomp_reference_results.sh",
        help="Reference-sync script path",
    )
    parser.add_argument(
        "--breadth-script",
        default="scripts/benchmark_vnncomp_all.sh",
        help="Breadth benchmark runner path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dry-run gate and write timestamped artifacts."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = timestamp_token()
    current_recorded_at = recorded_at()
    support_audit_path = REPORT_DIR / f"vnncomp_dry_run_support_{stamp}.json"

    audit_summary = run_support_audit(
        ny_bin=args.ny_bin,
        year=args.year,
        support_audit_path=support_audit_path,
    )
    reference_manifest = sync_reference_manifest(
        repo_root=args.repo_root,
        tool=args.tool,
        year=args.year,
        sync_script=args.sync_script,
    )
    reference_manifest_path = REFERENCE_DIR / "manifest.json"
    benchmark_summary, benchmark_summary_path = run_breadth_benchmark(
        breadth_script=args.breadth_script,
        year=args.year,
        categories=args.categories,
        limit=args.limit,
        start_from=args.start_from,
    )

    report = build_dry_run_report(
        audit_summary=audit_summary,
        benchmark_summary=benchmark_summary,
        reference_manifest=reference_manifest,
        benchmark_year=args.year,
        support_audit_path=support_audit_path,
        reference_manifest_path=reference_manifest_path,
        benchmark_summary_path=benchmark_summary_path,
        recorded_timestamp=current_recorded_at,
    )

    dry_run_json_path, dry_run_markdown_path = write_report_artifacts(
        report=report,
        stamp=stamp,
        run_scope=str(benchmark_summary.get("run_scope", "full")),
        year=args.year,
    )

    logger.info("Status: %s", report["status"])
    logger.info("JSON: %s", display_path(dry_run_json_path))
    logger.info("Markdown: %s", display_path(dry_run_markdown_path))

    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
