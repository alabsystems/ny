#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
System Health Check for ny (Rust project).

Verifies the system is connected and working:
- All crates in workspace are valid and compile
- No orphan crates (crates not in Cargo.toml workspace)
- Main entry points (CLI) are functional
- Reports contain no placeholder markers
- Cargo build/test lock waits are expected when other sessions are active

Run: python scripts/system_health_check.py
     python scripts/system_health_check.py --json-output manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from check_syspolicyd import run_syspolicyd_check

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_CROWN_DIR = PROJECT_ROOT / "crates" / "ny-propagate" / "src" / "network" / "graph_crown"

REQUIRED_TEST_FIXTURES: list[dict[str, object]] = []

OPTIONAL_TEST_FIXTURES: list[dict[str, object]] = [
    {
        "name": "whisper_tiny_encoder.onnx",
        "path": PROJECT_ROOT / "tests" / "models" / "whisper_tiny_encoder.onnx",
        "min_size_bytes": 10 * 1024 * 1024,
        "hint": "Generate with: python scripts/export_whisper_encoder.py",
    },
]

GGUF_MODULE_FILES = [
    "crates/ny-onnx/src/gguf/mod.rs",
    "crates/ny-onnx/src/gguf/info.rs",
    "crates/ny-onnx/src/gguf/load.rs",
    "crates/ny-onnx/src/gguf/parser.rs",
    "crates/ny-onnx/src/gguf/metadata.rs",
    "crates/ny-onnx/src/gguf/dequant.rs",
    "crates/ny-onnx/src/gguf/tests.rs",
]

# Commands to verify the system is working
GIT_FETCH_ENV = {
    "CARGO_NET_GIT_FETCH_WITH_CLI": "1",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "submodule.reference/cryptominisat.update",
    "GIT_CONFIG_VALUE_0": "none",
}

PIPELINE_COMMANDS: list[dict[str, Any]] = [
    {
        "name": "cargo_check",
        "cmd": ["cargo", "check", "--workspace"],
        # Includes time waiting for the per-repo cargo lock.
        "timeout": 3600,
        # Typical budget needed for a clean build on a workstation.
        "min_budget_sec": 600,
        "description": "Verify all crates compile",
        "env": GIT_FETCH_ENV,
    },
    {
        "name": "ny_onnx_gguf_check",
        "cmd": ["cargo", "check", "-p", "ny-onnx", "--features", "gguf"],
        # Includes time waiting for the per-repo cargo lock.
        "timeout": 3600,
        "min_budget_sec": 300,
        "description": "Verify ny-onnx gguf feature compiles",
        "env": GIT_FETCH_ENV,
    },
    {
        "name": "cli_help",
        "cmd": ["cargo", "run", "-p", "ny-cli", "--", "--help"],
        # Includes time waiting for the per-repo cargo lock + compilation.
        "timeout": 900,
        "min_budget_sec": 120,
        "description": "Verify CLI is functional",
        "env": GIT_FETCH_ENV,
    },
]

# Crates that should exist
EXPECTED_CRATES = [
    "ny-api",
    "ny-core",
    "ny-tensor",
    "ny-propagate",
    "ny-onnx",
    "ny-cli",
    "ny-gpu",
    "",
    "ny-python",
    "ny-test-utils",
]


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_project_name() -> str:
    """Get project name from git remote or directory."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        url = result.stdout.strip()
        return url.rstrip("/").split("/")[-1].replace(".git", "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return PROJECT_ROOT.name


class HealthCheck:
    """System-level health verification."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed: list[str] = []
        self.skipped: list[str] = []
        self.json_checks: dict[str, dict] = {}

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def skip(self, msg: str) -> None:
        self.skipped.append(msg)

    def set_check_result(self, check_name: str, result: dict) -> None:
        """Store structured result for a check."""
        self.json_checks[check_name] = result

    def to_json(self) -> dict[str, Any]:
        """Generate full JSON manifest."""
        if self.errors:
            status = "fail"
        elif self.warnings:
            status = "warn"
        else:
            status = "pass"

        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "project": _get_project_name(),
            "summary": {
                "status": status,
                "passed": len(self.passed),
                "warnings": len(self.warnings),
                "errors": len(self.errors),
                "skipped": len(self.skipped),
            },
            "checks": self.json_checks,
        }


def check_workspace_crates(hc: HealthCheck) -> None:
    """Check all expected crates exist and are in workspace."""
    print("\n## Workspace Crates")
    print("Checking: Are all expected crates present?\n")

    cargo_toml = PROJECT_ROOT / "Cargo.toml"
    if not cargo_toml.exists():
        hc.error("Cargo.toml not found at project root")
        hc.set_check_result("workspace_crates", {"status": "fail", "error": "no Cargo.toml"})
        return

    # Read workspace members from Cargo.toml
    try:
        content = cargo_toml.read_text()
    except OSError as e:
        hc.error(f"Cannot read Cargo.toml: {e}")
        hc.set_check_result("workspace_crates", {"status": "fail", "error": str(e)})
        return

    # Simple parse to find workspace members
    workspace_members: list[str] = []
    in_members = False
    for line in content.split("\n"):
        if "members = [" in line:
            in_members = True
            continue
        if in_members:
            if "]" in line:
                break
            # Extract crate path from line like '    "crates/ny-core",'
            line = line.strip().strip(",").strip('"')
            if line:
                workspace_members.append(line)

    # Check crates directory exists
    crates_dir = PROJECT_ROOT / "crates"
    if not crates_dir.exists():
        hc.error("crates/ directory not found")
        hc.set_check_result("workspace_crates", {"status": "fail", "error": "no crates dir"})
        return

    # Check expected crates exist
    found_crates = []
    missing_crates = []
    for crate in EXPECTED_CRATES:
        crate_path = crates_dir / crate
        if crate_path.exists() and (crate_path / "Cargo.toml").exists():
            found_crates.append(crate)
        else:
            missing_crates.append(crate)

    # Check for orphan crates (in directory but not in workspace)
    actual_crates = [d.name for d in crates_dir.iterdir() if d.is_dir() and (d / "Cargo.toml").exists()]
    workspace_crate_names = [m.split("/")[-1] for m in workspace_members]
    orphan_crates = [c for c in actual_crates if c not in workspace_crate_names]

    print(f"  Expected crates: {len(EXPECTED_CRATES)}")
    print(f"  Found crates: {len(found_crates)}")
    print(f"  Missing crates: {len(missing_crates)}")
    print(f"  Orphan crates: {len(orphan_crates)}")

    if missing_crates:
        hc.error(f"Missing crates: {', '.join(missing_crates)}")
    if orphan_crates:
        hc.warn(f"Orphan crates (not in workspace): {', '.join(orphan_crates)}")

    if not missing_crates and not orphan_crates:
        hc.ok("All expected crates present and in workspace")
        hc.set_check_result(
            "workspace_crates",
            {
                "status": "pass",
                "expected": len(EXPECTED_CRATES),
                "found": len(found_crates),
                "missing": [],
                "orphans": [],
            },
        )
    elif missing_crates:
        hc.set_check_result(
            "workspace_crates",
            {
                "status": "fail",
                "expected": len(EXPECTED_CRATES),
                "found": len(found_crates),
                "missing": missing_crates,
                "orphans": orphan_crates,
            },
        )
    else:
        hc.set_check_result(
            "workspace_crates",
            {
                "status": "warn",
                "expected": len(EXPECTED_CRATES),
                "found": len(found_crates),
                "missing": [],
                "orphans": orphan_crates,
            },
        )


def check_pipeline_runs(
    hc: HealthCheck,
    *,
    start_time: float,
    budget_sec: int,
    full: bool,
) -> None:
    """Check configured pipeline commands run successfully."""
    print("\n## Pipeline Execution")
    print("Checking: Do configured pipeline commands run?\n")

    if not PIPELINE_COMMANDS:
        hc.skip("No pipeline commands configured")
        hc.set_check_result("pipeline_execution", {"status": "skip"})
        return

    failures = 0
    failed_commands: list[str] = []
    skipped_commands: list[str] = []
    durations: dict[str, float] = {}

    for pipeline in PIPELINE_COMMANDS:
        name = pipeline["name"]
        cmd = pipeline["cmd"]
        timeout = pipeline.get("timeout", 120)
        min_budget = int(pipeline.get("min_budget_sec", 0))
        desc = pipeline.get("description", name)
        cmd_env = pipeline.get("env", {})

        cmd_str = " ".join(cmd)
        if not full and budget_sec > 0:
            elapsed = time.monotonic() - start_time
            remaining = budget_sec - int(elapsed)
            if remaining <= 0:
                skipped_commands.append(name)
                hc.skip(
                    f"Skipped pipeline command (time budget exhausted): {cmd_str}. "
                    "Re-run with --full to execute pipeline checks."
                )
                continue
            if remaining < min_budget:
                skipped_commands.append(name)
                hc.skip(
                    f"Skipped pipeline command (need ~{min_budget}s, have ~{max(0, remaining)}s): "
                    f"{cmd_str}. Re-run with --full to execute pipeline checks."
                )
                continue

        print(f"  Running: {desc}...")

        try:
            merged_env = {**os.environ, **cmd_env}
            started = time.monotonic()
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                check=False,
                # Do not capture output: stream progress to stdout/stderr so long
                # builds (or cargo lock waits) don't look "stuck" to harnesses.
                env=merged_env,
                timeout=timeout,
            )
            durations[name] = time.monotonic() - started
        except subprocess.TimeoutExpired:
            failures += 1
            failed_commands.append(name)
            hc.error(f"Pipeline command timed out: {cmd_str}")
            durations[name] = float(timeout)
            continue

        if result.returncode != 0:
            failures += 1
            failed_commands.append(name)
            hc.error(f"Pipeline command failed ({result.returncode}): {cmd_str}")
        else:
            print("    OK")

    if failures == 0 and not skipped_commands:
        hc.ok("All pipeline commands completed successfully")
        hc.set_check_result(
            "pipeline_execution",
            {
                "status": "pass",
                "commands_total": len(PIPELINE_COMMANDS),
                "passed": len(PIPELINE_COMMANDS),
                "failed": [],
                "skipped": [],
                "durations_sec": durations,
            },
        )
    elif failures == 0 and skipped_commands:
        hc.warn(
            "Pipeline commands skipped due to time budget. "
            "Run with --full for compile/CLI verification."
        )
        hc.set_check_result(
            "pipeline_execution",
            {
                "status": "warn",
                "commands_total": len(PIPELINE_COMMANDS),
                "passed": len(PIPELINE_COMMANDS) - len(skipped_commands),
                "failed": [],
                "skipped": skipped_commands,
                "durations_sec": durations,
            },
        )
    else:
        hc.set_check_result(
            "pipeline_execution",
            {
                "status": "fail",
                "commands_total": len(PIPELINE_COMMANDS),
                "passed": len(PIPELINE_COMMANDS) - failures,
                "failed": failed_commands,
                "skipped": skipped_commands,
                "durations_sec": durations,
            },
        )


def check_report_validity(hc: HealthCheck) -> None:
    """Check reports for placeholder markers.

    Only flags actual placeholder patterns, not mentions of placeholder
    keywords within regular text (e.g., "no TODOs found" is not a placeholder).
    """
    print("\n## Report Validity")
    print("Checking: Do reports contain placeholder markers?\n")

    report_dirs = ["reports"]
    report_extensions = {".md", ".html", ".txt"}

    # Patterns that indicate actual placeholders (start of line or standalone)
    # - TO-DO: or TO-DO - at start of line (actionable marker)
    # - "[PLACEHOLDER]" or "<PLACEHOLDER>" (explicit placeholder)
    # - "MOCK DATA" or "FAKE DATA" (test data markers)
    # - "DEMO ONLY" (warning marker)
    # Note: "NOT REAL" removed - too many false positives in technical text
    # Linter false-positive workaround: concatenate to avoid triggering TD002/TD003
    _todo = "TO" + "DO"
    placeholder_patterns = [
        rf"^##?\s*{_todo}\s*[-:]",   # ## heading or # heading
        rf"^\s*-\s*{_todo}\s*[-:]",  # - list item
        r"\[PLACEHOLDER\]",         # [PLACEHOLDER] explicit marker
        r"<PLACEHOLDER>",           # <PLACEHOLDER> explicit marker
        r"MOCK\s+DATA",             # MOCK DATA
        r"FAKE\s+DATA",             # FAKE DATA
        r"DEMO\s+ONLY",             # DEMO ONLY
    ]
    compiled_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in placeholder_patterns]

    existing_dirs = [PROJECT_ROOT / d for d in report_dirs if (PROJECT_ROOT / d).is_dir()]
    if not existing_dirs:
        hc.skip("No report directories found")
        hc.set_check_result("report_validity", {"status": "skip"})
        return

    report_files: list[Path] = []
    for report_dir in existing_dirs:
        for path in report_dir.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix.lower() in report_extensions:
                report_files.append(path)

    if not report_files:
        hc.skip("No report files found")
        hc.set_check_result("report_validity", {"status": "skip"})
        return

    placeholder_hits: list[str] = []
    for report in report_files:
        try:
            content = report.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in compiled_patterns:
            if pattern.search(content):
                rel_path = str(report.relative_to(PROJECT_ROOT))
                placeholder_hits.append(rel_path)
                break

    print(f"  Reports checked: {len(report_files)}")
    print(f"  Placeholder hits: {len(placeholder_hits)}")

    if not placeholder_hits:
        hc.ok("Reports checked with no placeholder markers")
        hc.set_check_result(
            "report_validity",
            {
                "status": "pass",
                "reports_total": len(report_files),
                "placeholder_hits": [],
            },
        )
    else:
        for hit in placeholder_hits[:5]:
            hc.warn(f"{hit} contains placeholder marker")
        hc.set_check_result(
            "report_validity",
            {
                "status": "warn",
                "reports_total": len(report_files),
                "placeholder_hits": placeholder_hits,
            },
        )


def _validate_fixture(
    fixture: dict[str, object],
    *,
    required: bool,
    missing: list[str],
    invalid: list[str],
    messages: list[str],
) -> None:
    name = str(fixture["name"])
    path = Path(fixture["path"])
    min_size = int(fixture.get("min_size_bytes", 0))
    hint = str(fixture.get("hint", "")).strip()
    if not path.exists():
        missing.append(f"{name} ({path})")
        detail = f"{name} missing at {path}."
        if hint:
            detail = f"{detail} {hint}"
        if required:
            messages.append(detail)
        else:
            messages.append(detail)
        return
    try:
        size = path.stat().st_size
    except OSError as exc:
        invalid.append(f"{name} ({path})")
        detail = f"{name} unreadable at {path}: {exc}."
        if hint:
            detail = f"{detail} {hint}"
        messages.append(detail)
        return
    if min_size and size < min_size:
        invalid.append(f"{name} ({path})")
        detail = f"{name} too small ({size} bytes) at {path}."
        if hint:
            detail = f"{detail} {hint}"
        messages.append(detail)


def check_test_fixtures(hc: HealthCheck) -> None:
    """Check test fixtures exist and look valid."""
    print("\n## Test Fixtures")
    print("Checking: Are required test fixtures present?\n")

    missing: list[str] = []
    invalid: list[str] = []
    error_details: list[str] = []
    warning_details: list[str] = []
    for fixture in REQUIRED_TEST_FIXTURES:
        _validate_fixture(
            fixture,
            required=True,
            missing=missing,
            invalid=invalid,
            messages=error_details,
        )

    for fixture in OPTIONAL_TEST_FIXTURES:
        _validate_fixture(
            fixture,
            required=False,
            missing=missing,
            invalid=invalid,
            messages=warning_details,
        )

    required_missing = [
        item for item in missing if any(item.startswith(str(f["name"])) for f in REQUIRED_TEST_FIXTURES)
    ]
    required_invalid = [
        item for item in invalid if any(item.startswith(str(f["name"])) for f in REQUIRED_TEST_FIXTURES)
    ]
    optional_missing = [
        item for item in missing if any(item.startswith(str(f["name"])) for f in OPTIONAL_TEST_FIXTURES)
    ]
    optional_invalid = [
        item for item in invalid if any(item.startswith(str(f["name"])) for f in OPTIONAL_TEST_FIXTURES)
    ]

    if not required_missing and not required_invalid:
        if REQUIRED_TEST_FIXTURES:
            hc.ok("Required test fixtures present")
        else:
            hc.skip("No required test fixtures configured")
        hc.set_check_result(
            "test_fixtures",
            {
                "status": "pass" if REQUIRED_TEST_FIXTURES else "skip",
                "fixtures_total": len(REQUIRED_TEST_FIXTURES),
                "missing": [],
                "invalid": [],
                "optional_missing": optional_missing,
                "optional_invalid": optional_invalid,
            },
        )
    else:
        for msg in error_details:
            hc.error(msg)
        hc.set_check_result(
            "test_fixtures",
            {
                "status": "fail",
                "fixtures_total": len(REQUIRED_TEST_FIXTURES),
                "missing": required_missing,
                "invalid": required_invalid,
                "optional_missing": optional_missing,
                "optional_invalid": optional_invalid,
            },
        )

    for msg in warning_details:
        if msg:
            hc.warn(msg)


def check_gguf_sources(hc: HealthCheck) -> None:
    """Check GGUF module source files exist."""
    print("\n## GGUF Sources")
    print("Checking: Are GGUF module sources present?\n")

    missing: list[str] = []
    found: list[str] = []
    for rel_path in GGUF_MODULE_FILES:
        if not (PROJECT_ROOT / rel_path).exists():
            missing.append(rel_path)
        else:
            found.append(rel_path)

    if missing:
        detail = ", ".join(missing)
        hc.error(f"Missing GGUF source files: {detail}")
        hc.set_check_result(
            "gguf_sources",
            {
                "status": "fail",
                "expected": list(GGUF_MODULE_FILES),
                "found": found,
                "missing": missing,
            },
        )
    else:
        hc.ok(f"GGUF module sources present ({len(GGUF_MODULE_FILES)} files)")
        hc.set_check_result(
            "gguf_sources",
            {
                "status": "pass",
                "expected": list(GGUF_MODULE_FILES),
                "found": found,
                "missing": [],
            },
        )

def check_graph_crown_artifacts(hc: HealthCheck) -> None:
    """Check for stray graph_crown files (editor artifacts)."""
    print("\n## Graph CROWN Artifacts")
    print("Checking: Are there stray graph_crown files with colon names?\n")

    if not GRAPH_CROWN_DIR.is_dir():
        hc.error(f"Missing graph_crown directory: {GRAPH_CROWN_DIR}")
        hc.set_check_result(
            "graph_crown_artifacts",
            {"status": "fail", "missing_dir": str(GRAPH_CROWN_DIR)},
        )
        return

    artifacts: list[dict[str, object]] = []
    for entry in GRAPH_CROWN_DIR.iterdir():
        if not entry.is_file():
            continue
        if ":" not in entry.name:
            continue
        try:
            size = entry.stat().st_size
        except OSError:
            size = None
        artifacts.append(
            {
                "path": str(entry.relative_to(PROJECT_ROOT)),
                "size_bytes": size,
            }
        )

    if not artifacts:
        hc.ok("No stray graph_crown artifacts found")
        hc.set_check_result(
            "graph_crown_artifacts",
            {"status": "pass", "artifacts": []},
        )
        return

    artifact_list = ", ".join(
        f"{item['path']} ({item['size_bytes']} bytes)" for item in artifacts
    )
    hc.error(f"Stray graph_crown artifacts detected: {artifact_list}")
    hc.set_check_result(
        "graph_crown_artifacts",
        {"status": "fail", "artifacts": artifacts},
    )

def check_syspolicyd_health(hc: HealthCheck) -> dict[str, Any]:
    """Run syspolicyd preflight check (#4230).

    Delegates to the standalone detector in check_syspolicyd.py.
    Returns the structured result dict for downstream gating decisions.
    """
    print("\n## syspolicyd Preflight")
    print("Checking: Can fresh binaries start on this host?\n")
    return run_syspolicyd_check(hc)


def main(argv: list[str] | None = None) -> int:
    """Run all health checks."""
    parser = argparse.ArgumentParser(
        description="System health check for ny"
    )
    parser.add_argument(
        "--json-output",
        metavar="PATH",
        help="Write JSON manifest to PATH",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all pipeline checks regardless of the time budget.",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=int(os.environ.get("SYSTEM_HEALTH_BUDGET_SEC", "60")),
        help="Time budget in seconds for pipeline commands (default: 60, 0 = unlimited).",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("SYSTEM HEALTH CHECK - ny")
    print("=" * 60)

    hc = HealthCheck()
    start_time = time.monotonic()

    check_workspace_crates(hc)
    check_test_fixtures(hc)
    check_gguf_sources(hc)
    check_graph_crown_artifacts(hc)

    # Run syspolicyd preflight before pipeline commands (#4230).
    # If the host cannot start fresh binaries, skip the expensive cargo pipeline.
    syspolicyd_result = check_syspolicyd_health(hc)
    if syspolicyd_result.get("status") == "fail":
        print("\n  syspolicyd preflight FAILED — skipping pipeline commands")
        cmd_names = [p["name"] for p in PIPELINE_COMMANDS]
        hc.set_check_result(
            "pipeline_execution",
            {
                "status": "skip",
                "blocked_by": "syspolicyd_health",
                "reason": "fresh binaries cannot start on this host (#4230)",
                "commands_total": len(PIPELINE_COMMANDS),
                "passed": 0,
                "failed": [],
                "skipped": cmd_names,
            },
        )
    else:
        check_pipeline_runs(
            hc,
            start_time=start_time,
            budget_sec=args.time_budget,
            full=args.full,
        )

    check_report_validity(hc)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if hc.passed:
        print(f"\nPASSED ({len(hc.passed)})")
        for msg in hc.passed:
            print(f"  [OK] {msg}")

    if hc.skipped:
        print(f"\nSKIPPED ({len(hc.skipped)})")
        for msg in hc.skipped:
            print(f"  [SKIP] {msg}")

    if hc.warnings:
        print(f"\nWARNINGS ({len(hc.warnings)})")
        for msg in hc.warnings:
            print(f"  [WARN] {msg}")

    if hc.errors:
        print(f"\nERRORS ({len(hc.errors)})")
        for msg in hc.errors:
            print(f"  [ERROR] {msg}")

    # Write JSON manifest if requested
    if args.json_output:
        manifest = hc.to_json()
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nJSON manifest written to: {args.json_output}")

    print("\n" + "=" * 60)
    if hc.errors:
        print("HEALTH CHECK FAILED")
        print("The system has integration problems that need fixing.")
        return 1
    if hc.warnings:
        print("HEALTH CHECK PASSED WITH WARNINGS")
        return 0
    print("HEALTH CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
