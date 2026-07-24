# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Regression tests for syspolicyd preflight gate in system_health_check.py (#4230).

Verifies that:
1. When syspolicyd detector returns "fail", pipeline commands are skipped.
2. When syspolicyd detector returns "warn" or "pass", pipeline proceeds normally.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _make_syspolicyd_side_effect(
    status: str, cpu: float, canary_ok: bool,
):
    """Create a side_effect that mimics run_syspolicyd_check behavior on hc."""
    result = {
        "status": status,
        "syspolicyd_cpu_pct": cpu,
        "canary_ok": canary_ok,
        "detail": "mocked",
    }

    def side_effect(hc):
        if hc is not None:
            if status == "fail":
                hc.error("syspolicyd preflight fail (mocked)")
            elif status == "warn":
                hc.warn("syspolicyd preflight warn (mocked)")
            else:
                hc.ok("syspolicyd healthy (mocked)")
            hc.set_check_result("syspolicyd_health", result)
        return result

    return side_effect


def _run_main_with_mock_syspolicyd(
    syspolicyd_status: str,
    *,
    cpu: float = 0.0,
    canary_ok: bool = True,
) -> tuple[int, dict, bool]:
    """Run system_health_check.main() with a mocked syspolicyd check.

    Returns (exit_code, json_manifest, pipeline_called).
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name

    try:
        import system_health_check

        side_effect = _make_syspolicyd_side_effect(
            syspolicyd_status, cpu, canary_ok,
        )
        with patch.object(
            system_health_check,
            "run_syspolicyd_check",
            side_effect=side_effect,
        ), patch.object(
            system_health_check,
            "check_pipeline_runs",
        ) as mock_pipeline:
            exit_code = system_health_check.main(
                ["--json-output", json_path, "--time-budget", "0"]
            )
            manifest = json.loads(Path(json_path).read_text())
            pipeline_called = mock_pipeline.called
    finally:
        Path(json_path).unlink(missing_ok=True)

    return exit_code, manifest, pipeline_called


class TestSyspolicydPreflightGate:
    """Tests for the syspolicyd preflight gate wired into system_health_check."""

    def test_fail_skips_pipeline_and_exits_nonzero(self):
        """When syspolicyd returns fail, pipeline is skipped and exit != 0."""
        exit_code, manifest, pipeline_called = _run_main_with_mock_syspolicyd(
            "fail", cpu=90.0, canary_ok=False,
        )

        assert exit_code == 1, f"Expected exit 1 on syspolicyd fail, got {exit_code}"
        assert not pipeline_called, "check_pipeline_runs should NOT be called on fail"

        # JSON manifest must contain syspolicyd_health
        checks = manifest.get("checks", {})
        assert "syspolicyd_health" in checks, (
            f"Missing syspolicyd_health in checks: {list(checks.keys())}"
        )
        assert checks["syspolicyd_health"]["status"] == "fail", (
            f"Expected syspolicyd_health status 'fail', got {checks['syspolicyd_health']['status']!r}"
        )

        # Pipeline execution must be skip with blocked_by metadata
        assert "pipeline_execution" in checks, (
            f"Missing pipeline_execution in checks: {list(checks.keys())}"
        )
        pe = checks["pipeline_execution"]
        assert pe["status"] == "skip", f"Expected pipeline status 'skip', got {pe['status']!r}"
        assert pe["blocked_by"] == "syspolicyd_health", (
            f"Expected blocked_by 'syspolicyd_health', got {pe['blocked_by']!r}"
        )
        assert pe["passed"] == 0, f"Expected 0 passed commands, got {pe['passed']}"

    def test_warn_still_runs_pipeline(self):
        """When syspolicyd returns warn, pipeline commands still execute."""
        exit_code, manifest, pipeline_called = _run_main_with_mock_syspolicyd(
            "warn", cpu=60.0, canary_ok=True,
        )

        assert pipeline_called, "check_pipeline_runs SHOULD be called on warn"

        checks = manifest.get("checks", {})
        assert "syspolicyd_health" in checks, (
            f"Missing syspolicyd_health in checks on warn: {list(checks.keys())}"
        )

    def test_pass_still_runs_pipeline(self):
        """When syspolicyd returns pass, pipeline commands execute normally."""
        exit_code, manifest, pipeline_called = _run_main_with_mock_syspolicyd(
            "pass", cpu=2.0, canary_ok=True,
        )

        assert pipeline_called, "check_pipeline_runs SHOULD be called on pass"

        checks = manifest.get("checks", {})
        assert "syspolicyd_health" in checks, (
            f"Missing syspolicyd_health in checks on pass: {list(checks.keys())}"
        )
        assert checks["syspolicyd_health"]["status"] == "pass", (
            f"Expected status 'pass', got {checks['syspolicyd_health']['status']!r}"
        )
