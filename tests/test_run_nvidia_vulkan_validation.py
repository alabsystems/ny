#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NVIDIA/Vulkan validation orchestrator manifest contract.

The shell orchestrator (scripts/run_nvidia_vulkan_validation.sh) is designed to
run on real NVIDIA hosts and shells out to cargo, benchmark scripts, etc. These
tests validate:
1. The manifest JSON schema contract that the orchestrator produces
2. The renderer's ability to consume each manifest variant
3. The shell script's syntactic validity

The orchestrator subprocess tests are intentionally omitted because the cargo
wrapper, PATH-based tool shims, and CI sandbox make reliable fake-binary
subprocess testing infeasible on macOS dev hosts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "benchmark_reports"
    / "nvidia_vulkan_validation"
)
SCRIPT = REPO_ROOT / "scripts" / "run_nvidia_vulkan_validation.sh"
RENDERER = REPO_ROOT / "scripts" / "render_nvidia_vulkan_validation_report.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from render_nvidia_vulkan_validation_report import (
    _compute_verdict,
    compute_cersyve_totals,
    load_manifest,
    render_report,
)


class TestShellScriptSyntax:
    """The orchestrator shell script is syntactically valid."""

    def test_bash_n_passes(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


class TestManifestContract:
    """The manifest JSON schema is consumed correctly by the renderer."""

    REQUIRED_KEYS = {
        "schema",
        "verdict",
        "blocker",
        "host_info_path",
        "measure_log_path",
        "measure_csv_path",
        "vulkan_confirmed",
        "adapter_line",
        "compare_backends_cersyve_csv",
        "compare_backends_metaroom_csv",
        "reference_blocker",
        "reference_cersyve_real_seconds",
    }

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "go_manifest.json",
            "nogo_manifest.json",
            "blocked_manifest.json",
            "blocked_reference_manifest.json",
        ],
    )
    def test_fixture_has_all_required_keys(self, fixture_name: str):
        with open(FIXTURES / fixture_name, encoding="utf-8") as f:
            data = json.load(f)
        missing = self.REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing keys in {fixture_name}: {missing}"

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "go_manifest.json",
            "nogo_manifest.json",
            "blocked_manifest.json",
            "blocked_reference_manifest.json",
        ],
    )
    def test_fixture_has_correct_schema(self, fixture_name: str):
        manifest = load_manifest(FIXTURES / fixture_name)
        assert manifest["schema"] == "nvidia_vulkan_validation_manifest_v1", (
            f"Unexpected schema in {fixture_name}: {manifest['schema']}"
        )

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "go_manifest.json",
            "nogo_manifest.json",
            "blocked_manifest.json",
            "blocked_reference_manifest.json",
        ],
    )
    def test_renderer_produces_valid_markdown(self, fixture_name: str):
        manifest = load_manifest(FIXTURES / fixture_name)
        report = render_report(manifest, FIXTURES)
        assert "## Summary" in report, f"Missing Summary section in {fixture_name}"
        assert "## Verdict" in report, f"Missing Verdict section in {fixture_name}"
        assert len(report) > 100, f"Report too short for {fixture_name}: {len(report)} chars"

    def test_non_vulkan_manifest_produces_blocked_verdict(self):
        manifest = load_manifest(FIXTURES / "blocked_manifest.json")
        verdict = _compute_verdict(manifest, {})
        assert verdict == "blocked", f"Expected blocked, got {verdict!r}"

    def test_missing_reference_produces_blocked_verdict(self):
        manifest = load_manifest(FIXTURES / "blocked_reference_manifest.json")
        totals = compute_cersyve_totals([])
        verdict = _compute_verdict(manifest, totals)
        assert verdict == "blocked", f"Expected blocked, got {verdict!r}"

    def test_go_manifest_round_trip(self):
        """Manifest + CSV -> render -> verdict matches expected."""
        manifest = load_manifest(FIXTURES / "go_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**go**" in report, "go manifest round-trip missing go verdict"

    def test_nogo_manifest_round_trip(self):
        manifest = load_manifest(FIXTURES / "nogo_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**no-go**" in report, "nogo manifest round-trip missing no-go verdict"

    def test_blocked_manifest_round_trip(self):
        manifest = load_manifest(FIXTURES / "blocked_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**blocked**" in report, "blocked manifest round-trip missing verdict"


class TestEndToEndCLI:
    """CLI integration: manifest -> renderer -> report file."""

    @pytest.mark.parametrize(
        "fixture_name,expected_verdict",
        [
            ("go_manifest.json", "**go**"),
            ("nogo_manifest.json", "**no-go**"),
            ("blocked_manifest.json", "**blocked**"),
            ("blocked_reference_manifest.json", "**blocked**"),
        ],
    )
    def test_cli_produces_expected_verdict(
        self, fixture_name: str, expected_verdict: str
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(RENDERER),
                    "--manifest",
                    str(FIXTURES / fixture_name),
                    "--output-dir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            output = (
                Path(tmpdir) / "issue-4359-nvidia-vulkan-validation-current.md"
            )
            assert output.exists(), f"Expected report file at {output}"
            content = output.read_text()
            assert expected_verdict in content, (
                f"Expected {expected_verdict!r} in CLI output for {fixture_name}"
            )
