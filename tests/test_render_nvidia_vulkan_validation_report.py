#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/render_nvidia_vulkan_validation_report.py.

Covers:
- go verdict from fixture data (ny competitive with reference)
- no-go verdict (ny >>5x slower than reference)
- blocked verdict (Vulkan not confirmed)
- blocked verdict (reference comparator unavailable)
- cersyve totals computation from CSV rows
- manifest schema validation
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
SCRIPT = REPO_ROOT / "scripts" / "render_nvidia_vulkan_validation_report.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from render_nvidia_vulkan_validation_report import (
    _compute_verdict,
    compute_cersyve_totals,
    load_csv_rows,
    load_manifest,
    render_report,
)


def _load_manifest(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


class TestComputeCersyveTotals:
    """Totals computation from cersyve compare-backends rows."""

    def test_normal_rows(self):
        rows = load_csv_rows(FIXTURES / "cersyve_compare.csv")
        totals = compute_cersyve_totals(rows)
        assert totals["ny_cpu_cersyve_total"] == pytest.approx(3.50, abs=0.01), (
            f"Expected cpu total ~3.50, got {totals['ny_cpu_cersyve_total']}"
        )
        assert totals["ny_wgpu_cersyve_total"] == pytest.approx(2.80, abs=0.01), (
            f"Expected wgpu total ~2.80, got {totals['ny_wgpu_cersyve_total']}"
        )
        assert totals["backend_speedup_total"] == pytest.approx(1.25, abs=0.01), (
            f"Expected speedup ~1.25, got {totals['backend_speedup_total']}"
        )

    def test_empty_rows(self):
        totals = compute_cersyve_totals([])
        assert totals["ny_cpu_cersyve_total"] is None, "Expected None for empty cpu total"
        assert totals["ny_wgpu_cersyve_total"] is None, "Expected None for empty wgpu total"
        assert totals["backend_speedup_total"] is None, "Expected None for empty speedup"

    def test_non_cersyve_rows_excluded(self):
        rows = [
            {"category": "metaroom_2023", "backend": "cpu", "wall_seconds": "10.0"},
            {"category": "metaroom_2023", "backend": "wgpu", "wall_seconds": "8.0"},
        ]
        totals = compute_cersyve_totals(rows)
        assert totals["ny_cpu_cersyve_total"] is None, (
            "Non-cersyve rows should be excluded from totals"
        )


class TestComputeVerdict:
    """Verdict computation from manifest + totals."""

    def test_go_verdict(self):
        manifest = _load_manifest("go_manifest.json")
        rows = load_csv_rows(FIXTURES / "cersyve_compare.csv")
        totals = compute_cersyve_totals(rows)
        verdict = _compute_verdict(manifest, totals)
        assert verdict == "go", f"Expected go verdict, got {verdict!r}"

    def test_nogo_verdict(self):
        manifest = _load_manifest("nogo_manifest.json")
        rows = load_csv_rows(FIXTURES / "cersyve_compare_slow.csv")
        totals = compute_cersyve_totals(rows)
        verdict = _compute_verdict(manifest, totals)
        assert verdict == "no-go", f"Expected no-go verdict, got {verdict!r}"

    def test_blocked_no_vulkan(self):
        manifest = _load_manifest("blocked_manifest.json")
        verdict = _compute_verdict(manifest, {})
        assert verdict == "blocked", f"Expected blocked verdict, got {verdict!r}"

    def test_blocked_no_reference(self):
        manifest = _load_manifest("blocked_reference_manifest.json")
        rows = load_csv_rows(FIXTURES / "cersyve_compare.csv")
        totals = compute_cersyve_totals(rows)
        verdict = _compute_verdict(manifest, totals)
        assert verdict == "blocked", f"Expected blocked verdict, got {verdict!r}"


class TestRenderReport:
    """Full report rendering from fixture data."""

    def test_go_report_contains_verdict(self):
        manifest = _load_manifest("go_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**go**" in report, "Report missing go verdict marker"
        assert "## Verdict" in report, "Report missing Verdict section"
        assert "## Summary" in report, "Report missing Summary section"
        assert "## Derived Comparison" in report, "Report missing Derived Comparison"
        assert "ny_cpu_cersyve_total" in report, "Report missing cpu total"
        assert "ny_wgpu_cersyve_total" in report, "Report missing wgpu total"

    def test_blocked_report_contains_blocker(self):
        manifest = _load_manifest("blocked_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**blocked**" in report, "Report missing blocked verdict marker"
        assert "Vulkan" in report, "Report missing Vulkan mention"

    def test_blocked_reference_report(self):
        manifest = _load_manifest("blocked_reference_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**blocked**" in report, "Report missing blocked verdict marker"
        assert "alpha-beta-CROWN" in report, "Report missing reference blocker"

    def test_nogo_report(self):
        manifest = _load_manifest("nogo_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**no-go**" in report, "Report missing no-go verdict marker"

    def test_report_section_order(self):
        manifest = _load_manifest("go_manifest.json")
        report = render_report(manifest, FIXTURES)
        sections = [
            "## Summary",
            "## Commands",
            "## Artifacts",
            "## Host Facts",
            "## Derived Comparison",
            "## Divergence Gate",
            "## Verdict",
        ]
        positions = [report.index(s) for s in sections]
        assert positions == sorted(positions), (
            f"Sections out of order: {list(zip(sections, positions))}"
        )


class TestManifestValidation:
    """Manifest loading and schema validation."""

    def test_valid_manifest_loads(self):
        manifest = load_manifest(FIXTURES / "go_manifest.json")
        assert manifest["schema"] == "nvidia_vulkan_validation_manifest_v1", (
            f"Unexpected schema: {manifest['schema']}"
        )

    def test_invalid_schema_rejected(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"schema": "wrong_schema_v1"}, f)
            f.flush()
            with pytest.raises(ValueError, match="Unknown manifest schema"):
                load_manifest(Path(f.name))


class TestCLIIntegration:
    """CLI integration via subprocess."""

    def test_cli_go_verdict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--manifest", str(FIXTURES / "go_manifest.json"),
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Renderer failed: {result.stderr}"
            output = Path(tmpdir) / "issue-4359-nvidia-vulkan-validation-current.md"
            assert output.exists(), f"Expected report at {output}"
            content = output.read_text()
            assert "**go**" in content, "CLI output missing go verdict"

    def test_cli_blocked_verdict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--manifest", str(FIXTURES / "blocked_manifest.json"),
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Renderer failed: {result.stderr}"
            output = Path(tmpdir) / "issue-4359-nvidia-vulkan-validation-current.md"
            content = output.read_text()
            assert "**blocked**" in content, "CLI output missing blocked verdict"
