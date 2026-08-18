#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/render_backend_benchmark_report.py (Packets A0 + B).

Covers:
- Golden output comparison for valid compare-backends input
- Metadata validation (unknown keys, empty sections, row-fact keys, bad dates)
- CSV pair integrity (duplicate backends, half-pairs, identity mismatches)
- Unsupported lane rejection
- CLI integration via subprocess
- Profile-lane rendering, validation, and fail-closed rejection (Packet B)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "benchmark_reports" / "compare_backends"
PROFILE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "benchmark_reports" / "profile_lane"
SCRIPT = REPO_ROOT / "scripts" / "render_backend_benchmark_report.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _benchmark_report_helpers import (
    render_profile_report,
    render_report,
    validate_metadata,
    validate_profile_metadata,
    validate_profile_rows,
    validate_rows,
)
from render_backend_benchmark_report import load_csv_rows


def _load_valid_meta() -> dict:
    with open(FIXTURES / "valid_metadata.json") as f:
        return json.load(f)


def _load_rows(filename: str) -> list[dict[str, str]]:
    return load_csv_rows([FIXTURES / filename])


def _run_cli(csv_filename: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "issue-4282-test-current.md"
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--metadata",
                str(FIXTURES / "valid_metadata.json"),
                "--csv",
                str(FIXTURES / csv_filename),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )


class TestValidateMetadata:
    """Metadata validation for benchmark_report_metadata_v1."""

    def test_valid_metadata_passes(self):
        meta = _load_valid_meta()
        errors = validate_metadata(meta, Path("issue-4282-gpu-backend-delta-current.md"))
        assert errors == [], f"Expected no errors, got {errors}"

    def test_unknown_keys_rejected(self):
        with open(FIXTURES / "invalid_metadata_unknown_keys.json") as f:
            meta = json.load(f)
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("unknown metadata keys" in e for e in errors), f"Expected unknown key error, got {errors}"
        assert any("extra_field" in e for e in errors), f"Expected extra_field mentioned, got {errors}"

    def test_row_fact_keys_rejected(self):
        with open(FIXTURES / "invalid_metadata_unknown_keys.json") as f:
            meta = json.load(f)
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("row-fact keys" in e for e in errors), f"Expected row-fact error, got {errors}"

    def test_empty_sections_rejected(self):
        with open(FIXTURES / "invalid_metadata_empty_sections.json") as f:
            meta = json.load(f)
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("summary_lines must not be empty" in e for e in errors), f"Missing summary error: {errors}"
        assert any("divergence_notes must not be empty" in e for e in errors), f"Missing divergence error: {errors}"
        assert any("verdict_lines must not be empty" in e for e in errors), f"Missing verdict error: {errors}"

    def test_wrong_schema_version_rejected(self):
        meta = _load_valid_meta()
        meta["schema_version"] = "wrong_v2"
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("schema_version" in e for e in errors), f"Expected schema_version error, got {errors}"

    def test_bad_date_format_rejected(self):
        meta = _load_valid_meta()
        meta["report_date"] = "March 21 2026"
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("YYYY-MM-DD" in e for e in errors), f"Expected date format error, got {errors}"

    def test_non_integer_issue_rejected(self):
        meta = _load_valid_meta()
        meta["issue"] = "4282"
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("issue must be an integer" in e for e in errors), f"Expected integer error, got {errors}"

    def test_issue_filename_mismatch_rejected(self):
        meta = _load_valid_meta()
        errors = validate_metadata(meta, Path("issue-9999-wrong-name-current.md"))
        assert any("does not match" in e for e in errors), f"Expected filename mismatch, got {errors}"

    def test_profile_findings_rejected(self):
        meta = _load_valid_meta()
        meta["profile_findings"] = [{"heading": "h", "bullets": ["b"]}]
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("profile_findings" in e for e in errors), f"Expected profile_findings error, got {errors}"

    def test_missing_command_fields_rejected(self):
        meta = _load_valid_meta()
        meta["commands"] = [{"label": "only label"}]
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("'label' and 'shell'" in e for e in errors), f"Expected command field error, got {errors}"

    def test_missing_artifact_fields_rejected(self):
        meta = _load_valid_meta()
        meta["artifact_notes"] = [{"path": "only path"}]
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("'label' and 'path'" in e for e in errors), f"Expected artifact field error, got {errors}"

    def test_report_notes_empty_bullets_rejected(self):
        meta = _load_valid_meta()
        meta["report_notes"] = [{"heading": "H", "bullets": []}]
        errors = validate_metadata(meta, Path("issue-4282-test.md"))
        assert any("non-empty 'bullets'" in e for e in errors), f"Expected empty bullets error, got {errors}"


class TestValidateRows:
    """CSV row integrity for compare-backends lanes."""

    def test_valid_rows_pass(self):
        rows = _load_rows("valid_rows.csv")
        errors = validate_rows(rows)
        assert errors == [], f"Expected no errors, got {errors}"

    def test_duplicate_backend_rejected(self):
        rows = _load_rows("duplicate_backend_rows.csv")
        errors = validate_rows(rows)
        assert any(
            "expected exactly one cpu row and one wgpu row" in e and "['cpu', 'cpu', 'wgpu']" in e
            for e in errors
        ), f"Expected exact-pair error, got {errors}"

    def test_half_pair_rejected(self):
        rows = _load_rows("half_pair_rows.csv")
        errors = validate_rows(rows)
        assert any(
            "expected exactly one cpu row and one wgpu row" in e and "['cpu']" in e
            for e in errors
        ), f"Expected exact-pair error, got {errors}"

    def test_identity_mismatch_rejected(self):
        rows = _load_rows("identity_mismatch_rows.csv")
        errors = validate_rows(rows)
        assert any("category" in e and "disagrees" in e for e in errors), f"Expected mismatch, got {errors}"

    def test_unsupported_lane_rejected(self):
        rows = _load_rows("unsupported_lane_rows.csv")
        errors = validate_rows(rows)
        assert any("not supported in Packet A0" in e for e in errors), f"Expected lane error, got {errors}"

    def test_empty_csv_rejected(self):
        errors = validate_rows([])
        assert any("no CSV rows found" in e for e in errors), f"Expected empty error, got {errors}"

    def test_mixed_lanes_rejected(self):
        rows_a = _load_rows("valid_rows.csv")
        rows_b = _load_rows("unsupported_lane_rows.csv")
        errors = validate_rows(rows_a + rows_b)
        assert any("mixed lanes" in e for e in errors), f"Expected mixed lanes error, got {errors}"

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("wrong_schema_version_rows.csv", "schema_version must be 'backend_benchmark_row_v1'"),
            ("missing_status_rows.csv", "required field 'status' is empty or missing"),
            ("missing_wall_seconds_rows.csv", "required field 'wall_seconds' is empty or missing"),
            ("mismatched_subject_id_rows.csv", "subject_id must equal comparison_key"),
            ("extra_backend_rows.csv", "backend must be one of ['cpu', 'wgpu']"),
            ("nonnumeric_wall_seconds_rows.csv", "wall_seconds must parse as float"),
            ("missing_lane_header_rows.csv", "required field 'lane' is empty or missing"),
            (
                "missing_comparison_key_header_rows.csv",
                "required field 'comparison_key' is empty or missing",
            ),
            ("missing_backend_header_rows.csv", "required field 'backend' is empty or missing"),
        ],
    )
    def test_malformed_rows_fail_closed(self, filename: str, expected: str):
        errors = validate_rows(_load_rows(filename))
        assert any(expected in e for e in errors), f"Expected {expected!r}, got {errors}"


class TestRenderReport:
    """Golden output comparison for rendered Markdown."""

    def test_golden_output_matches(self):
        meta = _load_valid_meta()
        rows = load_csv_rows([FIXTURES / "valid_rows.csv"])
        result = render_report(meta, rows)
        golden = (FIXTURES / "golden_output.md").read_text(encoding="utf-8")
        assert result == golden, "Rendered output does not match golden fixture"

    def test_report_has_required_sections(self):
        meta = _load_valid_meta()
        rows = load_csv_rows([FIXTURES / "valid_rows.csv"])
        result = render_report(meta, rows)
        for section in [
            "## Summary", "## Commands", "## Artifacts", "## Row Identity",
            "## Derived Comparison", "## Divergence Gate", "## Verdict",
        ]:
            assert section in result, f"Missing section: {section}"

    def test_report_section_order(self):
        meta = _load_valid_meta()
        rows = load_csv_rows([FIXTURES / "valid_rows.csv"])
        result = render_report(meta, rows)
        sections = [
            "## Summary", "## Commands", "## Artifacts", "## Row Identity",
            "## Derived Comparison", "## Divergence Gate", "## Verdict",
        ]
        positions = [result.index(s) for s in sections]
        assert positions == sorted(positions), f"Sections not in contract order: {positions}"


class TestCLIIntegration:
    """End-to-end CLI via subprocess."""

    def test_valid_run_produces_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "issue-4282-test-current.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--metadata",
                 str(FIXTURES / "valid_metadata.json"), "--csv",
                 str(FIXTURES / "valid_rows.csv"), "--output", str(output)],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, f"Expected exit 0, stderr: {result.stderr}"
            assert output.exists(), "Output file was not created"
            assert "## Summary" in output.read_text(), "Output missing Summary section"

    def test_invalid_metadata_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "issue-4282-test-current.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--metadata",
                 str(FIXTURES / "invalid_metadata_unknown_keys.json"), "--csv",
                 str(FIXTURES / "valid_rows.csv"), "--output", str(output)],
                capture_output=True, text=True,
            )
            assert result.returncode != 0, f"Expected nonzero exit, got {result.returncode}"
            assert "unknown metadata keys" in result.stderr, f"Expected error in stderr: {result.stderr}"

    def test_duplicate_backend_exits_nonzero(self):
        result = _run_cli("duplicate_backend_rows.csv")
        assert result.returncode != 0, f"Expected nonzero exit, got {result.returncode}"
        assert "expected exactly one cpu row and one wgpu row" in result.stderr, (
            f"Expected error in stderr: {result.stderr}"
        )

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("wrong_schema_version_rows.csv", "schema_version must be 'backend_benchmark_row_v1'"),
            ("missing_status_rows.csv", "required field 'status' is empty or missing"),
            ("missing_wall_seconds_rows.csv", "required field 'wall_seconds' is empty or missing"),
            ("mismatched_subject_id_rows.csv", "subject_id must equal comparison_key"),
            ("extra_backend_rows.csv", "backend must be one of ['cpu', 'wgpu']"),
            ("nonnumeric_wall_seconds_rows.csv", "wall_seconds must parse as float"),
            ("missing_lane_header_rows.csv", "required field 'lane' is empty or missing"),
            (
                "missing_comparison_key_header_rows.csv",
                "required field 'comparison_key' is empty or missing",
            ),
            ("missing_backend_header_rows.csv", "required field 'backend' is empty or missing"),
        ],
    )
    def test_malformed_compare_rows_exit_nonzero_without_traceback(
        self,
        filename: str,
        expected: str,
    ):
        result = _run_cli(filename)
        assert result.returncode != 0, f"Expected nonzero exit, got {result.returncode}"
        assert expected in result.stderr, f"Expected error in stderr: {result.stderr}"
        assert "Traceback" not in result.stderr, f"Unexpected traceback in stderr: {result.stderr}"
        assert "KeyError" not in result.stderr, f"Unexpected KeyError in stderr: {result.stderr}"

    def test_non_json_metadata_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_meta = Path(tmpdir) / "meta.yaml"
            yaml_meta.write_text("schema_version: test\n")
            output = Path(tmpdir) / "issue-4282-test-current.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--metadata", str(yaml_meta),
                 "--csv", str(FIXTURES / "valid_rows.csv"), "--output", str(output)],
                capture_output=True, text=True,
            )
            assert result.returncode != 0, f"Expected nonzero exit, got {result.returncode}"
            assert "must be a .json file" in result.stderr, f"Expected error in stderr: {result.stderr}"


# --- Packet B: Profile-lane tests ---


def _load_profile_meta() -> dict:
    with open(PROFILE_FIXTURES / "valid_profile_metadata.json") as f:
        return json.load(f)


def _load_profile_rows(filename: str) -> list[dict[str, str]]:
    return load_csv_rows([PROFILE_FIXTURES / filename])


class TestValidateProfileMetadata:
    """Profile-mode metadata validation (Packet B)."""

    PROFILE_OUTPUT = Path("reports/benchmarks/issue-4291-metaroom-host-profile-current.md")

    def test_valid_profile_metadata_passes(self):
        meta = _load_profile_meta()
        errors = validate_profile_metadata(meta, self.PROFILE_OUTPUT)
        assert errors == [], f"Expected no errors, got {errors}"

    def test_report_notes_forbidden_in_profile_mode(self):
        meta = _load_profile_meta()
        meta["report_notes"] = [{"heading": "H", "bullets": ["b"]}]
        errors = validate_profile_metadata(meta, self.PROFILE_OUTPUT)
        assert any("report_notes is forbidden" in e for e in errors), f"Expected report_notes error, got {errors}"

    def test_missing_profile_findings_rejected(self):
        meta = _load_profile_meta()
        del meta["profile_findings"]
        errors = validate_profile_metadata(meta, self.PROFILE_OUTPUT)
        assert any("profile_findings" in e for e in errors), f"Expected missing profile_findings, got {errors}"

    def test_empty_profile_findings_rejected(self):
        meta = _load_profile_meta()
        meta["profile_findings"] = []
        errors = validate_profile_metadata(meta, self.PROFILE_OUTPUT)
        assert any("profile_findings must be a non-empty list" in e for e in errors), f"Expected error, got {errors}"

    def test_profile_findings_missing_heading_rejected(self):
        meta = _load_profile_meta()
        meta["profile_findings"] = [{"bullets": ["b"]}]
        errors = validate_profile_metadata(meta, self.PROFILE_OUTPUT)
        assert any("must have 'heading'" in e for e in errors), f"Expected heading error, got {errors}"

    def test_profile_findings_empty_bullets_rejected(self):
        meta = _load_profile_meta()
        meta["profile_findings"] = [{"heading": "H", "bullets": []}]
        errors = validate_profile_metadata(meta, self.PROFILE_OUTPUT)
        assert any("non-empty 'bullets'" in e for e in errors), f"Expected bullets error, got {errors}"


class TestValidateProfileRows:
    """Profile-lane CSV row validation (Packet B)."""

    PROFILE_OUTPUT = Path("reports/benchmarks/issue-4291-metaroom-host-profile-current.md")

    def test_valid_profile_row_passes(self):
        rows = _load_profile_rows("valid_profile_row.csv")
        errors = validate_profile_rows(rows, self.PROFILE_OUTPUT)
        assert errors == [], f"Expected no errors, got {errors}"

    def test_multi_row_rejected(self):
        rows = _load_profile_rows("multi_row_profile.csv")
        errors = validate_profile_rows(rows, self.PROFILE_OUTPUT)
        assert any("exactly 1 row" in e for e in errors), f"Expected multi-row error, got {errors}"

    def test_wrong_lane_rejected(self):
        rows = _load_profile_rows("wrong_lane_profile.csv")
        errors = validate_profile_rows(rows, self.PROFILE_OUTPUT)
        assert any("not a profile lane" in e for e in errors), f"Expected lane error, got {errors}"

    def test_empty_csv_rejected(self):
        errors = validate_profile_rows([], self.PROFILE_OUTPUT)
        assert any("no CSV rows found" in e for e in errors), f"Expected empty error, got {errors}"

    def test_header_only_csv_rejected(self):
        rows = _load_profile_rows("header_only_profile.csv")
        errors = validate_profile_rows(rows, self.PROFILE_OUTPUT)
        assert any("no CSV rows found" in e for e in errors), f"Expected header-only error, got {errors}"

    def test_output_path_mismatch_rejected(self):
        rows = _load_profile_rows("valid_profile_row.csv")
        errors = validate_profile_rows(rows, Path("wrong/output/path.md"))
        assert any("does not match" in e for e in errors), f"Expected path mismatch, got {errors}"


class TestRenderProfileReport:
    """Profile-lane rendering (Packet B)."""

    def test_golden_profile_output_matches(self):
        meta = _load_profile_meta()
        rows = _load_profile_rows("valid_profile_row.csv")
        result = render_profile_report(meta, rows)
        golden = (PROFILE_FIXTURES / "golden_profile_output.md").read_text(encoding="utf-8")
        assert result == golden, "Rendered profile output does not match golden fixture"

    def test_profile_report_has_profile_findings_not_derived_comparison(self):
        meta = _load_profile_meta()
        rows = _load_profile_rows("valid_profile_row.csv")
        result = render_profile_report(meta, rows)
        assert "## Profile Findings" in result, "Missing Profile Findings section"
        assert "## Derived Comparison" not in result, "Profile report must not have Derived Comparison"

    def test_profile_report_section_order(self):
        meta = _load_profile_meta()
        rows = _load_profile_rows("valid_profile_row.csv")
        result = render_profile_report(meta, rows)
        sections = [
            "## Summary", "## Commands", "## Artifacts", "## Row Identity",
            "## Profile Findings", "## Divergence Gate", "## Verdict",
        ]
        positions = [result.index(s) for s in sections]
        assert positions == sorted(positions), f"Sections not in contract order: {positions}"

    def test_profile_row_identity_from_csv(self):
        meta = _load_profile_meta()
        rows = _load_profile_rows("valid_profile_row.csv")
        result = render_profile_report(meta, rows)
        assert "`schema_version`: `backend_benchmark_row_v1`" in result, "Missing schema_version in row identity"
        assert "`lane`: `metaroom_host_profile`" in result, "Missing lane in row identity"
        assert "`status`: `timeout`" in result, "Missing status in row identity"
        assert "`wall_seconds`: `221.70`" in result, "Missing wall_seconds in row identity"

    def test_profile_findings_headings_from_metadata(self):
        meta = _load_profile_meta()
        rows = _load_profile_rows("valid_profile_row.csv")
        result = render_profile_report(meta, rows)
        assert "Early sample dominant stack families:" in result, "Missing early sample heading"
        assert "Late sample dominant stack families:" in result, "Missing late sample heading"
        assert "Hotspot-to-issue mapping:" in result, "Missing hotspot heading"

    def test_empty_actual_method_renders_not_emitted(self):
        meta = _load_profile_meta()
        rows = _load_profile_rows("valid_profile_row.csv")
        result = render_profile_report(meta, rows)
        assert "`actual_method`: not emitted" in result, "Missing not emitted for empty actual_method"


class TestProfileCLIIntegration:
    """Profile-mode end-to-end CLI via subprocess (Packet B)."""

    def test_profile_valid_run_produces_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Output path must match the fixture CSV's profile_artifact_path
            # (resolved relative to cwd). Use the path from the fixture.
            output = REPO_ROOT / "reports" / "benchmarks" / "issue-4291-metaroom-host-profile-current.md"
            backup = output.read_text(encoding="utf-8") if output.exists() else None
            try:
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--metadata",
                     str(PROFILE_FIXTURES / "valid_profile_metadata.json"), "--csv",
                     str(PROFILE_FIXTURES / "valid_profile_row.csv"), "--output", str(output)],
                    capture_output=True, text=True,
                    cwd=str(REPO_ROOT),
                )
                assert result.returncode == 0, f"Expected exit 0, stderr: {result.stderr}"
                assert output.exists(), "Output file was not created"
                content = output.read_text()
                assert "## Profile Findings" in content, "Output missing Profile Findings section"
                assert "## Derived Comparison" not in content, "Profile output must not have Derived Comparison"
            finally:
                # Restore original report content
                if backup is not None:
                    output.write_text(backup, encoding="utf-8")

    def test_profile_multi_row_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "issue-4291-test-current.md"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--metadata",
                 str(PROFILE_FIXTURES / "valid_profile_metadata.json"), "--csv",
                 str(PROFILE_FIXTURES / "multi_row_profile.csv"), "--output", str(output)],
                capture_output=True, text=True,
            )
            assert result.returncode != 0, f"Expected nonzero exit, got {result.returncode}"
            assert "exactly 1 row" in result.stderr, f"Expected error in stderr: {result.stderr}"
