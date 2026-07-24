#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Ny and compare-run provenance regression tests for compare-backends benchmark reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "benchmark_reports" / "compare_backends"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _benchmark_report_helpers import render_report, validate_rows
from render_backend_benchmark_report import load_csv_rows


def _load_valid_meta() -> dict:
    with open(FIXTURES / "valid_metadata.json", encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(filename: str) -> list[dict[str, str]]:
    return load_csv_rows([FIXTURES / filename])


class TestCompareBackendNyProvenance:
    def test_matching_provenance_rows_pass(self) -> None:
        errors = validate_rows(_load_rows("valid_rows_with_ny_provenance.csv"))
        assert errors == [], f"Expected matching ny provenance to pass, got {errors}"

    def test_mismatched_digest_rejected(self) -> None:
        errors = validate_rows(_load_rows("mismatched_ny_sha_rows.csv"))
        assert any("ny_sha256" in error for error in errors), errors

    def test_mismatched_version_rejected(self) -> None:
        rows = _load_rows("valid_rows_with_ny_provenance.csv")
        rows[1]["notes"] = rows[1]["notes"].replace("ny 0.1.0", "ny 0.2.0")
        errors = validate_rows(rows)
        assert any("ny_version" in error for error in errors), errors

    def test_one_sided_provenance_rejected(self) -> None:
        errors = validate_rows(_load_rows("one_sided_ny_provenance_rows.csv"))
        assert any("presence mismatch" in error for error in errors), errors

    def test_partial_provenance_rejected(self) -> None:
        errors = validate_rows(_load_rows("partial_ny_provenance_rows.csv"))
        assert any("malformed ny provenance" in error for error in errors), errors

    def test_duplicate_ny_tag_rejected(self) -> None:
        rows = _load_rows("valid_rows_with_ny_provenance.csv")
        rows[0]["notes"] = (
            f"{rows[0]['notes']}; ny_sha256=shadow_digest"
        )
        errors = validate_rows(rows)
        assert any("malformed ny provenance" in error for error in errors), errors

    def test_legacy_rows_omit_ny_provenance_section(self) -> None:
        result = render_report(_load_valid_meta(), _load_rows("valid_rows.csv"))
        assert "### Ny Provenance" not in result, (
            "Legacy compare-backends rows should not render the provenance subsection"
        )

    def test_rendered_report_matches_provenance_golden(self) -> None:
        result = render_report(_load_valid_meta(), _load_rows("valid_rows_with_ny_provenance.csv"))
        golden = (FIXTURES / "golden_output_with_ny_provenance.md").read_text(encoding="utf-8")
        assert result == golden, "Rendered compare-backends provenance output drifted from the golden fixture"


class TestCompareBackendCompareRunId:
    """Regression tests for compare_run_id pair-integrity gate (#4383)."""

    def test_matching_compare_run_id_passes(self) -> None:
        errors = validate_rows(_load_rows("valid_rows_with_compare_run_id.csv"))
        assert errors == [], f"Expected matching compare_run_id to pass, got {errors}"

    def test_mismatched_compare_run_id_rejected(self) -> None:
        errors = validate_rows(_load_rows("mismatched_compare_run_id_rows.csv"))
        assert any("compare_run_id mismatch" in e for e in errors), errors

    def test_one_sided_compare_run_id_rejected(self) -> None:
        errors = validate_rows(_load_rows("one_sided_compare_run_id_rows.csv"))
        assert any("one-sided compare_run_id" in e for e in errors), errors

    def test_duplicate_compare_run_id_rejected(self) -> None:
        rows = _load_rows("valid_rows_with_compare_run_id.csv")
        rows[0]["notes"] = f"{rows[0]['notes']}; compare_run_id=shadow"
        errors = validate_rows(rows)
        assert any("malformed compare_run_id" in e for e in errors), errors

    def test_empty_compare_run_id_rejected(self) -> None:
        rows = _load_rows("valid_rows_with_compare_run_id.csv")
        for idx in (0, 1):
            rows[idx]["notes"] = rows[idx]["notes"].replace(
                "compare_run_id=cersyve_20260322_120000_12345",
                "compare_run_id=",
            )
        errors = validate_rows(rows)
        assert any("malformed compare_run_id" in e for e in errors), errors

    def test_legacy_rows_without_compare_run_id_pass(self) -> None:
        errors = validate_rows(_load_rows("valid_rows.csv"))
        assert not any("compare_run_id" in e for e in errors), (
            f"Legacy rows without compare_run_id should not trigger errors: {errors}"
        )

    def test_rendered_report_surfaces_compare_run_provenance(self) -> None:
        result = render_report(
            _load_valid_meta(),
            _load_rows("valid_rows_with_compare_run_id.csv"),
        )
        assert "### Compare Run Provenance" in result, (
            "Rows with compare_run_id should render a Compare Run Provenance subsection"
        )
        assert "cersyve_20260322_120000_12345" in result, (
            "Rendered report should contain the compare_run_id token"
        )

    def test_legacy_rows_omit_compare_run_provenance_section(self) -> None:
        result = render_report(_load_valid_meta(), _load_rows("valid_rows.csv"))
        assert "### Compare Run Provenance" not in result, (
            "Legacy rows should not render the Compare Run Provenance subsection"
        )
