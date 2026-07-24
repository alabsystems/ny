#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Profile-lane helpers for render_backend_benchmark_report.py (Packet B).

Split from _benchmark_report_helpers.py to stay under file-size limits.
Not a public API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _benchmark_report_helpers import (
    PROFILE_LANES,
    PROFILE_METADATA_REQUIRED_KEYS,
    PROFILE_ROW_REQUIRED_FIELDS,
    ROW_FACT_KEYS,
    _check_meta_filename,
    _check_meta_types,
    _render_artifacts,
    _render_commands,
    _render_header,
    _render_summary,
)


# --- Profile metadata validation ---


def _check_profile_meta_keys(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if meta.get("schema_version") != "benchmark_report_metadata_v1":
        errors.append(
            f"schema_version must be 'benchmark_report_metadata_v1', "
            f"got {meta.get('schema_version')!r}"
        )
    unknown = set(meta.keys()) - PROFILE_METADATA_REQUIRED_KEYS
    if unknown:
        errors.append(f"unknown metadata keys: {sorted(unknown)}")
    missing = PROFILE_METADATA_REQUIRED_KEYS - set(meta.keys())
    if missing:
        errors.append(f"missing required metadata keys: {sorted(missing)}")
    row_facts = ROW_FACT_KEYS & set(meta.keys())
    if row_facts:
        errors.append(f"row-fact keys not allowed in metadata: {sorted(row_facts)}")
    if "report_notes" in meta:
        errors.append("report_notes is forbidden in profile mode; use profile_findings")
    return errors


def _check_profile_findings(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    findings = meta.get("profile_findings")
    if not isinstance(findings, list) or len(findings) == 0:
        errors.append("profile_findings must be a non-empty list")
        return errors
    for i, section in enumerate(findings):
        if not isinstance(section, dict):
            errors.append(f"profile_findings[{i}] must be an object")
        elif "heading" not in section:
            errors.append(f"profile_findings[{i}] must have 'heading'")
        elif not section.get("bullets"):
            errors.append(f"profile_findings[{i}] must have a non-empty 'bullets' list")
    return errors


def validate_profile_metadata(meta: dict[str, Any], output_path: Path) -> list[str]:
    """Validate profile-mode benchmark_report_metadata_v1. Returns error list."""
    errors: list[str] = []
    errors.extend(_check_profile_meta_keys(meta))
    errors.extend(_check_meta_types(meta))
    errors.extend(_check_profile_findings(meta))
    for i, cmd in enumerate(meta.get("commands", [])):
        if not isinstance(cmd, dict):
            errors.append(f"commands[{i}] must be an object")
        elif "label" not in cmd or "shell" not in cmd:
            errors.append(f"commands[{i}] must have 'label' and 'shell'")
    for i, note in enumerate(meta.get("artifact_notes", [])):
        if not isinstance(note, dict):
            errors.append(f"artifact_notes[{i}] must be an object")
        elif "label" not in note or "path" not in note:
            errors.append(f"artifact_notes[{i}] must have 'label' and 'path'")
    errors.extend(_check_meta_filename(meta, output_path))
    return errors


# --- Profile row validation ---


def validate_profile_rows(
    rows: list[dict[str, str]], output_path: Path,
) -> list[str]:
    """Validate profile-lane row integrity. Returns error list."""
    if not rows:
        return ["no CSV rows found"]
    lanes = {r.get("lane", "") for r in rows}
    if len(lanes) > 1:
        return [f"mixed lanes in CSV input: {sorted(lanes)}"]
    lane = lanes.pop()
    if lane not in PROFILE_LANES:
        return [f"lane '{lane}' is not a profile lane. Supported: {sorted(PROFILE_LANES)}"]
    if len(rows) > 1:
        return [f"profile mode requires exactly 1 row, got {len(rows)}"]
    row = rows[0]
    errors: list[str] = []
    for field in PROFILE_ROW_REQUIRED_FIELDS:
        if not row.get(field, "").strip():
            errors.append(f"required field '{field}' is empty or missing")
    if row.get("subject_id", "") != row.get("comparison_key", ""):
        errors.append(
            f"subject_id must equal comparison_key in profile mode, "
            f"got {row.get('subject_id')!r} vs {row.get('comparison_key')!r}"
        )
    artifact = row.get("profile_artifact_path", "")
    if artifact:
        artifact_resolved = Path(artifact).resolve()
        output_resolved = output_path.resolve()
        if artifact_resolved != output_resolved:
            errors.append(
                f"profile_artifact_path '{artifact}' does not match "
                f"output path '{output_path}'"
            )
    return errors


# --- Profile rendering ---


def _render_profile_row_identity(row: dict[str, str], lines: list[str]) -> None:
    lines.append("## Row Identity")
    lines.append("")
    for field in (
        "schema_version", "lane", "subject_kind", "subject_id",
        "comparison_key", "model_path", "property_path", "preset_path",
        "backend", "timeout_seconds", "status", "actual_method",
        "wall_seconds", "domains_explored", "profile_artifact_path",
    ):
        val = row.get(field, "")
        if field == "actual_method" and not val:
            val = "empty (no final method line before timeout)"
        lines.append(f"- `{field}`: `{val}`")
    lines.append("")


def _render_profile_findings(
    meta: dict[str, Any], row: dict[str, str], lines: list[str],
) -> None:
    lines.append("## Profile Findings")
    lines.append("")
    actual = row.get("actual_method", "")
    lines.append("Final verifier facts:")
    lines.append("")
    lines.append(f"- `status`: `{row.get('status', '')}`")
    lines.append(f"- `wall_seconds`: `{row.get('wall_seconds', '')}`")
    lines.append(f"- `domains_explored`: `{row.get('domains_explored', '')}`")
    lines.append(
        f"- `actual_method`: {actual if actual else 'not emitted'}"
    )
    lines.append("")
    for section in meta["profile_findings"]:
        lines.append(f"{section['heading']}:")
        lines.append("")
        for bullet in section["bullets"]:
            lines.append(f"- {bullet}")
        lines.append("")


def _render_profile_footer(meta: dict[str, Any], lines: list[str]) -> None:
    lines.append("## Divergence Gate")
    lines.append("")
    for note in meta["divergence_notes"]:
        lines.append(note)
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for v in meta["verdict_lines"]:
        lines.append(v)
    lines.append("")


def render_profile_report(meta: dict[str, Any], rows: list[dict[str, str]]) -> str:
    """Render a profile-lane Markdown report from metadata and one CSV row."""
    row = rows[0]
    lines: list[str] = []
    _render_header(meta, lines)
    _render_summary(meta, lines)
    _render_commands(meta, lines)
    _render_artifacts(meta, lines)
    _render_profile_row_identity(row, lines)
    _render_profile_findings(meta, row, lines)
    _render_profile_footer(meta, lines)
    return "\n".join(lines)
