#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Ny-binary and compare-run provenance helpers for compare-backends benchmark reports."""
from __future__ import annotations

NY_PROVENANCE_KEYS = (
    "ny_source", "ny_bin", "ny_version", "ny_sha256",
)

COMPARE_RUN_ID_KEY = "compare_run_id"


def _iter_note_tags(row: dict[str, str]):
    for segment in row.get("notes", "").split(";"):
        segment = segment.strip()
        if "=" not in segment:
            continue
        yield tuple(part.strip() for part in segment.split("=", 1))


def _parse_ny_provenance(row: dict[str, str]) -> tuple[str, dict[str, str] | None]:
    found: dict[str, str] = {}
    for key, value in _iter_note_tags(row):
        if key in NY_PROVENANCE_KEYS:
            if key in found:
                return "malformed", None
            found[key] = value
    if not found:
        return "absent", None
    if len(found) != len(NY_PROVENANCE_KEYS):
        return "malformed", None
    if any(not found[key] for key in NY_PROVENANCE_KEYS):
        return "malformed", None
    return "present", found


def check_ny_provenance(
    key: str, cpu: dict[str, str], wgpu: dict[str, str],
) -> list[str]:
    cpu_state, cpu_provenance = _parse_ny_provenance(cpu)
    wgpu_state, wgpu_provenance = _parse_ny_provenance(wgpu)
    if cpu_state == "absent" and wgpu_state == "absent":
        return []
    if "malformed" in (cpu_state, wgpu_state):
        return [
            f"comparison_key {key!r}: malformed ny provenance: "
            f"cpu={cpu_state} vs wgpu={wgpu_state}"
        ]
    if "absent" in (cpu_state, wgpu_state):
        return [
            f"comparison_key {key!r}: ny provenance presence mismatch: "
            f"cpu={cpu_state} vs wgpu={wgpu_state}"
        ]
    assert cpu_provenance is not None and wgpu_provenance is not None
    errors: list[str] = []
    for field in ("ny_sha256", "ny_version"):
        if cpu_provenance[field] != wgpu_provenance[field]:
            errors.append(
                f"comparison_key {key!r}: ny provenance mismatch: "
                f"cpu {field}={cpu_provenance[field]!r} "
                f"vs wgpu {field}={wgpu_provenance[field]!r}"
            )
    return errors


def _parse_compare_run_id(row: dict[str, str]) -> tuple[str, str | None]:
    """Extract compare_run_id from notes as present/absent/malformed state."""
    found: list[str] = []
    for key, value in _iter_note_tags(row):
        if key == COMPARE_RUN_ID_KEY:
            found.append(value)
    if not found:
        return "absent", None
    if len(found) != 1 or not found[0]:
        return "malformed", None
    return "present", found[0]


def check_compare_run_id(
    key: str, cpu: dict[str, str], wgpu: dict[str, str],
) -> list[str]:
    """Validate compare_run_id consistency between a cpu/wgpu pair."""
    cpu_state, cpu_id = _parse_compare_run_id(cpu)
    wgpu_state, wgpu_id = _parse_compare_run_id(wgpu)
    if cpu_state == "absent" and wgpu_state == "absent":
        return []
    if "malformed" in (cpu_state, wgpu_state):
        return [
            f"comparison_key {key!r}: malformed compare_run_id: "
            f"cpu={cpu_state} vs wgpu={wgpu_state}"
        ]
    if "absent" in (cpu_state, wgpu_state):
        present_side = "cpu" if cpu_state == "present" else "wgpu"
        return [
            f"comparison_key {key!r}: one-sided compare_run_id "
            f"(only {present_side} has it)"
        ]
    assert cpu_id is not None and wgpu_id is not None
    if cpu_id != wgpu_id:
        return [
            f"comparison_key {key!r}: compare_run_id mismatch: "
            f"cpu={cpu_id!r} vs wgpu={wgpu_id!r}"
        ]
    return []


def render_compare_run_provenance(rows: list[dict[str, str]], lines: list[str]) -> None:
    """Render a Compare Run Provenance subsection under Row Identity."""
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["comparison_key"], {})[row["backend"]] = row
    saw_run_id = False
    rendered_rows: list[str] = []
    for key in sorted(groups.keys()):
        pair = groups[key]
        cpu_state, cpu_id = _parse_compare_run_id(pair.get("cpu", {}))
        wgpu_state, wgpu_id = _parse_compare_run_id(pair.get("wgpu", {}))
        run_id = cpu_id if cpu_state == "present" else wgpu_id if wgpu_state == "present" else None
        if run_id is not None:
            saw_run_id = True
            match = "yes" if cpu_state == wgpu_state == "present" and cpu_id == wgpu_id else "**NO**"
            rendered_rows.append(f"| {key} | {run_id} | {match} |")
    if not saw_run_id:
        return
    lines.append("### Compare Run Provenance")
    lines.append("")
    lines.append("| Comparison Key | Compare Run ID | Paired |")
    lines.append("|---|---|---|")
    lines.extend(rendered_rows)
    lines.append("")


def _format_ny_provenance_value(
    cpu: dict[str, str], wgpu: dict[str, str], field: str,
) -> str:
    if cpu[field] == wgpu[field]:
        return cpu[field]
    return f"cpu={cpu[field]}; wgpu={wgpu[field]}"


def render_ny_provenance(rows: list[dict[str, str]], lines: list[str]) -> None:
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["comparison_key"], {})[row["backend"]] = row
    rendered_rows: list[str] = []
    saw_structured_provenance = False
    for key in sorted(groups.keys()):
        pair = groups[key]
        cpu_state, cpu_provenance = _parse_ny_provenance(pair["cpu"])
        wgpu_state, wgpu_provenance = _parse_ny_provenance(pair["wgpu"])
        if "present" in (cpu_state, wgpu_state):
            saw_structured_provenance = True
        if cpu_state == "absent" and wgpu_state == "absent":
            rendered_rows.append(
                f"| {key} | legacy-unrecorded | legacy-unrecorded "
                f"| legacy-unrecorded | legacy-unrecorded |"
            )
            continue
        assert cpu_provenance is not None and wgpu_provenance is not None
        rendered_rows.append(
            f"| {key} | {cpu_provenance['ny_sha256']} | {cpu_provenance['ny_version']} "
            f"| {_format_ny_provenance_value(cpu_provenance, wgpu_provenance, 'ny_bin')} "
            f"| {_format_ny_provenance_value(cpu_provenance, wgpu_provenance, 'ny_source')} |"
        )
    if not saw_structured_provenance:
        return
    lines.append("### Ny Provenance")
    lines.append("")
    lines.append("| Comparison Key | Ny SHA256 | Ny Version | Ny Bin | Ny Source |")
    lines.append("|---|---|---|---|---|")
    lines.extend(rendered_rows)
    lines.append("")
