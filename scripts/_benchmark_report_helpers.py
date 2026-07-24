#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Internal helpers for render_backend_benchmark_report.py.

Split from the main script to stay under file-size and function-complexity
limits. Not a public API.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from _benchmark_report_compare_provenance import (
    check_compare_run_id,
    check_ny_provenance,
    render_compare_run_provenance,
    render_ny_provenance,
)

COMPARE_LANES = frozenset({
    "vnncomp_compare_backends",
    "avoice_kokoro_backend_delta",
})

PROFILE_LANES = frozenset({
    "metaroom_host_profile",
})

METADATA_REQUIRED_KEYS = frozenset({
    "schema_version", "report_date", "issue", "title", "epic",
    "summary_lines", "commands", "artifact_notes",
    "divergence_notes", "report_notes", "verdict_lines",
})

PROFILE_METADATA_REQUIRED_KEYS = frozenset({
    "schema_version", "report_date", "issue", "title", "epic",
    "summary_lines", "commands", "artifact_notes",
    "divergence_notes", "profile_findings", "verdict_lines",
})

ROW_FACT_KEYS = frozenset({
    "status", "actual_method", "wall_seconds", "comparison_key",
    "speedup", "delta_pct", "width_ratio",
})

SHARED_IDENTITY_FIELDS = (
    "subject_kind", "subject_id", "category", "workload",
    "model_path", "property_path", "preset_path", "timeout_seconds",
)

PROFILE_ROW_REQUIRED_FIELDS = (
    "schema_version", "subject_kind", "subject_id", "comparison_key",
    "model_path", "property_path", "preset_path", "backend",
    "timeout_seconds", "status", "wall_seconds", "domains_explored",
    "profile_artifact_path",
)

COMPARE_ROW_REQUIRED_FIELDS = (
    "schema_version",
    "lane",
    "subject_kind",
    "subject_id",
    "comparison_key",
    "backend",
    "status",
    "wall_seconds",
)

ALLOWED_BACKENDS = frozenset({"cpu", "wgpu"})


# --- Metadata validation (split into sub-checks) ---


def _check_meta_keys(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if meta.get("schema_version") != "benchmark_report_metadata_v1":
        errors.append(
            f"schema_version must be 'benchmark_report_metadata_v1', "
            f"got {meta.get('schema_version')!r}"
        )
    unknown = set(meta.keys()) - METADATA_REQUIRED_KEYS
    if unknown:
        errors.append(f"unknown metadata keys: {sorted(unknown)}")
    missing = METADATA_REQUIRED_KEYS - set(meta.keys())
    if missing:
        errors.append(f"missing required metadata keys: {sorted(missing)}")
    row_facts = ROW_FACT_KEYS & set(meta.keys())
    if row_facts:
        errors.append(f"row-fact keys not allowed in metadata: {sorted(row_facts)}")
    if "profile_findings" in meta:
        errors.append("profile_findings belongs to Packet B, not compare-backends")
    return errors


def _check_meta_types(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "report_date" in meta:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(meta["report_date"])):
            errors.append(f"report_date must be YYYY-MM-DD, got {meta['report_date']!r}")
    for key in ("issue", "epic"):
        if key in meta and not isinstance(meta[key], int):
            errors.append(f"{key} must be an integer, got {type(meta[key]).__name__}")
    for key in ("summary_lines", "divergence_notes", "verdict_lines"):
        val = meta.get(key)
        if isinstance(val, list) and len(val) == 0:
            errors.append(f"{key} must not be empty")
    return errors


def _check_meta_nested(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
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
    for i, section in enumerate(meta.get("report_notes", [])):
        if not isinstance(section, dict):
            errors.append(f"report_notes[{i}] must be an object")
        elif "heading" not in section:
            errors.append(f"report_notes[{i}] must have 'heading'")
        elif not section.get("bullets"):
            errors.append(f"report_notes[{i}] must have a non-empty 'bullets' list")
    return errors


def _check_meta_filename(meta: dict[str, Any], output_path: Path) -> list[str]:
    if "issue" not in meta or not isinstance(meta["issue"], int):
        return []
    m = re.match(r"issue-(\d+)-", output_path.stem)
    if m and int(m.group(1)) != meta["issue"]:
        return [
            f"metadata issue={meta['issue']} does not match output "
            f"filename issue number {m.group(1)}"
        ]
    return []


def validate_metadata(meta: dict[str, Any], output_path: Path) -> list[str]:
    """Validate compare-mode benchmark_report_metadata_v1. Returns error list."""
    errors: list[str] = []
    errors.extend(_check_meta_keys(meta))
    errors.extend(_check_meta_types(meta))
    errors.extend(_check_meta_nested(meta))
    errors.extend(_check_meta_filename(meta, output_path))
    return errors


# --- Row validation ---


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    """Validate compare-backends row integrity. Returns error list."""
    if not rows:
        return ["no CSV rows found"]

    row_errors = _validate_compare_rows(rows)
    if row_errors:
        return row_errors

    lanes = {r.get("lane", "") for r in rows}
    if len(lanes) > 1:
        return [f"mixed lanes in CSV input: {sorted(lanes)}"]

    lane = lanes.pop()
    if lane not in COMPARE_LANES:
        return [
            f"lane '{lane}' is not supported in Packet A0. "
            f"Supported: {sorted(COMPARE_LANES)}"
        ]

    errors: list[str] = []
    groups: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r.get("comparison_key", ""), []).append(r)

    for key, group in groups.items():
        backends = sorted(r.get("backend", "") for r in group)
        if backends != ["cpu", "wgpu"]:
            errors.append(
                f"comparison_key {key!r}: expected exactly one cpu row and one wgpu row; "
                f"got backends {backends!r}"
            )
            continue
        cpu = next(r for r in group if r.get("backend", "") == "cpu")
        wgpu = next(r for r in group if r.get("backend", "") == "wgpu")
        errors.extend(_check_identity_agreement(key, cpu, wgpu))
        errors.extend(check_ny_provenance(key, cpu, wgpu))
        errors.extend(check_compare_run_id(key, cpu, wgpu))
    return errors


def _validate_compare_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row_idx, row in enumerate(rows):
        for field in COMPARE_ROW_REQUIRED_FIELDS:
            if not row.get(field, "").strip():
                errors.append(f"row {row_idx}: required field '{field}' is empty or missing")

        backend = row.get("backend", "").strip()
        if backend and backend not in ALLOWED_BACKENDS:
            errors.append(
                f"row {row_idx}: backend must be one of {sorted(ALLOWED_BACKENDS)}, "
                f"got {backend!r}"
            )

        schema_version = row.get("schema_version", "").strip()
        if schema_version and schema_version != "backend_benchmark_row_v1":
            errors.append(
                f"row {row_idx}: schema_version must be 'backend_benchmark_row_v1', "
                f"got {schema_version!r}"
            )

        subject_id = row.get("subject_id", "").strip()
        comparison_key = row.get("comparison_key", "").strip()
        if subject_id and comparison_key and subject_id != comparison_key:
            errors.append(
                f"row {row_idx}: subject_id must equal comparison_key, "
                f"got {subject_id!r} vs {comparison_key!r}"
            )

        wall_seconds = row.get("wall_seconds", "").strip()
        if wall_seconds:
            try:
                float(wall_seconds)
            except ValueError:
                errors.append(
                    f"row {row_idx}: wall_seconds must parse as float, got {wall_seconds!r}"
                )
    return errors


def _check_identity_agreement(
    key: str, cpu: dict[str, str], wgpu: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for field in SHARED_IDENTITY_FIELDS:
        if cpu.get(field, "") != wgpu.get(field, ""):
            errors.append(
                f"comparison_key {key!r}: shared identity field "
                f"'{field}' disagrees: cpu={cpu.get(field, '')!r} "
                f"vs wgpu={wgpu.get(field, '')!r}"
            )
    return errors


# --- Rendering helpers ---


def _safe_float(val: str) -> float | None:
    if not val or not val.strip():
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _fmt(val: float | None, decimals: int = 2) -> str:
    return "N/A" if val is None else f"{val:.{decimals}f}"


def _fmt_pct(val: float | None) -> str:
    return "N/A" if val is None else f"{val:+.1f}%"


def _is_solved(status: str) -> bool:
    return status in ("verified", "violated")


def _render_header(meta: dict[str, Any], lines: list[str]) -> None:
    lines.extend([
        "<!--",
        "Copyright 2026 Andrew Yates",
        "Author: Andrew Yates <andrewyates.name@gmail.com>",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
    ])
    lines.append(f"# {meta['title']}")
    lines.append("")
    lines.append(f"**Issue:** #{meta['issue']}  ")
    lines.append(f"**Epic:** #{meta['epic']}  ")
    lines.append(f"**Date:** {meta['report_date']}")
    lines.append("")


def _render_summary(meta: dict[str, Any], lines: list[str]) -> None:
    lines.append("## Summary")
    lines.append("")
    for s in meta["summary_lines"]:
        lines.append(s)
    lines.append("")


def _render_commands(meta: dict[str, Any], lines: list[str]) -> None:
    lines.append("## Commands")
    lines.append("")
    for cmd in meta["commands"]:
        lines.append(f"**{cmd['label']}:**")
        lines.append("")
        lines.append("```bash")
        lines.append(cmd["shell"])
        lines.append("```")
        lines.append("")


def _render_artifacts(meta: dict[str, Any], lines: list[str]) -> None:
    lines.append("## Artifacts")
    lines.append("")
    for note in meta["artifact_notes"]:
        lines.append(f"- **{note['label']}:** `{note['path']}`")
    lines.append("")


def _render_row_identity(rows: list[dict[str, str]], lines: list[str]) -> None:
    lines.append("## Row Identity")
    lines.append("")
    lines.append("| Comparison Key | Category | Subject Kind | Backend | Status |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r.get("category", ""), r["comparison_key"], r["backend"])):
        lines.append(
            f"| {r['comparison_key']} | {r.get('category', '')} "
            f"| {r.get('subject_kind', '')} | {r['backend']} | {r.get('status', '')} |"
        )
    lines.append("")
    render_ny_provenance(rows, lines)
    render_compare_run_provenance(rows, lines)


def _compute_category_stats(keys: list[str], groups: dict) -> dict[str, Any]:
    cpu_solved = wgpu_solved = divergences = 0
    cpu_wall_total = wgpu_wall_total = 0.0
    cpu_wall_valid = wgpu_wall_valid = True
    for key in keys:
        pair = groups[key]
        cpu_r, wgpu_r = pair["cpu"], pair["wgpu"]
        if _is_solved(cpu_r.get("status", "")):
            cpu_solved += 1
        if _is_solved(wgpu_r.get("status", "")):
            wgpu_solved += 1
        cpu_w = _safe_float(cpu_r.get("wall_seconds", ""))
        wgpu_w = _safe_float(wgpu_r.get("wall_seconds", ""))
        if cpu_w is not None:
            cpu_wall_total += cpu_w
        else:
            cpu_wall_valid = False
        if wgpu_w is not None:
            wgpu_wall_total += wgpu_w
        else:
            wgpu_wall_valid = False
        if cpu_r.get("status", "") != wgpu_r.get("status", ""):
            divergences += 1
    return {
        "n": len(keys), "cpu_solved": cpu_solved, "wgpu_solved": wgpu_solved,
        "cpu_wall": cpu_wall_total if cpu_wall_valid else None,
        "wgpu_wall": wgpu_wall_total if wgpu_wall_valid else None,
        "divergences": divergences,
    }


def _render_category_overview(
    categories: dict[str, list[str]], groups: dict, lines: list[str],
) -> None:
    lines.append("### Category Overview")
    lines.append("")
    lines.append(
        "| Category | Samples | CPU Solved | WGPU Solved | CPU Wall "
        "| WGPU Wall | Delta | Speedup | Delta % | Divergences |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cat in sorted(categories.keys()):
        s = _compute_category_stats(categories[cat], groups)
        cw, ww = s["cpu_wall"], s["wgpu_wall"]
        if cw is not None and ww is not None:
            delta = ww - cw
            speedup = cw / ww if ww > 0 else None
            delta_pct = 100.0 * (cw - ww) / cw if cw > 0 else None
        else:
            delta = speedup = delta_pct = None
        lines.append(
            f"| {cat} | {s['n']} | {s['cpu_solved']} | {s['wgpu_solved']} "
            f"| {_fmt(cw)} | {_fmt(ww)} | {_fmt(delta)} "
            f"| {_fmt(speedup)}x | {_fmt_pct(delta_pct)} | {s['divergences']} |"
        )
    lines.append("")


def _render_detail_row(key: str, pair: dict, lines: list[str]) -> None:
    cpu_r, wgpu_r = pair["cpu"], pair["wgpu"]
    cpu_w = _safe_float(cpu_r.get("wall_seconds", ""))
    wgpu_w = _safe_float(wgpu_r.get("wall_seconds", ""))
    cpu_width = _safe_float(cpu_r.get("output_width_sum", ""))
    wgpu_width = _safe_float(wgpu_r.get("output_width_sum", ""))
    if cpu_w is not None and wgpu_w is not None:
        delta = wgpu_w - cpu_w
        speedup = cpu_w / wgpu_w if wgpu_w > 0 else None
        delta_pct = 100.0 * (cpu_w - wgpu_w) / cpu_w if cpu_w > 0 else None
    else:
        delta = speedup = delta_pct = None
    width_ratio = (
        wgpu_width / cpu_width
        if cpu_width is not None and wgpu_width is not None and cpu_width > 0
        else None
    )
    lines.append(
        f"| {key} "
        f"| {cpu_r.get('status', '')} | {_fmt(cpu_w)} "
        f"| {cpu_r.get('domains_explored', '') or 'N/A'} "
        f"| {_fmt(cpu_width)} "
        f"| {wgpu_r.get('status', '')} | {_fmt(wgpu_w)} "
        f"| {wgpu_r.get('domains_explored', '') or 'N/A'} "
        f"| {_fmt(wgpu_width)} "
        f"| {_fmt(delta)} | {_fmt(speedup)}x "
        f"| {_fmt_pct(delta_pct)} | {_fmt(width_ratio)} |"
    )


def _render_derived_comparison(rows: list[dict[str, str]], lines: list[str]) -> None:
    lines.append("## Derived Comparison")
    lines.append("")
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r["comparison_key"], {})[r["backend"]] = r
    categories: dict[str, list[str]] = {}
    for key, pair in groups.items():
        cat = pair.get("cpu", pair.get("wgpu", {})).get("category", "unknown")
        categories.setdefault(cat, []).append(key)
    _render_category_overview(categories, groups, lines)
    lines.append("### Per-Instance Detail")
    lines.append("")
    lines.append(
        "| Comparison Key | CPU Status | CPU Seconds | CPU Domains | CPU Width "
        "| WGPU Status | WGPU Seconds | WGPU Domains | WGPU Width "
        "| Delta | Speedup | Delta % | Width Ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups.keys()):
        _render_detail_row(key, groups[key], lines)
    lines.append("")


def _render_footer(meta: dict[str, Any], lines: list[str]) -> None:
    lines.append("## Divergence Gate")
    lines.append("")
    for note in meta["divergence_notes"]:
        lines.append(note)
    lines.append("")
    if meta.get("report_notes"):
        for section in meta["report_notes"]:
            lines.append(f"### {section['heading']}")
            lines.append("")
            for bullet in section["bullets"]:
                lines.append(f"- {bullet}")
            lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for v in meta["verdict_lines"]:
        lines.append(v)
    lines.append("")


def render_report(meta: dict[str, Any], rows: list[dict[str, str]]) -> str:
    """Render a compare-backends Markdown report from metadata and CSV rows."""
    lines: list[str] = []
    _render_header(meta, lines)
    _render_summary(meta, lines)
    _render_commands(meta, lines)
    _render_artifacts(meta, lines)
    _render_row_identity(rows, lines)
    _render_derived_comparison(rows, lines)
    _render_footer(meta, lines)
    return "\n".join(lines)


# Re-export profile-lane helpers for backwards compatibility
from _benchmark_report_profile_helpers import (  # noqa: E402
    render_profile_report,
    validate_profile_metadata,
    validate_profile_rows,
)
