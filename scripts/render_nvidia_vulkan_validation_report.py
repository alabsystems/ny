#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Render the #4359 NVIDIA/Vulkan validation report from the orchestrator manifest.

Reads the JSON manifest written by run_nvidia_vulkan_validation.sh plus any
compare-backends CSVs and produces the final Markdown report at
reports/benchmarks/issue-4359-nvidia-vulkan-validation-current.md.

CLI:
    python3 scripts/render_nvidia_vulkan_validation_report.py \
        --manifest reports/benchmarks/issue-4359-nvidia-vulkan-manifest.json \
        --output-dir reports/benchmarks
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_manifest(path: Path) -> dict:
    """Load and validate the orchestrator manifest."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != "nvidia_vulkan_validation_manifest_v1":
        raise ValueError(f"Unknown manifest schema: {data.get('schema')}")
    return data


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows from a backend_benchmark_row_v1 file."""
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_cersyve_totals(
    rows: list[dict[str, str]],
) -> dict[str, float | None]:
    """Compute ny CPU/wgpu totals from cersyve compare-backends rows."""
    cpu_total = 0.0
    wgpu_total = 0.0
    cpu_count = 0
    wgpu_count = 0
    for row in rows:
        cat = row.get("category", "")
        if "cersyve" not in cat.lower():
            continue
        backend = row.get("backend", "")
        wall = row.get("wall_seconds", "")
        if not wall:
            continue
        try:
            secs = float(wall)
        except ValueError:
            continue
        if backend == "cpu":
            cpu_total += secs
            cpu_count += 1
        elif backend == "wgpu":
            wgpu_total += secs
            wgpu_count += 1
    result: dict[str, float | None] = {
        "ny_cpu_cersyve_total": cpu_total if cpu_count > 0 else None,
        "ny_wgpu_cersyve_total": wgpu_total if wgpu_count > 0 else None,
    }
    if cpu_count > 0 and wgpu_count > 0 and wgpu_total > 0:
        result["backend_speedup_total"] = cpu_total / wgpu_total
    else:
        result["backend_speedup_total"] = None
    return result


def load_host_info(manifest_dir: Path, manifest: dict) -> str:
    """Load host info text, or return a placeholder."""
    host_path = manifest_dir / manifest.get("host_info_path", "")
    if host_path.is_file():
        return host_path.read_text(encoding="utf-8")
    return "(host info not available)"


def _fmt(val: float | None) -> str:
    """Format a float or return N/A."""
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _render_header_sections(manifest: dict) -> list[str]:
    """Render title, summary, and commands sections."""
    sections: list[str] = []
    vulkan_confirmed = manifest.get("vulkan_confirmed", False)
    verdict = manifest.get("verdict", "blocked")

    sections.append("# NVIDIA/Vulkan Validation Report — #4359\n")
    sections.append("## Summary\n")
    if not vulkan_confirmed:
        sections.append(
            "**BLOCKED**: wgpu did not select the Vulkan backend on this host.\n"
        )
    elif verdict == "blocked":
        sections.append(
            "Ny wgpu confirmed Vulkan backend. "
            "Reference comparator unavailable — verdict blocked.\n"
        )
    else:
        sections.append(
            "Ny wgpu confirmed Vulkan backend. "
            "Compare-backends and reference data collected.\n"
        )
    sections.append("## Commands\n")
    sections.append("```bash\n")
    sections.append("scripts/run_nvidia_vulkan_validation.sh\n")
    sections.append(
        "python3 scripts/render_nvidia_vulkan_validation_report.py "
        "--manifest <manifest.json> --output-dir <dir>\n"
    )
    sections.append("```\n")
    return sections


def _render_artifacts_and_host(manifest: dict, manifest_dir: Path) -> list[str]:
    """Render artifacts and host facts sections."""
    sections: list[str] = []
    sections.append("## Artifacts\n")
    for key in (
        "host_info_path", "measure_log_path", "measure_csv_path",
        "compare_backends_cersyve_csv", "compare_backends_metaroom_csv",
    ):
        val = manifest.get(key)
        if val:
            sections.append(f"- `{key}`: `{val}`\n")
        else:
            sections.append(f"- `{key}`: (not available)\n")

    adapter_line = manifest.get("adapter_line", "")
    vulkan_confirmed = manifest.get("vulkan_confirmed", False)
    sections.append("\n## Host Facts\n")
    sections.append(f"- Vulkan confirmed: **{vulkan_confirmed}**\n")
    if adapter_line:
        sections.append(f"- Adapter: `{adapter_line}`\n")

    host_info = load_host_info(manifest_dir, manifest)
    sections.append("\n<details><summary>Full host info</summary>\n\n")
    sections.append(f"```\n{host_info}```\n")
    sections.append("</details>\n")
    return sections


def _render_comparison(
    manifest: dict,
    manifest_dir: Path,
    totals: dict[str, float | None],
) -> list[str]:
    """Render derived comparison section."""
    sections: list[str] = []
    reference_real = manifest.get("reference_cersyve_real_seconds")

    sections.append("\n## Derived Comparison\n")
    sections.append(f"- `ny_cpu_cersyve_total`: {_fmt(totals.get('ny_cpu_cersyve_total'))}s\n")
    sections.append(f"- `ny_wgpu_cersyve_total`: {_fmt(totals.get('ny_wgpu_cersyve_total'))}s\n")
    sections.append(f"- `backend_speedup_total`: {_fmt(totals.get('backend_speedup_total'))}x\n")
    if reference_real is not None:
        sections.append(f"- `abcrown_cersyve_total`: {reference_real:.2f}s\n")
        ny_wgpu = totals.get("ny_wgpu_cersyve_total")
        if ny_wgpu is not None and ny_wgpu > 0:
            gap = reference_real / ny_wgpu
            sections.append(f"- `reference_gap_total`: {gap:.2f}x\n")
        else:
            sections.append("- `reference_gap_total`: N/A\n")
    else:
        sections.append("- `abcrown_cersyve_total`: N/A\n")
        sections.append("- `reference_gap_total`: N/A\n")
    return sections


def _render_verdict(manifest: dict, totals: dict[str, float | None]) -> list[str]:
    """Render divergence gate and verdict sections."""
    sections: list[str] = []
    vulkan_confirmed = manifest.get("vulkan_confirmed", False)
    cersyve_csv_name = manifest.get("compare_backends_cersyve_csv")

    sections.append("\n## Divergence Gate\n")
    if not vulkan_confirmed:
        sections.append("FAIL: Vulkan backend not confirmed.\n")
    elif not cersyve_csv_name:
        sections.append("SKIP: No compare-backends CSV available.\n")
    else:
        sections.append("PASS: Compare-backends rows collected on same host.\n")

    sections.append("\n## Verdict\n")
    final_verdict = _compute_verdict(manifest, totals)
    if final_verdict == "go":
        sections.append(
            "**go**: NVIDIA/Vulkan is close enough that #4258 should keep "
            "optimizing the existing wgpu path.\n"
        )
    elif final_verdict == "no-go":
        sections.append(
            "**no-go**: NVIDIA/Vulkan is materially non-competitive or fails "
            "to initialize reliably. Evaluate alternate backend strategy.\n"
        )
    else:
        blocker = manifest.get("blocker")
        reference_blocker = manifest.get("reference_blocker")
        reason = blocker or reference_blocker or "unknown blocker"
        sections.append(f"**blocked**: {reason}\n")
    return sections


def render_report(manifest: dict, manifest_dir: Path) -> str:
    """Render the full validation report Markdown."""
    cersyve_csv_name = manifest.get("compare_backends_cersyve_csv")
    totals: dict[str, float | None] = {}
    if cersyve_csv_name:
        cersyve_path = manifest_dir / cersyve_csv_name
        if cersyve_path.is_file():
            rows = load_csv_rows(cersyve_path)
            totals = compute_cersyve_totals(rows)

    sections: list[str] = []
    sections.extend(_render_header_sections(manifest))
    sections.extend(_render_artifacts_and_host(manifest, manifest_dir))
    sections.extend(_render_comparison(manifest, manifest_dir, totals))
    sections.extend(_render_verdict(manifest, totals))
    return "\n".join(sections)


def _compute_verdict(
    manifest: dict, totals: dict[str, float | None]
) -> str:
    """Compute go / no-go / blocked from manifest + totals."""
    if not manifest.get("vulkan_confirmed", False):
        return "blocked"
    if manifest.get("reference_blocker"):
        return "blocked"
    reference_real = manifest.get("reference_cersyve_real_seconds")
    if reference_real is None:
        return "blocked"
    ny_wgpu = totals.get("ny_wgpu_cersyve_total")
    if ny_wgpu is None or ny_wgpu <= 0:
        return "blocked"
    gap = reference_real / ny_wgpu
    if gap < 0.2:
        return "no-go"
    return "go"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Render NVIDIA/Vulkan validation report from orchestrator manifest."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to nvidia_vulkan_validation_manifest_v1 JSON",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for the report",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    manifest_dir = args.manifest.parent
    report = render_report(manifest, manifest_dir)

    output_path = args.output_dir / "issue-4359-nvidia-vulkan-validation-current.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
