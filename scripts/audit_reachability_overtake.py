#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Audit method-specific reachability targets in VNN-COMP 2025 results.

This is deliberately narrower than ``main16_gap_audit.py``.  It answers three
questions using occurrence-aware official rows and the published ZERO-TOL
result table:

* where did PyRAT report UNSAT while NY and alpha-beta-CROWN did not decide?
* where can a PyRAT-style constrained-zonotope implementation catch up?
* which CIFAR rows are suggested *only* by NNV's probabilistic ``cp-star`` run?

The last class is a hint queue, never ground-truth or proof authority.  A row is
bankable only after NY emits its own sound certificate.

Example::

  python3 scripts/audit_reachability_overtake.py \
    --official /data/vnncomp2025_results \
    --measured reports/measured
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ny_retroactive_scorecard as retro  # noqa: E402

REPO = SCRIPT_DIR.parent
SCHEMA = "ny_reachability_overtake_audit_v1"
DECISIONS = frozenset(("holds", "violated"))
STRATEGIES = (
    "pyrat_constrained_zono_overtake",
    "pyrat_constrained_zono_catchup",
    "nnv_cp_star_only_hint",
    "abc_unsat_catchup",
)


class AuditError(RuntimeError):
    """The requested audit inputs are missing or internally inconsistent."""


def _require_dir(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"{label} is unavailable: {path}") from error
    if not resolved.is_dir():
        raise AuditError(f"{label} is not a directory: {resolved}")
    return resolved


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise AuditError(f"required {label} is missing: {path}")
    return path


def _fingerprint(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    return {
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _load_official(
    root: Path,
) -> tuple[
    dict[str, list[tuple]],
    dict[str, dict[tuple, str]],
    dict[str, dict[str, dict[tuple, str]]],
    list[dict[str, object]],
]:
    root = _require_dir(root, "official result root")
    reference_csv = _require_file(
        root / "alpha_beta_crown" / "results.csv", "alpha-beta-CROWN results"
    )
    longtable = _require_file(
        root / "SCORING-ZERO-TOL" / "latex" / "longtable.tex",
        "ZERO-TOL longtable",
    )

    files = [reference_csv, longtable]
    tools: dict[str, dict[str, dict[tuple, str]]] = {}
    for tool in retro.OFFICIAL_TOOLS:
        result_csv = _require_file(root / tool / "results.csv", f"{tool} results")
        files.append(result_csv)
        tools[tool] = retro.load_tool_csv(result_csv)

    try:
        order = retro.load_reference_instance_order(reference_csv)
        truth = retro.load_published_ground_truth(longtable, order)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise AuditError(f"official result artifacts are invalid: {error}") from error

    missing = [category for category in retro.REGULAR if not order.get(category)]
    if missing:
        raise AuditError(
            "official reference has no rows for: " + ", ".join(sorted(missing))
        )
    fingerprints = [_fingerprint(path, root) for path in sorted(set(files))]
    return order, truth, tools, fingerprints


def _load_ny(
    root: Path,
) -> tuple[dict[str, dict[tuple, str]], list[dict[str, object]]]:
    root = _require_dir(root, "NY measured root")
    results: dict[str, dict[tuple, str]] = defaultdict(dict)
    files: list[Path] = []
    for category in retro.REGULAR:
        path = _require_file(root / f"{category}.csv", f"NY {category} results")
        files.append(path)
        loaded = retro.load_tool_csv(path)
        unexpected = sorted(set(loaded) - {category})
        if unexpected:
            raise AuditError(
                f"NY file {path} contains other categories: {', '.join(unexpected)}"
            )
        results[category].update(loaded.get(category, {}))
    return results, [_fingerprint(path, root) for path in files]


def _target_row(
    category: str,
    index: int,
    instance: tuple,
    truth: str,
    ny_verdict: str,
    tools: dict[str, dict[str, dict[tuple, str]]],
) -> dict[str, Any]:
    onnx, vnnlib, occurrence = instance
    return {
        "category": category,
        "instance_id": index,
        "onnx": onnx,
        "vnnlib": vnnlib,
        "occurrence": occurrence,
        "published_zero_tol_result": truth,
        "ny": ny_verdict,
        "official_tools": {
            tool: tools[tool].get(category, {}).get(instance, "unknown")
            for tool in retro.OFFICIAL_TOOLS
        },
    }


def build_audit(official_root: Path, measured_root: Path) -> dict[str, Any]:
    order, truth, tools, official_files = _load_official(official_root)
    ny, measured_files = _load_ny(measured_root)
    targets: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}

    for category in retro.REGULAR:
        reference_rows = order[category]
        reference_set = set(reference_rows)
        unexpected = sorted(set(ny.get(category, {})) - reference_set)
        if unexpected:
            raise AuditError(
                f"NY {category} has rows outside the official occurrence order"
            )

        for index, instance in enumerate(reference_rows):
            published = truth.get(category, {}).get(instance, "unknown")
            ny_verdict = ny.get(category, {}).get(instance, "unknown")
            if published != "holds" or ny_verdict in DECISIONS:
                continue

            verdicts = {
                tool: tools[tool].get(category, {}).get(instance, "unknown")
                for tool in retro.OFFICIAL_TOOLS
            }
            row = _target_row(
                category,
                index,
                instance,
                published,
                ny_verdict,
                tools,
            )

            if verdicts["pyrat"] == "holds":
                if verdicts["alpha_beta_crown"] not in DECISIONS:
                    targets["pyrat_constrained_zono_overtake"].append(row)
                elif verdicts["alpha_beta_crown"] == "holds":
                    targets["pyrat_constrained_zono_catchup"].append(row)

            if verdicts["alpha_beta_crown"] == "holds":
                targets["abc_unsat_catchup"].append(row)

            if category == "cifar100_2024" and verdicts["nnv"] == "holds":
                other_deciders = [
                    tool
                    for tool, verdict in verdicts.items()
                    if tool != "nnv" and verdict in DECISIONS
                ]
                if not other_deciders:
                    targets["nnv_cp_star_only_hint"].append(row)

    metadata = {
        "pyrat_constrained_zono_overtake": {
            "authority": "official_result_method_signal",
            "meaning": (
                "published UNSAT; PyRAT holds; alpha-beta-CROWN and NY have no decision"
            ),
        },
        "pyrat_constrained_zono_catchup": {
            "authority": "official_result_method_signal",
            "meaning": (
                "published UNSAT; PyRAT and alpha-beta-CROWN hold; NY has no decision"
            ),
        },
        "nnv_cp_star_only_hint": {
            "authority": "hint_only_not_proof_or_ground_truth",
            "meaning": (
                "CIFAR published UNSAT; NNV holds; every other official tool and NY "
                "have no decision"
            ),
        },
        "abc_unsat_catchup": {
            "authority": "official_method_target",
            "meaning": ("published UNSAT; alpha-beta-CROWN holds; NY has no decision"),
        },
    }
    summaries: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        by_category = Counter(row["category"] for row in targets[strategy])
        summaries.append(
            {
                "strategy": strategy,
                "authority": metadata[strategy]["authority"],
                "count": len(targets[strategy]),
                "by_category": dict(sorted(by_category.items())),
            }
        )

    return {
        "schema": SCHEMA,
        "method_note": (
            "NNV VNN-COMP 2025 CIFAR used probabilistic cp-star; its sole holds "
            "are prioritization hints only. NY must produce an independent sound proof."
        ),
        "definitions": metadata,
        "summary": summaries,
        "targets": targets,
        "inputs": {
            "official_root": str(official_root.resolve()),
            "measured_root": str(measured_root.resolve()),
            "official_files": official_files,
            "measured_files": measured_files,
        },
    }


def _render_table(audit: dict[str, Any]) -> str:
    lines = [
        "NY REACHABILITY OVERTAKE AUDIT",
        "strategy                              authority                         count  categories",
    ]
    for item in audit["summary"]:
        categories = (
            ", ".join(
                f"{category}={count}" for category, count in item["by_category"].items()
            )
            or "-"
        )
        lines.append(
            f"{item['strategy']:<37} {item['authority']:<33} "
            f"{item['count']:>5}  {categories}"
        )

    lines.extend(["", "PRIMARY PYRAT-ONLY OVERTAKE TARGETS"])
    primary = audit["targets"]["pyrat_constrained_zono_overtake"]
    if not primary:
        lines.append("(none)")
    else:
        for row in primary:
            verdicts = row["official_tools"]
            lines.append(
                f"{row['category']}:{row['instance_id']}  {row['onnx']}  "
                f"{row['vnnlib']}  ny={row['ny']} abc={verdicts['alpha_beta_crown']} "
                f"pyrat={verdicts['pyrat']}"
            )
    lines.extend(
        [
            "",
            "SAFETY: nnv_cp_star_only_hint is not proof or independent ground truth;",
            "bank a row only after NY emits its own sound certificate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official",
        type=Path,
        default=REPO / "external_tools" / "vnncomp2025_results",
        help="root of the official VNN-COMP 2025 result repository",
    )
    parser.add_argument(
        "--measured",
        type=Path,
        default=REPO / "reports" / "measured",
        help="root containing NY's per-category measured CSVs",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--json-out", type=Path, help="also write the complete audit as JSON"
    )
    args = parser.parse_args(argv)

    try:
        audit = build_audit(args.official, args.measured)
    except AuditError as error:
        parser.error(str(error))

    payload = _json_text(audit)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    if args.format == "json":
        sys.stdout.write(payload)
    else:
        sys.stdout.write(_render_table(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
