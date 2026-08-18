#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Emit the regular-track FIELD-FALSIFIED watchlist as machine-readable JSON.

WHY THIS EXISTS
---------------
``longtable.tex`` and ``True Result`` both print ``unsat`` on rows the VNN-COMP
field actually FALSIFIED.  ``process_results.py`` only promotes ``true_result``
to ``sat`` when some tool's counterexample graded exactly ``CORRECT``; a row
whose every accepted counterexample graded ``CORRECT_UP_TO_TOLERANCE`` keeps the
``unsat`` label even though the tools that produced those witnesses were each
paid +10 falsifier credit.  Reading the label alone therefore hands an ny
``unsat`` a silent +10 on an instance the field PROVED violable.

This script freezes the authoritative signal — the per-row ``is_fals`` credit in
``SCORING-ZERO-TOL/results.txt`` — into a file the banking path can enforce
against without re-parsing 200k lines of organizer log on every row.

Usage
-----
  scripts/emit_ce_falsified_watchlist.py
      [--official external_tools/vnncomp2025_results]
      [--out reports/ce_falsified_watchlist.json] [--check]

``--check`` regenerates in memory and exits nonzero if the on-disk file is stale,
so CI can prove the committed watchlist still matches the official artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Reuse the scorecard's official-results model rather than re-deriving it.
_SCORECARD_PATH = REPO / "scripts" / "ny_retroactive_scorecard.py"
_spec = importlib.util.spec_from_file_location("ny_retroactive_scorecard", _SCORECARD_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - repo layout invariant
    raise SystemExit(f"cannot import the scorecard module at {_SCORECARD_PATH}")
scorecard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = scorecard
_spec.loader.exec_module(scorecard)

WATCHLIST_VERSION = 1
DEFAULT_OUT = REPO / "reports" / "ce_falsified_watchlist.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(official_dir: Path) -> dict:
    """Build the watchlist payload from the official ZERO-TOL artifacts."""
    results_txt = official_dir / "SCORING-ZERO-TOL" / "results.txt"
    reference_csv = official_dir / "alpha_beta_crown" / "results.csv"
    reference_order = scorecard.load_reference_instance_order(reference_csv)
    falsified = scorecard.load_accepted_ce_falsifications(results_txt, reference_order)

    categories: dict[str, list[dict]] = {}
    total = mislabelled = 0
    for cat in scorecard.REGULAR:
        rows = falsified.get(cat, {})
        if not rows:
            continue
        entries = []
        for (onnx, vnnlib, occurrence), info in sorted(
            rows.items(), key=lambda kv: kv[1]["idx"]
        ):
            entries.append(
                {
                    "index": info["idx"],
                    "row_id": info["rid"],
                    "onnx": onnx,
                    "vnnlib": vnnlib,
                    "occurrence": occurrence,
                    "published_true_result": info["true"],
                    "falsifiers": info["falsifiers"],
                    "ce_grades": info["ce"],
                }
            )
        total += len(entries)
        mislabelled += sum(1 for e in entries if e["published_true_result"] == "unsat")
        categories[cat] = entries

    return {
        "version": WATCHLIST_VERSION,
        "what": (
            "regular-track VNN-COMP 2025 instances the field falsified with a "
            "counterexample the organizers ACCEPTED (is_fals credit paid). An ny "
            "'unsat' on any of these contradicts an accepted witness and must be "
            "refused at bank time."
        ),
        "source": {
            "results_txt": str(results_txt.relative_to(official_dir)),
            "results_txt_sha256": _sha256(results_txt),
            "reference_results_csv": str(reference_csv.relative_to(official_dir)),
            "reference_results_csv_sha256": _sha256(reference_csv),
        },
        "counts": {
            "falsified_rows": total,
            "labelled_unsat_anyway": mislabelled,
            "categories": len(categories),
        },
        "categories": categories,
    }


def load(path: Path | None = None) -> dict[str, set[tuple[str, str, int]]]:
    """Load the watchlist as ``cat -> {scorecard instance key}``.

    Returns an empty mapping when the file is absent so callers can distinguish
    "no watchlist available" from "watchlist says this row is clean"; every
    caller that enforces MUST treat the empty case as a hard error of its own.
    """
    path = path or DEFAULT_OUT
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != WATCHLIST_VERSION:
        raise ValueError(
            f"{path}: watchlist version {payload.get('version')!r} is not the "
            f"expected {WATCHLIST_VERSION}; regenerate with "
            "scripts/emit_ce_falsified_watchlist.py"
        )
    out: dict[str, set[tuple[str, str, int]]] = {}
    for cat, entries in payload.get("categories", {}).items():
        out[cat] = {
            (e["onnx"], e["vnnlib"], int(e["occurrence"])) for e in entries
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--official", default="external_tools/vnncomp2025_results")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the on-disk watchlist differs from a fresh build",
    )
    args = ap.parse_args()

    official_dir = Path(args.official)
    if not official_dir.is_absolute():
        official_dir = (REPO / args.official).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO / args.out).resolve()

    payload = build(official_dir)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not out_path.is_file():
            print(f"MISSING: {out_path} does not exist", file=sys.stderr)
            return 1
        if out_path.read_text(encoding="utf-8") != rendered:
            print(
                f"STALE: {out_path} differs from a fresh build of the official "
                "artifacts; regenerate with scripts/emit_ce_falsified_watchlist.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {out_path} matches the official artifacts")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    counts = payload["counts"]
    print(
        f"wrote {out_path}: {counts['falsified_rows']} field-falsified row(s) "
        f"across {counts['categories']} regular categories; "
        f"{counts['labelled_unsat_anyway']} of them still publish 'unsat'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
