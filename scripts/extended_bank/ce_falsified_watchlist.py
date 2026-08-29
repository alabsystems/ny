#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Build and query the EXTENDED-track CE-falsified watchlist.

WHY THIS EXISTS
---------------
The regular 16 tracks already have a bank-time refusal: a promotion whose
published ZERO-TOL truth is ``violated`` cannot be banked as ``unsat``
(``scripts/regular_bank_evidence.py``: "UNSAT promotion requires published
holds truth").  That gate is fed by ``SCORING-ZERO-TOL/latex/longtable.tex``,
which only tabulates the 16 SCORED categories.  The 9 extended tracks were
therefore blind at bank time: ``scripts/extended_bank/validate_bank.py`` banked
every source ``unsat`` unconditionally.

This module supplies the missing ground truth for the extended tracks, taken
from the same corpus the organizers published, and — crucially — NOT from the
``True Result:`` line.  ``process_results.py`` labels a row whose only accepted
counterexample was graded ``correct_up_to_tolerance`` as ``True Result: unsat``
even though a real, organizer-accepted counterexample exists and every
falsifier was paid +10 for it.  Banking ``unsat`` on such a row is a -150.

AUTHORITATIVE SIGNAL
--------------------
Per-tool markers inside ``Row: [...]``:

    Row: ['patch-1-spec_onnx_patch-1_idx_16972_2', '7.0 (v)', '32.5 (v)']

``(v)`` = that tool reported a violation; ``(h)`` = that tool reported holds.
A ``(v)`` claim alone is not falsification — the organizer still validates the
counterexample and pays -150 when it is rejected.  The row is treated as
FIELD-FALSIFIED only when some tool carries ``(v)`` AND the organizer credited
that same tool as a falsifier on that row (``is_fals: True`` with the +10
score).  That pair is the organizer's own accepted-counterexample decision.

BINDING
-------
The integer in the per-tool score lines (``<idx>: <tool> score: ...``) is the
row's position in the category's unfiltered ``instances.csv``.  Every binding
is verified, not assumed: the organizer's row id must be exactly
``<onnx stem>-<vnnlib stem>`` for the ``instances.csv`` row at that position.
``--verify`` reports the ratio and fails closed on a single mismatch.

Usage
-----
  scripts/extended_bank/ce_falsified_watchlist.py build --out <json>
  scripts/extended_bank/ce_falsified_watchlist.py verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_RESULTS_TXT = (
    REPO_ROOT / "external_tools" / "vnncomp2025_results" / "SCORING-ZERO-TOL" / "results.txt"
)
DEFAULT_BENCH_ROOT = REPO_ROOT / "benchmarks" / "vnncomp2025" / "benchmarks"
DEFAULT_WATCHLIST = SCRIPT_PATH.with_name("ce_falsified_extended_v1.json")

SCHEMA = "extended_ce_falsified_watchlist_v1"

# The 9 unscored "extended" tracks. Kept as a literal so this module has no
# import-time dependency on the scorecard; `verify` cross-checks the names
# against the categories actually present in results.txt.
EXTENDED_TRACKS = (
    "collins_aerospace_benchmark",
    "lsnc_relu",
    "relusplitter",
    "cctsdb_yolo_2023",
    "vggnet16_2022",
    "ml4acopf_2024",
    "vit_2023",
    "yolo_2023",
    "traffic_signs_recognition_2023",
)

_CATEGORY = re.compile(r"^Category 2025_(\S+):$")
_ROW = re.compile(r"^Row:\s*\[(.*)\]\s*$")
_CELL = re.compile(r"'([^']*)'")
_TRUE_RESULT = re.compile(r"^True Result:\s*(\S+)")
_SCORE = re.compile(
    r"^(\d+):\s+(\S+)\s+score:\s+(-?\d+),\s+is_ver:\s+(True|False),"
    r"\s+is_fals:\s+(True|False),"
)
_CE_VALIDITY = re.compile(r"^were violated counterexamples valid\?:\s*(\{.*\})\s*$")


class WatchlistError(RuntimeError):
    """The organizer corpus or a binding is not in the expected shape."""


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        pass
    # The corpus is commonly reached through a symlink into another checkout;
    # anchor on the corpus root so the recorded provenance is the same string
    # from every worktree. sha256 remains the integrity binding either way.
    parts = resolved.parts
    if "external_tools" in parts:
        index = len(parts) - 1 - parts[::-1].index("external_tools")
        return PurePosixPath(*parts[index:]).as_posix()
    return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_official_rows(results_txt: Path) -> dict[str, list[dict]]:
    """category -> ordered rows carrying the per-tool (v)/(h) markers.

    A row is emitted only once its score block has been seen, because the score
    lines carry the authoritative row index and the falsifier credit.
    """
    out: dict[str, list[dict]] = {}
    category: str | None = None
    pending: dict | None = None
    ce_validity: dict[str, str] = {}
    with results_txt.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            match = _CATEGORY.match(line)
            if match:
                category = match.group(1)
                out.setdefault(category, [])
                pending = None
                ce_validity = {}
                continue
            if category is None:
                continue
            match = _CE_VALIDITY.match(line)
            if match:
                try:
                    parsed = json.loads(match.group(1).replace("'", '"'))
                except ValueError:
                    parsed = {}
                if isinstance(parsed, dict):
                    ce_validity = {
                        str(tool): str(result) for tool, result in parsed.items()
                    }
                continue
            match = _ROW.match(line)
            if match:
                # Cells are POSITIONAL: cell 0 is the row id, then exactly one
                # cell per participating tool, in the declared tool order. A
                # cell is '<seconds> (v)', '<seconds> (h)', 'timeout' or '-';
                # only the first two carry a verdict claim.
                cells = _CELL.findall(match.group(1))
                pending = {
                    "row_id": cells[0] if cells else None,
                    "markers": [
                        "v" if "(v)" in cell else "h" if "(h)" in cell else None
                        for cell in cells[1:]
                    ],
                    "ce_validity": ce_validity,
                    "true_result": None,
                    "scores": [],
                }
                ce_validity = {}
                continue
            if pending is None:
                continue
            match = _TRUE_RESULT.match(line)
            if match:
                pending["true_result"] = match.group(1)
                continue
            match = _SCORE.match(line)
            if match:
                pending["scores"].append(
                    {
                        "index": int(match.group(1)),
                        "tool": match.group(2),
                        "points": int(match.group(3)),
                        "is_ver": match.group(4) == "True",
                        "is_fals": match.group(5) == "True",
                    }
                )
                continue
            if pending["scores"]:
                # Blank line after the score block closes the row.
                out[category].append(pending)
                pending = None
    if pending is not None and pending["scores"]:
        out[category].append(pending)
    return out


def field_falsified(row: dict, tools: Sequence[str]) -> tuple[bool, list[str]]:
    """Did the FIELD falsify this row with an organizer-ACCEPTED counterexample?

    Requires both halves: the tool's own ``(v)`` marker in the Row cell AND the
    organizer's falsifier credit for that same tool on that same row. A ``(v)``
    whose counterexample was rejected earns -150, not falsification, so it must
    not enter the watchlist.
    """
    # Some rows carry a trailing organizer annotation cell (observed:
    # '*multiple results*' on ml4acopf). Tool cells are the leading
    # len(tools) cells; anything after them is commentary, never a verdict.
    markers = row["markers"]
    if len(markers) < len(tools):
        raise WatchlistError(
            f"row {row['row_id']!r}: {len(markers)} cells for {len(tools)} tools"
        )
    violated = {
        tool
        for tool, mark in zip(tools, markers[: len(tools)], strict=True)
        if mark == "v"
    }
    credited = sorted(
        entry["tool"]
        for entry in row["scores"]
        if entry["is_fals"] and entry["tool"] in violated
    )
    return (bool(credited), credited)


def load_instances(bench_root: Path, track: str) -> list[tuple[str, str]]:
    path = bench_root / track / "instances.csv"
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) < 3:
                raise WatchlistError(f"{path}: short row {fields!r}")
            rows.append((fields[0].strip(), fields[1].strip()))
    if not rows:
        raise WatchlistError(f"{path}: no rows")
    return rows


def expected_row_id(onnx: str, vnnlib: str) -> str:
    return f"{PurePosixPath(onnx).stem}-{PurePosixPath(vnnlib).stem}"


def bind_track(
    track: str,
    official_rows: list[dict],
    instances: list[tuple[str, str]],
) -> tuple[list[dict], int, list[str]]:
    """Positionally bind organizer rows to instances.csv and VERIFY each one."""
    bound: list[dict] = []
    mismatches: list[str] = []
    for row in official_rows:
        indices = {entry["index"] for entry in row["scores"]}
        if len(indices) != 1:
            raise WatchlistError(
                f"{track}: row {row['row_id']!r} has score indices {sorted(indices)}"
            )
        index = indices.pop()
        if not 0 <= index < len(instances):
            mismatches.append(
                f"{track}[{index}]: index outside instances.csv "
                f"(0..{len(instances) - 1})"
            )
            continue
        onnx, vnnlib = instances[index]
        expected = expected_row_id(onnx, vnnlib)
        if row["row_id"] != expected:
            mismatches.append(
                f"{track}[{index}]: organizer row id {row['row_id']!r} != "
                f"instances.csv binding {expected!r}"
            )
            continue
        bound.append({"index": index, "onnx": onnx, "vnnlib": vnnlib, "row": row})
    seen = [entry["index"] for entry in bound]
    if len(set(seen)) != len(seen):
        raise WatchlistError(f"{track}: duplicate row indices in results.txt")
    return bound, len(official_rows), mismatches


def build(
    results_txt: Path,
    bench_root: Path,
    tracks: Sequence[str] = EXTENDED_TRACKS,
) -> dict:
    parsed = parse_official_rows(results_txt)
    entries: list[dict] = []
    verified = 0
    considered = 0
    mismatches: list[str] = []
    per_track: dict[str, dict] = {}
    for track in tracks:
        official_rows = parsed.get(track)
        if not official_rows:
            raise WatchlistError(f"results.txt has no scored rows for track {track!r}")
        tools = _tools_for(results_txt, track)
        instances = load_instances(bench_root, track)
        bound, total, track_mismatches = bind_track(track, official_rows, instances)
        considered += total
        verified += len(bound)
        mismatches.extend(track_mismatches)
        falsified = 0
        for item in bound:
            is_falsified, credited = field_falsified(item["row"], tools)
            if not is_falsified:
                continue
            falsified += 1
            entries.append(
                {
                    "track": track,
                    "index": item["index"],
                    "onnx": item["onnx"],
                    "vnnlib": item["vnnlib"],
                    "row_id": item["row"]["row_id"],
                    "falsifiers": credited,
                    "true_result_line": item["row"]["true_result"],
                    "ce_validity": item["row"]["ce_validity"],
                }
            )
        per_track[track] = {
            "instances": len(instances),
            "official_rows": total,
            "bound": len(bound),
            "falsified": falsified,
        }
    return {
        "schema": SCHEMA,
        "source": {
            # Repo-relative when possible: the absolute path is a property of
            # whichever worktree built the artifact, not of the evidence.
            "results_txt": _repo_relative(results_txt),
            "results_txt_sha256": _sha256(results_txt),
        },
        "binding": {
            "verified": verified,
            "considered": considered,
            "mismatches": mismatches,
        },
        "per_track": per_track,
        "entries": sorted(
            entries, key=lambda entry: (entry["track"], entry["index"])
        ),
    }


_TOOLS_CACHE: dict[tuple[Path, str], tuple[str, ...]] = {}


def _tools_for(results_txt: Path, track: str) -> tuple[str, ...]:
    cached = _TOOLS_CACHE.get((results_txt, track))
    if cached is not None:
        return cached
    header = re.compile(r"participating tools:\s*\[([^\]]*)\]")
    category: str | None = None
    tools: tuple[str, ...] | None = None
    with results_txt.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _CATEGORY.match(line.rstrip("\n"))
            if match:
                category = match.group(1)
                continue
            if category != track:
                continue
            match = header.search(line)
            if match:
                tools = tuple(re.findall(r"'([^']+)'", match.group(1)))
                break
    if not tools:
        raise WatchlistError(f"results.txt has no participating-tool list for {track!r}")
    _TOOLS_CACHE[(results_txt, track)] = tools
    return tools


def load_watchlist(path: Path = DEFAULT_WATCHLIST) -> dict[tuple[str, str, str], dict]:
    """(track, onnx, vnnlib) -> entry, for the bank-time refusal.

    Missing file is a hard error: a silently empty watchlist is exactly the
    blind spot this exists to close.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise WatchlistError(f"{path}: unexpected schema {data.get('schema')!r}")
    if data.get("binding", {}).get("mismatches"):
        raise WatchlistError(f"{path}: watchlist was built with unverified bindings")
    return {
        (entry["track"], entry["onnx"], entry["vnnlib"]): entry
        for entry in data["entries"]
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("build", "verify", "show"))
    parser.add_argument("--results-txt", type=Path, default=DEFAULT_RESULTS_TXT)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--track", action="append", default=None)
    args = parser.parse_args(argv)

    tracks = tuple(args.track) if args.track else EXTENDED_TRACKS
    try:
        payload = build(args.results_txt, args.bench_root, tracks)
    except WatchlistError as error:
        print(f"ce_falsified_watchlist: error: {error}", file=sys.stderr)
        return 2

    binding = payload["binding"]
    print(
        f"binding verified {binding['verified']}/{binding['considered']} "
        f"with {len(binding['mismatches'])} mismatches"
    )
    for track, stats in payload["per_track"].items():
        print(
            f"  {track:34s} instances={stats['instances']:4d} "
            f"bound={stats['bound']:4d} CE-FALSIFIED={stats['falsified']:3d}"
        )
    print(f"  {'TOTAL':34s} CE-FALSIFIED={len(payload['entries'])}")
    if binding["mismatches"]:
        for line in binding["mismatches"][:20]:
            print(f"  MISMATCH {line}", file=sys.stderr)
        return 1
    if args.action == "show":
        for entry in payload["entries"]:
            print(
                f"{entry['track']},{entry['index']},{entry['onnx']},"
                f"{entry['vnnlib']},{','.join(entry['falsifiers'])}"
            )
    if args.action == "build":
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
