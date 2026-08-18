#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Banked-row reproduction check: do the ledgers still reproduce at HEAD?

WHY THIS EXISTS
---------------
Two fail-closed gates each silently zeroed a category that was already banked,
and nothing noticed for months:

* ``25dee0c5`` (2026-07-25) added a Cast-dtype admission gate; ``cctsdb_yolo_2023``
  (banked at 100.0 normalized on 2026-07-21) stopped LOADING.
* ``1ede1d30`` (2026-07-26) quarantined the WGPU proof adapter; 16 presets that
  declare ``device: wgpu`` silently fell back to the CPU verifier, taking
  ``vit_2023`` from 9 banked unsats to 1 and breaking the MANDATED
  ``soundnessbench`` 3/3-sat gate itself.

Both gates are RIGHT about soundness. The defect is that their CAPABILITY COST
was never measured. This script is the cheap, routine measurement: it replays a
small, information-dense subset of currently banked SOLVED rows through the real
competition entry point (``ny vnncomp``) with its preset, and — crucially —
NAMES THE CAUSE of every non-reproduction, because a load failure, a silent
backend substitution and a genuine bound regression are three different
emergencies.

Pairs with the two STATIC guards, which need no runs at all and would have
caught both instances on the commit that introduced them:

* ``preset::model_load_smoke_tests``    — every shipped preset's models still load
* ``preset::backend_capability_tests``  — every preset's declared ``device:`` is
  honoured at runtime or covered by a dated waiver in
  ``configs/backend_capability_waivers.yaml``

Usage
-----
  scripts/check_banked_rows.py --bin /path/to/release/ny [--per-category 2]
      [--categories vit_2023,cctsdb_yolo_2023] [--budget-cap 60]
      [--out reports/banked_row_check.json]

MEASUREMENT HYGIENE (enforced, not advisory)
--------------------------------------------
* ``--bin`` must be a release binary outside the repo working tree.
* ``--configs-dir`` must be the ``configs/`` ROOT, never ``configs/vnncomp25``:
  the subdirectory silently loads NO preset and runs auto defaults.
* every run is wrapped in ``timeout -k 30`` and its return code inspected
  (137 = killed, which is NOT a verdict).
* a run whose log lacks ``Loading preset:`` is discarded as INVALID, never
  reported as a regression.
* runs are SERIAL by default; this box is shared.
* the official ground truth powers the ONLY wrong-verdict check here
  (``UNSOUND-VERDICT-FLIP``). Missing it never degrades silently: fatal when it
  was asked for, and a loud PARTIAL verdict otherwise.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OFFICIAL = REPO / "external_tools/vnncomp2025_results"

# --- reuse, do not reinvent -------------------------------------------------
# scripts/ny_retroactive_scorecard.py owns the official-results model (raw
# verdict canonicalisation, subdir-aware instance keys, exact per-occurrence
# official budgets). Import it rather than re-deriving any of that here.
_SCORECARD_PATH = REPO / "scripts" / "ny_retroactive_scorecard.py"
_spec = importlib.util.spec_from_file_location("ny_retroactive_scorecard", _SCORECARD_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - repo layout invariant
    raise SystemExit(f"cannot import the scorecard module at {_SCORECARD_PATH}")
scorecard = importlib.util.module_from_spec(_spec)
# Register before exec: the module defines frozen dataclasses, whose creation
# looks the owning module up in sys.modules.
sys.modules[_spec.name] = scorecard
_spec.loader.exec_module(scorecard)

_WATCHLIST_PATH = REPO / "scripts" / "emit_ce_falsified_watchlist.py"
_wspec = importlib.util.spec_from_file_location(
    "emit_ce_falsified_watchlist", _WATCHLIST_PATH
)
if _wspec is None or _wspec.loader is None:  # pragma: no cover - repo layout invariant
    raise SystemExit(f"cannot import the watchlist module at {_WATCHLIST_PATH}")
watchlist_mod = importlib.util.module_from_spec(_wspec)
sys.modules[_wspec.name] = watchlist_mod
_wspec.loader.exec_module(watchlist_mod)

SOLVED = ("sat", "unsat")

# Cause taxonomy. The whole point of this script: a diff is not a diagnosis.
CAUSE_INVALID_NO_PRESET = "INVALID-RUN(preset-not-loaded)"
CAUSE_MODEL_LOAD = "MODEL-LOAD-FAILURE"
CAUSE_BACKEND_OVERRIDE = "BACKEND-OVERRIDE"
CAUSE_UNSOUND = "UNSOUND-VERDICT-FLIP"
CAUSE_HARNESS_KILLED = "HARNESS-KILLED(not-a-verdict)"
CAUSE_CRASH = "CRASH"
CAUSE_BUDGET_WALL = "BOUND-REGRESSION(budget-wall)"
CAUSE_EARLY_GIVE_UP = "BOUND-REGRESSION(early-give-up)"
CAUSE_MISSING_INPUT = "MISSING-INPUT"
CAUSE_UNCLASSIFIED = "UNCLASSIFIED-REGRESSION"

PRESET_MARKER = "Loading preset:"
BACKEND_OVERRIDE_MARKER = "NY-HARNESS: BACKEND-OVERRIDE"
LOAD_FAILURE_MARKERS = (
    "NY-HARNESS: MODEL-LOAD-FAILURE",
    "Model loading failed",
    "MODEL-LOAD-FAILURE",
)
CRASH_MARKERS = (
    "panicked at",
    "SIGSEGV",
    "memory allocation of",
    "Illegal instruction",
)
# Lines worth surfacing verbatim: both real instances announced themselves.
EVIDENCE_PATTERNS = (
    re.compile(r"^.*NY-HARNESS: MODEL-LOAD-FAILURE.*$", re.MULTILINE),
    re.compile(r"^.*NY-HARNESS: BACKEND-OVERRIDE.*$", re.MULTILINE),
    re.compile(r"^.*quarantined.*overriding the requested backend.*$", re.MULTILINE),
    re.compile(r"^.*targets unsupported dtype.*$", re.MULTILINE),
    re.compile(r"^.*Model loading failed.*$", re.MULTILINE),
    re.compile(r"^.*panicked at.*$", re.MULTILINE),
    re.compile(r"^.*no preset found for category.*$", re.MULTILINE),
)


@dataclass
class BankedRow:
    """One banked SOLVED ledger row."""

    category: str
    onnx: str
    vnnlib: str
    verdict: str
    seconds: float
    ledger: str


@dataclass
class RowResult:
    category: str
    onnx: str
    vnnlib: str
    banked_verdict: str
    banked_seconds: float
    ledger: str
    budget_seconds: float
    observed_verdict: str
    elapsed_seconds: float
    return_code: int
    preset_loaded: str | None
    reproduced: bool
    cause: str | None
    # Whether the log carried a silent backend substitution, reproduced or not.
    backend_override: bool
    official_ground_truth: str | None
    evidence: list[str] = field(default_factory=list)
    log_path: str | None = None


def load_ledger(path: Path) -> list[BankedRow]:
    """Parse a headerless measured/extended ledger CSV.

    Accepted schemas (see scripts/extended_bank/validate_bank.py):
      measured:      track,onnx,vnnlib,prepared,verdict,seconds[,run_id]
      extended bank: track,onnx,vnnlib,verdict,seconds
    """
    rows: list[BankedRow] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for parts in csv.reader(fh):
            if len(parts) < 5 or not parts[0].strip():
                continue
            if parts[0].strip().lower() in ("track", "category", "cat"):
                continue
            category, onnx, vnnlib = (parts[0].strip(), parts[1].strip(), parts[2].strip())
            if len(parts) >= 6 and parts[3].strip().lower() in ("0", "prepared"):
                verdict, seconds = parts[4].strip().lower(), parts[5].strip()
            else:
                verdict, seconds = parts[3].strip().lower(), parts[4].strip()
            if verdict not in SOLVED:
                continue
            # test_nano/test_tiny are unscored harness-overhead instances: the
            # official processor drops them, and reproducing one proves nothing
            # about a category's capability.
            if scorecard.is_harness_test_instance(onnx, vnnlib):
                continue
            try:
                elapsed = float(seconds)
            except ValueError:
                elapsed = float("nan")
            rows.append(
                BankedRow(
                    category=category,
                    onnx=onnx,
                    vnnlib=vnnlib,
                    verdict=verdict,
                    seconds=elapsed,
                    ledger=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                )
            )
    return rows


def audit_field_falsified_unsats(
    rows: list[BankedRow], watchlist_path: Path
) -> list[BankedRow]:
    """Banked ``unsat`` rows on instances the field FALSIFIED. Needs no runs.

    This is the STATIC half of the field-falsified gate: ny_measured_sweep.py
    refuses to write such a row, and this catches any that predate the gate or
    arrived through another path.  Matching is on the subdir-aware instance key
    at ANY occurrence, because a deduplicated ledger row cannot say which
    occurrence of a repeated ONNX/VNN-LIB pair it measured.
    """
    listed = watchlist_mod.load(watchlist_path)
    if not listed:
        return []
    bases: dict[str, set[tuple[str, str]]] = {
        cat: {(onnx, vnnlib) for onnx, vnnlib, _occ in keys}
        for cat, keys in listed.items()
    }
    offenders = []
    for row in rows:
        if row.verdict != "unsat":
            continue
        if scorecard.key(row.onnx, row.vnnlib) in bases.get(row.category, set()):
            offenders.append(row)
    return offenders


def select_subset(rows: list[BankedRow], per_category: int) -> list[BankedRow]:
    """Pick the highest-information-value rows per category.

    Information value, in priority order:
      1. BOTH VERDICT DIRECTIONS — an unsat exercises the bound path, a sat the
         falsifier/witness path. A gate can break one and leave the other.
      2. DISTINCT MODELS — distinct load paths. cctsdb's regression was a LOAD
         failure, so a subset that reuses one ONNX file per category is blind to
         a per-model gate.
      3. FASTEST FIRST within a bucket — the check must stay cheap enough to run
         routinely, and a row banked at 4 s reproduces in ~4 s.
    """
    chosen: list[BankedRow] = []
    for category in sorted({row.category for row in rows}):
        pool = [row for row in rows if row.category == category]

        def sort_key(row: BankedRow) -> tuple:
            seconds = row.seconds if row.seconds == row.seconds else 1e9
            # Prefer the FASTEST STRICTLY-POSITIVE solve: a row banked at 0.0 s
            # reproduces without the verifier doing measurable work, so it is a
            # weak signal for a bound regression (it stays a perfect signal for a
            # load failure or a backend substitution, hence the fallback).
            return (seconds <= 0.0, seconds, row.onnx, row.vnnlib)

        picked: list[BankedRow] = []
        used_models: set[str] = set()
        # Pass 1: one fastest row per (verdict, model) pair, alternating verdicts
        # so both directions appear before a second row of either.
        by_verdict: dict[str, list[BankedRow]] = defaultdict(list)
        for row in sorted(pool, key=sort_key):
            by_verdict[row.verdict].append(row)
        order = [v for v in ("unsat", "sat") if by_verdict[v]]
        cursor = {v: 0 for v in order}
        while len(picked) < per_category and order:
            progressed = False
            for verdict in list(order):
                if len(picked) >= per_category:
                    break
                bucket = by_verdict[verdict]
                index = cursor[verdict]
                while index < len(bucket) and bucket[index].onnx in used_models:
                    index += 1
                if index < len(bucket):
                    row = bucket[index]
                    picked.append(row)
                    used_models.add(row.onnx)
                    cursor[verdict] = index + 1
                    progressed = True
                else:
                    cursor[verdict] = index
                    order.remove(verdict)
            if not progressed:
                break
        # Pass 2: top up with the fastest remaining rows (model reuse allowed).
        if len(picked) < per_category:
            for row in sorted(pool, key=sort_key):
                if len(picked) >= per_category:
                    break
                if row not in picked:
                    picked.append(row)
        chosen.extend(picked)
    return chosen


def official_budgets(benchmark_root: Path, category: str) -> dict[tuple[str, str], float]:
    """Exact official per-instance budgets, keyed subdir-aware."""
    try:
        occurrences = scorecard.load_official_instance_occurrences(benchmark_root, category)
    except Exception:  # noqa: BLE001 - a missing/odd list must not abort the check
        return {}
    out: dict[tuple[str, str], float] = {}
    for occurrence in occurrences:
        out.setdefault(
            scorecard.key(occurrence.onnx, occurrence.vnnlib),
            float(occurrence.timeout_seconds),
        )
    return out


def official_ground_truth(official_dir: Path) -> dict[str, dict[tuple, str]]:
    """cat -> instance key -> {violated, holds, unknown} from the official field.

    Raw-verdict model, identical to scripts/ny_retroactive_scorecard.py: an
    instance is ``violated`` if any official tool reported sat, else ``holds`` if
    any reported unsat.
    """
    tools = [path.name for path in sorted(official_dir.iterdir()) if path.is_dir()]
    merged: dict[str, dict[tuple, str]] = defaultdict(dict)
    for tool in tools:
        results = scorecard.load_tool_csv(official_dir / tool / "results.csv")
        for category, instances in results.items():
            for instance, verdict in instances.items():
                current = merged[category].get(instance)
                if verdict == "violated":
                    merged[category][instance] = "violated"
                elif verdict == "holds" and current != "violated":
                    merged[category][instance] = "holds"
                elif current is None:
                    merged[category][instance] = "unknown"
    return merged


def extract_evidence(log: str) -> list[str]:
    seen: list[str] = []
    for pattern in EVIDENCE_PATTERNS:
        for match in pattern.findall(log):
            line = match.strip()
            if line and line not in seen:
                seen.append(line)
    return seen[:12]


def parse_result_file(path: Path) -> str:
    """Read the competition result file ny writes (first token is the verdict)."""
    if not path.is_file():
        return "no-result-file"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return "no-result-file"
    first = text.splitlines()[0].strip().lower()
    for token in ("unsat", "sat", "timeout", "unknown", "error"):
        if first.startswith(token):
            return token
    return first[:32] or "unknown"


def has_backend_override(log: str) -> bool:
    return BACKEND_OVERRIDE_MARKER in log or (
        "quarantined" in log and "overriding the requested backend" in log
    )


def classify(
    row: BankedRow,
    observed: str,
    log: str,
    return_code: int,
    elapsed: float,
    budget: float,
) -> tuple[bool, str | None]:
    """Reproduced? If not, WHICH EMERGENCY is it?

    Returns a PRIMARY cause, optionally suffixed with ``+BACKEND-OVERRIDE`` as a
    PROVENANCE MODIFIER. The modifier is deliberately not a primary cause: the
    substitution marker appears in every wgpu preset's log, including rows that
    still reproduce, so its presence proves the row's provenance is invalid — not
    that it caused this particular verdict change.
    """
    if PRESET_MARKER not in log:
        # Never report this as a regression: the row is simply invalid.
        return False, CAUSE_INVALID_NO_PRESET
    if observed == row.verdict:
        return True, None
    modifier = f"+{CAUSE_BACKEND_OVERRIDE}" if has_backend_override(log) else ""
    opposite = {"sat": "unsat", "unsat": "sat"}[row.verdict]
    if observed == opposite:
        # Soundness beats every other consideration; no modifier noise.
        return False, CAUSE_UNSOUND
    if any(marker in log for marker in LOAD_FAILURE_MARKERS):
        return False, f"{CAUSE_MODEL_LOAD}{modifier}"
    if return_code in (124, 137, -9):
        return False, f"{CAUSE_HARNESS_KILLED}{modifier}"
    if any(marker in log for marker in CRASH_MARKERS):
        return False, f"{CAUSE_CRASH}{modifier}"
    if observed in ("timeout", "unknown"):
        primary = CAUSE_BUDGET_WALL if elapsed >= 0.85 * budget else CAUSE_EARLY_GIVE_UP
        return False, f"{primary}{modifier}"
    return False, f"{CAUSE_UNCLASSIFIED}{modifier}"


def run_row(
    row: BankedRow,
    *,
    binary: Path,
    configs_dir: Path,
    benchmark_root: Path,
    budget: float,
    log_dir: Path,
    grace: int,
) -> RowResult:
    category_dir = benchmark_root / row.category
    onnx = category_dir / row.onnx.lstrip("./")
    vnnlib = category_dir / row.vnnlib.lstrip("./")
    stem = f"{row.category}__{Path(row.onnx).stem}__{Path(row.vnnlib).stem}"
    log_path = log_dir / f"{stem}.log"
    result_path = log_dir / f"{stem}.res.txt"

    if not onnx.is_file() or not vnnlib.is_file():
        missing = [str(path) for path in (onnx, vnnlib) if not path.is_file()]
        return RowResult(
            category=row.category,
            onnx=row.onnx,
            vnnlib=row.vnnlib,
            banked_verdict=row.verdict,
            banked_seconds=row.seconds,
            ledger=row.ledger,
            budget_seconds=budget,
            observed_verdict="missing-input",
            elapsed_seconds=0.0,
            return_code=-1,
            preset_loaded=None,
            reproduced=False,
            cause=CAUSE_MISSING_INPUT,
            backend_override=False,
            official_ground_truth=None,
            evidence=[f"missing input: {path}" for path in missing],
        )

    command = [
        "timeout",
        "-k",
        "30",
        str(int(budget) + grace),
        str(binary),
        "vnncomp",
        "v1",
        row.category,
        str(onnx),
        str(vnnlib),
        str(result_path),
        str(int(budget)),
        "--configs-dir",
        str(configs_dir),
    ]
    started = time.monotonic()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    log = f"$ {' '.join(command)}\n{completed.stdout}\n{completed.stderr}"
    log_path.write_text(log, encoding="utf-8")

    observed = parse_result_file(result_path)
    if completed.returncode in (124, 137) and observed == "no-result-file":
        observed = "killed"
    reproduced, cause = classify(row, observed, log, completed.returncode, elapsed, budget)
    preset_line = None
    for line in log.splitlines():
        if PRESET_MARKER in line:
            preset_line = line.split(PRESET_MARKER, 1)[1].strip()
            break
    return RowResult(
        category=row.category,
        onnx=row.onnx,
        vnnlib=row.vnnlib,
        banked_verdict=row.verdict,
        banked_seconds=row.seconds,
        ledger=row.ledger,
        budget_seconds=budget,
        observed_verdict=observed,
        elapsed_seconds=round(elapsed, 2),
        return_code=completed.returncode,
        preset_loaded=preset_line,
        reproduced=reproduced,
        cause=cause,
        backend_override=has_backend_override(log),
        official_ground_truth=None,
        evidence=extract_evidence(log),
        log_path=str(log_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bin", type=Path, help="release ny binary (required unless --list-only)"
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=REPO / "configs",
        help="preset ROOT (configs/), never configs/vnncomp25",
    )
    parser.add_argument("--measured", type=Path, default=REPO / "reports/measured")
    parser.add_argument("--ext-measured", type=Path, default=REPO / "reports/measured-ext")
    parser.add_argument(
        "--benchmark-root", type=Path, default=REPO / "benchmarks/vnncomp2025/benchmarks"
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=None,
        help=(
            f"official VNN-COMP results tree (default {DEFAULT_OFFICIAL}); it is the "
            f"ground truth for the {CAUSE_UNSOUND} check. Passing this explicitly makes "
            "that check MANDATORY — a missing tree is then fatal."
        ),
    )
    parser.add_argument(
        "--require-ground-truth",
        action="store_true",
        help=(
            f"fail instead of warning when no official ground truth is available, i.e. "
            f"refuse to run the check at all without the {CAUSE_UNSOUND} arm"
        ),
    )
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=watchlist_mod.DEFAULT_OUT,
        help=(
            "field-falsified watchlist JSON; a banked unsat on any listed "
            "instance fails the check before any row is run"
        ),
    )
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--categories", default="", help="comma-separated allowlist")
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=0.0,
        help="cap each row's budget (0 = use the exact official budget). A cap makes the check "
        "cheap but can turn a slow solve into a false 'budget-wall' regression, so a capped run "
        "reports CAPPED and must not be used to declare a bound regression.",
    )
    parser.add_argument("--grace", type=int, default=120, help="seconds added to the kill timeout")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write a JSON report here")
    parser.add_argument("--list-only", action="store_true", help="print the subset and exit")
    args = parser.parse_args()

    binary = args.bin.resolve() if args.bin else None
    if not args.list_only:
        if binary is None or not binary.is_file():
            print(f"FATAL: --bin is not a file: {binary}", file=sys.stderr)
            return 2
        # Hygiene: a ledger row is only valid through a release binary + preset.
        if "/release/" not in str(binary):
            print(
                f"FATAL: --bin must be a --release build (debug fails gemm-f16 fullfp16): {binary}",
                file=sys.stderr,
            )
            return 2
    configs_dir = args.configs_dir.resolve()
    if configs_dir.name.startswith("vnncomp"):
        print(
            f"FATAL: --configs-dir must be the configs/ ROOT, not {configs_dir.name}/. The "
            "subdirectory silently loads NO preset and runs auto defaults.",
            file=sys.stderr,
        )
        return 2
    if not (configs_dir / "vnncomp25").is_dir():
        print(f"FATAL: {configs_dir} does not contain vnncomp25/", file=sys.stderr)
        return 2
    if shutil.which("timeout") is None:
        print("FATAL: coreutils `timeout` is required", file=sys.stderr)
        return 2

    rows: list[BankedRow] = []
    for ledger_dir in (args.measured, args.ext_measured):
        if not ledger_dir.is_dir():
            continue
        for path in sorted(ledger_dir.glob("*.csv")):
            rows.extend(load_ledger(path))
    if args.categories:
        allow = {name.strip() for name in args.categories.split(",") if name.strip()}
        rows = [row for row in rows if row.category in allow]
    if not rows:
        print("FATAL: no banked SOLVED rows found", file=sys.stderr)
        return 2

    # Deduplicate identical (category, onnx, vnnlib, verdict) across ledgers,
    # preferring the extended ledger's row when both hold it.
    unique: dict[tuple[str, str, str], BankedRow] = {}
    for row in rows:
        unique.setdefault((row.category, row.onnx, row.vnnlib), row)

    # STATIC GATE (no runs): an unsat on an instance the field falsified with an
    # ACCEPTED counterexample is unsound regardless of whether it reproduces.
    if not args.watchlist.is_file():
        print(
            f"FATAL: no field-falsified watchlist at {args.watchlist}; generate it "
            "with scripts/emit_ce_falsified_watchlist.py (pass "
            "--watchlist to point elsewhere)",
            file=sys.stderr,
        )
        return 2
    falsified_unsats = audit_field_falsified_unsats(
        list(unique.values()), args.watchlist
    )
    if falsified_unsats:
        print(
            f"FATAL: {len(falsified_unsats)} banked unsat row(s) sit on instances "
            "the VNN-COMP field FALSIFIED with an ACCEPTED counterexample:",
            file=sys.stderr,
        )
        for row in falsified_unsats:
            print(
                f"  {row.category} {row.onnx} {row.vnnlib}  ({row.ledger})",
                file=sys.stderr,
            )
        print(
            "  Those rows credit +10 against a witness the organizers accepted "
            "and paid a falsifier for. Re-measure them; do not bank them.",
            file=sys.stderr,
        )
        return 2

    subset = select_subset(list(unique.values()), args.per_category)

    if args.list_only:
        for row in subset:
            print(f"{row.category},{row.onnx},{row.vnnlib},{row.verdict},{row.seconds}")
        print(f"# {len(subset)} rows across {len({r.category for r in subset})} categories")
        return 0

    # GROUND-TRUTH GATE. `official_ground_truth` feeds the one check here that
    # catches a WRONG verdict rather than a slow one (CAUSE_UNSOUND). This used
    # to degrade to `{}` whenever the tree was absent, so the script printed PASS
    # while that arm tested nothing — the worst failure mode a safety check has.
    # Absence is now either fatal or unmissable, never silent.
    official_dir = args.official if args.official is not None else DEFAULT_OFFICIAL
    ground_truth = official_ground_truth(official_dir) if official_dir.is_dir() else {}
    if not ground_truth:
        reason = (
            f"no such directory: {official_dir}"
            if not official_dir.is_dir()
            else f"no tool results.csv found under {official_dir}"
        )
        if args.official is not None or args.require_ground_truth:
            print(
                f"FATAL: official ground truth was requested but is unusable — {reason}. "
                f"The {CAUSE_UNSOUND} check cannot run without it. Fetch the official "
                "results tree, or drop --official/--require-ground-truth to run the "
                "reproduction check alone.",
                file=sys.stderr,
            )
            return 2
        print(
            "\n!! =================================================================\n"
            f"!! NO OFFICIAL GROUND TRUTH — {reason}\n"
            f"!! DISABLED: {CAUSE_UNSOUND}. This run can only detect a banked row that\n"
            "!! stops REPRODUCING; it CANNOT detect a wrong verdict. A PASS below is\n"
            "!! not a soundness result.\n"
            "!! Fetch external_tools/vnncomp2025_results (or pass --official) before\n"
            "!! using this script as a soundness gate.\n"
            "!! =================================================================\n",
            file=sys.stderr,
        )

    log_dir = args.log_dir or (REPO / "scratchpad" / f"banked_row_check_{int(time.time())}")
    log_dir.mkdir(parents=True, exist_ok=True)
    budgets_by_category: dict[str, dict[tuple[str, str], float]] = {}

    results: list[RowResult] = []
    print(f"banked-row reproduction check — {len(subset)} rows, logs in {log_dir}")
    for index, row in enumerate(subset, 1):
        if row.category not in budgets_by_category:
            budgets_by_category[row.category] = official_budgets(args.benchmark_root, row.category)
        budget = budgets_by_category[row.category].get(
            scorecard.key(row.onnx, row.vnnlib), 0.0
        )
        capped = False
        if budget <= 0:
            budget = args.budget_cap if args.budget_cap > 0 else 300.0
        if args.budget_cap > 0 and budget > args.budget_cap:
            budget = args.budget_cap
            capped = True
        result = run_row(
            row,
            binary=binary,
            configs_dir=configs_dir,
            benchmark_root=args.benchmark_root,
            budget=budget,
            log_dir=log_dir,
            grace=args.grace,
        )
        # A capped budget cannot distinguish "slower than the bank" from "would
        # still have solved at the official budget", so say so in the cause
        # instead of letting a cheap run masquerade as a bound regression.
        # `cause` is a composite string, hence the prefix test.
        if capped and result.cause and result.cause.startswith(
            (CAUSE_BUDGET_WALL, CAUSE_EARLY_GIVE_UP)
        ):
            result.cause = f"{result.cause}+CAPPED-BUDGET(inconclusive)"
        gt = ground_truth.get(row.category, {})
        gt_verdict = gt.get((*scorecard.key(row.onnx, row.vnnlib), 0))
        result.official_ground_truth = gt_verdict
        if gt_verdict and result.observed_verdict in SOLVED:
            expected = {"unsat": "holds", "sat": "violated"}[result.observed_verdict]
            if gt_verdict in ("holds", "violated") and expected != gt_verdict:
                result.reproduced = False
                result.cause = CAUSE_UNSOUND
                result.evidence.insert(
                    0,
                    f"official ground truth is {gt_verdict}, this run said "
                    f"{result.observed_verdict}",
                )
        results.append(result)
        status = "OK  " if result.reproduced else "FAIL"
        print(
            f"[{index}/{len(subset)}] {status} {row.category} {Path(row.onnx).name} "
            f"{Path(row.vnnlib).name} banked={row.verdict} observed={result.observed_verdict} "
            f"budget={budget:g}s elapsed={result.elapsed_seconds:g}s"
            + (f" cause={result.cause}" if result.cause else "")
        )
        for line in result.evidence[:3]:
            print(f"        evidence: {line}")

    failures = [result for result in results if not result.reproduced]
    invalid = [result for result in failures if result.cause == CAUSE_INVALID_NO_PRESET]
    by_cause: dict[str, list[RowResult]] = defaultdict(list)
    for result in failures:
        by_cause[result.cause or CAUSE_UNCLASSIFIED].append(result)

    print("\n==================== BANKED-ROW CHECK ====================")
    print(f"rows checked   : {len(results)}")
    print(f"reproduced     : {len(results) - len(failures)}")
    print(f"NOT reproduced : {len(failures)}")
    if ground_truth:
        print(f"ground truth   : {official_dir}")
    else:
        print(f"ground truth   : ABSENT — {CAUSE_UNSOUND} NOT CHECKED (see warning above)")
    overridden = [result for result in results if result.backend_override]
    if overridden:
        print(
            f"PROVENANCE     : {len(overridden)} row(s) ran on a SUBSTITUTED backend "
            f"({len([r for r in overridden if r.reproduced])} of them still reproduced). Their "
            "preset asked for an accelerator this binary refuses to use, so the row is not a "
            "measurement of the configuration it claims."
        )
    if invalid:
        print(
            f"INVALID RUNS   : {len(invalid)} (no preset loaded — these are NOT regressions; "
            "fix --configs-dir and re-run)"
        )
    for cause in sorted(by_cause):
        entries = by_cause[cause]
        print(f"\n--- {cause}  ({len(entries)} row(s)) ---")
        for result in entries:
            print(
                f"  {result.category} {Path(result.onnx).name} {Path(result.vnnlib).name}: "
                f"banked {result.banked_verdict} @ {result.banked_seconds:g}s -> "
                f"{result.observed_verdict} @ {result.elapsed_seconds:g}s "
                f"(budget {result.budget_seconds:g}s, rc {result.return_code})"
            )
            for line in result.evidence[:4]:
                print(f"      {line}")
            if result.log_path:
                print(f"      log: {result.log_path}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "binary": str(binary),
                    "configs_dir": str(configs_dir),
                    "per_category": args.per_category,
                    "budget_cap": args.budget_cap,
                    "ground_truth_dir": str(official_dir) if ground_truth else None,
                    "ground_truth_checked": bool(ground_truth),
                    "rows": [asdict(result) for result in results],
                    "causes": {cause: len(entries) for cause, entries in by_cause.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.out}")

    if any(result.cause == CAUSE_UNSOUND for result in failures):
        print("\nVERDICT: UNSOUND — a replayed row contradicts its bank/official GT. STOP.")
        return 3
    if failures:
        print("\nVERDICT: FAIL — banked rows no longer reproduce at HEAD.")
        return 1
    if not ground_truth:
        print(
            "\nVERDICT: PASS (PARTIAL) — every sampled banked row reproduces at HEAD, but "
            f"{CAUSE_UNSOUND} was never checked: no official ground truth. NOT a soundness "
            "result."
        )
        return 0
    print("\nVERDICT: PASS — every sampled banked row reproduces at HEAD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
