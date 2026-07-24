#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""
VNN-COMP 2025 regular-track competitive scoring model.

The scoring core is a clean-room reimplementation of the algorithm used by
the VNN-COMP 2025 organizers, as published in ``VNN-COMP/vnncomp2025_results``
(``SCORING-ZERO-TOL/process_results.py`` and ``SCORING-SMALL-TOL/...``).  The
two published scoreboards (ZERO-TOL and SMALL-TOL) share byte-identical
``process_results.py`` and ``settings.py``; the only difference is the
counterexample-tolerance gate in ``counterexamples.py`` that decides whether a
counterexample is fully ``CORRECT`` (falsifies disagreeing ``holds`` verdicts)
or merely ``CORRECT_UP_TO_TOLERANCE`` (scores the SAT tool but does not falsify
the UNSAT tools).  We model that distinction as the per-counterexample
``CounterexampleResult`` rather than re-deriving it from ONNX execution.

Input limitation: the public tools' plain ``results.csv`` rows contain verdicts
but not organizer counterexample-validation outcomes.  When a ``sat`` row has
no explicit ``ce_status`` column this script assumes its witness was strictly
``CORRECT``.  A report made from such raw CSVs is therefore a counterfactual
model, not the published ZERO-TOL or SMALL-TOL scoreboard.  Exact reproduction
requires explicit CE statuses produced by the organizer's witness checker.
Harness-only ``test_nano`` and ``test_tiny`` rows are never scored.

Algorithm summary (authoritative, from get_score() lines 719-825):

  Per (benchmark, instance, tool):
    * result not in {holds, violated}      -> 0
    * correct "holds" (no CORRECT CE)       -> +POINTS_CORRECT (10)
    * correct "violated" (valid CE)         -> +10
    * incorrect (three cases below)         -> -150 (PENALTY_INCORRECT)
        (a) "violated" but no valid CE when one is required
        (b) "violated" but the witness is invalid for this tool
        (c) "holds" but ANOTHER tool produced a CORRECT (not within-tolerance) CE

  There is NO time bonus in 2025 (the get_score() time-bonus block is gated on
  ``add_time_bonus = False`` and never runs).  Every correct result is exactly
  +10; the legacy 1-point "randgen" case does not exist in 2025.

  Per-benchmark normalization (process_results.py lines 464-475):
    raw[tool]  = sum of per-instance scores for the tool on the benchmark
    max_score  = max raw over participating tools
    percent    = max(0, 100 * raw / max_score)  if max_score > 0 else 0.0
    (winner -> 100.0; net-negative tools floored to 0.0; whole benchmark
     contributes 0.0 to everyone if no tool scored positively.)

  Overall total (lines 240-475):
    total[tool] = sum of per-benchmark ``percent`` across all scored benchmarks.
    With 16 regular benchmarks the theoretical max is 1600.0.

Usage:
    # a directory of per-tool per-instance result CSVs -- the only input that
    # supports a competitive comparison (scripts/ny_measured_sweep.py emits
    # ny's own sweep in this format)
    python3 scripts/vnncomp_competitive_score.py \\
        --results-dir vnncomp2025_results/ --target 1566.9

    # ny's own canonical metrics JSON: per-category aggregate counts, so this
    # is a single-tool self-normalized board.  Winner-relative normalization
    # makes the sole participant the winner of every benchmark it scores on, so
    # its percents are 100.0 by construction, no target comparison is reported,
    # and the total is not comparable to a published multi-tool total.  Counts
    # carry no instance identities, so --json cannot be combined with
    # --results-dir.
    python3 scripts/vnncomp_competitive_score.py \\
        --json metrics/benchmarks/vnncomp_latest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings transcribed from the authoritative VNN-COMP 2025 settings.py.
# ---------------------------------------------------------------------------

#: Points awarded for a correct ``holds``/``violated`` verdict (settings line:
#: get_score returns 10 for every correct result in 2025).
POINTS_CORRECT = 10

#: Penalty for an incorrect verdict (settings.py: ``PENALTY_INCORRECT = -150``).
PENALTY_INCORRECT = -150

#: Floor applied to normalized per-benchmark percentages
#: (process_results.py line 222: ``min_percent = 0``).
MIN_PERCENT = 0.0

#: Tools excluded from scoring (settings.py: ``SKIP_TOOLS = ['rover']``).
SKIP_TOOLS = frozenset({"rover"})

#: Number of regular-track benchmarks (informational; theoretical max = 1600.0).
NUM_REGULAR_BENCHMARKS = 16

#: Exact 2025 regular-track set from the organizer's settings.py. Raw result
#: repositories also contain extended and harness-test categories; including
#: them would produce a plausible-looking but impossible >1600 "regular" total.
REGULAR_BENCHMARKS = frozenset(
    {
        "acasxu_2023",
        "cersyve",
        "cgan_2023",
        "cifar100_2024",
        "collins_rul_cnn_2022",
        "cora_2024",
        "dist_shift_2023",
        "linearizenn_2024",
        "malbeware",
        "metaroom_2023",
        "nn4sys",
        "safenlp_2024",
        "sat_relu",
        "soundnessbench",
        "tinyimagenet_2024",
        "tllverifybench_2023",
    }
)

#: Theoretical maximum overall total: win every one of the 16 benchmarks.
THEORETICAL_MAX_TOTAL = NUM_REGULAR_BENCHMARKS * 100.0

#: Official published overall totals for alpha-beta-CROWN, the 2025 winner.
#: Reference total (alpha-beta-CROWN, VNN-COMP 2025 published scoreboard).
#: ZERO-TOL is the strict scoreboard; SMALL-TOL
#: allows a 1e-7 absolute / 1e-6 relative tolerance on counterexamples and so
#: lets alpha-beta-CROWN sweep every benchmark (1600.0).
OFFICIAL_TARGET_TOTALS = {
    "alpha-beta-CROWN": {"zero_tol": 1566.9, "small_tol": 1600.0},
}

#: Default ``--target`` reference total (alpha-beta-CROWN ZERO-TOL).
DEFAULT_TARGET = OFFICIAL_TARGET_TOTALS["alpha-beta-CROWN"]["zero_tol"]


class Tolerance(str, Enum):
    """Which of the two published scoreboards / tolerance gates to apply."""

    ZERO_TOL = "zero_tol"
    SMALL_TOL = "small_tol"


# ---------------------------------------------------------------------------
# Result mapping (process_results.py load(), settings.py CSV_SUBSTITUTIONS).
# Raw CSV result strings are lowercased and prefix-substituted into the canonical
# set {holds, violated, timeout, error, unknown}.  Only ``holds`` and
# ``violated`` are *scored*; everything else yields 0.
# ---------------------------------------------------------------------------

#: Longest-prefix-first so e.g. ``prepare_instance_timeout`` is mapped before
#: ``prepare_instance_error_``.  Order is enforced explicitly in _normalize.
_CSV_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("unsat", "holds"),
    ("sat", "violated"),
    ("no_result_in_file", "unknown"),
    ("prepare_instance_timeout", "timeout"),
    ("prepare_instance_error_", "unknown"),
    ("run_instance_timeout", "timeout"),
    ("error_exit_code_", "error"),
    ("error_nonmaximal", "unknown"),
)

#: Canonical post-substitution result set.
VALID_RESULTS = frozenset({"holds", "violated", "timeout", "error", "unknown"})

#: Results that are *scored* at all (every other result -> 0 points).
SCORABLE_RESULTS = frozenset({"holds", "violated"})


def normalize_result(raw: str) -> str:
    """Map a raw CSV result string into the canonical VNN-COMP result set.

    Mirrors ``process_results.py`` ``load()``: lowercase, apply the
    longest-matching prefix substitution, and treat the empty string as
    ``unknown``.  Anything not resolving to the canonical set is treated as
    ``unknown`` (defensive; the official pipeline asserts validity).
    """
    result = (raw or "").strip().lower()
    if result == "":
        return "unknown"
    for prefix, replacement in _CSV_SUBSTITUTIONS:
        if result.startswith(prefix):
            return replacement
    if result in VALID_RESULTS:
        return result
    return "unknown"


class CounterexampleResult(str, Enum):
    """Validation outcome of a tool's reported counterexample (witness).

    Transcribed from ``counterexamples.py``.  Only ``CORRECT`` and
    ``CORRECT_UP_TO_TOLERANCE`` count as "valid for this tool".  A strictly
    ``CORRECT`` counterexample additionally falsifies any disagreeing ``holds``
    verdict; a ``CORRECT_UP_TO_TOLERANCE`` one does not.
    """

    CORRECT = "correct"
    CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"
    EXEC_DOESNT_MATCH = "exec_doesnt_match"
    SPEC_NOT_VIOLATED = "spec_not_violated"
    NO_COUNTEREXAMPLE = "no_counterexample"


#: Counterexample outcomes that are "valid for this tool" (line 756): the SAT
#: tool keeps its +10 and is not penalized.
_VALID_FOR_THIS_TOOL = frozenset(
    {CounterexampleResult.CORRECT, CounterexampleResult.CORRECT_UP_TO_TOLERANCE},
)


@dataclass(frozen=True)
class InstanceResult:
    """One tool's outcome on one (benchmark, instance) pair.

    Attributes:
        tool: Tool / team name.
        benchmark: Benchmark (category) name.
        instance: Instance key (e.g. ``"onnx/x.onnx|vnnlib/p.vnnlib"``).
        result: Raw or canonical result string; normalized on ingest.
        counterexample: Validation outcome for this tool's witness, when the
            tool reports ``violated``.  ``None`` means "no CE was produced".
            For a ``violated`` verdict that *requires* a counterexample, a
            missing or non-valid CE is an incorrect result (-150).
        ce_required: Whether a counterexample is required for a ``violated``
            verdict on this instance (true for almost all VNN-COMP benchmarks).
        runtime: Solve time in seconds (recorded only; no time bonus in 2025).
    """

    tool: str
    benchmark: str
    instance: str
    result: str
    counterexample: CounterexampleResult | None = None
    ce_required: bool = True
    runtime: float = 0.0

    def normalized(self) -> InstanceResult:
        """Return a copy with ``result`` mapped to the canonical set."""
        canonical = normalize_result(self.result)
        if canonical == self.result:
            return self
        return InstanceResult(
            tool=self.tool,
            benchmark=self.benchmark,
            instance=self.instance,
            result=canonical,
            counterexample=self.counterexample,
            ce_required=self.ce_required,
            runtime=self.runtime,
        )


# ---------------------------------------------------------------------------
# Per-instance scoring.
# ---------------------------------------------------------------------------


def _any_tool_has_correct_ce(
    instance_results: list[InstanceResult],
) -> bool:
    """Whether *some* tool produced a strictly CORRECT counterexample.

    Mirrors ``valid_ce_any_tool`` (line 754): a within-tolerance CE does NOT
    count here, so it neither penalizes the SAT tool nor falsifies the UNSAT
    tools.  Only a fully ``CORRECT`` witness falsifies disagreeing ``holds``.
    """
    return any(
        r.result == "violated" and r.counterexample is CounterexampleResult.CORRECT
        for r in instance_results
    )


def score_instance(
    target: InstanceResult,
    instance_results: list[InstanceResult],
) -> int:
    """Score one tool's verdict on one instance (get_score(), lines 719-825).

    ``target`` is the tool being scored; ``instance_results`` is *every* tool's
    result on the same (benchmark, instance), used to detect a falsifying
    counterexample from another tool.

    Returns one of ``{0, POINTS_CORRECT, PENALTY_INCORRECT}``.
    """
    target = target.normalized()
    results = [r.normalized() for r in instance_results]

    # (line 763) Anything other than a holds/violated verdict scores 0.
    if target.result not in SCORABLE_RESULTS:
        return 0

    if target.result == "violated":
        ce = target.counterexample
        # (a) violated but no valid CE produced when one is required (766-773).
        if target.ce_required and ce is None:
            return PENALTY_INCORRECT
        # (b) violated but the witness is invalid for this tool (774-780).
        if ce is not None and ce not in _VALID_FOR_THIS_TOOL:
            return PENALTY_INCORRECT
        # Correct violated: valid (or not-required) witness -> +10 (line 805).
        return POINTS_CORRECT

    # target.result == "holds":
    # (c) holds but ANOTHER tool produced a strictly CORRECT CE (781-786).
    if _any_tool_has_correct_ce(results):
        return PENALTY_INCORRECT
    # Correct holds -> +10 (line 805).
    return POINTS_CORRECT


# ---------------------------------------------------------------------------
# Per-benchmark raw scoring + normalization.
# ---------------------------------------------------------------------------


def score_benchmark(results: list[InstanceResult]) -> dict[str, int]:
    """Compute per-tool *raw* scores for a single benchmark.

    ``results`` are all (tool, instance) results for one benchmark.  Tools in
    ``SKIP_TOOLS`` are excluded.  Returns ``{tool: raw_score}`` where each raw
    score is the sum of that tool's per-instance points.

    The instance-level cross-tool view (for falsifying counterexamples) is built
    per instance, so the order of ``results`` does not matter.
    """
    by_instance: dict[str, list[InstanceResult]] = defaultdict(list)
    tools: set[str] = set()
    for r in results:
        if r.tool in SKIP_TOOLS:
            continue
        by_instance[r.instance].append(r)
        tools.add(r.tool)

    raw: dict[str, int] = dict.fromkeys(tools, 0)
    for instance_results in by_instance.values():
        for target in instance_results:
            if target.tool in SKIP_TOOLS:
                continue
            raw[target.tool] += score_instance(target, instance_results)
    return raw


def normalize_benchmark(raw_scores: dict[str, int]) -> dict[str, float]:
    """Normalize a benchmark's raw scores to the winner-relative 0..100 scale.

    Implements process_results.py lines 464-475:

        max_score = max(raw over participating tools)
        if max_score > 0:
            percent[tool] = max(0, 100 * raw / max_score)
        else:
            percent[tool] = 0.0    # whole benchmark contributes 0 to everyone

    The winner gets exactly 100.0; net-negative tools are floored to 0.0.
    """
    if not raw_scores:
        return {}
    max_score = max(raw_scores.values())
    if max_score <= 0:
        # No tool scored positively: benchmark contributes 0.0 to everyone.
        return dict.fromkeys(raw_scores, 0.0)
    return {
        tool: max(MIN_PERCENT, 100.0 * raw / max_score)
        for tool, raw in raw_scores.items()
    }


def overall_scoreboard(
    all_benchmarks: dict[str, list[InstanceResult]],
    *,
    tolerance: Tolerance = Tolerance.ZERO_TOL,
) -> dict[str, float]:
    """Compute overall totals = sum of normalized per-benchmark percents.

    ``all_benchmarks`` maps ``benchmark_name -> [InstanceResult, ...]``.  The
    ``tolerance`` argument is accepted for symmetry; the tolerance distinction
    is encoded in each result's ``counterexample`` field (CORRECT vs
    CORRECT_UP_TO_TOLERANCE), so the same data scored under different tolerance
    *interpretations* must arrive pre-classified.  See ``downgrade_tolerance``.

    Returns ``{tool: total}`` (a float total per tool).  A benchmark a tool did
    not participate in simply contributes 0.
    """
    totals: dict[str, float] = defaultdict(float)
    for results in all_benchmarks.values():
        scored = (
            [_apply_zero_tol(r) for r in results]
            if tolerance is Tolerance.ZERO_TOL
            else results
        )
        raw = score_benchmark(scored)
        for tool, percent in normalize_benchmark(raw).items():
            totals[tool] += percent
    return dict(totals)


def _apply_zero_tol(result: InstanceResult) -> InstanceResult:
    """Reinterpret a SMALL-TOL-classified result under the ZERO-TOL gate.

    The published data we ingest is typically classified for one scoreboard.
    The only cross-scoreboard difference is the CORRECT vs
    CORRECT_UP_TO_TOLERANCE gate.  When data is supplied with an explicit
    per-tolerance classification (the JSON/CSV ingest paths attach the right
    ``CounterexampleResult`` for the requested tolerance), this is a no-op.
    It is retained as the single hook where a stricter gate could downgrade a
    borderline witness.
    """
    return result


# ---------------------------------------------------------------------------
# Input ingest: official per-tool CSVs and ny's vnncomp_latest.json.
# ---------------------------------------------------------------------------

#: Counterexample classification per scoreboard, keyed by an explicit CE-status
#: field that may appear in extended CSVs.  Falls back to CORRECT for a plain
#: ``violated`` verdict (the common published case where the CE validated).
_CE_STATUS_ALIASES = {
    "correct": CounterexampleResult.CORRECT,
    "correct_up_to_tolerance": CounterexampleResult.CORRECT_UP_TO_TOLERANCE,
    "within_tolerance": CounterexampleResult.CORRECT_UP_TO_TOLERANCE,
    "exec_doesnt_match": CounterexampleResult.EXEC_DOESNT_MATCH,
    "spec_not_violated": CounterexampleResult.SPEC_NOT_VIOLATED,
    "no_counterexample": CounterexampleResult.NO_COUNTEREXAMPLE,
    "none": None,
    "": None,
}


def _ce_for_result(
    canonical_result: str,
    ce_status: str | None,
    *,
    ce_required: bool,
) -> CounterexampleResult | None:
    """Resolve the CounterexampleResult for a ``violated`` row.

    A ``violated`` verdict with no explicit CE status is assumed to carry a
    valid (``CORRECT``) witness. This is an optimistic raw-CSV modeling
    assumption, not evidence that the organizer checker accepted the witness.
    Set an explicit CE status column to override.
    """
    if canonical_result != "violated":
        return None
    if ce_status is None:
        return CounterexampleResult.CORRECT if ce_required else None
    key = ce_status.strip().lower()
    if key in _CE_STATUS_ALIASES:
        return _CE_STATUS_ALIASES[key]
    return CounterexampleResult.CORRECT


def _tail2(path: str) -> str:
    """The last two non-empty path components of ``path`` (``subdir/basename``)."""
    parts = [c for c in path.split("/") if c not in ("", ".")]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else path


def _instance_key(onnx: str, vnnlib: str) -> str:
    """Cross-tool identity of one (onnx, vnnlib) instance.

    Keyed on ``subdir/basename`` rather than the basename alone: safenlp_2024
    ships medical/ and ruarobot/ subdirs whose ONNX and vnnlib basenames are
    identical (``perturbations_0.onnx``, ``hyperrectangle_N.vnnlib``), so a
    basename-only key merges two distinct scored instances into one bucket --
    and since merging can only add counterexamples to a bucket, a valid witness
    on ruarobot/hyperrectangle_N would falsify a correct ``holds`` on
    medical/hyperrectangle_N and charge PENALTY_INCORRECT for a right answer.
    ``subdir/basename`` is the common suffix of ny's relative
    (``onnx/medical/x``) and the official tools' absolute
    (``<root>/onnx/medical/x``) path styles, and collapses to ``onnx/x`` on the
    flat benchmarks, so it disambiguates the nested ones without splitting rows
    that belong together.
    """
    return f"{_tail2(onnx)}|{_tail2(vnnlib)}"


def _is_harness_test_instance(onnx: str, vnnlib: str) -> bool:
    """Whether this row is an unscored harness-overhead instance."""
    return any(
        marker in onnx or marker in vnnlib for marker in ("test_nano", "test_tiny")
    )


def load_results_dir(
    results_dir: str,
) -> dict[str, list[InstanceResult]]:
    """Load a directory of official per-tool VNN-COMP result CSVs.

    Expected layout (mirrors ``vnncomp2025_results``): per-tool result CSVs with
    rows ``category,onnx,vnnlib,prepare_time,result,runtime`` and an optional
    trailing ``ce_status`` column.  The tool name is taken from a ``tool``
    column if present, else from the file stem.  Returns
    ``{benchmark: [InstanceResult, ...]}``.
    """
    benchmarks: dict[str, list[InstanceResult]] = defaultdict(list)
    root = Path(results_dir)
    csv_paths = set(root.glob("*.csv"))
    csv_paths.update(root.rglob("results.csv"))
    for path in sorted(csv_paths):
        relative = path.relative_to(root)
        # The official repository also contains scorer-generated results.csv
        # files under SCORING-*; they are not tool measurements.
        if any(part.upper().startswith("SCORING") for part in relative.parts):
            continue
        tool_from_name = (
            relative.parts[0]
            if len(relative.parts) > 1
            else os.path.splitext(relative.name)[0]
        )
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            cols = [c.strip().lower() for c in header]
            has_header = "result" in cols or "category" in cols
            if has_header:
                idx = {name: i for i, name in enumerate(cols)}
                rows = reader
            else:
                # Headerless positional CSV (the harness results.csv format).
                idx = {
                    "category": 0,
                    "onnx_path": 1,
                    "vnnlib_path": 2,
                    "prepare_runtime": 3,
                    "result": 4,
                    "runtime": 5,
                }
                rows = [header, *list(reader)]
            for row in rows:
                if not row or len(row) <= idx.get("result", 4):
                    continue
                benchmark = row[idx.get("category", 0)].strip()
                onnx = row[idx.get("onnx_path", idx.get("onnx", 1))].strip()
                vnnlib = row[idx.get("vnnlib_path", idx.get("vnnlib", 2))].strip()
                if _is_harness_test_instance(onnx, vnnlib):
                    continue
                raw_result = row[idx["result"]].strip()
                tool = (
                    row[idx["tool"]].strip()
                    if "tool" in idx and len(row) > idx["tool"]
                    else tool_from_name
                )
                ce_status = (
                    row[idx["ce_status"]].strip()
                    if "ce_status" in idx and len(row) > idx["ce_status"]
                    else None
                )
                runtime = _safe_float(
                    row[idx["runtime"]]
                    if "runtime" in idx and len(row) > idx["runtime"]
                    else "0",
                )
                canonical = normalize_result(raw_result)
                benchmarks[benchmark].append(
                    InstanceResult(
                        tool=tool,
                        benchmark=benchmark,
                        instance=_instance_key(onnx, vnnlib),
                        result=canonical,
                        counterexample=_ce_for_result(
                            canonical,
                            ce_status,
                            ce_required=True,
                        ),
                        ce_required=True,
                        runtime=runtime,
                    ),
                )
    return dict(benchmarks)


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def load_ny_json(
    json_path: str,
    *,
    tool_name: str = "ny",
) -> dict[str, list[InstanceResult]]:
    """Load ny's own ``vnncomp_latest.json`` into single-tool benchmark results.

    The canonical artifact (see ``aggregate_vnncomp_results.py``) stores
    per-category *aggregate counts* (``verified``, ``falsified``, ``timeout``,
    ``unknown``, ``error``) rather than per-instance rows.  We expand those
    counts into synthetic per-instance results for ny so the same scoring
    machinery applies: ``verified`` -> holds, ``falsified`` -> violated (with a
    validated CE), and everything else -> unknown (0 points).

    The artifact carries only ny's own aggregate counts, so the resulting
    scoreboard is single-tool and self-normalized: ny wins every benchmark it
    scores any points on by construction.  The counts also carry no instance
    identities, so these rows cannot be matched against another tool's rows --
    ny's ``holds`` verdicts can never be cross-checked against a rival's
    counterexample, and a board built from them can never charge ny the -150
    incorrect-verdict penalty.  Such a board is therefore not comparable to a
    published multi-tool total, and these rows must not be merged with
    ``load_results_dir`` rows.  To score ny against rivals, run
    ``scripts/ny_measured_sweep.py`` and feed its per-instance CSV through
    ``--results-dir``.
    """
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    benchmarks: dict[str, list[InstanceResult]] = {}
    for cat_name, stats in (payload.get("categories") or {}).items():
        rows: list[InstanceResult] = []
        rows += _synthetic_rows(tool_name, cat_name, "holds", stats.get("verified", 0))
        rows += _synthetic_rows(
            tool_name,
            cat_name,
            "violated",
            stats.get("falsified", 0),
        )
        for bucket in ("timeout", "unknown", "error"):
            rows += _synthetic_rows(tool_name, cat_name, bucket, stats.get(bucket, 0))
        benchmarks[cat_name] = rows
    return benchmarks


def _synthetic_rows(
    tool: str,
    benchmark: str,
    canonical_result: str,
    count: int,
) -> list[InstanceResult]:
    """Expand an aggregate count into ``count`` synthetic per-instance rows.

    The counts name no instance and carry no witness, so the rows get counter
    keys that deliberately match nothing else: they are scoreable only against
    each other.  A ``violated`` row is assumed to carry a valid witness because
    the artifact records no CE-validation outcome -- unlike the published CSVs,
    where the organizers validated every counterexample before scoring.  Both
    assumptions are why these rows can only ever score ny
    ``10 * (verified + falsified)``, and why they may not be merged into a
    cross-tool field.
    """
    ce = CounterexampleResult.CORRECT if canonical_result == "violated" else None
    return [
        InstanceResult(
            tool=tool,
            benchmark=benchmark,
            instance=f"{canonical_result}-{i}",
            result=canonical_result,
            counterexample=ce,
            ce_required=True,
        )
        for i in range(int(count))
    ]


# ---------------------------------------------------------------------------
# Reporting / CLI.
# ---------------------------------------------------------------------------


#: Printed in place of the vs-target verdict when only one tool participated.
#: Winner-relative normalization scores every tool against the best raw score in
#: the field, so a sole participant is the winner of each benchmark it scores on
#: and its total says nothing about a field it was never measured against.
_SINGLE_TOOL_NOTICE = (
    "Single-tool scoreboard: one participant, so winner-relative normalization\n"
    "scored it against itself -- every benchmark it scores any points on is\n"
    "100.0 by construction and the total is self-referential.  These numbers\n"
    "are NOT comparable to the published multi-tool totals, so no target\n"
    "comparison is reported.  To score against rivals, pass --results-dir over\n"
    "per-tool per-instance result CSVs (scripts/ny_measured_sweep.py emits\n"
    "ny's sweep in that format)."
)


@dataclass
class ScoreReport:
    """A full scoring pass under one tolerance gate."""

    tolerance: Tolerance
    per_benchmark: dict[str, dict[str, float]] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)


def build_report(
    all_benchmarks: dict[str, list[InstanceResult]],
    tolerance: Tolerance,
) -> ScoreReport:
    """Score every benchmark and accumulate totals under ``tolerance``."""
    report = ScoreReport(tolerance=tolerance)
    totals: dict[str, float] = defaultdict(float)
    for benchmark in sorted(all_benchmarks):
        results = all_benchmarks[benchmark]
        scored = (
            [_apply_zero_tol(r) for r in results]
            if tolerance is Tolerance.ZERO_TOL
            else results
        )
        normalized = normalize_benchmark(score_benchmark(scored))
        report.per_benchmark[benchmark] = normalized
        for tool, percent in normalized.items():
            totals[tool] += percent
    report.totals = dict(totals)
    return report


def _ranked(totals: dict[str, float]) -> list[tuple[str, float]]:
    """Tools ranked by descending total (process_results.py reversed(sorted))."""
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))


def format_report(report: ScoreReport, *, target: float) -> str:
    """Render a per-benchmark table + overall ranking as plain text."""
    lines: list[str] = []
    label = report.tolerance.value.upper().replace("_", "-")
    lines.append(f"=== VNN-COMP 2025 scoring model [{label}] ===")
    lines.append(
        "CE note: SAT rows without explicit ce_status are assumed strictly CORRECT; "
        "such a raw-CSV board is not an organizer-validated official scoreboard."
    )

    tools = sorted({t for norm in report.per_benchmark.values() for t in norm})
    if report.per_benchmark and tools:
        header = "benchmark".ljust(28) + "".join(t[:12].rjust(13) for t in tools)
        lines.append(header)
        for benchmark in sorted(report.per_benchmark):
            norm = report.per_benchmark[benchmark]
            row = benchmark[:27].ljust(28)
            row += "".join(f"{norm.get(t, 0.0):13.1f}" for t in tools)
            lines.append(row)

    lines.append("")
    lines.append("Overall ranking (sum of normalized per-benchmark percents):")
    for rank, (tool, total) in enumerate(_ranked(report.totals), start=1):
        lines.append(f"  {rank:>2}. {tool:<24} {total:8.1f}")

    lines.append("")
    if len(report.totals) < 2:
        # A target is a rival's total from a real field; there is nothing here
        # to compare it against.
        lines.append(_SINGLE_TOOL_NOTICE)
        return "\n".join(lines)

    lines.append(f"Reference total: {target:.1f}")
    best_tool, best_total = _ranked(report.totals)[0]
    margin = best_total - target
    verdict = "AHEAD" if margin >= 0 else "BEHIND"
    lines.append(
        f"Leader {best_tool} = {best_total:.1f} ({verdict} by {abs(margin):.1f})",
    )
    # ny-specific margin, if ny participated.
    if "ny" in report.totals:
        ny_margin = report.totals["ny"] - target
        lines.append(
            f"ny = {report.totals['ny']:.1f} "
            f"(needs {max(0.0, -ny_margin):.1f} more to reach target)",
        )
    return "\n".join(lines)


def _load_inputs(args: argparse.Namespace) -> dict[str, list[InstanceResult]]:
    """Load the single input source main() has already validated.

    The two sources are mutually exclusive rather than merged: their instance
    keys live in disjoint universes (see ``_synthetic_rows``), so merging them
    would seat ny in a cross-tool table it could never be cross-checked in.
    """
    if args.results_dir:
        loaded = load_results_dir(args.results_dir)
    else:
        loaded = load_ny_json(args.json)
    return {
        benchmark: rows
        for benchmark, rows in loaded.items()
        if benchmark in REGULAR_BENCHMARKS
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VNN-COMP 2025 regular-track competitive scoring",
    )
    parser.add_argument(
        "--results-dir",
        help="Directory of per-tool per-instance VNN-COMP result CSVs; the "
        "only input that supports a competitive comparison",
    )
    parser.add_argument(
        "--json",
        help="Path to ny's metrics/benchmarks/vnncomp_latest.json. Holds "
        "aggregate counts, so it scores a single-tool self-normalized "
        "board (percents 100.0 by construction, no target comparison) "
        "and cannot be combined with --results-dir",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET,
        help=f"Overall total to beat (default {DEFAULT_TARGET}, "
        "alpha-beta-CROWN ZERO-TOL); not reported for a single-tool board",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.results_dir and not args.json:
        parser.error("provide --results-dir or --json")
    if args.results_dir and args.json:
        parser.error(
            "--json and --results-dir cannot be combined: --json carries "
            "per-category aggregate counts, not instance identities, so its "
            "rows never match a --results-dir row. ny would be scored beside "
            "the other tools without ever being cross-checked against their "
            "counterexamples -- unable to take the -150 incorrect-verdict "
            "penalty, and reported as if it had competed. Run "
            "scripts/ny_measured_sweep.py and score its per-instance CSV "
            "through --results-dir instead.",
        )

    benchmarks = _load_inputs(args)
    if not benchmarks:
        parser.error("no benchmark results loaded from the given inputs")

    for tolerance in (Tolerance.ZERO_TOL, Tolerance.SMALL_TOL):
        report = build_report(benchmarks, tolerance)
        print(format_report(report, target=args.target))
        print()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
