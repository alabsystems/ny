#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""#phase-cost census: plan a stratified sample, judge its runs, aggregate the per-phase cost, and rank bottlenecks.

The binary side (`ny_core::phase_cost`, serialized as `phase_cost` in `<results>.flight.json`, schema
`ny-core/phase-cost/v1`) records, per scored row, each phase's OCCUPIED wall (inclusive of nested phases), thread
time, self time, enters, work units and sampled RSS, plus run-level work counters. This tool turns a set of such rows
into a ranked BOTTLENECK REPORT: per category the phase-time distribution and the cost per useful unit, phases whose
share is anomalous against the cross-category median, and the evidence rows behind every claim.

Subcommands (pure python3, no cargo, no ny binary, no box):

  plan       --ny-root R --out SPEC.tsv [--year 2026] [--rows-per-category 2] [--cheap-budget 120]
             [--max-budget 1800] [--categories a,b]   write the stratified row spec (deterministic)
  aggregate  --out-dir D --expect-commit C [--report R.md] [--json J.json]
             judge every run in D (plan.tsv + res/ + raw/) and write the report

A run is VOID (never counted, never silently zero) when it cannot prove what it measured: no flight record, no
`phase_cost`, an unknown schema, a census level of `off`, a build_id that does not name the expected commit or is
dirty, a non-empty NY_* ambient environment, no `[phase-cost] level=` line on its stderr (the census's own
observable), no `Result:` line, or an instance line that names another row.

Rows are read per run; nothing is inferred from an absence. See docs/PHASE_COST_CENSUS_*.md for the method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys

SCHEMA = "ny-core/phase-cost/v1"
SPEC_SCHEMA = "costcensus-spec/v1"
SPEC_COLUMNS = ["tag", "year", "category", "index", "onnx", "vnnlib", "budget", "bank_verdict", "bank_wall", "stratum"]
SOLVED = ("unsat", "sat")
OPEN = ("timeout", "unknown", "error")
# Phases that are umbrellas over others (their share is not "a cost center" by itself): the BaB loop and the
# verifier invocation. Their self time is NOT coverage: it is wall inside them that no phase claims.
UMBRELLAS = ("bab_loop", "verify")
# Phases in the order the report prints them.
PHASE_ORDER = [
    "load", "root_bounds", "alpha_opt", "beta_opt", "interm_refresh", "branch_select", "branch_prescore",
    "branch_sim", "child_bound", "bab_loop", "attack", "mip", "lowdim", "certify", "publish", "exact", "verify",
    "unattributed",
]
# The pseudo-phase for wall no instrumented phase covers (1 - coverage): process startup, device/session
# initialisation, preset and router work, and any lane without a phase boundary. It is ranked like a phase.
UNATTRIBUTED = "unattributed"
# Anomaly rule: a category's mean share of a phase is anomalous when it is at least ANOM_RATIO x the cross-category
# median share of that phase, exceeds the median by ANOM_ABS, and is itself >= ANOM_MIN.
ANOM_RATIO = 2.0
ANOM_ABS = 0.15
ANOM_MIN = 0.10


class CensusError(Exception):
    """A refusal: the census will not produce an artefact it cannot vouch for."""


# ------------------------------------------------------------------------------------------------------------------
# plan
# ------------------------------------------------------------------------------------------------------------------


def _m0lib(ny_root: str):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "m0"), os.path.join(ny_root, "scripts", "m0")):
        if os.path.isfile(os.path.join(cand, "m0lib.py")):
            sys.path.insert(0, os.path.abspath(cand))
            import m0lib  # noqa: E402  (located at runtime)

            return m0lib
    raise CensusError("scripts/m0/m0lib.py not found (the row-source library)")


def _bank_walls(path: str) -> list[float | None]:
    walls: list[float | None] = []
    with open(path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 5 or r[0].strip().lower() in ("cat", "category", "benchmark"):
                continue
            try:
                walls.append(float(r[5]) if len(r) >= 6 else None)
            except ValueError:
                walls.append(None)
    return walls


def select_rows(rows: list[dict], per_category: int, cheap_budget: int, max_budget: int) -> list[dict]:
    """Deterministic stratified pick for ONE category. `rows` carry index, budget, bank_verdict, bank_wall.

    * rows over `max_budget` are never picked (the census must fit a box hold);
    * a category whose smallest budget exceeds `cheap_budget` gets ONE row (box frugality), else `per_category`;
    * the first row is SOLVED (banked unsat when the category has one, else sat — a proof exercises the BaB
      phases a falsification skips; the median banked wall among those rows of the smallest budget class — a
      representative proof, neither the trivial nor the extreme one); the second is OPEN (banked
      timeout/unknown, smallest budget, lowest index — the bottleneck exemplar). A missing stratum is filled
      from the other so the count holds when the category allows it.
    """
    eligible = [r for r in rows if r["budget"] <= max_budget]
    if not eligible:
        return []
    want = per_category if min(r["budget"] for r in eligible) <= cheap_budget else 1
    solved = [r for r in eligible if r["bank_verdict"] == "unsat"] or [r for r in eligible if r["bank_verdict"] == "sat"]
    opened = [r for r in eligible if r["bank_verdict"] in OPEN]

    def median_solved(pool: list[dict]) -> dict | None:
        if not pool:
            return None
        bmin = min(r["budget"] for r in pool)
        cls = sorted((r for r in pool if r["budget"] == bmin),
                     key=lambda r: (r["bank_wall"] if r["bank_wall"] is not None else math.inf, r["index"]))
        return cls[(len(cls) - 1) // 2]

    def first_open(pool: list[dict]) -> dict | None:
        return min(pool, key=lambda r: (r["budget"], r["index"])) if pool else None

    picks: list[dict] = []
    order = [("solved", median_solved(solved)), ("open", first_open(opened))]
    for stratum, row in order:
        if row is not None and len(picks) < want:
            picks.append(dict(row, stratum=stratum))
    # fill from whichever stratum still has rows (deterministic: smallest budget, lowest index)
    taken = {r["index"] for r in picks}
    rest = sorted((r for r in eligible if r["index"] not in taken), key=lambda r: (r["budget"], r["index"]))
    for row in rest:
        if len(picks) >= want:
            break
        picks.append(dict(row, stratum="fill"))
    return picks


def plan(ny_root: str, out: str, year: int, per_category: int, cheap_budget: int, max_budget: int,
         categories: list[str] | None) -> list[dict]:
    if year != 2026:
        raise CensusError("only the 2026 stratum is planned by this tool today")
    L = _m0lib(ny_root)
    man = json.load(open(os.path.join(ny_root, "reports/measured-2026/bank_manifest.json")))
    cats = sorted(man.get("banks", {}))
    if categories:
        unknown = sorted(set(categories) - set(cats))
        if unknown:
            raise CensusError(f"categories not in the 2026 bank manifest: {unknown}")
        cats = [c for c in cats if c in categories]
    spec: list[dict] = []
    for cat in cats:
        inst = L.load_instances(L.scored_instances_csv(ny_root, year, cat))
        bank_path = L.bank_csv(ny_root, year, cat)
        verdicts = L.align_bank(inst, L.load_bank(bank_path), f"{year}/{cat}")
        walls = _bank_walls(bank_path)
        if L.ALIGN_MODES.get(f"{year}/{cat}") != "by-index":
            walls = [None] * len(inst)  # walls align by index only; a permuted bank keeps verdicts, drops walls
        rows = [dict(index=r.index, onnx=r.onnx, vnnlib=r.vnnlib, budget=r.budget, bank_verdict=v,
                     bank_wall=(walls[i] if i < len(walls) else None))
                for i, (r, v) in enumerate(zip(inst, verdicts))]
        for pick in select_rows(rows, per_category, cheap_budget, max_budget):
            pick.update(year=year, category=cat, tag=f"{cat}__i{pick['index']}")
            spec.append(pick)
    write_spec(out, spec)
    return spec


def write_spec(path: str, spec: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(f"# {SPEC_SCHEMA}\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(SPEC_COLUMNS)
        for r in spec:
            w.writerow([("" if r.get(c) is None else r.get(c)) for c in SPEC_COLUMNS])


def read_spec(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        first = fh.readline().strip()
        if first != f"# {SPEC_SCHEMA}":
            raise CensusError(f"{path}: not a {SPEC_SCHEMA} spec (first line {first!r})")
        rd = csv.DictReader(fh, delimiter="\t")
        if rd.fieldnames != SPEC_COLUMNS:
            raise CensusError(f"{path}: columns {rd.fieldnames} != {SPEC_COLUMNS}")
        out = []
        for r in rd:
            r = dict(r)
            r["year"] = int(r["year"])
            r["index"] = int(r["index"])
            r["budget"] = int(r["budget"])
            r["bank_wall"] = float(r["bank_wall"]) if r["bank_wall"] else None
            out.append(r)
    tags = [r["tag"] for r in out]
    if len(set(tags)) != len(tags):
        raise CensusError(f"{path}: duplicate tags")
    return out


# ------------------------------------------------------------------------------------------------------------------
# judge one run
# ------------------------------------------------------------------------------------------------------------------

RESULT_RE = re.compile(r"^Result:\s*(\S+)", re.M)
RUN_LINE_RE = re.compile(r"^Running VNN-COMP instance:.*$", re.M)
PRESET_RE = re.compile(r"^Loading preset: (\S+)", re.M)
COST_LINE_RE = re.compile(r"^\[phase-cost\] level=(\S+) ", re.M)
PROFILE_LINE = "[proof-profile] mode=competition source=cli"
DECIDED = ("sat", "unsat")


def _res_token(text: str | None) -> str | None:
    if not text:
        return None
    first = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    return first if first in ("sat", "unsat", "unknown", "timeout", "error") else None


def judge_run(row: dict, flight: dict | None, log: str | None, res_text: str | None, expect_commit: str) -> dict:
    """One run -> {status: OK|VOID, why: [...], verdict, wall_s, cost} (cost = the phase_cost snapshot)."""
    why: list[str] = []
    out = {"tag": row["tag"], "category": row["category"], "budget": row["budget"], "stratum": row["stratum"],
           "bank_verdict": row["bank_verdict"], "status": "VOID", "why": why, "verdict": None, "wall_s": None,
           "partial": False, "cost": None}
    if flight is None:
        why.append("no flight record")
        return out
    cost = flight.get("phase_cost")
    if not isinstance(cost, dict):
        why.append("flight record has no phase_cost (a binary without the census, or the census was off)")
    elif cost.get("schema") != SCHEMA:
        why.append(f"phase_cost schema {cost.get('schema')!r} != {SCHEMA}")
    elif cost.get("level") == "off":
        why.append("phase_cost level=off")
    bid = str(flight.get("build_id", ""))
    if f"commit={expect_commit}" not in bid and not re.search(r"commit=" + re.escape(expect_commit[:9]), bid):
        why.append(f"build_id does not name commit {expect_commit[:12]}")
    if "dirty=true" in bid:
        why.append("dirty build")
    amb = flight.get("ambient_env")
    if not isinstance(amb, dict):
        why.append("no ambient_env")
    else:
        ny = sorted(k for k in amb if k.startswith("NY_"))
        if ny:
            why.append(f"NY_* in the ambient environment: {ny}")
        if amb.get("OMP_NUM_THREADS") != "1" or any(k.startswith("RAYON_") for k in amb):
            why.append(f"ambient env is not run_instance.sh's (OMP_NUM_THREADS=1 only): {sorted(amb)}")
    if flight.get("category") != row["category"]:
        why.append(f"flight category {flight.get('category')!r} != {row['category']!r}")
    log_verdicts: list[str] = []
    if log is None:
        why.append("no run log")
    else:
        if not COST_LINE_RE.search(log):
            why.append("no `[phase-cost] level=` line in the run log (the census's own observable)")
        log_verdicts = RESULT_RE.findall(log)
        if not log_verdicts:
            why.append("no `Result:` line in the run log")
        elif len(set(log_verdicts)) > 1:
            why.append(f"multiple Result tokens: {','.join(log_verdicts)}")
        run_line = RUN_LINE_RE.search(log)
        relational = any(isinstance(e, dict) and e.get("method") == "relational_route"
                         for e in ((flight or {}).get("events") or []))
        if run_line and f"vnnlib='{row['vnnlib']}'" not in run_line.group(0):
            why.append("the `Running VNN-COMP instance:` line names another row")
        elif not run_line and not relational:
            why.append("missing `Running VNN-COMP instance:` line")
        # the dual-network (relational) route prints no instance line; its identity is the per-row results file
        # and flight record the runner moved aside before the run (never a previous attempt's)
        out["relational"] = relational
        if PROFILE_LINE not in log:
            why.append(f"no `{PROFILE_LINE}` line")
        pre = PRESET_RE.search(log)
        out["preset"] = pre.group(1) if pre else None
    events = flight.get("events") or []
    terminal = [e for e in events if e.get("method") in ("run_complete", "run_incomplete")]
    if terminal:
        out["wall_s"] = float(terminal[-1].get("at_secs") or 0.0)
        out["verdict"] = terminal[-1].get("reason") if terminal[-1]["method"] == "run_complete" else None
    status = flight.get("record_status")
    if isinstance(status, dict):
        out["partial"] = True
        out["wall_s"] = float(status.get("flushed_at_secs") or out["wall_s"] or 0.0)
    tok = _res_token(res_text)
    if tok is not None:
        out["verdict"] = tok
    if log_verdicts and tok is not None and log_verdicts[-1] != tok:
        why.append(f"results file {tok} != log Result {log_verdicts[-1]}")
    if out["wall_s"] is None or out["wall_s"] <= 0:
        why.append("no terminal wall in the flight record")
    if isinstance(cost, dict):
        out["cost"] = cost
    # Not a moat, but a census run that contradicts a banked DECIDED verdict is surfaced, never averaged in.
    claimed = {v for v in log_verdicts + ([tok] if tok else []) if v in DECIDED}
    bank = row["bank_verdict"]
    if bank in DECIDED and claimed - {bank}:
        out["status"] = "EMERGENCY"
        why.insert(0, f"BANK-FLIP: banked {bank}, run stated {'/'.join(sorted(claimed))}")
        return out
    if len(claimed) > 1:
        out["status"] = "EMERGENCY"
        why.insert(0, f"SELF-CONTRADICTION: {'/'.join(sorted(claimed))}")
        return out
    # review(cost-census): a row the bank holds OPEN that this run DECIDED is not a conversion and never silent: an
    # arm64 `sat` is not adjudicated (x86 parity: smart_turn's arm64 sats were demoted after a certified-x86 replay
    # disagreed), and an `unsat` is unbanked until a both-year moat. Surfaced on every judged run, OK or VOID.
    if bank in OPEN and len(claimed) == 1:
        out["new_decision"] = next(iter(claimed))
    if not why:
        out["status"] = "OK"
    return out


# ------------------------------------------------------------------------------------------------------------------
# aggregate
# ------------------------------------------------------------------------------------------------------------------


def _phase_rows(cost: dict) -> dict[str, dict]:
    return {p["phase"]: p for p in cost.get("phases", []) if isinstance(p, dict) and "phase" in p}


def row_metrics(run: dict) -> dict:
    """Shares and unit costs of one OK run. Shares are INCLUSIVE occupied wall / row wall (clamped to 1)."""
    cost = run["cost"]
    wall = float(run["wall_s"])
    phases = _phase_rows(cost)
    share = {name: min(1.0, float(p.get("wall_s", 0.0)) / wall) for name, p in phases.items()}
    # top-level = a non-umbrella phase most of whose thread time was entered with no same-thread parent or directly
    # under an umbrella (bab_loop / verify): the phases that partition a thread's work.
    root_in: dict[str, float] = {}
    tot_in: dict[str, float] = {}
    for e in cost.get("edges", []):
        tot_in[e["child"]] = tot_in.get(e["child"], 0.0) + float(e["thread_s"])
        if e["parent"] == "root" or e["parent"] in UMBRELLAS:
            root_in[e["child"]] = root_in.get(e["child"], 0.0) + float(e["thread_s"])
    top = [n for n in phases if n not in UMBRELLAS and tot_in.get(n, 0.0) > 0
           and root_in.get(n, 0.0) >= 0.5 * tot_in[n]]
    top_sum = sum(float(phases[n]["wall_s"]) for n in top) / wall if wall > 0 else 0.0
    coverage = min(1.0, top_sum)
    share[UNATTRIBUTED] = max(0.0, 1.0 - coverage)
    # Bounding and optimizer time spent INSIDE branch selection (the kFSB shape: simulated children are bounded
    # by the same `child_bound` / `beta_opt` code as real ones). Same-thread edges only: a lower bound.
    # review(cost-census): an edge carries THREAD time summed over every thread that ran the child under a branch
    # parent, while a share is OCCUPIED wall; subtracting the one from the other is the rayon double count (four
    # threads selecting at once hold 4 s of nested thread time inside 1 s of wall). Each child's nested thread time
    # is therefore converted to its own wall units by the fraction of that child's thread time it represents.
    branch = ("branch_select", "branch_prescore", "branch_sim")
    nested_thread: dict[str, float] = {}
    for e in cost.get("edges", []):
        if e["parent"] in branch and e["child"] not in branch:
            nested_thread[e["child"]] = nested_thread.get(e["child"], 0.0) + float(e["thread_s"])

    def nested_frac(child: str) -> float:
        thread = float(phases.get(child, {}).get("thread_s", 0.0))
        return min(1.0, nested_thread.get(child, 0.0) / thread) if thread > 0 else 0.0

    nested = {c: float(phases.get(c, {}).get("wall_s", 0.0)) * nested_frac(c) for c in nested_thread}
    bound_wall = float(phases.get("child_bound", {}).get("wall_s", 0.0))
    bound_net = bound_wall * (1.0 - nested_frac("child_bound"))
    # The longest stretch of the row that the instrumented phases certainly did NOT occupy, from each phase's
    # first entry and last exit (or the snapshot, for a phase still OPEN when the record was written: its span runs
    # to the end). A phase's [first, last] span hides its own internal gaps, so a DENSE span (occupied wall >= half
    # the span) is treated as occupied throughout: this is a LOWER bound on the longest uninstrumented stretch.
    # review(cost-census): a SPARSE span (occupied wall < half the span) is NOT treated as occupancy -- `publish` is
    # entered once for the placeholder at t~0.01 s and once for the verdict at the end, so its span covers the whole
    # row and, counted as occupancy, hid every stretch on every row (census2's 42.4 s acasxu stretch read 0.00).
    # Instead each candidate stretch between dense spans is charged the occupied wall of every sparse phase whose
    # span overlaps it (at most the overlap): what remains is certain -- no instrumented phase ran for at least that
    # many seconds inside the stretch.
    elapsed = float(cost.get("elapsed_s") or wall)

    def span_of(p: dict) -> tuple[float, float] | None:
        lo = p.get("first_enter_s")
        hi = elapsed if int(p.get("open", 0) or 0) > 0 else p.get("last_exit_s")
        if lo is None or hi is None:
            return None
        return float(lo), max(float(lo), float(hi))

    dense: list[tuple[float, float, str]] = []
    sparse: list[tuple[float, float, float]] = []
    for n, p in phases.items():
        sp = span_of(p) if n not in UMBRELLAS else None
        if sp is None:
            continue
        occupied = float(p.get("wall_s", 0.0))
        if occupied >= 0.5 * (sp[1] - sp[0]):
            dense.append((sp[0], sp[1], n))
        else:
            sparse.append((sp[0], sp[1], occupied))
    stretches: list[tuple[float, float, str, str]] = []
    cur_end, cur_name = 0.0, "start"
    for lo, hi, n in sorted(dense):
        if lo > cur_end:
            stretches.append((cur_end, lo, cur_name, n))
        if hi > cur_end:
            cur_end, cur_name = hi, n
    end = max(wall, elapsed)
    if end > cur_end:
        stretches.append((cur_end, end, cur_name, "end"))
    gap, gap_after, gap_before = 0.0, None, None
    for lo, hi, after, before in stretches:
        charged = sum(min(occ, max(0.0, min(hi, shi) - max(lo, slo))) for slo, shi, occ in sparse)
        certain = (hi - lo) - charged
        if certain > gap:
            gap, gap_after, gap_before = certain, after, before
    counters = cost.get("counters", {})
    verified = int(counters.get("domains_verified", 0))
    explored = int(counters.get("domains_explored", 0))
    enters = sum(int(p.get("enters", 0)) for p in phases.values())
    return {
        "share": share,
        "bound_net_share": min(1.0, bound_net / wall) if wall > 0 else 0.0,
        "in_select_share": min(1.0, sum(nested.values()) / wall) if wall > 0 else 0.0,
        "in_select": nested,
        "top_sum": top_sum,
        "max_gap_s": gap,
        "gap_after": gap_after,
        "gap_before": gap_before,
        "top": sorted(top),
        "coverage": coverage,
        "s_per_verified": (wall / verified) if verified else None,
        "s_per_explored": (wall / explored) if explored else None,
        "verified": verified,
        "explored": explored,
        "scored": int(counters.get("candidates_scored", 0)),
        "simulated": int(counters.get("candidates_simulated", 0)),
        "enters": enters,
        "rss_peak_mb": (cost.get("rss_peak_kb") or 0) / 1024.0,
        "hwm_mb": (cost.get("process_hwm_kb") or 0) / 1024.0,
        "phases": phases,
    }


def overhead_bound(run: dict, guard_ns: float, sample_us: float) -> float:
    """Upper-bound estimate of the census's own cost on one row, as a fraction of its wall: every entered guard
    priced at the measured per-enter cost `guard_ns` (an armed enter+exit pair, release build) and every RSS sample
    at `sample_us` (one /proc/self/statm read)."""
    cost = run["cost"]
    enters = sum(int(p.get("enters", 0)) for p in cost.get("phases", []))
    secs = enters * guard_ns * 1e-9 + int(cost.get("rss_samples", 0)) * sample_us * 1e-6
    return secs / float(run["wall_s"])


def aggregate(runs: list[dict], guard_ns: float | None = None, sample_us: float = 20.0) -> dict:
    ok = [r for r in runs if r["status"] == "OK"]
    for r in ok:
        r["m"] = row_metrics(r)
        if guard_ns is not None:
            r["m"]["overhead"] = overhead_bound(r, guard_ns, sample_us)
    cats = sorted({r["category"] for r in ok})
    per_cat: dict[str, dict] = {}
    for c in cats:
        rs = [r for r in ok if r["category"] == c]
        names = sorted({n for r in rs for n in r["m"]["share"]}, key=lambda n: PHASE_ORDER.index(n)
                       if n in PHASE_ORDER else 99)
        mean_share = {n: statistics.fmean(r["m"]["share"].get(n, 0.0) for r in rs) for n in names}
        wall = sum(float(r["wall_s"]) for r in rs)
        ver = sum(r["m"]["verified"] for r in rs)
        exp = sum(r["m"]["explored"] for r in rs)
        per_cat[c] = {
            "bound_net_share": statistics.fmean(r["m"]["bound_net_share"] for r in rs),
            "in_select_share": statistics.fmean(r["m"]["in_select_share"] for r in rs),
            "rows": [r["tag"] for r in rs],
            "wall_s": wall,
            "mean_share": mean_share,
            "coverage": statistics.fmean(r["m"]["coverage"] for r in rs),
            "s_per_verified": (wall / ver) if ver else None,
            "s_per_explored": (wall / exp) if exp else None,
            "verified": ver,
            "explored": exp,
            "rss_peak_mb": max(r["m"]["rss_peak_mb"] for r in rs),
            "hwm_mb": max(r["m"]["hwm_mb"] for r in rs),
        }
    # cross-category median share per phase (a category that never entered a phase contributes 0)
    phases = sorted({n for c in per_cat.values() for n in c["mean_share"]},
                    key=lambda n: PHASE_ORDER.index(n) if n in PHASE_ORDER else 99)
    median = {n: statistics.median([per_cat[c]["mean_share"].get(n, 0.0) for c in cats]) for n in phases} if cats else {}
    findings = []
    for c in cats:
        for n, s in per_cat[c]["mean_share"].items():
            if n in UMBRELLAS:
                continue
            m = median.get(n, 0.0)
            if s >= ANOM_MIN and s - m >= ANOM_ABS and (m == 0 or s >= ANOM_RATIO * m):
                excess_s = (s - m) * per_cat[c]["wall_s"]
                ev = sorted((r for r in ok if r["category"] == c),
                            key=lambda r: -r["m"]["share"].get(n, 0.0))
                findings.append({
                    "kind": "share",
                    "category": c,
                    "phase": n,
                    "share": s,
                    "median": m,
                    "excess_s": excess_s,
                    "speedup_if_10x": 1.0 / (1.0 - s + s / 10.0),
                    "evidence": [_evidence(r, n) for r in ev],
                })
        # structural signature: selection costs at least as much as bounding (the kFSB shape)
        sel = per_cat[c]["mean_share"].get("branch_select", 0.0)
        # child bounding NET of the bounding done inside selection (simulated children are not explored ones)
        bnd = per_cat[c]["bound_net_share"]
        if sel >= ANOM_MIN and sel >= bnd:
            ev = sorted((r for r in ok if r["category"] == c), key=lambda r: -r["m"]["share"].get("branch_select", 0))
            findings.append({
                "kind": "select>=bound",
                "category": c,
                "phase": "branch_select",
                "share": sel,
                "median": median.get("branch_select", 0.0),
                "excess_s": max(0.0, sel - bnd) * per_cat[c]["wall_s"],
                "speedup_if_10x": 1.0 / (1.0 - sel + sel / 10.0),
                "evidence": [_evidence(r, "branch_select") for r in ev],
                "note": f"branch_select {sel:.2f} >= child_bound (net of selection) {bnd:.2f}: choosing the split costs at "
                        f"least as much as bounding the children; {per_cat[c]['in_select_share']:.2f} of the wall is "
                        f"bounding/optimizer work nested inside selection",
            })
    findings.sort(key=lambda f: (-f["excess_s"], f["category"], f["phase"]))
    # SYSTEMIC cost centres: a phase that is expensive in MOST categories is invisible to the median rule above.
    total_wall = sum(float(r["wall_s"]) for r in ok)
    systemic = []
    for n in phases:
        if n in UMBRELLAS:
            continue
        secs = sum(r["m"]["share"].get(n, 0.0) * float(r["wall_s"]) for r in ok)
        med = median.get(n, 0.0)
        if med >= ANOM_MIN or (total_wall > 0 and secs / total_wall >= ANOM_MIN):
            systemic.append({"phase": n, "median_share": med, "seconds": secs,
                             "share_of_census_wall": secs / total_wall if total_wall else 0.0,
                             "categories_over_0p1": sorted(c for c, d in per_cat.items()
                                                           if d["mean_share"].get(n, 0.0) >= ANOM_MIN)})
    systemic.sort(key=lambda x: -x["seconds"])
    # ATTACK ON PROVED ROWS: occupied attack wall on rows the run itself proved unsat (an attack cannot win them; it
    # either delays the proof or, when concurrent, holds threads the proof could use).
    def attack_on(verdicts: tuple[str, ...]) -> dict:
        rows = [r for r in ok if r["verdict"] in verdicts]
        return {
            "rows": len(rows),
            "wall_s": sum(float(r["wall_s"]) for r in rows),
            "attack_s": sum(r["m"]["share"].get("attack", 0.0) * float(r["wall_s"]) for r in rows),
            "worst": sorted(({"tag": r["tag"], "attack_share": r["m"]["share"].get("attack", 0.0),
                              "wall_s": float(r["wall_s"]), "concurrent": r["m"]["top_sum"] > 1.05}
                             for r in rows), key=lambda x: -x["attack_share"])[:8],
        }

    attack_on_unsat = attack_on(("unsat",))
    # rows the run did NOT decide: there the attack's wall is budget the proof never got
    attack_on_open = attack_on(("timeout", "unknown"))
    overhead = None
    if guard_ns is not None and ok:
        ov = sorted(r["m"]["overhead"] for r in ok)
        worst = max(ok, key=lambda r: r["m"]["overhead"])
        overhead = {"guard_ns": guard_ns, "sample_us": sample_us, "max": ov[-1], "median": statistics.median(ov),
                    "worst_tag": worst["tag"], "rows_over_0p5pct": sum(1 for x in ov if x > 0.005)}
    return {"ok": ok, "per_category": per_cat, "median": median, "phases": phases, "findings": findings,
            "overhead": overhead, "systemic": systemic, "attack_on_unsat": attack_on_unsat,
            "attack_on_open": attack_on_open}


def _evidence(run: dict, phase: str) -> dict:
    p = run["m"]["phases"].get(phase, {})
    units = int(p.get("units", 0))
    w = float(p.get("wall_s", 0.0))
    return {
        "tag": run["tag"],
        "verdict": run["verdict"],
        "wall_s": float(run["wall_s"]),
        "phase_wall_s": w,
        "share": run["m"]["share"].get(phase, 0.0),
        "enters": int(p.get("enters", 0)),
        "units": units,
        "unit": p.get("unit", ""),
        "s_per_unit": (w / units) if units else None,
        "max_inflight": int(p.get("max_inflight", 0)),
    }


# ------------------------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------------------------


def _f(x, nd=2, dash="-"):
    if x is None:
        return dash
    if isinstance(x, float):
        if x != 0 and (abs(x) >= 1e4 or abs(x) < 10 ** (-nd)):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def render(agg: dict, runs: list[dict], meta: dict) -> str:
    L: list[str] = []
    L.append(f"# Phase-cost census — {meta.get('title', 'bottleneck report')}")
    L.append("")
    L.append("Generated by `scripts/costcensus/costcensus.py aggregate` from the run artefacts listed below; every number"
             " is re-derivable from them. Shares are INCLUSIVE occupied wall over the row's wall (a phase contains"
             " the phases nested in it: `bab_loop` contains `branch_select` contains `branch_sim` contains"
             " `beta_opt`), so shares of different phases do not sum to 1.")
    L.append("")
    for k in ("pin", "pin_sha256", "commit", "stratum", "configs_dir", "profile", "host", "generated"):
        if meta.get(k):
            L.append(f"- **{k}**: `{meta[k]}`")
    nvoid = sum(1 for r in runs if r["status"] == "VOID")
    nemerg = sum(1 for r in runs if r["status"] == "EMERGENCY")
    L.append(f"- **rows**: {len(runs)} planned/run, {len(agg['ok'])} OK, {nvoid} VOID, {nemerg} EMERGENCY")
    if nemerg:
        L.append("")
        L.append("**SOUNDNESS EMERGENCY: at least one run contradicted a banked decided verdict (see Rows).**")
    newly = [r for r in runs if r.get("new_decision")]
    if newly:
        L.append("")
        L.append(f"**Rows the bank holds open that this census DECIDED ({len(newly)}) -- NOT conversions:** "
                 + "; ".join(f"`{r['tag']}` banked {r['bank_verdict']} -> {r['new_decision']}" for r in newly)
                 + ". A census run is not a moat: an arm64 `sat` must be replayed on the certified x86 checker before it"
                 " is banked (x86 parity), and an `unsat` needs a both-year moat.")
    L.append("")
    ov = agg.get("overhead")
    if ov:
        L.append(f"- **census overhead bound** (every enter priced at {ov['guard_ns']:.0f} ns, every RSS sample at"
                 f" {ov['sample_us']:.0f} us): max {100 * ov['max']:.4f}% of row wall (`{ov['worst_tag']}`), median"
                 f" {100 * ov['median']:.4f}%, rows over 0.5%: {ov['rows_over_0p5pct']}")
    L.append("")
    L.append("## Ranked bottlenecks")
    L.append("")
    if not agg["findings"]:
        L.append("No phase share is anomalous under the rule (share >= "
                 f"{ANOM_MIN}, >= {ANOM_RATIO}x and >= +{ANOM_ABS} over the cross-category median).")
    else:
        L.append("Rule: a category's mean share of a phase is flagged when it is >= "
                 f"{ANOM_MIN}, at least {ANOM_RATIO}x the cross-category median share and at least +{ANOM_ABS} above"
                 " it; `select>=bound` flags categories where branch selection costs at least as much wall as child"
                 " bounding. `excess_s` = (share - median) x the category's census wall: the seconds that phase spent"
                 " above a typical category, summed over the sampled rows. `x10` = the row speedup if that phase alone"
                 " became 10x cheaper (Amdahl).")
        L.append("")
        L.append("| rank | category | phase | kind | share | median | excess_s | x10 | evidence (tag: phase_wall/row_wall, enters, s/unit) |")
        L.append("|---:|---|---|---|---:|---:|---:|---:|---|")
        for i, f in enumerate(agg["findings"], 1):
            ev = "; ".join(
                f"`{e['tag']}` {e['verdict']}: {_f(e['phase_wall_s'],1)}/{_f(e['wall_s'],1)} s, {e['enters']} enters"
                + (f", {_f(e['s_per_unit'],4)} s/{e['unit'].rstrip('s')}" if e["s_per_unit"] is not None else "")
                for e in f["evidence"][:3])
            L.append(f"| {i} | {f['category']} | {f['phase']} | {f['kind']} | {_f(f['share'])} | {_f(f['median'])} |"
                     f" {_f(f['excess_s'],1)} | {_f(f['speedup_if_10x'])} | {ev} |")
        notes = [f for f in agg["findings"] if f.get("note")]
        if notes:
            L.append("")
            for f in notes:
                L.append(f"- {f['category']}: {f['note']}")
    L.append("")
    L.append("## Systemic cost centres")
    L.append("")
    L.append("A phase expensive in MOST categories is invisible to the median rule above; listed here when its"
             f" cross-category median share is >= {ANOM_MIN} or it holds >= {ANOM_MIN} of all census wall"
             " (occupied wall: concurrent phases each count their own wall).")
    L.append("")
    L.append("| phase | median share | seconds | share of census wall | categories with share >= 0.1 |")
    L.append("|---|---:|---:|---:|---|")
    for x in agg.get("systemic", []):
        L.append(f"| {x['phase']} | {_f(x['median_share'])} | {_f(x['seconds'],1)} |"
                 f" {_f(x['share_of_census_wall'])} | {', '.join(x['categories_over_0p1'])} |")
    for key, title in (("attack_on_unsat", "Attack on rows the run proved unsat"),
                       ("attack_on_open", "Attack on rows the run did not decide (timeout/unknown)")):
        au = agg.get(key)
        if au and au["rows"]:
            L.append("")
            L.append(f"**{title}**: {_f(au['attack_s'],1)} s of attack wall across {au['rows']} rows totalling"
                     f" {_f(au['wall_s'],1)} s ({_f(au['attack_s'] / au['wall_s'] if au['wall_s'] else 0.0)} of"
                     " their wall). Worst:"
                     + ";".join(f" `{w['tag']}` {_f(w['attack_share'])} of {_f(w['wall_s'],1)} s"
                                + (" (concurrent with other phases)" if w["concurrent"] else " (serial)")
                                for w in au["worst"]) + ".")
    L.append("")
    L.append("## Per-category phase shares (mean over OK rows)")
    L.append("")
    ph = agg["phases"]
    L.append("| category | rows | wall_s | coverage | " + " | ".join(ph)
             + " | bound_net | in_select | s/verified | s/explored | rss_peak_mb |")
    L.append("|---|---:|---:|---:|" + "---:|" * len(ph) + "---:|---:|---:|---:|---:|")
    for c, d in agg["per_category"].items():
        cells = " | ".join(_f(d["mean_share"].get(n)) if n in d["mean_share"] else "" for n in ph)
        L.append(f"| {c} | {len(d['rows'])} | {_f(d['wall_s'],1)} | {_f(d['coverage'])} | {cells} |"
                 f" {_f(d['bound_net_share'])} | {_f(d['in_select_share'])} |"
                 f" {_f(d['s_per_verified'],3)} | {_f(d['s_per_explored'],3)} | {_f(d['rss_peak_mb'],0)} |")
    L.append("| **median** | | | | " + " | ".join(_f(agg["median"].get(n)) for n in ph) + " | | | | | |")
    L.append("")
    L.append("`bound_net` = child_bound share net of the bounding done INSIDE branch selection (kFSB simulates"
             " candidate children with the same code); `in_select` = share of the wall spent in non-branch phases"
             " (child_bound, beta_opt, alpha_opt, ...) nested inside branch_select/prescore/sim on the same thread.")
    L.append("")
    L.append("`coverage` = occupied wall of the non-umbrella phases entered at a thread's top level or directly"
             " under an umbrella (`bab_loop`, `verify`), over the row wall (clamped at 1): how much of the row the census"
             " attributes to a cost centre. It is an UPPER bound (top-level phases that ran concurrently -- an attack"
             " beside a concurrent lane's root bounds, or a lane's stack-neutral spans recorded under `root` -- each"
             " count their own wall), so `unattributed` = 1 - coverage is a LOWER bound; it is ranked like a phase (process startup,"
             " device/session init, preset/router work, BaB bookkeeping, uninstrumented lanes). Compare it with the"
             " `verify` umbrella's share: wall inside `verify` but in no sub-phase is an uninstrumented lane.")
    L.append("")
    L.append("## Rows")
    L.append("")
    L.append("`gap` = the longest stretch during which no non-umbrella phase was open (from first-entry / last-exit"
             " spans, a still-open phase running to the snapshot: a lower bound on the longest uninstrumented stretch),"
             " with the phases on either side. A SPARSE span (occupied wall < half the span, e.g. `publish`: placeholder"
             " at the start, verdict at the end) is not occupancy; its occupied wall is charged against the stretch.")
    L.append("")
    L.append("| tag | stratum | bank | verdict | wall_s | budget | status | top phases (share) | explored | verified | note |")
    L.append("|---|---|---|---|---:|---:|---|---|---:|---:|---|")
    for r in runs:
        if r["status"] == "OK":
            sh = sorted(r["m"]["share"].items(), key=lambda kv: -kv[1])[:4]
            tops = ", ".join(f"{n} {_f(s)}" for n, s in sh)
            L.append(f"| `{r['tag']}` | {r['stratum']} | {r['bank_verdict']} | {r['verdict']}"
                     f"{' (partial)' if r['partial'] else ''} | {_f(r['wall_s'],1)} | {r['budget']} | OK | {tops} |"
                     f" {r['m']['explored']} | {r['m']['verified']} |"
                     + (f" gap {_f(r['m']['max_gap_s'],1)} s ({r['m']['gap_after']} -> {r['m']['gap_before']});"
                        if r["m"]["max_gap_s"] >= 1.0 else "")
                     + (f" overhead<={100 * r['m']['overhead']:.4f}%" if "overhead" in r["m"] else "") + " |")
        else:
            L.append(f"| `{r['tag']}` | {r['stratum']} | {r['bank_verdict']} | {r['verdict']} | {_f(r['wall_s'],1)} |"
                     f" {r['budget']} | {r['status']} | | | | {'; '.join(r['why'])} |")
    L.append("")
    return "\n".join(L)


def load_runs(out_dir: str, expect_commit: str) -> list[dict]:
    spec = read_spec(os.path.join(out_dir, "plan.tsv"))
    runs = []
    for row in spec:
        res = os.path.join(out_dir, "res", f"{row['tag']}.txt")
        flight_path = res + ".flight.json"
        log_path = os.path.join(out_dir, "raw", f"{row['tag']}.log")
        flight = None
        if os.path.isfile(flight_path):
            try:
                flight = json.load(open(flight_path))
            except json.JSONDecodeError:
                flight = None
        log = open(log_path, errors="replace").read() if os.path.isfile(log_path) else None
        res_text = open(res, errors="replace").read() if os.path.isfile(res) else None
        runs.append(judge_run(row, flight, log, res_text, expect_commit))
    return runs


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--ny-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--rows-per-category", type=int, default=2)
    p.add_argument("--cheap-budget", type=int, default=120)
    p.add_argument("--max-budget", type=int, default=1800)
    p.add_argument("--categories", default="")
    a = sub.add_parser("aggregate")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--expect-commit", required=True)
    a.add_argument("--report", default=None)
    a.add_argument("--json", default=None)
    a.add_argument("--title", default="bottleneck report")
    a.add_argument("--meta", action="append", default=[], help="key=value lines for the report header")
    a.add_argument("--guard-ns", type=float, default=None,
                   help="measured per-enter guard cost (ns), for the overhead bound")
    a.add_argument("--sample-us", type=float, default=20.0, help="cost of one RSS sample (us)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            cats = [c for c in args.categories.split(",") if c] or None
            spec = plan(args.ny_root, args.out, args.year, args.rows_per_category, args.cheap_budget,
                        args.max_budget, cats)
            worst = sum(r["budget"] for r in spec)
            print(f"COSTCENSUS_PLAN_DONE rc=0 rows={len(spec)} categories={len({r['category'] for r in spec})} "
                  f"worst_case_wall_s={worst} out={args.out}")
            return 0
        runs = load_runs(args.out_dir, args.expect_commit)
        agg = aggregate(runs, args.guard_ns, args.sample_us)
        meta = {"title": args.title}
        for kv in args.meta:
            k, _, v = kv.partition("=")
            meta[k] = v
        md = render(agg, runs, meta)
        if args.report:
            with open(args.report, "w") as fh:
                fh.write(md)
        if args.json:
            slim = {
                "schema": "costcensus-aggregate/v1",
                "meta": meta,
                "runs": [{k: v for k, v in r.items() if k not in ("cost", "m")} for r in runs],
                "per_category": agg["per_category"],
                "median": agg["median"],
                "findings": agg["findings"],
                "overhead": agg["overhead"],
                "systemic": agg["systemic"],
                "attack_on_unsat": agg["attack_on_unsat"],
                "attack_on_open": agg["attack_on_open"],
            }
            with open(args.json, "w") as fh:
                json.dump(slim, fh, indent=1, sort_keys=True, default=str)
        nvoid = sum(1 for r in runs if r["status"] == "VOID")
        nemerg = sum(1 for r in runs if r["status"] == "EMERGENCY")
        rc = 3 if nemerg else (0 if runs and nvoid == 0 else (5 if agg["ok"] else 6))
        newly = sum(1 for r in runs if r.get("new_decision"))
        print(f"COSTCENSUS_AGG_DONE rc={rc} rows={len(runs)} ok={len(agg['ok'])} void={nvoid} emergency={nemerg} "
              f"findings={len(agg['findings'])} categories={len(agg['per_category'])} newly_decided={newly}")
        return rc
    except CensusError as exc:
        print(f"COSTCENSUS_DONE rc=2 refused: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
