#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""M0 item 5 / section 1.7: the BOTH-YEAR moat row list for scripts/m0/moat_both_profiles.sh.

For each scored category of each year, align three sources BY INDEX, asserting the key at every index:

  * instances.csv -- 2026: the version named by reports/measured-2026/bank_manifest.json (rows checked against
    the manifest); 2025: benchmarks/vnncomp2025/benchmarks/<cat>/instances.csv;
  * the bank -- reports/measured-2026/<cat>.csv or reports/measured/<cat>.csv (same length, same onnx/vnnlib at
    every index, or refused: repeated keys such as sat_relu's are exactly why the key is asserted per index);
  * field truth -- the VNN-COMP results.txt (2026 SCORING, 2025 SCORING-ZERO-TOL): the scorer's row key
    `<onnx stem>-<vnnlib stem>` must equal the instances.csv key at the same index. truth=sat when a
    counterexample scored, unsat when a verification scored and no counterexample did.

and emit one MOAT spec row per verdict-bearing row with the official per-row budget and the preset the resolver
would load from --configs-dir. Selected by default: every banked sat/unsat row AND every field-SAT row whatever its
bank (`select_row`). The field-SAT rows ny banks timeout/unknown are the rows on which a newly armed tier runs to
the end on a truly-SAT instance with nothing to short-circuit it -- the one place a false unsat can appear that a
banked-rows-only moat never visits (64 rows on the 2026 banks, 7 of them soundnessbench's planted sats).
--banked-only restores the banked-rows-only list; --all-rows takes every row. The runner compares each run's
verdict against the bank (a sat<->unsat flip is an EMERGENCY) and the field truth (a contradiction is an
EMERGENCY), in both ProofProfiles.

A category missing from the field results is reported (its rows carry truth='-'), never silently passed; a
results file that yields no row at all is refused.

    python3 scripts/m0/moat_rows.py --configs-dir <home>/ny-wt-<pin>/configs \
        --out <home>/ny/scratchpad/m0/moat_<pin>.tsv --years 2026,2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m0lib as L  # noqa: E402


def categories(ny_root: str, year: int) -> tuple[list[str], list[str]]:
    """(banked scored categories, scored categories with no bank) of one year."""
    if year == 2026:
        man = os.path.join(ny_root, "reports/measured-2026/bank_manifest.json")
        if not os.path.isfile(man):
            raise L.M0Error(f"bank manifest missing: {man}")
        d = json.load(open(man))
        scored = list(d["scored_categories"])
        return [c for c in scored if c in d["banks"]], [c for c in scored if c not in d["banks"]]
    out, unbanked = [], []
    root = os.path.join(ny_root, "benchmarks/vnncomp2025/benchmarks")
    for c in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, c)) or c == "test":
            continue
        (out if os.path.isfile(os.path.join(ny_root, "reports/measured", c + ".csv")) else unbanked).append(c)
    return out, unbanked


def select_row(bank: str, truth_token: str, all_rows: bool = False, banked_only: bool = False) -> bool:
    """A moat row carries a verdict to defend: a banked sat/unsat, or (unless banked_only) a field-SAT row -- where
    an unsat is the kill line whatever ny banked."""
    if all_rows:
        return True
    if bank in L.DECIDED:
        return True
    return not banked_only and truth_token == "sat"


def build(ny_root: str, configs_dir: str, years: list[int], cats_filter: set[str] | None, truth_paths: dict,
          all_rows: bool, reps_c: int, reps_p: int, prefix: str, banked_only: bool = False
          ) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    report: list[str] = []
    for year in years:
        truth = L.parse_results_txt(truth_paths[year], year)
        if not truth:
            raise L.M0Error(f"{truth_paths[year]}: no {year} category parsed (a truth gate that compares against "
                            "nothing is a false pass)")
        banked, unbanked = categories(ny_root, year)
        for cat in unbanked:
            if not cats_filter or cat in cats_filter:
                report.append(f"  {year} {cat:44s} SKIPPED: scored but no bank (nothing to moat)")
        for cat in banked:
            if cats_filter and cat not in cats_filter:
                continue
            inst = L.load_instances(L.scored_instances_csv(ny_root, year, cat))
            bank = L.align_bank(inst, L.load_bank(L.bank_csv(ny_root, year, cat)), f"{year}/{cat} bank")
            tr = L.align_truth(inst, truth.get(cat), f"{year}/{cat} truth")
            preset = L.resolve_preset(configs_dir, cat)
            n_truth = sum(1 for t in tr if t is not None)
            picked = sat_extra = 0
            for r, b, t in zip(inst, bank, tr):
                tok = t.token() if t else "-"
                if not select_row(b, tok, all_rows, banked_only):
                    continue
                picked += 1
                sat_extra += int(b not in L.DECIDED and tok == "sat")
                rows.append(L.spec_row(
                    tag=f"{prefix}{year % 100}_{cat}_{r.index:04d}", kind="MOAT", tier="-", year=year, category=cat,
                    onnx=r.onnx, vnnlib=r.vnnlib, budget=r.budget, configs_dir=configs_dir, preset=preset or "-",
                    truth=tok, bank=b, index=r.index, reps_c=reps_c, reps_p=reps_p, need="-", alarm="-",
                    flight="-", group=f"moat/{year}",
                    note=f"true_result={t.true_result if t else '-'} fals_ok={len(t.fals_ok) if t else 0} "
                         f"ver_ok={len(t.ver_ok) if t else 0}"))
            report.append(f"  {year} {cat:44s} rows={len(inst):5d} picked={picked:5d} field_sat_unbanked={sat_extra:3d}"
                          f" with_truth={n_truth:5d}"
                          f" preset={preset or 'NONE (auto defaults)'}"
                          + ("" if cat in truth else "  NOTE: category absent from field results"))
    return rows, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs-dir", required=True, help="the arming pin's configs dir")
    ap.add_argument("--out", required=True, help="spec to write (refused if it exists)")
    ap.add_argument("--years", default="2026,2025")
    ap.add_argument("--categories", default="all")
    ap.add_argument("--ny-root", default="<home>/ny")
    ap.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    ap.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    ap.add_argument("--all-rows", action="store_true", help="every row, not only the verdict-bearing ones")
    ap.add_argument("--banked-only", action="store_true",
                    help="only banked sat/unsat rows (drops the field-SAT rows ny banks undecided: NOT a full moat)")
    ap.add_argument("--reps-competition", type=int, default=3,
                    help="section 1.7 mode identity: a Proof verdict must hold in >= 2 of 3 Competition reps")
    ap.add_argument("--reps-proof", type=int, default=1)
    ap.add_argument("--tag-prefix", default="MOAT")
    a = ap.parse_args(argv)
    try:
        years = [int(y) for y in a.years.split(",") if y.strip()]
        cats = None if a.categories == "all" else {c.strip() for c in a.categories.split(",") if c.strip()}
        cfg = os.path.abspath(a.configs_dir)
        if not os.path.isdir(cfg):
            raise L.M0Error(f"configs dir missing: {cfg}")
        if a.all_rows and a.banked_only:
            raise L.M0Error("--all-rows and --banked-only contradict each other")
        rows, report = build(a.ny_root, cfg, years, cats, {2026: a.truth_2026, 2025: a.truth_2025}, a.all_rows,
                             a.reps_competition, a.reps_proof, a.tag_prefix, a.banked_only)
        if not rows:
            raise L.M0Error("no rows selected")
        if cats:
            seen = {r["category"] for r in rows}
            missing = cats - seen
            if missing:
                raise L.M0Error(f"categories selected nothing: {sorted(missing)}")
        L.write_spec(a.out, rows)
        print("\n".join(report))
        runs = sum(int(r["reps_c"]) + int(r["reps_p"]) for r in rows)
        print(f"MOATROWS_DONE rc=0 rows={len(rows)} runs={runs} spec={a.out}")
        return 0
    except L.M0Error as e:
        print(f"MOATROWS_REFUSED: {e}", file=sys.stderr)
        print("MOATROWS_DONE rc=2 rows=0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
