#!/usr/bin/env python3
"""Audit every banked non-solved row against its OFFICIAL per-instance budget.

WHY THIS EXISTS
---------------
A row banked as `timeout` is normally a statement that ny spent the whole
budget and still could not decide. That is a capability fact, and the campaign
plans around it.

But a `timeout` whose wall is a small fraction of the budget is not that fact at
all. It is one of two bugs, and both have bitten us:

  1. A STALE SWEEP CAP. Old sweeps ran with a harness cap (~101s, ~106s, ~110s
     fingerprints appear across cgan, cifar100 and nn4sys) and banked the result
     as though it were the official budget. nn4sys carries 22 such rows, banked
     at ~110s against budgets of 380-800s. They stop at 110-112s whether the
     budget is 380s or 800s -- a real timeout tracks the budget; a cap does not.

  2. AN EARLY QUIT. A deadline raised inside a bound walk propagates out and the
     instance ends with no verdict while budget remains. cgan lost 1456s across
     seven rows this way.

Either way the banked number is false, and a false bank is worse than no bank:
the next agent measures against it and does not know. This script finds them.

Reads the banks (`reports/measured/`, `reports/measured-ext/`,
`reports/measured-2026/`) and joins to the corpus `instances.csv` on
(onnx basename, vnnlib basename).

  2026 CORPUS (the default). `benchmarks/vnncomp2026_benchmarks/benchmarks/
  <bench>/<version>/instances.csv`. A row that matches here has a
  content-stable 2026 identity; a row that does not is reported as UNMATCHED
  and is NOT evidence of anything.

  2025 CORPUS (`--corpus 2025`). The historical flat layout. NOTE that
  `benchmarks/vnncomp2025/benchmarks/` in this checkout holds exactly ONE
  family, so this mode loads ~200 budgets, fails to match ~570 banked rows,
  and reports ZERO hits board-wide. It was the only mode this script had until
  2026-08-25, which is why the 48-row underspend it exists to find had never
  once been printed by it.

Three bank schemas are read, not one:
  measured-5  category,onnx,vnnlib,verdict,seconds            (measured-ext)
  measured-6  category,onnx,vnnlib,prepare_status,verdict,seconds
  measured-7  ... ,run_id
The same row is banked in more than one directory (all 69 ml4acopf rows appear
in both `measured/` and `measured-ext/`), so rows are DEDUPED on
(bench, onnx basename, vnnlib basename) before counting. Without that the
per-family counts double.

Usage:  python3 scripts/audit_banked_budget_use.py [--threshold 0.75] [--top 20]
                                                   [--corpus 2026|2025]

Exit status is always 0 -- this is a reporting tool, not a gate. It is
deliberately NOT a test, because a legitimately-refused row (an unsupported op
declined in 2s) is correct behaviour and should not fail a build.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVED = {"unsat", "sat"}


def base(p):
    return p.split("/")[-1]


# NO benchmark-name aliasing. It is tempting to map the 2025 bank files
# `relusplitter` -> `relusplitter_2026` and `soundnessbench` ->
# `soundnessbench_2026`, but those pairs share ZERO instance identities: the
# 2025 relusplitter bank is mnist_fc/cifar rows, the 2026 corpus is
# `model_2_2~...`; 2025 soundnessbench is `vnnlib/model_0.vnnlib`, 2026 is
# `vnnlib/properties/property_000.vnnlib`. Aliasing does not make them match --
# it only files 187 un-auditable 2025 rows under a 2026 family name, which is
# exactly the wrong-lane trap this repo keeps paying for. The 2026 rows for
# both families are banked under their own names in `reports/measured-2026/`.

CORPORA = {
    "2026": ("benchmarks/vnncomp2026/benchmarks/*/*/instances.csv", -3),
    "2025": ("benchmarks/vnncomp2025/benchmarks/*/instances.csv", -2),
}

# PRECEDENCE, most authoritative FIRST. `reports/measured-2026/` holds rows
# measured against the 2026 corpus at its official budgets and wins every tie.
# This is not cosmetic: `reports/measured/lsnc_relu.csv` carries 69 rows whose
# walls are whole integers (10..15) at under half the 25 s budget, and
# `reports/measured-2026/lsnc_relu_NOTES.md` establishes those rows are
# FABRICATED, not measured -- every genuinely measured row in this repo carries
# a decimal. Reading the 2025 file first files 21 fabricated rows as an
# underspend finding and hides the real 2026 measurement of the same 80 rows.
BANK_DIRS = ("reports/measured-2026", "reports/measured-ext", "reports/measured")

VERDICT_TOKENS = {"unsat", "sat", "timeout", "unknown", "error", "holds", "violated"}


def load_budgets(corpus="2026"):
    """(bench, onnx, vnnlib) -> official per-instance timeout in seconds.

    The 2026 corpus is VERSIONED (`<bench>/1.0/`, `<bench>/2.0/`) and the two
    lists can disagree, so the LARGER budget is kept: this audit's whole point
    is "was budget discarded", and crediting the smaller budget would hide a
    real underspend behind a version choice this script cannot make.

    The budget is the LAST field, not field 2: relational families quote a
    python-style pair list in field 0 that contains commas.
    """
    pat, bench_at = CORPORA[corpus]
    budgets = {}
    for inst in glob.glob(os.path.join(REPO, pat)):
        bench = inst.split("/")[bench_at]
        with open(inst) as fh:
            for row in csv.reader(fh):
                if len(row) < 3:
                    continue
                try:
                    secs = float(row[-1])
                except ValueError:
                    continue
                key = (bench, base(row[0]), base(row[1]))
                budgets[key] = max(secs, budgets.get(key, 0.0))
    return budgets


def parse_bank_row(row):
    """(verdict, seconds) from a measured-5 / -6 / -7 row, or None."""
    if len(row) < 5:
        return None
    if row[3].strip() in VERDICT_TOKENS:      # measured-5: no prepare_status
        return row[3].strip(), row[4]
    if len(row) < 6:
        return None
    return row[4].strip(), row[5]


def load_banked_rows():
    """(bench, onnx, vnnlib) -> (verdict, seconds), deduped across bank dirs."""
    rows = {}
    for d in BANK_DIRS:
        for path in sorted(glob.glob(os.path.join(REPO, d, "*.csv"))):
            bench = os.path.basename(path)[:-4]
            with open(path) as fh:
                for row in csv.reader(fh):
                    parsed = parse_bank_row(row)
                    if parsed is None:
                        continue
                    key = (bench, base(row[1]), base(row[2]))
                    rows.setdefault(key, parsed)
    return rows


def audit(threshold, overrun=1.0, corpus="2026"):
    """Return (under-budget hits, over-budget hits, unmatched, n_budgets).

    Two opposite failures, both of which invalidate the banked number:

      UNDER  -- a stale sweep cap or an early quit (see module docstring).
      OVER   -- the run exceeded the limit the competition enforces. Those
                seconds do not exist: the harness kills at the budget, so the
                row scores zero no matter what the solver later concluded, and
                a solver that believes it has more time is itself the bug.
                safenlp is the live example -- 44 rows at 120-155% of a 20s
                budget, which the under-budget check by construction cannot see.

    UNMATCHED is per-benchmark and is reported, not swallowed. A bank row with
    no corpus identity is UN-AUDITED, not clean: 162 of the 200 banked vit_2023
    rows -- every single pgd_2_3_16 row -- name a vnnlib the 2026 corpus does
    not contain, so no statement about their budget use is even meaningful.
    """
    budgets = load_budgets(corpus)
    hits = []
    over = []
    unmatched = defaultdict(int)
    for (bench, onnx, vnnlib), (verdict, seconds) in load_banked_rows().items():
        if verdict in SOLVED:
            continue
        budget = budgets.get((bench, onnx, vnnlib))
        if budget is None:
            unmatched[bench] += 1
            continue
        try:
            wall = float(seconds)
        except (TypeError, ValueError):
            continue
        rec = (bench, vnnlib, verdict, wall, budget, wall / budget)
        if wall < threshold * budget:
            hits.append(rec)
        elif wall > overrun * budget:
            over.append(rec)
    return hits, over, unmatched, len(budgets)


def print_cap_verdict(hits):
    """Apply the docstring's OWN discriminator instead of leaving it to the reader.

    The two under-budget causes have opposite fingerprints and want opposite
    follow-ups, and until now this script printed the rule but never evaluated
    it:

      CAP        the wall is flat while the budget varies -- a harness limit was
                 recorded as if it were the official budget. The banked number
                 is FALSE and the row is UN-MEASURED. Re-measure; do NOT read it
                 as capability.
      EARLY-QUIT the wall varies and tracks nothing -- the solver stopped on its
                 own. Find the refusal in the run's own log.

    Needs >= 3 rows and a >= 1.5x budget spread before it will call anything,
    and says INCONCLUSIVE otherwise rather than guessing: a flat wall across a
    FLAT budget is not evidence of a cap, only of a hard instance. 1.5x is the
    smallest spread at which a <1.25x wall spread already proves the wall is
    not tracking the budget; nn4sys's flagged rows span 340-600 s (1.76x) and a
    2x rule would have declined to call the one cluster in the tree that is
    already CONFIRMED a cap (see 28552d53c: those rows re-run at their true
    budgets land at 536-797 s).
    """
    groups = defaultdict(list)
    for bench, vnnlib, _verdict, wall, budget, _f in hits:
        groups[bench].append((wall, budget, vnnlib))

    print("\nper-benchmark verdict on THAT rule:")
    for bench in sorted(groups):
        rows = groups[bench]
        walls = [w for w, _b, _v in rows]
        budgets = [b for _w, b, _v in rows]
        if len(rows) < 3 or max(budgets) < 1.5 * min(budgets):
            print(f"  {bench:18s} INCONCLUSIVE  ({len(rows)} rows, budgets "
                  f"{min(budgets):.0f}-{max(budgets):.0f}s: too few rows or too "
                  f"little budget spread to tell a cap from a quit)")
            continue
        wall_spread = max(walls) / max(min(walls), 1e-9)
        budget_spread = max(budgets) / max(min(budgets), 1e-9)
        if wall_spread < 1.25:
            print(f"  {bench:18s} CAP           (walls {min(walls):.0f}-{max(walls):.0f}s "
                  f"flat across budgets {min(budgets):.0f}-{max(budgets):.0f}s) "
                  f"-- banked number is FALSE, re-measure before quoting it")
        else:
            print(f"  {bench:18s} EARLY-QUIT    (walls {min(walls):.0f}-{max(walls):.0f}s vary "
                  f"across budgets {min(budgets):.0f}-{max(budgets):.0f}s) "
                  f"-- find the refusal in the run's own log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="flag rows using less than this fraction of budget")
    ap.add_argument("--top", type=int, default=20, help="worst-N individual rows to list")
    ap.add_argument("--overrun", type=float, default=1.0,
                    help="flag rows whose wall exceeds this multiple of budget")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="2026",
                    help="which corpus supplies the official budgets (default: 2026)")
    args = ap.parse_args()

    hits, over, unmatched, nbud = audit(args.threshold, args.overrun, args.corpus)
    if nbud == 0:
        # An audit that loaded NO official budgets cannot clear anything. Every
        # row falls through to UN-AUDITED and the summary would read as a
        # board-wide all-clear built from zero evidence -- the precise shape of
        # the absence-as-evidence trap this repo keeps paying for. Refuse.
        pat, _ = CORPORA[args.corpus]
        print(f"VOID: corpus {args.corpus!r} loaded 0 official budgets.", file=sys.stderr)
        print(f"  glob: {os.path.join(REPO, pat)}", file=sys.stderr)
        print("  Nothing was audited; this is NOT an all-clear. Fix the corpus "
              "path or download the corpus, then re-run.", file=sys.stderr)
        return 2
    n_unmatched = sum(unmatched.values())
    print(f"corpus: {args.corpus}   official budgets loaded: {nbud}   "
          f"banked non-solved rows with NO corpus identity (UN-AUDITED): {n_unmatched}")
    if n_unmatched:
        worst = sorted(unmatched.items(), key=lambda kv: -kv[1])[:8]
        print("  un-audited by benchmark: " + ", ".join(f"{b}={n}" for b, n in worst))
    print(f"\nbanked NON-SOLVED rows using <{args.threshold:.0%} of official budget: {len(hits)}\n")

    agg = defaultdict(lambda: [0, 0.0])
    for bench, _v, _ver, wall, budget, _f in hits:
        agg[bench][0] += 1
        agg[bench][1] += budget - wall
    print(f"{'benchmark':22s} {'rows':>5s} {'budget_discarded_s':>19s}")
    for bench in sorted(agg, key=lambda b: -agg[b][1]):
        print(f"{bench:22s} {agg[bench][0]:5d} {agg[bench][1]:19.0f}")

    print(f"\nworst {args.top} individual rows:")
    for bench, vnnlib, verdict, wall, budget, frac in sorted(hits, key=lambda r: r[5])[: args.top]:
        print(f"  {bench:18s} {vnnlib[:40]:40s} {verdict:8s} "
              f"{wall:7.1f}s /{budget:7.0f}s = {frac * 100:5.1f}%")

    if hits:
        print("\nA uniform wall across DIFFERENT budgets means a cap, not a timeout: re-measure.")
        print("A wall far under budget with a varying value means an early quit: find the refusal.")
        print_cap_verdict(hits)

    print(f"\n{'=' * 70}\nbanked NON-SOLVED rows EXCEEDING {args.overrun:.0%} of official budget: {len(over)}")
    if over:
        agg_o = defaultdict(lambda: [0, 0.0])
        for bench, _v, _ver, wall, budget, frac in over:
            agg_o[bench][0] += 1
            agg_o[bench][1] = max(agg_o[bench][1], frac)
        print(f"\n{'benchmark':22s} {'rows':>5s} {'worst_frac':>11s}")
        for bench in sorted(agg_o, key=lambda b: -agg_o[b][0]):
            print(f"{bench:22s} {agg_o[bench][0]:5d} {agg_o[bench][1] * 100:10.0f}%")
        print(f"\nworst {args.top}:")
        for bench, vnnlib, verdict, wall, budget, frac in sorted(over, key=lambda r: -r[5])[: args.top]:
            print(f"  {bench:18s} {vnnlib[:40]:40s} {verdict:8s} "
                  f"{wall:7.1f}s /{budget:7.0f}s = {frac * 100:5.0f}%")
        print("\nThese seconds DO NOT EXIST: the harness kills at the budget, so the row")
        print("scores zero regardless of the verdict, and a solver that believes it has")
        print("more time is itself the defect. Check whether the wall is harness or solver.")


if __name__ == "__main__":
    sys.exit(main() or 0)
