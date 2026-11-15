#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Targeted verdict-row moat gate (#verdict-row-moat, 2026-08-29).

GT contradictions can ONLY come from verdict-bearing rows, so rerunning
exactly the banked sat/unsat rows at their official budgets IS the complete
soundness gate for a benchmark — at ~4% of a full sweep's cost. Receipts:
docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md ("HEAD moat receipt via
targeted verdict-row sweep"), first run 34/34 holds, zero contradictions.

Usage:
  scripts/verdict_row_moat.py --bank reports/sweeps/<dated-bank>.csv \
      --instances benchmarks/vnncomp2025/benchmarks/relusplitter/instances.csv \
      --gt-dir ~/vnncomp2025_results --category relusplitter \
      --benchmark-dir benchmarks/vnncomp2025/benchmarks/relusplitter

Exit codes: 0 = clean (all verdicts hold, zero GT contradictions);
2 = CONTRADICTION or verdict flip (soundness emergency per CLAUDE.md);
3 = drops only (rows no longer solve: capability regression, not soundness).
Always extract the verdict set from a DATED bank (the live results CSV is
gitignored and can be clobbered by an interrupted sweep). Inner runs read
stdin from /dev/null (a `while read` loop otherwise loses all but row 1).
"""
import argparse, csv, glob, json, os, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--instances', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--category', required=True)
    ap.add_argument('--benchmark-dir', required=True)
    ap.add_argument('--ny', default='./target/release/ny')
    ap.add_argument('--configs-dir', default='configs')
    a = ap.parse_args()

    def rel_under(prefix, p):
        # Preserve NESTED paths (e.g. safenlp's onnx/medical/x.onnx): keep
        # everything after the top-level onnx/ or vnnlib/ component.
        # basename() alone made all 969 safenlp rows VOID (subdir stripped)
        # and can collide across subdirs.
        p = p.lstrip('./')
        marker = prefix + '/'
        i = p.find(marker)
        return p[i + len(marker):] if i != -1 else os.path.basename(p)

    budgets = {}
    with open(a.instances) as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                budgets[(rel_under('onnx', row[0]), rel_under('vnnlib', row[1]))] = int(
                    float(row[2]))

    rows = []
    with open(a.bank) as f:
        for row in csv.reader(f):
            if len(row) >= 6 and row[4].strip() in ('sat', 'unsat'):
                onnx = rel_under('onnx', row[1])
                vnnlib = rel_under('vnnlib', row[2])
                rows.append((onnx, vnnlib, budgets.get((onnx, vnnlib), 60), row[4].strip()))
    if not rows:
        print('VOID: bank contains no verdict rows', file=sys.stderr)
        return 4

    gt = {}
    # Two layouts exist and only one is documented: the per-category split
    # `<tool>/2025_<cat>/results.csv` AND the flat `<tool>/results.csv` that
    # carries every category with the category in column 0. Accept both --
    # pointing this at a path that yields neither used to load ZERO ground truth
    # and then report "zero GT contradictions", which is a SOUNDNESS GATE
    # PASSING BECAUSE IT COMPARED AGAINST NOTHING. `SCORING-*` are re-scorings
    # of the field, not entrants, and are excluded.
    gt_paths = [
        p for p in glob.glob(os.path.join(a.gt_dir, '*', f'2025_{a.category}', '*.csv'))
        + glob.glob(os.path.join(a.gt_dir, '*', 'results.csv'))
        if not os.path.basename(os.path.dirname(p)).startswith('SCORING')
        and not os.path.basename(os.path.dirname(os.path.dirname(p))).startswith('SCORING')
    ]
    for path in gt_paths:
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) < 6 or row[0] != a.category:
                    continue
                k = (rel_under('onnx', row[1]), rel_under('vnnlib', row[2]))
                v = row[4].strip().lower()
                v = 'unsat' if v.startswith('unsat') else ('sat' if v.startswith('sat') or 'violated' in v else None)
                if v:
                    gt.setdefault(k, set()).add(v)
    consensus = {k: next(iter(v)) for k, v in gt.items() if len(v) == 1}
    if not consensus:
        print(
            f"verdict_row_moat: NO ground truth loaded for {a.category!r} from "
            f"--gt-dir {a.gt_dir!r} ({len(gt_paths)} candidate files).\n"
            "  This gate's whole output is 'do ny's verdicts contradict the field'.\n"
            "  With an empty consensus it would report ZERO CONTRADICTIONS having\n"
            "  compared against nothing -- a soundness gate passing vacuously.\n"
            "  Refusing (exit 5). Point --gt-dir at the published results tree.",
            file=sys.stderr,
        )
        sys.exit(5)
    print(f"verdict_row_moat: ground truth {len(consensus)} consensus rows "
          f"from {len(gt_paths)} file(s)")

    contra, flips, drops, holds = [], [], [], 0
    for onnx, vnnlib, budget, prev in rows:
        out = '/tmp/verdict_row_moat_out.txt'
        subprocess.run(
            [a.ny, 'vnncomp', 'v1', '--configs-dir', a.configs_dir, a.category,
             os.path.join(a.benchmark_dir, 'onnx', onnx),
             os.path.join(a.benchmark_dir, 'vnnlib', vnnlib), out, str(budget)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**os.environ, 'OMP_NUM_THREADS': '1'})
        now = open(out).readline().strip() if os.path.exists(out) else ''
        g = consensus.get((onnx, vnnlib))
        if now in ('sat', 'unsat'):
            if g is not None and g != now:
                contra.append((onnx, vnnlib, now, g))
            if now == prev:
                holds += 1
            else:
                flips.append((onnx, vnnlib, prev, now))
        else:
            drops.append((onnx, vnnlib, prev, now))
        print(f'{onnx}|{vnnlib}|{prev}|{now}', flush=True)

    print(f'SUMMARY rows={len(rows)} holds={holds} contradictions={len(contra)} '
          f'flips={len(flips)} drops={len(drops)}')
    for c in contra: print('  CONTRADICTION:', c)
    for fl in flips: print('  FLIP:', fl)
    for d in drops: print('  drop:', d)
    if contra or flips:
        return 2
    if drops:
        return 3
    return 0

if __name__ == '__main__':
    sys.exit(main())
