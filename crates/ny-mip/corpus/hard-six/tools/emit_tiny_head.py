# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# TINY "classification-head" rung generator for the ay MILP corpus — a
# definitely-solvable first rung BELOW the w2 window (which ay P0 returns
# unknown@120s on, and which is unknown even with 0 binaries because ay's
# exact-rational simplex chokes on the 100x2048 dense Gemm_56 rows).
#
# This emits the FINAL classification head only:
#
#     h in [l,u]  ->  Relu_57  ->  Gemm_58 margin
#
# i.e. the last (Gemm_56 output box) -> Relu_57 -> (W58[p]-W58[q]) margin.
# The z->h affine coupling (Gemm_56 = W56.z + b56, the 100x2048 dense block
# that is ay's wall) is DROPPED: h ranges freely over its refined+inflated
# box. Dropping constraints only ENLARGES the feasible set, so this is a
# sound OVER-approximation (coarser than any committed window):
#   * an UNSAT here would soundly verify the row on this coarsened model;
#   * a SAT is a coarsening artifact (the decoupled box admits margins the
#     real net cannot reach) — NOT a network counterexample; NY re-validates
#     every model, so a spurious SAT cannot leak into a verdict.
# On these boxes the head is SAT (min margin ~= -5.4, deeper than CROWN's
# -0.45 by the documented correlation loss). Purpose: a real-structure
# QF_LRA / big-M MILP with the genuine Relu_57 nonconvexity and real W58
# objective coefficients that ay solves in tens of milliseconds — the
# engine-loop bootstrap rung, not a verification target.
#
# --keep N   keep the N most-unstable Relu_57 neurons as binaries and
#            phase-PIN the rest (a BaB sub-domain leaf): each pinned neuron
#            is clamped past the DELTA inflation margin to its tighter side
#            (active if |l|<u else inactive), which drops its binary. Pinning
#            restricts h to one phase halfspace per neuron — a valid deeper
#            sub-domain of the coarsened head. Omit --keep to keep ALL
#            unstable neurons (the full un-pinned coarsened head).
#
# Reuses emit_hard_six.py wholesale (Model, add_relu, add_margin_column,
# emit_smt2/emit_milp, load_domain, clamp, infl, DELTA).
#
# Usage:
#   python emit_tiny_head.py --domain domains/prop8945_dom1_d8.npz \
#       --maps maps/ --row 99-67 --keep 6 --out instances/
import argparse
import os

import numpy as np

import emit_hard_six as E


def pin_domain(d, keep):
    """Return an augmented prem list that keeps the `keep` most-unstable
    Relu_57 neurons and phase-pins the rest to their tighter side, past the
    DELTA inflation margin so the pin actually drops the binary."""
    l, u = E.clamp(d)  # premise-clamped Gemm_56 output bounds
    uns = np.where((l < 0.0) & (u > 0.0))[0]
    # the existing split node (prem clamp to exactly 0) is re-destabilised by
    # DELTA inflation, so treat every j whose inflated box straddles 0 as a
    # pin candidate.
    infl_uns = set(int(j) for j in np.where((l - E.DELTA < 0.0) & (u + E.DELTA > 0.0))[0])
    cand = sorted(infl_uns)
    # rank by distance-to-stable (min(|l|, u)); keep the largest, pin the rest.
    cand.sort(key=lambda j: min(abs(l[j]), u[j]), reverse=True)
    kept = set(cand[:keep])
    pin = 10.0 * E.DELTA  # 1e-3: post-inflation margin 9e-4 > 0
    prem = list(d["prem"])
    for j in cand[keep:]:
        if abs(l[j]) < u[j]:
            prem.append((j, "+", pin))   # active: l[j] -> +1e-3
        else:
            prem.append((j, "-", -pin))  # inactive: u[j] -> -1e-3
    d = dict(d)
    d["prem"] = prem
    return d, sorted(kept)


def build_head(a, c, l, u):
    """Coarsest window: h free in the inflated Gemm_56 output box ->
    Relu_57 -> W58 margin. No z->h coupling."""
    li, ui = E.infl((l, u))
    m = E.Model()
    h_base = m.add_cols(li, ui)
    pos2, uns2, rh_base, _ = m.add_relu(h_base, li, ui)
    for i in pos2:
        m.obj[h_base + int(i)] = a[int(i)]
    for k, i in enumerate(uns2):
        m.obj[rh_base + k] = a[int(i)]
    obj_col = E.add_margin_column(m, c)
    return m, obj_col, list(map(int, uns2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--row", required=True, help="spec row, e.g. 99-67")
    ap.add_argument("--keep", type=int, default=None,
                    help="keep N most-unstable binaries, pin the rest")
    ap.add_argument("--int-scale", action="store_true",
                    help="power-of-2 integer-scaled lowering (see emit_hard_six)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    E.MAPS = args.maps
    prop, d, _zbox, _boxes = E.load_domain(args.domain)
    tag = "full" if args.keep is None else f"pin{args.keep}"
    if args.keep is not None:
        d, kept = pin_domain(d, args.keep)
    l, u = E.clamp(d)
    W58, b58 = E._ld("W58"), E._ld("b58")
    p, q = (int(x) for x in args.row.split("-"))
    a = W58[p] - W58[q]
    c = float(b58[p] - b58[q])
    m, obj_col, uns = build_head(a, c, l, u)
    A = m.rows_csr()
    tagn = f"full{len(uns)}" if args.keep is None else f"pin{len(uns)}"
    suffix = "_int" if args.int_scale else ""
    name = (f"cifar100med_prop{prop}_dom{d['dom']}_d{d['depth']}"
            f"_r{p}-{q}_whead_{tagn}{suffix}")
    print(f"{name}: cols={len(m.col_lo)} rows={m.nrows} nnz={A.nnz} "
          f"binaries={sum(m.integ)} uns={uns}")
    if args.int_scale:
        nchk, mbits = E.verify_int_scale(m, A)
        print(f"  int-scale VERIFIED exact on {nchk} rows; "
              f"max integer coefficient = {mbits} bits")
    for form in ("dec", "min"):
        E.emit_smt2(m, A, obj_col, form,
                    os.path.join(args.out, f"{name}_{form}.smt2"),
                    int_scale=args.int_scale)
    if not args.int_scale:
        E.emit_milp(m, A, os.path.join(args.out, f"{name}.milp"))


if __name__ == "__main__":
    main()
