# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Hard-six full-depth / window MIP instance generator for the ay MILP corpus.
#
# Builds the exact big-M MIP over a pinned BaB subdomain of the cifar100_2024
# CIFAR100_resnet_medium hard-six instances (window w2/w3/w5 or the FULL net
# from the exact vnnlib input box) and emits it in the two formats the ay P0
# program consumes (docs/AY_MIP_P0.md, docs/SOLVER_POLICY.md):
#
#   *.smt2  - SMT-LIB2 QF_LRA, byte-faithful to ny-mip's ay lowering dialect
#             (crates/ny-mip/src/ay/lower.rs): exact IEEE-754 rational
#             literals (never a rounded display), ReLU binaries as 0/1
#             disjunctions, bare-column (minimize) on an aux margin column.
#             Two forms per instance:
#               *_dec.smt2  decision form: margin <= 0 asserted; UNSAT <=>
#                           the subdomain is VERIFIED on that spec row.
#               *_min.smt2  OMT form: (minimize margin).
#   *.milp  - ny-mip dump.rs `milp v1` bit-pattern format, loadable by
#             `mip-diff` (the standing G0 differential gate).
#
# Soundness posture (same as the suffix/window gates that produced the
# HiGHS baselines): all intermediate boxes are inflated by DELTA=1e-4 to
# absorb the f32-net vs f64-affine semantic gap (measured <= 1.1e-5), and
# box inflation only WEAKENS the model, so an UNSAT/`>=0` verdict on these
# instances is sound for the real f32 network under the same caveat ny's
# own float paths carry. A SAT model is a candidate violation and must be
# revalidated by a concrete forward pass (NyVerdictAdmission posture).
#
# Model construction is a self-contained copy of the measured window_gate.py
# builder (session scratchpad sfx/), pointed at:
#   --maps DIR    the exact affine-map dumps (W56/b56/W58/b58, block maps
#                 C0/C3/C6/C8, L11_*/L12_*/L13_*, A20_*/S20, A21_*, A1_22...,
#                 built from the benchmark ONNX by build_block_maps*.py)
#   --domains DIR the per-domain box extracts (.npz, committed alongside)
#                 or --log FILE, a raw NY_SUFFIX_MIP_DUMP probe log.
#
# Usage:
#   extract:  python emit_hard_six.py extract --log probe_8945.u.log \
#                 --prop 8945 --max-doms 2 --out domains/
#   emit:     python emit_hard_six.py emit --domain domains/prop8945_dom1_d8.npz \
#                 --maps SFX --win full --vnnlib <path.vnnlib> --out instances/
#             (--win 2|3|5|full; full requires --vnnlib for the exact input box)
import argparse
import hashlib
import os
import re
import struct
import sys
import time

import numpy as np

# f32-net vs f64-affine semantic gap, measured <=1.1e-5 at Gemm_56 out.
DELTA = 1e-4

MAPS = None  # set from --maps


def _ld(name):
    return np.load(f"{MAPS}/{name}.npy")


def _lz(name):
    import scipy.sparse as sp

    return sp.load_npz(f"{MAPS}/{name}.npz")


# Residual-block chain, upstream -> downstream (CIFAR100_resnet_medium tail).
BLOCKS = [
    dict(inn="Add_10", pre="Conv_11", relu="Relu_13",
         A1=lambda: _lz("L11_1"), b1=lambda: _ld("L11_1_b"),
         A2=lambda: _lz("L11_2"), b2=lambda: _ld("L11_2_b"), sc=None),
    dict(inn="Add_16", pre="Conv_17", relu="Relu_19",
         A1=lambda: _lz("L12_1"), b1=lambda: _ld("L12_1_b"),
         A2=lambda: _lz("L12_2"), b2=lambda: _ld("L12_2_b"), sc=None),
    dict(inn="Add_22", pre="Conv_23", relu="Relu_25",
         A1=lambda: _lz("L13_1"), b1=lambda: _ld("L13_1_b"),
         A2=lambda: _lz("L13_2"), b2=lambda: _ld("L13_2_b"), sc=None),
    dict(inn="Add_28", pre="Conv_29", relu="Relu_31",
         A1=lambda: _ld("A20_1"), b1=lambda: _ld("A20_1_b"),
         A2=lambda: _ld("A20_2"), b2=lambda: _ld("A20_2_b"),
         sc=lambda: (_ld("S20"), _ld("S20_b"), "Conv_34")),
    dict(inn="Add_36", pre="Conv_37", relu="Relu_39",
         A1=lambda: _ld("A21_1"), b1=lambda: _ld("A21_1_b"),
         A2=lambda: _ld("A21_2"), b2=lambda: _ld("A21_2_b"), sc=None),
    dict(inn="Add_42", pre="Conv_43", relu="Relu_45",
         A1=lambda: _ld("A1_22"), b1=lambda: _ld("b1_22"),
         A2=lambda: _ld("A2_22"), b2=lambda: _ld("b2_22"), sc=None),
    dict(inn="Add_48", pre="Conv_49", relu="Relu_51",
         A1=lambda: _ld("A1_23"), b1=lambda: _ld("b1_23"),
         A2=lambda: _ld("A2_23"), b2=lambda: _ld("b2_23"), sc=None),
]


# ---------------------------------------------------------------- probe log


def parse(log):
    """Parse an NY_SUFFIX_MIP_DUMP probe log.
    Returns (inboxes, xboxes, doms); xboxes keyed (node, id)."""
    inboxes, xboxes, doms = {}, {}, []
    cur_spec, pend_xref = None, {}
    for line in open(log):
        if "[sfx] spec rows=" in line:
            m = re.search(r"rows=([\d:,\-]+)", line)
            cur_spec = [tuple(int(x) for x in p.split(":")) for p in m.group(1).split(",")]
        elif "[sfx] inbox" in line:
            m = re.search(r"id=(\d+) node=(\S+) n=(\d+) vals=\[([^\]]*)\]", line)
            arr = np.array([tuple(float(x) for x in p.split(":")) for p in m.group(4).split(",")])
            inboxes[int(m.group(1))] = (arr[:, 0], arr[:, 1])
        elif "[sfx] xbox" in line:
            m = re.search(r"id=(\d+) node=(\S+) n=(\d+) vals=\[([^\]]*)\]", line)
            arr = np.array([tuple(float(x) for x in p.split(":")) for p in m.group(4).split(",")])
            xboxes[(m.group(2), int(m.group(1)))] = (arr[:, 0], arr[:, 1])
        elif "[sfx] xref" in line:
            m = re.search(r"dom=(\d+) node=(\S+) id=(\d+)", line)
            pend_xref[m.group(2)] = int(m.group(3))
        elif "[sfx] dom=" in line:
            d = {
                "dom": int(re.search(r"dom=(\d+)", line).group(1)),
                "depth": int(re.search(r"depth=(\d+)", line).group(1)),
                "inbox": int(re.search(r"inbox=(-?\d+)", line).group(1)),
                "spec": cur_spec,
                "xref": dict(pend_xref),
            }
            pend_xref = {}
            prem = re.search(r"prem=\[([^\]]*)\]", line).group(1)
            d["prem"] = []
            for t in prem.split(","):
                if t:
                    m = re.match(r"(\d+)([+\-])@(\S+)", t)
                    d["prem"].append((int(m.group(1)), m.group(2), float(m.group(3))))
            lbs = re.search(r"lbs=\[([^\]]*)\]", line).group(1)
            d["lbs"] = [float(x) for x in lbs.split(",")] if lbs else []
            arr = np.array([tuple(float(x) for x in p.split(":"))
                            for p in re.search(r"seed=\[([^\]]*)\]", line).group(1).split(",")])
            d["l"], d["u"] = arr[:, 0], arr[:, 1]
            doms.append(d)
    return inboxes, xboxes, doms


def pick_domains(inboxes, xboxes, doms, max_doms):
    """Deepest frontier, most-negative CROWN bound first, deduped -- the
    pinned lineage (identical ordering to the measured window gates)."""
    doms.sort(key=lambda d: (-d["depth"], min(d["lbs"]) if d["lbs"] else 0))
    seen, picked = set(), []
    for d in doms:
        key = (d["depth"], tuple(d["prem"]), round(min(d["lbs"]), 5) if d["lbs"] else 0)
        if key in seen or d["inbox"] < 0 or not d["xref"]:
            continue
        seen.add(key)
        picked.append(d)
        if len(picked) >= max_doms:
            break
    return picked


def clamp(d):
    """Premise-clamped Gemm_56 output bounds (the BaB split literals)."""
    l, u = d["l"].copy(), d["u"].copy()
    for j, sgn, s in d["prem"]:
        if sgn == "+":
            l[j] = max(l[j], s)
        else:
            u[j] = min(u[j], s)
    return l, u


def save_domain(d, inboxes, xboxes, prop, outdir):
    """Compact committed extract: everything emit needs except the maps."""
    zl, zu = inboxes[d["inbox"]]
    kw = {
        "prop": prop, "dom": d["dom"], "depth": d["depth"],
        "spec": np.array(d["spec"], dtype=np.int64),
        "lbs": np.array(d["lbs"]),
        "prem_j": np.array([p[0] for p in d["prem"]], dtype=np.int64),
        "prem_sgn": np.array([1 if p[1] == "+" else -1 for p in d["prem"]], dtype=np.int64),
        "prem_s": np.array([p[2] for p in d["prem"]]),
        "seed_l": d["l"], "seed_u": d["u"], "z_l": zl, "z_u": zu,
        "xnodes": np.array(sorted(d["xref"]), dtype=object),
    }
    for node in d["xref"]:
        lo, hi = xboxes[(node, d["xref"][node])]
        kw[f"x_{node}_l"], kw[f"x_{node}_u"] = lo, hi
    path = os.path.join(outdir, f"prop{prop}_dom{d['dom']}_d{d['depth']}.npz")
    np.savez_compressed(path, **kw)
    print(f"wrote {path} ({os.path.getsize(path)/1e3:.0f} KB, "
          f"nodes={list(d['xref'])}, depth={d['depth']}, "
          f"min_lb={min(d['lbs']):.4f})")
    return path


def load_domain(path):
    z = np.load(path, allow_pickle=True)
    d = {
        "dom": int(z["dom"]), "depth": int(z["depth"]),
        "spec": [tuple(r) for r in z["spec"]],
        "lbs": list(z["lbs"]),
        "prem": [(int(j), "+" if s > 0 else "-", float(v))
                 for j, s, v in zip(z["prem_j"], z["prem_sgn"], z["prem_s"])],
        "l": z["seed_l"], "u": z["seed_u"],
    }
    zbox = (z["z_l"], z["z_u"])
    boxes = {str(n): (z[f"x_{n}_l"], z[f"x_{n}_u"]) for n in z["xnodes"]}
    return str(z["prop"]), d, zbox, boxes


# ------------------------------------------------------------------- model


class Model:
    """Sparse MIP builder; bulk COO-triplet assembly (window_gate.py copy)."""

    def __init__(self):
        self.col_lo, self.col_hi, self.obj, self.integ = [], [], [], []
        self.trip = []
        self.r_lo, self.r_hi = [], []
        self.nrows = 0

    def add_cols(self, lo, hi, obj=None, integer=False):
        n = len(lo)
        base = len(self.col_lo)
        self.col_lo.extend(lo)
        self.col_hi.extend(hi)
        self.obj.extend(obj if obj is not None else np.zeros(n))
        self.integ.extend([integer] * n)
        return base

    def add_row(self, idx, val, lo, hi):
        r = np.full(len(idx), self.nrows, dtype=np.int64)
        self.trip.append((r, np.asarray(idx, np.int64), np.asarray(val, np.float64)))
        self.r_lo.append(lo)
        self.r_hi.append(hi)
        self.nrows += 1

    @staticmethod
    def _coo(M):
        import scipy.sparse as sp

        return M.tocoo() if sp.issparse(M) else sp.coo_matrix(np.asarray(M))

    def add_affine_rows(self, out_base, b, in_specs):
        """Rows: out_i - (sum of specs) = b_i (see window_gate.py)."""
        n = len(b)
        row0 = self.nrows
        ar = np.arange(n, dtype=np.int64)
        self.trip.append((row0 + ar, out_base + ar, np.ones(n)))
        for spec in in_specs:
            kind = spec[0]
            if kind == "id":
                self.trip.append((row0 + ar, spec[1] + ar, -np.ones(n)))
                continue
            if kind == "dense":
                _, base, M = spec
                co = self._coo(M)
                cols = base + co.col.astype(np.int64)
            else:
                _, base, M, colsel = spec
                colsel = np.asarray(colsel, dtype=np.int64)
                import scipy.sparse as sp

                Msub = (M.tocsc()[:, colsel] if sp.issparse(M)
                        else np.asarray(M)[:, colsel])
                co = self._coo(Msub)
                cols = (base + colsel[co.col] if kind == "spread"
                        else base + co.col.astype(np.int64))
            self.trip.append((row0 + co.row.astype(np.int64), cols, -co.data))
        self.r_lo.extend(b)
        self.r_hi.extend(b)
        self.nrows += n

    def add_relu(self, p_base, pl, pu):
        """Big-M ReLU vars for pre-act columns p (bounds pl, pu)."""
        pos = np.where(pl >= 0.0)[0]
        uns = np.where((pl < 0.0) & (pu > 0.0))[0]
        n_b = uns.size
        r_base = self.add_cols(np.zeros(n_b), np.maximum(pu[uns], 0.0))
        d_base = self.add_cols(np.zeros(n_b), np.ones(n_b), integer=True)
        for k, i in enumerate(uns):
            i = int(i)
            self.add_row([r_base + k, p_base + i], [1.0, -1.0], 0.0, np.inf)
            self.add_row([r_base + k, d_base + k], [1.0, -pu[i]], -np.inf, 0.0)
            self.add_row([r_base + k, p_base + i, d_base + k], [1.0, -1.0, -pl[i]],
                         -np.inf, -pl[i])
        return pos, uns, r_base, d_base

    def rows_csr(self):
        """Aggregate triplets to CSR, summing duplicate (row, col) pairs."""
        import scipy.sparse as sp

        A = sp.coo_matrix(
            (np.concatenate([t[2] for t in self.trip]),
             (np.concatenate([t[0] for t in self.trip]),
              np.concatenate([t[1] for t in self.trip]))),
            shape=(self.nrows, len(self.col_lo))).tocsr()
        A.sum_duplicates()
        return A


def add_relu_vec(m, p_base, pl, pu):
    """Full-width relu output vector t = relu(p) (window_gate.py copy)."""
    pos, uns, r_base, _ = m.add_relu(p_base, pl, pu)
    t_base = m.add_cols(np.maximum(pl, 0.0), np.maximum(pu, 0.0))
    if pos.size:
        po = pos.astype(np.int64)
        m.trip.append((m.nrows + np.arange(pos.size), t_base + po, np.ones(pos.size)))
        m.trip.append((m.nrows + np.arange(pos.size), p_base + po, -np.ones(pos.size)))
        m.r_lo.extend([0.0] * pos.size)
        m.r_hi.extend([0.0] * pos.size)
        m.nrows += pos.size
    if uns.size:
        un = uns.astype(np.int64)
        m.trip.append((m.nrows + np.arange(uns.size), t_base + un, np.ones(uns.size)))
        m.trip.append((m.nrows + np.arange(uns.size),
                       r_base + np.arange(uns.size), -np.ones(uns.size)))
        m.r_lo.extend([0.0] * uns.size)
        m.r_hi.extend([0.0] * uns.size)
        m.nrows += uns.size
    return t_base, uns


def infl(box):
    lo, hi = box
    return lo - DELTA, hi + DELTA


def build_window(win, a, l, u, zbox, boxes, xin=None):
    """win in {2,3,5}: window spans the last (win-1) residual blocks + the
    Gemm_56/Relu_57/Gemm_58 tail. win='full': whole net from the EXACT
    vnnlib input box xin (stem + all blocks) -- ground truth modulo the
    DELTA-inflated intermediate big-M boxes. (window_gate.py copy)"""
    boxes = {k: infl(v) for k, v in boxes.items()}
    zbox = infl(zbox)
    l, u = l - DELTA, u + DELTA
    m = Model()
    stats = []
    full = win == "full"
    chain = BLOCKS if full else (BLOCKS[len(BLOCKS) - (win - 1):] if win > 1 else [])
    x_base = None
    if full:
        xl, xu = xin  # exact, NOT inflated
        x0 = m.add_cols(xl, xu)
        p0l, p0u = boxes["Conv_0"]
        p0 = m.add_cols(p0l, p0u)
        m.add_affine_rows(p0, _ld("C0_b"), [("dense", x0, _lz("C0"))])
        t121, uns2_ = add_relu_vec(m, p0, p0l, p0u)
        stats.append(("Relu_2.uns", int(uns2_.size)))
        p3l, p3u = boxes["Conv_3"]
        p3 = m.add_cols(p3l, p3u)
        m.add_affine_rows(p3, _ld("C3_b"), [("dense", t121, _lz("C3"))])
        r5, uns5_ = add_relu_vec(m, p3, p3l, p3u)
        stats.append(("Relu_5.uns", int(uns5_.size)))
        a10l, a10u = boxes["Add_10"]
        x_base = m.add_cols(a10l, a10u)
        m.add_affine_rows(x_base, _ld("C6_b") + _ld("C8_b"),
                          [("dense", r5, _lz("C6")), ("dense", t121, _lz("C8"))])
    for bi, blk in enumerate(chain):
        if x_base is None:
            xl, xu = boxes[blk["inn"]]
            x_base = m.add_cols(xl, xu)
        pl, pu = boxes[blk["pre"]]
        p_base = m.add_cols(pl, pu)
        m.add_affine_rows(p_base, blk["b1"](), [("dense", x_base, blk["A1"]())])
        pos, uns, r_base, _ = m.add_relu(p_base, pl, pu)
        stats.append((blk["relu"] + ".uns", int(uns.size)))
        last = bi == len(chain) - 1
        ol, ou = zbox if last else boxes[chain[bi + 1]["inn"]]
        o_base = m.add_cols(ol, ou)
        A2, b2 = blk["A2"](), blk["b2"]()
        specs = [("spread", p_base, A2, pos), ("gather", r_base, A2, uns)]
        if blk["sc"] is None:
            specs.append(("id", x_base))
        else:
            Ssc, csc, sc_node = blk["sc"]()
            scl, scu = boxes[sc_node]
            sc_base = m.add_cols(scl, scu)
            m.add_affine_rows(sc_base, csc, [("dense", x_base, Ssc)])
            specs.append(("id", sc_base))
        m.add_affine_rows(o_base, b2, specs)
        x_base = o_base
    z_base = x_base if x_base is not None else m.add_cols(*zbox)
    h_base = m.add_cols(l, u)
    m.add_affine_rows(h_base, _ld("b56"), [("dense", z_base, _ld("W56"))])
    pos2, uns2, rh_base, _ = m.add_relu(h_base, l, u)
    stats.append(("Relu_57.uns", int(uns2.size)))
    for i in pos2:
        m.obj[h_base + int(i)] = a[int(i)]
    for k, i in enumerate(uns2):
        m.obj[rh_base + k] = a[int(i)]
    return m, stats


def add_margin_column(m, c):
    """Aux column obj = margin (incl. constant c): obj - sum(a_i x_i) = c.
    Moves the per-column objective into an equality row so both output
    formats optimize/constrain a single column (ay's exact minimize lane)."""
    coefs = [(i, w) for i, w in enumerate(m.obj) if w != 0.0]
    obj_col = m.add_cols([-np.inf], [np.inf])
    idx = [obj_col] + [i for i, _ in coefs]
    val = [1.0] + [-w for _, w in coefs]
    m.add_row(idx, val, float(c), float(c))
    for i, _ in coefs:
        m.obj[i] = 0.0
    m.obj[obj_col] = 1.0
    return obj_col


# ---------------------------------------------------------------- emitters


_LIT_CACHE = {}


def real_literal(x):
    """Exact SMT-LIB Real literal for a finite f64 (lower.rs mirror)."""
    x = float(x)
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    hit = _LIT_CACHE.get(bits)
    if hit is not None:
        return hit
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {x}")
    if x == 0.0:
        lit = "0.0"
    else:
        neg = bits >> 63 == 1
        exp_bits = (bits >> 52) & 0x7FF
        frac = bits & ((1 << 52) - 1)
        if exp_bits == 0:
            mant, exp = frac, -1074
        else:
            mant, exp = frac | (1 << 52), exp_bits - 1075
        while mant & 1 == 0 and exp < 0:
            mant >>= 1
            exp += 1
        body = f"{mant << exp}.0" if exp >= 0 else f"(/ {mant}.0 {1 << (-exp)}.0)"
        lit = f"(- {body})" if neg else body
    _LIT_CACHE[bits] = lit
    return lit


def hex64(x):
    return format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "016x")


# ----------------------------------------------------- integer-scaled lowering
#
# Every finite IEEE-754 f64 is a dyadic rational mant * 2**exp, so its
# denominator is a POWER OF 2. Multiplying a whole affine constraint row by
# 2**maxk (maxk = max denominator exponent over the row's coefficients AND its
# rhs bounds) clears every denominator EXACTLY -- pure bit-shifts -- turning
# the row into INTEGER literals with no `(/ ...)`. Scaling a constraint by a
# positive constant is equivalence-preserving, so the transform is sound; it
# hands ay a pure-integer coefficient matrix (fraction-free / Bareiss-friendly)
# instead of dense rows of large-denominator rationals.

_DYADIC_CACHE = {}


def _dyadic(x):
    """Exact (mant, exp) with float(x) == mant * 2**exp; mant signed, odd
    unless zero, matching real_literal's reduction (so denominators are the
    same power of 2 the rational lowering emits)."""
    x = float(x)
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    hit = _DYADIC_CACHE.get(bits)
    if hit is not None:
        return hit
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {x}")
    if x == 0.0:
        res = (0, 0)
    else:
        neg = bits >> 63 == 1
        exp_bits = (bits >> 52) & 0x7FF
        frac = bits & ((1 << 52) - 1)
        if exp_bits == 0:
            mant, exp = frac, -1074
        else:
            mant, exp = frac | (1 << 52), exp_bits - 1075
        while mant & 1 == 0 and exp < 0:
            mant >>= 1
            exp += 1
        res = (-mant if neg else mant, exp)
    _DYADIC_CACHE[bits] = res
    return res


def _den_k(x):
    """Denominator exponent k so that |x| == odd / 2**k (k=0 for integers)."""
    _, exp = _dyadic(x)
    return -exp if exp < 0 else 0


def _scaled_int(x, k):
    """Exact integer x * 2**k (requires k >= _den_k(x))."""
    mant, exp = _dyadic(x)
    e = exp + k
    if e < 0:
        raise ValueError(f"under-scale {x}: exp={exp} k={k}")
    return mant << e


def int_literal(n):
    """Exact integer -> SMT-LIB Real literal (QF_LRA numerals are >= 0)."""
    n = int(n)
    return f"{n}.0" if n >= 0 else f"(- {-n}.0)"


def _row_maxk(data, lo, hi):
    """LCD exponent for one affine row: max denominator exponent over its
    coefficients and its finite rhs bound(s) (all powers of 2)."""
    maxk = 0
    for v in data:
        dk = _den_k(v)
        if dk > maxk:
            maxk = dk
    for b in (lo, hi):
        if np.isfinite(b):
            dk = _den_k(b)
            if dk > maxk:
                maxk = dk
    return maxk


def verify_int_scale(m, A, n_sample=96):
    """Assert the integer-scaling transform is EXACT on a sample of rows:
    scaled_coef / 2**maxk == the original f64's exact rational (via
    fractions.Fraction). Returns (rows_checked, max_int_bitlength)."""
    from fractions import Fraction

    indptr, indices, data = A.indptr, A.indices, A.data
    rows = range(m.nrows)
    if m.nrows > n_sample:
        # deterministic spread + the densest rows (the Gemm_56 wall).
        nnz = np.diff(indptr)
        dense = list(np.argsort(nnz)[-n_sample // 2:])
        stride = list(range(0, m.nrows, max(1, m.nrows // (n_sample // 2))))
        rows = sorted(set(int(r) for r in dense + stride))
    checked, max_bits = 0, 0
    for r in rows:
        s, e = indptr[r], indptr[r + 1]
        if s == e:
            continue
        lo, hi = m.r_lo[r], m.r_hi[r]
        maxk = _row_maxk(data[s:e], lo, hi)
        scale = Fraction(1, 1 << maxk)
        for k in range(s, e):
            si = _scaled_int(data[k], maxk)
            assert Fraction(si) * scale == Fraction(float(data[k])), (r, k)
            b = si.bit_length()
            if b > max_bits:
                max_bits = b
        for b in (lo, hi):
            if np.isfinite(b):
                si = _scaled_int(b, maxk)
                assert Fraction(si) * scale == Fraction(float(b)), (r, "rhs")
                bl = si.bit_length()
                if bl > max_bits:
                    max_bits = bl
        checked += 1
    return checked, max_bits


def _scaled_bound(f, i, x, op, int_scale):
    """Emit a single-variable bound assert (op c_i x). In int_scale mode the
    row 1*c_i {op} x is multiplied by 2**_den_k(x) so x becomes an integer
    literal (no `(/ ...)`); the variable then carries the 2**k coefficient."""
    if not int_scale:
        f.write(f"(assert ({op} c{i} {real_literal(x)}))\n")
        return
    k = _den_k(x)
    rhs = int_literal(_scaled_int(x, k))
    if k == 0:
        f.write(f"(assert ({op} c{i} {rhs}))\n")
    else:
        f.write(f"(assert ({op} (* {1 << k}.0 c{i}) {rhs}))\n")


def emit_smt2(m, A, obj_col, form, path, int_scale=False):
    """form: 'dec' (assert margin<=0; UNSAT <=> verified) or 'min' (OMT).
    int_scale: clear every row's power-of-2 denominators -> integer literals."""
    ncols = len(m.col_lo)
    with open(path, "w") as f:
        f.write("(set-logic QF_LRA)\n")
        for i in range(ncols):
            f.write(f"(declare-const c{i} Real)\n")
            lo, hi = m.col_lo[i], m.col_hi[i]
            if m.integ[i]:
                can0, can1 = lo <= 0.0, hi >= 1.0
                if can0 and can1:
                    f.write(f"(assert (or (= c{i} 0.0) (= c{i} 1.0)))\n")
                elif can0:
                    f.write(f"(assert (= c{i} 0.0))\n")
                elif can1:
                    f.write(f"(assert (= c{i} 1.0))\n")
                else:
                    raise ValueError(f"binary col {i} excludes 0 and 1")
            else:
                if np.isfinite(lo):
                    _scaled_bound(f, i, lo, ">=", int_scale)
                if np.isfinite(hi):
                    _scaled_bound(f, i, hi, "<=", int_scale)
        indptr, indices, data = A.indptr, A.indices, A.data
        for r in range(m.nrows):
            s, e = indptr[r], indptr[r + 1]
            if s == e:
                if m.r_lo[r] > 0.0 or m.r_hi[r] < 0.0:
                    f.write("(assert false)\n")
                continue
            lo, hi = m.r_lo[r], m.r_hi[r]
            if int_scale:
                maxk = _row_maxk(data[s:e], lo, hi)
                terms = [f"(* {int_literal(_scaled_int(data[k], maxk))} c{indices[k]})"
                         for k in range(s, e)]
            else:
                terms = [f"(* {real_literal(data[k])} c{indices[k]})" for k in range(s, e)]
            body = terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"

            def _rhs(x):
                return int_literal(_scaled_int(x, maxk)) if int_scale else real_literal(x)

            if lo == hi and np.isfinite(lo):
                f.write(f"(assert (= {body} {_rhs(lo)}))\n")
            else:
                if np.isfinite(lo):
                    f.write(f"(assert (>= {body} {_rhs(lo)}))\n")
                if np.isfinite(hi):
                    f.write(f"(assert (<= {body} {_rhs(hi)}))\n")
        if form == "dec":
            f.write(f"(assert (<= c{obj_col} 0.0))\n")
        else:
            f.write(f"(minimize c{obj_col})\n")
        f.write("(check-sat)\n")
        f.write("(get-value (" + " ".join(f"c{i}" for i in range(ncols)) + "))\n")


def emit_milp(m, A, path):
    """dump.rs `milp v1` bit-pattern format (echo comments omitted)."""
    ncols = len(m.col_lo)
    pinf, ninf = hex64(np.inf), hex64(-np.inf)
    with open(path, "w") as f:
        f.write("milp v1\n")
        f.write(f"cols {ncols}\n")
        for i in range(ncols):
            lo = hex64(m.col_lo[i]) if np.isfinite(m.col_lo[i]) else ninf
            hi = hex64(m.col_hi[i]) if np.isfinite(m.col_hi[i]) else pinf
            f.write(f"{lo} {hi} {hex64(m.obj[i])} {1 if m.integ[i] else 0}\n")
        f.write(f"rows {m.nrows}\n")
        indptr, indices, data = A.indptr, A.indices, A.data
        for r in range(m.nrows):
            s, e = indptr[r], indptr[r + 1]
            lo = hex64(m.r_lo[r]) if np.isfinite(m.r_lo[r]) else ninf
            hi = hex64(m.r_hi[r]) if np.isfinite(m.r_hi[r]) else pinf
            parts = [f"{lo} {hi} {e - s}"]
            parts.extend(f"{indices[k]} {hex64(data[k])}" for k in range(s, e))
            f.write(" ".join(parts) + "\n")


def parse_vnnlib_box(path):
    txt = open(path).read()
    hi = {int(m.group(1)): float(m.group(2)) for m in
          re.finditer(r"\(assert \(<= X_(\d+) (\S+)\)\)", txt)}
    lo = {int(m.group(1)): float(m.group(2)) for m in
          re.finditer(r"\(assert \(>= X_(\d+) (\S+)\)\)", txt)}
    nx = max(hi) + 1
    return (np.array([lo[i] for i in range(nx)]),
            np.array([hi[i] for i in range(nx)]))


# -------------------------------------------------------------------- main


def cmd_extract(args):
    inboxes, xboxes, doms = parse(args.log)
    picked = pick_domains(inboxes, xboxes, doms, args.max_doms)
    print(f"{len(picked)} pinned domain(s) picked from {args.log}")
    for d in picked:
        save_domain(d, inboxes, xboxes, args.prop, args.out)


def cmd_emit(args):
    global MAPS
    MAPS = args.maps
    prop, d, zbox, boxes = load_domain(args.domain)
    win = "full" if args.win == "full" else int(args.win)
    xin = parse_vnnlib_box(args.vnnlib) if win == "full" else None
    if win == "full":
        need = ([b["inn"] for b in BLOCKS] + [b["pre"] for b in BLOCKS]
                + ["Conv_0", "Conv_3", "Conv_34"])
        missing = [n for n in need if n not in boxes]
        if missing:
            sys.exit(f"domain extract lacks nodes {missing}; re-dump with "
                     f"NY_SUFFIX_MIP_DUMP_NODES")
    l, u = clamp(d)
    W58, b58 = _ld("W58"), _ld("b58")
    for ri, (p, q) in enumerate(d["spec"]):
        if ri >= len(d["lbs"]) or d["lbs"][ri] >= 0:
            continue
        if args.row and f"{p}-{q}" != args.row:
            continue
        a = W58[p] - W58[q]
        c = float(b58[p] - b58[q])
        t0 = time.time()
        m, stats = build_window(win, a, l, u, zbox, boxes, xin=xin)
        obj_col = add_margin_column(m, c)
        A = m.rows_csr()
        isc = getattr(args, "int_scale", False)
        suffix = "_int" if isc else ""
        name = (f"cifar100med_prop{prop}_dom{d['dom']}_d{d['depth']}"
                f"_r{p}-{q}_w{args.win}{suffix}")
        print(f"{name}: cols={len(m.col_lo)} rows={m.nrows} nnz={A.nnz} "
              f"binaries={sum(m.integ)} crown_lb={d['lbs'][ri]:.4f} "
              f"stats={stats} build={time.time()-t0:.0f}s", flush=True)
        if isc:
            nchk, mbits = verify_int_scale(m, A)
            print(f"  int-scale VERIFIED exact on {nchk} rows; "
                  f"max integer coefficient = {mbits} bits", flush=True)
        outs = []
        for form in ("dec", "min"):
            path = os.path.join(args.out, f"{name}_{form}.smt2")
            emit_smt2(m, A, obj_col, form, path, int_scale=isc)
            outs.append(path)
        if not isc:
            # milp (dump.rs f64 bit-pattern) cannot hold the scaled integers
            # exactly (they exceed 2**53); int-scale is an SMT2-only lowering.
            path = os.path.join(args.out, f"{name}.milp")
            emit_milp(m, A, path)
            outs.append(path)
        for path in outs:
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            print(f"  {os.path.basename(path)}: "
                  f"{os.path.getsize(path)/1e6:.1f} MB sha256={h}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="probe log -> per-domain .npz extracts")
    ex.add_argument("--log", required=True)
    ex.add_argument("--prop", required=True)
    ex.add_argument("--max-doms", type=int, default=2)
    ex.add_argument("--out", required=True)
    ex.set_defaults(fn=cmd_extract)
    em = sub.add_parser("emit", help=".npz extract -> .smt2 + .milp instances")
    em.add_argument("--domain", required=True, help="per-domain .npz extract")
    em.add_argument("--maps", required=True, help="affine-map dump dir")
    em.add_argument("--win", required=True, choices=["2", "3", "5", "full"])
    em.add_argument("--vnnlib", help="exact input box (required for --win full)")
    em.add_argument("--row", help="only this spec row, e.g. 99-67")
    em.add_argument("--int-scale", action="store_true",
                    help="power-of-2 integer-scaled lowering: clear each row's "
                         "denominators -> pure-integer coefficients, no `(/ ...)` "
                         "(SMT2-only; milp/f64 cannot hold the scaled integers)")
    em.add_argument("--out", required=True)
    em.set_defaults(fn=cmd_emit)
    args = ap.parse_args()
    if args.cmd == "emit" and args.win == "full" and not args.vnnlib:
        ap.error("--win full requires --vnnlib")
    args.fn(args)


if __name__ == "__main__":
    main()
