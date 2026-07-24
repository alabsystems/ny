#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""
vggnet16 forward-mode feasibility probe (#vggnet16-forward-route).

Reference implementation (numpy, OUT OF THE VERDICT PATH) of the three
candidate root-bound methods for the `vggnet16_2022` VNN-COMP category, used to
decide which forward-mode architecture ny should implement.  Nothing here feeds
a verdict; it exists so the decisive root-bound numbers are reproducible.

Background
----------
Every `vggnet16_2022` spec fixes almost every input pixel: only `k` pixels carry
a non-degenerate interval (measured k: 1, 5, 10, 20, 100 for specs 0-14; the
full 150528 for specs 15-17).  The TRUE margin varies by ~1e-6 across the whole
box, yet ny's root bound is `-1.8e18` because the large-conv gate routes the
bootstrap to plain IBP and VGG16 IBP intermediates explode to ~1e13.

Methods implemented (`--method`)
--------------------------------
`deeppoly`
    Sparse-column forward-linear (DeepPoly-style lower/upper affine pair) over
    a k-column input basis — i.e. exactly what
    `network/core/graph/forward_linear.rs` computes, but with the fixed input
    coordinates folded into the bias instead of carried as 150528 dense
    columns.  Verdict: the k-column basis makes forward mode AFFORDABLE, but
    the bounds are still vacuous, because every crossing-ReLU relaxation
    intercept lands in the scalar bias INTERVAL, which then widens by the
    layer's L1 norm (~8x per layer) exactly like IBP.

`zonotope`
    Same k-column seed, but each crossing ReLU / non-dominated maxpool window
    spends ONE NEW GENERATOR instead of a bias interval, so the relaxation
    error stays symbolic and cancels downstream.  Verdict: near-exact — see the
    module docstring of `--method zonotope` output.

`exact`
    Concrete evaluation at both box endpoints, per layer, to separate honest
    sensitivity growth from relaxation looseness.

Usage
-----
    python3 scripts/vggnet16_forward_zonotope_probe.py \
        --method zonotope \
        benchmarks/vnncomp2025/benchmarks/vggnet16_2022/vnnlib/spec1_Scottish_deerhound.vnnlib

NOTE ON SOUNDNESS: this probe computes in plain f64 with no certified rounding
term.  A rigorous implementation cannot certify f64 by interval error
propagation at VGG16 depth (`err' = |W| err` grows ~42x per conv layer, so a
1e-14 first-layer injection reaches ~1e12 at the logits).  See the commit
message for the precision analysis.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import onnx
from numpy.lib.stride_tricks import as_strided

DEFAULT_ONNX = (
    "benchmarks/vnncomp2025/benchmarks/vggnet16_2022/onnx/vgg16-7.onnx"
)


def load_spec(path: str):
    """Parse a vggnet16 vnnlib file into (x_lower, x_upper, [(a, b), ...]).

    Each `(a, b)` encodes the counter-example condition `Y_a >= Y_b`, i.e. the
    property margin is `Y_b - Y_a > 0`.
    """
    lo: dict[int, float] = {}
    hi: dict[int, float] = {}
    ge_y: list[tuple[int, int]] = []
    for line in open(path):
        line = line.strip()
        m = re.match(r"\(assert \(>= X_(\d+) ([-\d.eE+]+)\)\)", line)
        if m:
            lo[int(m.group(1))] = float(m.group(2))
            continue
        m = re.match(r"\(assert \(<= X_(\d+) ([-\d.eE+]+)\)\)", line)
        if m:
            hi[int(m.group(1))] = float(m.group(2))
            continue
        m = re.match(r"\(assert \(>= Y_(\d+) Y_(\d+)\)\)", line)
        if m:
            ge_y.append((int(m.group(1)), int(m.group(2))))
    n = len(lo)
    return (
        np.array([lo[i] for i in range(n)]),
        np.array([hi[i] for i in range(n)]),
        ge_y,
    )


def im2col_idx(shape, kh, kw, pad, stride):
    """Index map (C*kh*kw, OH*OW) for an im2col gather; -1 marks zero padding."""
    c, h, w = shape
    idx = np.arange(c * h * w, dtype=np.int64).reshape(c, h, w)
    idxp = np.pad(idx, ((0, 0), (pad, pad), (pad, pad)), constant_values=-1)
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    s = idxp.strides
    view = as_strided(
        idxp,
        shape=(c, kh, kw, oh, ow),
        strides=(s[0], s[1], s[2], s[1] * stride, s[2] * stride),
    )
    return np.ascontiguousarray(view.reshape(c * kh * kw, oh * ow)), oh, ow


def gather(vec, idx):
    return np.where(idx >= 0, vec[np.maximum(idx, 0)], 0.0)


def load_weights(path: str):
    model = onnx.load(path)
    init = {
        t.name: onnx.numpy_helper.to_array(t).astype(np.float64)
        for t in model.graph.initializer
    }
    return model, init


# --------------------------------------------------------------------------
# method: exact
# --------------------------------------------------------------------------
def run_exact(model, init, xl, xu, ge_y):
    def forward(x):
        cur = x.reshape(3, 224, 224)
        shape = (3, 224, 224)
        out = {}
        for nd in model.graph.node:
            op = nd.op_type
            if op == "Conv":
                wt, b = init[nd.input[1]], init[nd.input[2]]
                col, oh, ow = im2col_idx(shape, 3, 3, 1, 1)
                cur = wt.reshape(wt.shape[0], -1) @ gather(cur.reshape(-1), col)
                cur = (cur + b[:, None]).reshape(wt.shape[0], oh, ow)
                shape = cur.shape
            elif op == "Relu":
                cur = np.maximum(cur, 0.0)
            elif op == "MaxPool":
                col, oh, ow = im2col_idx(shape, 2, 2, 0, 2)
                cur = gather(cur.reshape(-1), col).reshape(shape[0], 4, oh * ow)
                cur = cur.max(axis=1).reshape(shape[0], oh, ow)
                shape = cur.shape
            elif op == "Flatten":
                cur = cur.reshape(-1)
                shape = (cur.size,)
            elif op == "Gemm":
                wt, b = init[nd.input[1]], init[nd.input[2]]
                cur = wt @ cur.reshape(-1) + b
                shape = (cur.size,)
            elif op == "Dropout":
                pass
            else:
                raise SystemExit(f"unhandled op {op}")
            out[nd.name] = cur.reshape(-1).copy()
        return out

    a, b = forward(xl), forward(xu)
    for name in a:
        d = np.abs(a[name] - b[name])
        print(f"{name:26s} maxTrueVar={d.max():.6g}  maxAbs={np.abs(a[name]).max():.6g}")


# --------------------------------------------------------------------------
# method: deeppoly (sparse-column forward-linear, lower/upper affine pair)
# --------------------------------------------------------------------------
class Affine:
    def __init__(self, al, bl, au, bu, shape):
        self.al, self.bl, self.au, self.bu, self.shape = al, bl, au, bu, shape

    def concretize(self, zc, zr):
        return (
            self.bl + self.al @ zc - np.abs(self.al) @ zr,
            self.bu + self.au @ zc + np.abs(self.au) @ zr,
        )


def run_deeppoly(model, init, xl, xu, ge_y, pert):
    k = len(pert)
    zc = (xl[pert] + xu[pert]) / 2.0
    zr = (xu[pert] - xl[pert]) / 2.0
    al = np.zeros((len(xl), k))
    al[pert, np.arange(k)] = 1.0
    b = xl.copy()
    b[pert] = 0.0
    cur = Affine(al, b.copy(), al.copy(), b.copy(), (3, 224, 224))
    t0 = time.time()
    for nd in model.graph.node:
        op = nd.op_type
        if op == "Conv":
            wt, bias = init[nd.input[1]], init[nd.input[2]]
            oc = wt.shape[0]
            wm = wt.reshape(oc, -1)
            wp, wn = np.maximum(wm, 0.0), np.minimum(wm, 0.0)
            idx, oh, ow = im2col_idx(cur.shape, 3, 3, 1, 1)

            def cp(va, vb, addb):
                out = wp @ gather(va, idx) + wn @ gather(vb, idx)
                if addb:
                    out = out + bias[:, None]
                return out.reshape(-1)

            nbl, nbu = cp(cur.bl, cur.bu, True), cp(cur.bu, cur.bl, True)
            nal = np.stack([cp(cur.al[:, j], cur.au[:, j], False) for j in range(k)], 1)
            nau = np.stack([cp(cur.au[:, j], cur.al[:, j], False) for j in range(k)], 1)
            cur = Affine(nal, nbl, nau, nbu, (oc, oh, ow))
        elif op == "Gemm":
            wt, bias = init[nd.input[1]], init[nd.input[2]]
            wp, wn = np.maximum(wt, 0.0), np.minimum(wt, 0.0)
            cur = Affine(
                wp @ cur.al + wn @ cur.au,
                wp @ cur.bl + wn @ cur.bu + bias,
                wp @ cur.au + wn @ cur.al,
                wp @ cur.bu + wn @ cur.bl + bias,
                (wt.shape[0],),
            )
        elif op == "Relu":
            pl, pu = cur.concretize(zc, zr)
            cross = (pl < 0) & (pu > 0)
            den = np.where(cross, pu - pl, 1.0)
            s = np.where(cross, pu / den, (pl >= 0).astype(float))
            t = np.where(cross, -pu * pl / den, 0.0)
            a = np.where(cross, (pu > -pl).astype(float), (pl >= 0).astype(float))
            print(
                f"  {nd.name}: crossing={int(cross.sum())}/{len(pl)} "
                f"maxwidth={(pu - pl).max():.6g}"
            )
            cur = Affine(
                cur.al * a[:, None], cur.bl * a,
                cur.au * s[:, None], cur.bu * s + t, cur.shape,
            )
        elif op == "MaxPool":
            pl, pu = cur.concretize(zc, zr)
            sel, slack, out_shape = maxpool_select(cur.shape, pl, pu)
            cur = Affine(
                cur.al[sel], cur.bl[sel], cur.au[sel], cur.bu[sel] + slack, out_shape
            )
        elif op == "Flatten":
            cur.shape = (int(np.prod(cur.shape)),)
        elif op == "Dropout":
            pass
        else:
            raise SystemExit(f"unhandled op {op}")
        sys.stdout.flush()
    print(f"[deeppoly] pass {time.time() - t0:.1f}s")
    for (a_idx, b_idx) in ge_y:
        row = np.zeros(cur.bl.size)
        row[b_idx], row[a_idx] = 1.0, -1.0
        rp, rn = np.maximum(row, 0), np.minimum(row, 0)
        mal = rp @ cur.al + rn @ cur.au
        mbl = rp @ cur.bl + rn @ cur.bu
        print(
            f"MARGIN Y_{b_idx} - Y_{a_idx}: deeppoly lower bound = "
            f"{mbl + mal @ zc - np.abs(mal) @ zr:.6f}"
        )


def maxpool_select(shape, pl, pu):
    """2x2/stride-2 max-pool relaxation shared by both forward methods.

    Picks `i* = argmax lower`, so `max >= x_{i*}` (exact lower row) and
    `max <= x_{i*} + sum_{i != i*} relu(u_i - l_{i*})` (upper slack, ZERO when
    the window is strictly dominated — the overwhelmingly common case here).
    """
    c, h, w = shape
    idx, oh, ow = im2col_idx(shape, 2, 2, 0, 2)
    idx = idx.reshape(c, 4, oh * ow)
    lw, uw = pl[idx], pu[idx]
    star = np.argmax(lw, axis=1)
    sel = np.take_along_axis(idx, star[:, None, :], axis=1)[:, 0, :]
    ls = np.take_along_axis(lw, star[:, None, :], axis=1)
    slack = np.maximum(uw - ls, 0.0).sum(axis=1)
    slack -= np.maximum(
        np.take_along_axis(uw, star[:, None, :], axis=1)[:, 0, :] - ls[:, 0, :], 0.0
    )
    return sel.reshape(-1), slack.reshape(-1), (c, oh, ow)


# --------------------------------------------------------------------------
# method: zonotope (DeepZ — one generator per crossing ReLU)
# --------------------------------------------------------------------------
def run_zonotope(model, init, xl, xu, ge_y, pert, max_gen):
    k = len(pert)
    c = (xl + xu) / 2.0
    gens = np.zeros((len(xl), k))
    gens[pert, np.arange(k)] = (xu[pert] - xl[pert]) / 2.0
    shape = (3, 224, 224)
    t0 = time.time()
    for nd in model.graph.node:
        op = nd.op_type
        if op == "Conv":
            wt, bias = init[nd.input[1]], init[nd.input[2]]
            oc = wt.shape[0]
            wm = wt.reshape(oc, -1)
            idx, oh, ow = im2col_idx(shape, 3, 3, 1, 1)
            c = (wm @ gather(c, idx) + bias[:, None]).reshape(-1)
            gens = np.stack(
                [(wm @ gather(gens[:, j], idx)).reshape(-1) for j in range(gens.shape[1])],
                axis=1,
            )
            shape = (oc, oh, ow)
        elif op == "Gemm":
            wt, bias = init[nd.input[1]], init[nd.input[2]]
            c = wt @ c + bias
            gens = wt @ gens
            shape = (wt.shape[0],)
        elif op == "Relu":
            rad = np.abs(gens).sum(axis=1)
            lo, up = c - rad, c + rad
            cross = (lo < 0) & (up > 0)
            n_cross = int(cross.sum())
            den = np.where(cross, up - lo, 1.0)
            lam = np.where(cross, up / den, (lo >= 0).astype(float))
            mu = np.where(cross, -lam * lo / 2.0, 0.0)
            c = lam * c + mu
            gens = gens * lam[:, None]
            if n_cross:
                if n_cross > max_gen:
                    print(f"  {nd.name}: crossing={n_cross} exceeds --max-gen; abort")
                    return
                ci = np.nonzero(cross)[0]
                add = np.zeros((len(c), n_cross))
                add[ci, np.arange(n_cross)] = mu[ci]
                gens = np.concatenate([gens, add], axis=1)
            print(
                f"  {nd.name}: crossing={n_cross}/{len(c)} gens={gens.shape[1]} "
                f"maxwidth={2 * np.abs(gens).sum(axis=1).max():.6g}"
            )
        elif op == "MaxPool":
            rad = np.abs(gens).sum(axis=1)
            sel, slack, out_shape = maxpool_select(shape, c - rad, c + rad)
            c = c[sel] + slack / 2.0
            gens = gens[sel]
            nz = np.nonzero(slack > 0)[0]
            if len(nz):
                if len(nz) > max_gen:
                    print(f"  {nd.name}: tied windows={len(nz)} exceeds --max-gen; abort")
                    return
                add = np.zeros((len(c), len(nz)))
                add[nz, np.arange(len(nz))] = slack[nz] / 2.0
                gens = np.concatenate([gens, add], axis=1)
            print(f"  {nd.name}: tied_windows={len(nz)} gens={gens.shape[1]}")
            shape = out_shape
        elif op == "Flatten":
            shape = (c.size,)
        elif op == "Dropout":
            pass
        else:
            raise SystemExit(f"unhandled op {op}")
        sys.stdout.flush()
    print(f"[zonotope] pass {time.time() - t0:.1f}s gens={gens.shape[1]}")
    for (a_idx, b_idx) in ge_y:
        row = np.zeros(len(c))
        row[b_idx], row[a_idx] = 1.0, -1.0
        mc = float(row @ c)
        mg = np.abs(row @ gens).sum()
        print(
            f"MARGIN Y_{b_idx} - Y_{a_idx}: zonotope bound = "
            f"[{mc - mg:.6f}, {mc + mg:.6f}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec", help="path to a vggnet16 .vnnlib file")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--method", choices=("zonotope", "deeppoly", "exact"),
                    default="zonotope")
    ap.add_argument("--max-gen", type=int, default=600,
                    help="abort when the generator count exceeds this")
    args = ap.parse_args()

    xl, xu, ge_y = load_spec(args.spec)
    pert = np.nonzero(xu - xl > 1e-12)[0]
    print(f"spec={args.spec.split('/')[-1]} perturbed_inputs={len(pert)}/{len(xl)}")
    model, init = load_weights(args.onnx)
    if args.method == "exact":
        run_exact(model, init, xl, xu, ge_y)
    elif len(pert) > args.max_gen:
        print(f"k={len(pert)} exceeds --max-gen={args.max_gen}: dense-basis case, skipped")
    elif args.method == "deeppoly":
        run_deeppoly(model, init, xl, xu, ge_y, pert)
    else:
        run_zonotope(model, init, xl, xu, ge_y, pert, args.max_gen)


if __name__ == "__main__":
    main()
