#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""
vggnet16 DeepZ-zonotope CERTIFIED-ROUNDING probe (#vggnet16-forward-route).

Reference-only (numpy, scripts/, never in a verdict path).  Answers ONE
question that `vggnet16_forward_zonotope_probe.py` deliberately left open:

    if the sparse-generator zonotope forward pass carries a RIGOROUS
    outward rounding-error channel, how wide is the certified margin
    interval, as a function of the working-precision unit roundoff `u`?

Method
------
Runs the SAME DeepZ pass as `--method zonotope` in the sibling probe, but
carries two extra per-element nonnegative channels:

    ec[i] >= |c_exact[i] - c_computed[i]|                (center error)
    eg[i] >= sum_j |g_exact[i][j] - g_computed[i][j]|    (generator error)

so the certified concretization of element i is

    [ c[i] - sum_j|g[i][j]| - ec[i] - eg[i],
      c[i] + sum_j|g[i][j]| + ec[i] + eg[i] ].

Per-op accounting (all of it OUTWARD, all of it standard Higham):

  Conv / Gemm (exact affine map W, dot length K):
      ec' = |W| ec + gamma_K(u) * (|W| |c|)
      eg' = |W| eg + gamma_K(u) * (|W| sum_j|g|)
    gamma_K(u) = K u / (1 - K u).  This is the ONLY error-amplifying op:
    the error channel is pushed through |W|, i.e. through *exactly* the IBP
    transfer operator.

  ReLU (DeepZ):  lambda in [0,1] computed from the CERTIFIED [lo,up];
    mu = -lambda*lo/2 rounded UP (any larger mu stays sound).
      ec' = lambda*ec + 2u|lambda c + mu| ;  eg' = lambda*eg + u*lambda*sum|g|
    lambda <= 1, so ReLU CONTRACTS the error channel.

  MaxPool: gather (no amplification) + an outward-rounded slack.

The pass itself is executed in plain f64 (the *values* only need to be
representative; `u` is a free parameter of the error channel), so the reported
certified width for a given `u` is what a working-precision-`u` implementation
would certify, up to the value-dependence of `S`.  That dependence is second
order: `S` is set by the weights and activations, not by the precision.

`--ug` gives the GENERATOR channel its own unit roundoff, so the mixed design
"center in double-double, generator columns in plain f64" can be measured
directly.  `--tau` is the new-generator spend threshold: a relaxation term
smaller than `tau` is folded into the interval channel instead of buying a
zonotope column (without it, the rounding channel itself manufactures spurious
crossing ReLUs / tied max-pool windows and the generator count explodes).

Usage
-----
    python3 scripts/vggnet16_zonotope_rounding_probe.py \
        --u 1.1102230246251565e-16 \
        benchmarks/vnncomp2025/benchmarks/vggnet16_2022/vnnlib/spec1_*.vnnlib

    # u = 2^-53   f64        = 1.1102230246251565e-16
    # u = 2^-105  double-double (conservative; Dekker DD is ~2^-106)
    #                        = 2.465190328815662e-32
    # u = 2^-113  float128 / quad
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vggnet16_forward_zonotope_probe import (  # noqa: E402
    DEFAULT_ONNX,
    gather,
    im2col_idx,
    load_spec,
    load_weights,
    maxpool_select,
)


def gamma(k: int, u: float) -> float:
    """Higham gamma_k = k u / (1 - k u); +inf if the denominator is not positive."""
    d = 1.0 - k * u
    if d <= 0.0:
        return float("inf")
    return k * u / d


def run(model, init, xl, xu, ge_y, pert, u, ug, tau, verbose, seed_u=None):
    k = len(pert)
    c = (xl + xu) / 2.0
    gens = np.zeros((len(xl), k))
    gens[pert, np.arange(k)] = (xu[pert] - xl[pert]) / 2.0
    # Seed: the box endpoints are given exactly (f64-representable vnnlib
    # decimals are NOT exact in general, so seed one ulp of slack per entry).
    #
    # `seed_u` is the relative representation error of the INPUT BOX as the
    # implementation actually receives it, which is a DIFFERENT quantity from
    # the working precision `u`.  It defaults to `u` (the box is re-parsed at
    # working precision), but an implementation that receives the box already
    # rounded to some narrower type must pass that type's unit roundoff here:
    # the *fixed* pixels then carry a real uncertainty that can only be
    # represented in the interval channel `ec`, which is pushed through |W| with
    # no cancellation.  Seeding at f32 (`seed_u = 2**-24`) is what NY's engine
    # would do today, since `ny_core::Bound` is f32.
    if seed_u is None:
        seed_u = u
    ec = np.abs(c) * seed_u
    eg = np.abs(gens).sum(axis=1) * ug
    shape = (3, 224, 224)
    t0 = time.time()
    for nd in model.graph.node:
        op = nd.op_type
        if op == "Conv":
            wt, bias = init[nd.input[1]], init[nd.input[2]]
            oc = wt.shape[0]
            wm = wt.reshape(oc, -1)
            awm = np.abs(wm)
            idx, oh, ow = im2col_idx(shape, 3, 3, 1, 1)
            kdot = wm.shape[1] + 1  # +1 for the bias add
            g = gamma(kdot, u)
            gg = gamma(kdot, ug)
            rad = np.abs(gens).sum(axis=1)
            s_c = (awm @ gather(np.abs(c), idx)).reshape(-1)
            s_g = (awm @ gather(rad, idx)).reshape(-1)
            ec = (awm @ gather(ec, idx)).reshape(-1) + g * (s_c + np.abs(bias)[:, None].repeat(oh * ow, 1).reshape(-1))
            eg = (awm @ gather(eg, idx)).reshape(-1) + gg * s_g
            c = (wm @ gather(c, idx) + bias[:, None]).reshape(-1)
            gens = np.stack(
                [(wm @ gather(gens[:, j], idx)).reshape(-1) for j in range(gens.shape[1])],
                axis=1,
            )
            shape = (oc, oh, ow)
        elif op == "Gemm":
            wt, bias = init[nd.input[1]], init[nd.input[2]]
            awt = np.abs(wt)
            g = gamma(wt.shape[1] + 1, u)
            gg = gamma(wt.shape[1] + 1, ug)
            rad = np.abs(gens).sum(axis=1)
            ec = awt @ ec + g * (awt @ np.abs(c) + np.abs(bias))
            eg = awt @ eg + gg * (awt @ rad)
            c = wt @ c + bias
            gens = wt @ gens
            shape = (wt.shape[0],)
        elif op == "Relu":
            rad = np.abs(gens).sum(axis=1)
            lo, up = c - rad - ec - eg, c + rad + ec + eg  # CERTIFIED
            cross = (lo < 0) & (up > 0)
            n_cross = int(cross.sum())
            den = np.where(cross, up - lo, 1.0)
            lam = np.where(cross, up / den, (lo >= 0).astype(float))
            lam = np.clip(lam, 0.0, 1.0)
            mu = np.where(cross, -lam * lo / 2.0, 0.0)
            mu = np.maximum(mu, 0.0) * (1.0 + 4.0 * u)  # outward
            newc = lam * c + mu
            ec = lam * ec + 2.0 * u * np.abs(newc)
            eg = lam * eg + ug * lam * rad
            c = newc
            gens = gens * lam[:, None]
            # Generator-spend policy: a new generator is only worth its column
            # when mu is materially larger than the interval channel already
            # carried.  Sub-threshold mu is folded into `ec` (sound: an
            # interval is a valid over-approximation of a +-mu symbol).
            spend = cross & (mu > tau)
            fold = cross & ~spend
            ec = ec + np.where(fold, mu, 0.0)
            n_spend = int(spend.sum())
            if n_spend:
                ci = np.nonzero(spend)[0]
                add = np.zeros((len(c), n_spend))
                add[ci, np.arange(n_spend)] = mu[ci]
                gens = np.concatenate([gens, add], axis=1)
            if verbose:
                print(
                    f"  {nd.name:24s} cross={n_cross:6d} spend={n_spend:4d} "
                    f"gens={gens.shape[1]:4d} "
                    f"maxrad={np.abs(gens).sum(axis=1).max():.4g} "
                    f"maxEc={ec.max():.4g} maxEg={eg.max():.4g} "
                    f"maxAbsC={np.abs(c).max():.4g}"
                )
        elif op == "MaxPool":
            rad = np.abs(gens).sum(axis=1)
            lo, up = c - rad - ec - eg, c + rad + ec + eg
            sel, slack, out_shape = maxpool_select(shape, lo, up)
            slack = np.maximum(slack, 0.0) * (1.0 + 4.0 * u)
            c = c[sel] + slack / 2.0
            gens = gens[sel]
            ec = ec[sel] + 2.0 * u * np.abs(c)
            eg = eg[sel]
            half = slack / 2.0
            nz = np.nonzero(half > tau)[0]
            ec = ec + np.where(half > tau, 0.0, half)
            if len(nz):
                add = np.zeros((len(c), len(nz)))
                add[nz, np.arange(len(nz))] = half[nz]
                gens = np.concatenate([gens, add], axis=1)
            if verbose:
                print(
                    f"  {nd.name:24s} tied={int((slack > 0).sum()):6d} "
                    f"spend={len(nz):4d} gens={gens.shape[1]:4d} "
                    f"maxEc={ec.max():.4g} maxEg={eg.max():.4g}"
                )
            shape = out_shape
        elif op == "Flatten":
            shape = (c.size,)
        elif op == "Dropout":
            pass
        else:
            raise SystemExit(f"unhandled op {op}")
        sys.stdout.flush()
    print(f"[rounding] u={u:.4g} pass {time.time() - t0:.1f}s gens={gens.shape[1]}")
    for (a_idx, b_idx) in ge_y:
        row = np.zeros(len(c))
        row[b_idx], row[a_idx] = 1.0, -1.0
        mc = float(row @ c)
        mg = float(np.abs(row @ gens).sum())
        me = float(np.abs(row) @ (ec + eg))
        print(
            f"MARGIN Y_{b_idx}-Y_{a_idx}: value=[{mc - mg:.6f},{mc + mg:.6f}] "
            f"half-width(relax)={mg:.4g}  half-width(ROUNDING)={me:.6g}  "
            f"certified=[{mc - mg - me:.6g},{mc + mg + me:.6g}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--u", type=float, default=2.0**-53)
    ap.add_argument("--ug", type=float, default=None,
                    help="separate unit roundoff for the GENERATOR channel "
                         "(default: same as --u)")
    ap.add_argument("--tau", type=float, default=1e-9,
                    help="new-generator spend threshold; sub-threshold relaxation\n"
                         "terms are folded into the interval error channel")
    ap.add_argument("--seed-u", type=float, default=None,
                    help="relative representation error of the INPUT BOX as received\n"
                         "(default: same as --u).  Use 2**-24 to model an f32 input box.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    xl, xu, ge_y = load_spec(a.spec)
    model, init = load_weights(a.onnx)
    pert = np.nonzero(xu - xl)[0]
    print(f"spec={a.spec} k={len(pert)} u={a.u:.6g} ug={a.ug} tau={a.tau:.6g} "
          f"seed_u={a.seed_u}")
    run(model, init, xl, xu, ge_y, pert, a.u, a.ug if a.ug is not None else a.u, a.tau,
        not a.quiet, a.seed_u)


if __name__ == "__main__":
    main()
