// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vgg-gpu-crown`: the certified patches-mode CROWN backward for a chunk of
//! rows.
//!
//! A ROW is one linear objective `c . v_t` over a value `v_t` (a signed unit
//! vector for an intermediate neuron bound, or a spec row for the output).
//! Walking the chain backward, each row keeps a coefficient vector `lambda`
//! over the current value, restricted to a WINDOW — `(C, qh, qw)` at a
//! per-row origin — outside which it is exactly zero (the patches
//! representation: a conv target's support grows by the kernel per layer and
//! is clipped to the image, never materialised densely). Every row also
//! carries a bias accumulator `beta` and a nonnegative charge `kappa`, with
//! the invariant
//!
//! ```text
//!   (INV)  for every input x in the box:   c . v_t(x) >= lambda . v(x) + beta_exact - kappa_exact
//! ```
//!
//! where `beta_exact` is the exact sum of the product terms whose rounded sum
//! is `beta`, and `kappa_exact <= kappa` after certification. Each step keeps
//! (INV) with EXACT algebra on the STORED numbers:
//!
//! * **affine** `v' = W v + b`: `lambda . v' = (W^T lambda) . v + lambda . b`,
//!   and with the computed `mu` and certified `err >= |W^T lambda - mu|`,
//!   `(W^T lambda) . v >= mu . v - sum err_j |v_j|max`. The charge is taken
//!   against the box of `v` — the value it multiplies — NOT transported
//!   through the rest of the network (this is what keeps the rounding charge
//!   at the size of one layer's `gamma * |A||B|` instead of the IBP-amplified
//!   `~2^66 u` measured for the zonotope's `|W|`-transported error channel).
//! * **ReLU** `h = relu(z)`, box `[l, u]` of `z`: inactive / active are exact;
//!   crossing with `lambda > 0` uses `h >= alpha z`, `alpha in {0, 1}`
//!   (adaptive, exact); crossing with `lambda < 0` uses the chord
//!   `h <= s z + t` with `s >= u / (u - l)` and `t >= -s l` both rounded UP
//!   (valid for any such `s`, `t`), stores `fl(lambda s)` and charges its
//!   rounding against `|z|max`, and adds the product `lambda t` to `beta`.
//!   A non-finite chord falls back to `h <= u` (`lambda h >= lambda u`).
//! * **max-pool** (non-overlapping): `max >= v_k` with `k = argmax l_p`
//!   (exact), and for `lambda < 0` the upper bound
//!   `max <= v_k + max(0, max_{p != k} u_p - l_k)` (constant rounded UP).
//! * **reshape**: identity on the row-major data.
//! * **stop** at value `d` (the input, or a truncation value): `lambda . v >=
//!   sum min(lambda_j l_j, lambda_j u_j)`.
//!
//! The published lower bound is `next_down(fl(conc + beta) - charge)`, where
//! `charge` upper-bounds `kappa` plus the Higham `gamma_n * sum|terms|` of the
//! nearest-rounded `beta` and `conc` sums (plus underflow terms), everything
//! rounded upward. A non-finite result publishes `-inf` (always sound).

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Instant;

use rayon::prelude::*;

use super::boxes::{BoxView, Boxes};
use super::plan::{ChainPlan, ConvSpec, LinearSpec, OpKind, PoolSpec, Shape3};
use super::product::{self, gamma_up, ConvGeom, Ctx, Stop, ETA_F64};
use crate::margin_row::rounding::{certify_up, next_down, next_up, slack16, UNIT};

/// Per-row accumulators of (INV).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RowAcc {
    beta: f64,
    beta_abs: f64,
    beta_terms: u64,
    kappa: f64,
    kappa_terms: u64,
}

impl RowAcc {
    /// Add one product term (already rounded once) to `beta`.
    #[inline]
    fn add_beta(&mut self, p: f64) {
        self.beta += p;
        self.beta_abs += p.abs();
        self.beta_terms += 1;
    }

    /// Add one nonnegative charge term to `kappa`.
    #[inline]
    fn add_kappa(&mut self, t: f64) {
        self.kappa += t;
        self.kappa_terms += 1;
    }
}

/// A chunk of rows sharing one window shape over the current value.
pub(crate) struct Rows {
    pub(crate) n: usize,
    pub(crate) value: usize,
    pub(crate) qh: usize,
    pub(crate) qw: usize,
    pub(crate) oy: Vec<usize>,
    pub(crate) ox: Vec<usize>,
    /// `n x (C * qh * qw)`, channel-major within a row.
    pub(crate) lam: Vec<f64>,
    pub(crate) acc: Vec<RowAcc>,
}

/// Clip a backward window: rows `[s*o - p, s*(o + q - 1) - p + k)` of an
/// input axis of length `len`, shifted into `[0, len)` with its size kept.
/// Returns `(q2, origins)`; the shifted window always CONTAINS the in-range
/// support, so no nonzero coefficient is ever dropped.
pub(crate) fn back_window(
    q: usize,
    origins: &[usize],
    s: usize,
    p: usize,
    k: usize,
    len: usize,
) -> (usize, Vec<usize>) {
    let q2 = (s * (q - 1) + k).min(len);
    let hi = (len - q2) as isize;
    let o2 = origins
        .iter()
        .map(|&o| ((s * o) as isize - p as isize).clamp(0, hi) as usize)
        .collect();
    (q2, o2)
}

/// Seed rows for intermediate neurons of the affine value `vt`, taking the
/// first backward step (through `ops[vt - 1]`) EXACTLY: `mu = sign * W^T e_j`
/// is a sign flip of stored weights, and `beta = sign * b_j`.
pub(crate) fn seed_neurons(
    plan: &ChainPlan<'_>,
    vt: usize,
    targets: &[(usize, bool)],
) -> Result<Rows, Stop> {
    let op = &plan.ops[vt - 1];
    let (inp, out) = (op.input, op.output);
    let n = targets.len();
    let mut acc = vec![RowAcc::default(); n];
    match &op.kind {
        OpKind::Conv(spec) => {
            let qh = spec.kh.min(inp.h);
            let qw = spec.kw.min(inp.w);
            let len = spec.cin * qh * qw;
            let mut lam = vec![0.0_f64; n * len];
            let mut oy = Vec::with_capacity(n);
            let mut ox = Vec::with_capacity(n);
            for (r, &(j, negative)) in targets.iter().enumerate() {
                if j >= out.numel() {
                    return Err(Stop::Refused("seed index"));
                }
                let co = j / out.plane();
                let y = (j % out.plane()) / out.w;
                let x = j % out.w;
                let (_, oy2) = back_window(1, &[y], spec.sh, spec.ph, spec.kh, inp.h);
                let (_, ox2) = back_window(1, &[x], spec.sw, spec.pw, spec.kw, inp.w);
                let (oy2, ox2) = (oy2[0], ox2[0]);
                let sign = if negative { -1.0 } else { 1.0 };
                let row = &mut lam[r * len..(r + 1) * len];
                for ci in 0..spec.cin {
                    for y2 in 0..qh {
                        let ky = (oy2 + y2) as isize - ((spec.sh * y) as isize - spec.ph as isize);
                        if ky < 0 || ky >= spec.kh as isize {
                            continue;
                        }
                        for x2 in 0..qw {
                            let kx =
                                (ox2 + x2) as isize - ((spec.sw * x) as isize - spec.pw as isize);
                            if kx < 0 || kx >= spec.kw as isize {
                                continue;
                            }
                            row[(ci * qh + y2) * qw + x2] =
                                sign * spec.weight(co, ci, ky as usize, kx as usize);
                        }
                    }
                }
                acc[r].add_beta(sign * spec.bias[co]);
                oy.push(oy2);
                ox.push(ox2);
            }
            Ok(Rows {
                n,
                value: vt - 1,
                qh,
                qw,
                oy,
                ox,
                lam,
                acc,
            })
        }
        OpKind::Linear(spec) => {
            let len = spec.nin;
            let mut lam = vec![0.0_f64; n * len];
            for (r, &(j, negative)) in targets.iter().enumerate() {
                if j >= spec.nout {
                    return Err(Stop::Refused("seed index"));
                }
                let sign = if negative { -1.0 } else { 1.0 };
                for (i, slot) in lam[r * len..(r + 1) * len].iter_mut().enumerate() {
                    *slot = sign * spec.weight(j, i);
                }
                acc[r].add_beta(sign * spec.bias[j]);
            }
            Ok(Rows {
                n,
                value: vt - 1,
                qh: inp.h,
                qw: inp.w,
                oy: vec![0; n],
                ox: vec![0; n],
                lam,
                acc,
            })
        }
        _ => Err(Stop::Refused("seed on a non-affine value")),
    }
}

/// Rows for objective vectors over the OUTPUT value (full window).
pub(crate) fn seed_objectives(plan: &ChainPlan<'_>, objectives: &[Vec<f32>]) -> Result<Rows, Stop> {
    let v = plan.output_value();
    let shape = plan.value_shape(v);
    let len = shape.numel();
    let n = objectives.len();
    let mut lam = Vec::with_capacity(n * len);
    for obj in objectives {
        if obj.len() != len {
            return Err(Stop::Refused("objective arity"));
        }
        lam.extend(obj.iter().map(|&c| ny_core::f32_to_f64_exact(c)));
    }
    if lam.iter().any(|v| !v.is_finite()) {
        return Err(Stop::Refused("non-finite objective"));
    }
    Ok(Rows {
        n,
        value: v,
        qh: shape.h,
        qw: shape.w,
        oy: vec![0; n],
        ox: vec![0; n],
        lam,
        acc: vec![RowAcc::default(); n],
    })
}

/// Per-row peak bytes of the walk from `start` (window `(qh, qw)`) down to
/// value `0` or the truncation point: the admission estimate for one row.
pub(crate) fn row_bytes(
    plan: &ChainPlan<'_>,
    start: usize,
    qh: usize,
    qw: usize,
    depth: usize,
    affine_seen: usize,
) -> usize {
    let mut v = start;
    let (mut qh, mut qw) = (qh, qw);
    let mut affine = affine_seen;
    let mut numel = plan.value_shape(v).c * qh * qw;
    let mut peak = numel * 8;
    while v > 0 {
        if depth > 0 && affine >= depth {
            break;
        }
        let op = &plan.ops[v - 1];
        let inp = op.input;
        let next = match &op.kind {
            OpKind::Relu => numel,
            OpKind::Reshape | OpKind::Linear(_) => {
                qh = inp.h;
                qw = inp.w;
                inp.numel()
            }
            OpKind::MaxPool(p) => {
                qh = ((qh - 1) * p.sh + p.kh).min(inp.h);
                qw = ((qw - 1) * p.sw + p.kw).min(inp.w);
                inp.c * qh * qw
            }
            OpKind::Conv(spec) => {
                qh = (spec.sh * (qh - 1) + spec.kh).min(inp.h);
                qw = (spec.sw * (qw - 1) + spec.kw).min(inp.w);
                inp.c * qh * qw
            }
        };
        let step = match &op.kind {
            OpKind::Conv(_) | OpKind::Linear(_) => {
                affine += 1;
                // lambda + mu + err, plus lambda's size again for the linear
                // GPU route's two f32 copies of lambda (`|lambda|` rounded up
                // and, on the f32 route, the signed narrowing; 4 + 4 bytes).
                (2 * numel + 2 * next) * 8
            }
            OpKind::MaxPool(_) => (numel + next) * 8,
            _ => numel * 8,
        };
        peak = peak.max(step);
        numel = next;
        v -= 1;
    }
    // Origins, accumulators and the concretization pass.
    peak + 64
}

/// Walk `rows` down to value `0` (or `depth` affine ops, counting
/// `affine_seen` already taken) and publish one certified lower bound per row.
pub(crate) fn run_rows(
    plan: &ChainPlan<'_>,
    boxes: &Boxes,
    ctx: &Ctx<'_>,
    mut rows: Rows,
    depth: usize,
    affine_seen: usize,
) -> Result<Vec<f64>, Stop> {
    let mut affine = affine_seen;
    while rows.value > 0 {
        if depth > 0 && affine >= depth {
            break;
        }
        if Instant::now() >= ctx.deadline {
            return Err(Stop::Expired);
        }
        let op = &plan.ops[rows.value - 1];
        let (inp, out) = (op.input, op.output);
        let view = boxes.view(rows.value - 1);
        match &op.kind {
            OpKind::Relu => step_relu(&mut rows, inp, view),
            OpKind::MaxPool(p) => step_pool(&mut rows, p, inp, out, view)?,
            OpKind::Reshape => step_reshape(&mut rows, inp, out)?,
            OpKind::Conv(spec) => {
                step_conv(&mut rows, spec, inp, out, view, ctx)?;
                affine += 1;
            }
            OpKind::Linear(spec) => {
                step_linear(&mut rows, spec, inp, out, view, ctx)?;
                affine += 1;
            }
        }
        rows.value -= 1;
    }
    let shape = plan.value_shape(rows.value);
    Ok(concretize(&rows, shape, boxes.view(rows.value)))
}

/// The ReLU relaxation of one coefficient (see the module docs).
#[inline]
fn relax_relu(slot: &mut f64, acc: &mut RowAcc, view: BoxView<'_>, j: usize) {
    let l = *slot;
    let lo = view.lo(j);
    let hi = view.hi(j);
    if hi <= 0.0 {
        *slot = 0.0;
        return;
    }
    if lo >= 0.0 {
        return;
    }
    if l > 0.0 {
        if hi < -lo {
            *slot = 0.0;
        }
        return;
    }
    // `d <= hi - lo` (exact) and `s >= hi / (hi - lo)` are what make the
    // chord valid; an overflowed `d = +inf` would give `s = 0+`, which is NOT
    // an upper chord, so a non-finite `d` takes the constant fallback below.
    let d = next_down(hi - lo);
    let s = if d.is_finite() && d > 0.0 {
        next_up(hi / d)
    } else {
        f64::INFINITY
    };
    let t = next_up(s * (-lo));
    let nl = l * s;
    if !(s.is_finite() && t.is_finite() && nl.is_finite()) {
        *slot = 0.0;
        acc.add_beta(l * hi);
        return;
    }
    acc.add_kappa((2.0 * UNIT * nl.abs() + ETA_F64) * view.mag(j));
    acc.add_beta(l * t);
    *slot = nl;
}

fn step_relu(rows: &mut Rows, shape: Shape3, view: BoxView<'_>) {
    let (qh, qw) = (rows.qh, rows.qw);
    let len = shape.c * qh * qw;
    rows.lam
        .par_chunks_mut(len)
        .zip(rows.acc.par_iter_mut())
        .zip(rows.oy.par_iter().zip(rows.ox.par_iter()))
        .for_each(|((lam, acc), (&oy, &ox))| {
            for c in 0..shape.c {
                for y in 0..qh {
                    let base = c * shape.plane() + (oy + y) * shape.w + ox;
                    for x in 0..qw {
                        let i = (c * qh + y) * qw + x;
                        if lam[i] != 0.0 {
                            relax_relu(&mut lam[i], acc, view, base + x);
                        }
                    }
                }
            }
        });
}

fn step_pool(
    rows: &mut Rows,
    spec: &PoolSpec,
    inp: Shape3,
    out: Shape3,
    view: BoxView<'_>,
) -> Result<(), Stop> {
    let (qh, qw) = (rows.qh, rows.qw);
    let (qh2, oy2) = back_window(qh, &rows.oy, spec.sh, 0, spec.kh, inp.h);
    let (qw2, ox2) = back_window(qw, &rows.ox, spec.sw, 0, spec.kw, inp.w);
    let old_len = out.c * qh * qw;
    let new_len = inp.c * qh2 * qw2;
    let mut lam2 = vec![0.0_f64; rows.n * new_len];
    let bad = AtomicBool::new(false);
    lam2.par_chunks_mut(new_len)
        .zip(rows.lam.par_chunks(old_len))
        .zip(rows.acc.par_iter_mut())
        .zip(
            rows.oy
                .par_iter()
                .zip(rows.ox.par_iter())
                .zip(oy2.par_iter().zip(ox2.par_iter())),
        )
        .for_each(|(((dst, src), acc), ((&oy, &ox), (&oy2, &ox2)))| {
            for c in 0..out.c {
                for y in 0..qh {
                    for x in 0..qw {
                        let l = src[(c * qh + y) * qw + x];
                        if l == 0.0 {
                            continue;
                        }
                        let (py, px) = (oy + y, ox + x);
                        let at = |dy: usize, dx: usize| {
                            c * inp.plane() + (py * spec.sh + dy) * inp.w + px * spec.sw + dx
                        };
                        let mut best = (0usize, 0usize);
                        let mut best_lo = f64::NEG_INFINITY;
                        for dy in 0..spec.kh {
                            for dx in 0..spec.kw {
                                let lo = view.lo(at(dy, dx));
                                if lo > best_lo {
                                    best_lo = lo;
                                    best = (dy, dx);
                                }
                            }
                        }
                        if l < 0.0 {
                            let mut other_hi = f64::NEG_INFINITY;
                            for dy in 0..spec.kh {
                                for dx in 0..spec.kw {
                                    if (dy, dx) != best {
                                        other_hi = other_hi.max(view.hi(at(dy, dx)));
                                    }
                                }
                            }
                            if other_hi > best_lo {
                                let cst = next_up(other_hi - best_lo);
                                acc.add_beta(l * cst);
                            }
                        }
                        let ty = (py * spec.sh + best.0) as isize - oy2 as isize;
                        let tx = (px * spec.sw + best.1) as isize - ox2 as isize;
                        if ty < 0 || ty >= qh2 as isize || tx < 0 || tx >= qw2 as isize {
                            bad.store(true, Relaxed);
                            continue;
                        }
                        dst[(c * qh2 + ty as usize) * qw2 + tx as usize] += l;
                    }
                }
            }
        });
    if bad.load(Relaxed) {
        return Err(Stop::Refused("max-pool window"));
    }
    rows.lam = lam2;
    rows.qh = qh2;
    rows.qw = qw2;
    rows.oy = oy2;
    rows.ox = ox2;
    Ok(())
}

fn step_reshape(rows: &mut Rows, inp: Shape3, out: Shape3) -> Result<(), Stop> {
    let full = rows.qh == out.h
        && rows.qw == out.w
        && rows.oy.iter().all(|&o| o == 0)
        && rows.ox.iter().all(|&o| o == 0);
    if !full || inp.numel() != out.numel() {
        return Err(Stop::Refused("reshape of a partial window"));
    }
    rows.qh = inp.h;
    rows.qw = inp.w;
    Ok(())
}

/// `beta += lambda . b` over the current (affine OUTPUT) window.
fn add_bias(rows: &mut Rows, shape: Shape3, bias: &[f64]) {
    let plane = rows.qh * rows.qw;
    let len = shape.c * plane;
    rows.lam
        .par_chunks(len)
        .zip(rows.acc.par_iter_mut())
        .for_each(|(lam, acc)| {
            for (c, &b) in bias.iter().enumerate().take(shape.c) {
                if b == 0.0 {
                    continue;
                }
                for &l in &lam[c * plane..(c + 1) * plane] {
                    if l != 0.0 {
                        acc.add_beta(l * b);
                    }
                }
            }
        });
}

/// `kappa += sum err_j |v_j|max` over the new window of the affine INPUT.
fn charge(
    acc: &mut [RowAcc],
    err: &[f64],
    shape: Shape3,
    qh: usize,
    qw: usize,
    oy: &[usize],
    ox: &[usize],
    view: BoxView<'_>,
) {
    let len = shape.c * qh * qw;
    acc.par_iter_mut()
        .zip(err.par_chunks(len))
        .zip(oy.par_iter().zip(ox.par_iter()))
        .for_each(|((a, e), (&oy, &ox))| {
            let mut sum = 0.0_f64;
            let mut terms = 0u64;
            for c in 0..shape.c {
                for y in 0..qh {
                    let base = c * shape.plane() + (oy + y) * shape.w + ox;
                    for x in 0..qw {
                        let m = view.mag(base + x);
                        if m != 0.0 {
                            sum += e[(c * qh + y) * qw + x] * m;
                            terms += 1;
                        }
                    }
                }
            }
            a.kappa += sum;
            a.kappa_terms += terms;
        });
}

fn step_conv(
    rows: &mut Rows,
    spec: &ConvSpec,
    inp: Shape3,
    out: Shape3,
    view: BoxView<'_>,
    ctx: &Ctx<'_>,
) -> Result<(), Stop> {
    add_bias(rows, out, &spec.bias);
    let (qh2, oy2) = back_window(rows.qh, &rows.oy, spec.sh, spec.ph, spec.kh, inp.h);
    let (qw2, ox2) = back_window(rows.qw, &rows.ox, spec.sw, spec.pw, spec.kw, inp.w);
    let cert = {
        let geom = ConvGeom {
            rows: rows.n,
            qh: rows.qh,
            qw: rows.qw,
            oy: &rows.oy,
            ox: &rows.ox,
            qh2,
            qw2,
            oy2: &oy2,
            ox2: &ox2,
        };
        product::conv_product(ctx, spec, &geom, &rows.lam)?
    };
    charge(&mut rows.acc, &cert.err, inp, qh2, qw2, &oy2, &ox2, view);
    rows.lam = cert.mu;
    rows.qh = qh2;
    rows.qw = qw2;
    rows.oy = oy2;
    rows.ox = ox2;
    Ok(())
}

fn step_linear(
    rows: &mut Rows,
    spec: &LinearSpec<'_>,
    inp: Shape3,
    out: Shape3,
    view: BoxView<'_>,
    ctx: &Ctx<'_>,
) -> Result<(), Stop> {
    if rows.qh != 1 || rows.qw != 1 || out.numel() != spec.nout {
        return Err(Stop::Refused("linear output window"));
    }
    add_bias(rows, out, &spec.bias);
    let cert = product::linear_product(ctx, spec, rows.n, &rows.lam)?;
    let zeros = vec![0usize; rows.n];
    charge(
        &mut rows.acc,
        &cert.err,
        inp,
        inp.h,
        inp.w,
        &zeros,
        &zeros,
        view,
    );
    rows.lam = cert.mu;
    rows.qh = inp.h;
    rows.qw = inp.w;
    rows.oy = zeros.clone();
    rows.ox = zeros;
    Ok(())
}

/// Publish `next_down(fl(conc + beta) - charge)` (see the module docs).
fn finish(acc: &RowAcc, conc: f64, conc_abs: f64, conc_terms: u64) -> f64 {
    let (Some(gb), Some(gk), Some(gc)) = (
        gamma_up(acc.beta_terms + 2, UNIT),
        gamma_up(acc.kappa_terms + 2, UNIT),
        gamma_up(conc_terms + 2, UNIT),
    ) else {
        return f64::NEG_INFINITY;
    };
    let eta_b = ETA_F64 * (acc.beta_terms as f64 + 2.0);
    let eta_k = ETA_F64 * (acc.kappa_terms as f64 + 2.0);
    let eta_c = ETA_F64 * (conc_terms as f64 + 2.0);
    let beta_err = slack16(gb * (certify_up(acc.beta_abs, gb) + eta_b)) + eta_b;
    let kappa_up = certify_up(acc.kappa, gk) + eta_k;
    let conc_err = slack16(gc * (certify_up(conc_abs, gc) + eta_c)) + eta_c;
    let total = slack16(kappa_up + beta_err + conc_err);
    let head = next_down(conc + acc.beta);
    let lb = next_down(head - total);
    if lb.is_finite() && total.is_finite() {
        lb
    } else {
        f64::NEG_INFINITY
    }
}

fn concretize(rows: &Rows, shape: Shape3, view: BoxView<'_>) -> Vec<f64> {
    let (qh, qw) = (rows.qh, rows.qw);
    let len = shape.c * qh * qw;
    rows.lam
        .par_chunks(len)
        .zip(rows.acc.par_iter())
        .zip(rows.oy.par_iter().zip(rows.ox.par_iter()))
        .map(|((lam, acc), (&oy, &ox))| {
            let (mut s, mut sa, mut n) = (0.0_f64, 0.0_f64, 0u64);
            for c in 0..shape.c {
                for y in 0..qh {
                    let base = c * shape.plane() + (oy + y) * shape.w + ox;
                    for x in 0..qw {
                        let l = lam[(c * qh + y) * qw + x];
                        if l == 0.0 {
                            continue;
                        }
                        let j = base + x;
                        let p = if l > 0.0 {
                            l * view.lo(j)
                        } else {
                            l * view.hi(j)
                        };
                        s += p;
                        sa += p.abs();
                        n += 1;
                    }
                }
            }
            finish(acc, s, sa, n)
        })
        .collect()
}
