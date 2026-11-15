// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The STATIC split order of the frozen dual BaB: score every unstable row by
//! the closed-form box + single-halfspace child lower bound of its worse child
//! (port of the P1 replica's `score_box_halfspace` / `_lmin`, itself a dense
//! port of Vibecheck's `verify_gen_lp.py:1604-1741`), and split in descending
//! score order at every depth.
//!
//! ORDERING ONLY. Nothing here is certified and nothing here may be: a score
//! decides WHICH row is split next, never a bound. Every node bound is still
//! [`super::problem::FrozenDualProblem::eval_node`]'s certified value.

use rayon::prelude::*;

use super::build::FrozenDualState;

/// `np.sign` semantics (`0` at zero), unlike `f64::signum`.
fn sign0(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `max_{lambda >= 0} min_{e in [-1,1]^n} c0 + (d + lambda a) . e - lambda beta`
/// (the exact one-halfspace dual of `min c0 + d.e  s.t.  a.e <= beta`):
/// `+inf` when the constraint makes the box infeasible.
fn lmin(a: &[f64], d: &[f64], c0: f64, beta: f64, scratch: &mut Vec<(f64, f64)>) -> f64 {
    let g0 = c0 - d.iter().map(|v| v.abs()).sum::<f64>();
    let mut dot = 0.0f64;
    for (&av, &dv) in a.iter().zip(d) {
        let eff = if dv != 0.0 { sign0(dv) } else { sign0(av) };
        dot += av * eff;
    }
    let gp0 = -beta - dot;
    if gp0 <= 0.0 {
        return g0;
    }
    scratch.clear();
    for (&av, &dv) in a.iter().zip(d) {
        if av != 0.0 && dv != 0.0 {
            let lam = -dv / av;
            if lam > 0.0 {
                scratch.push((lam, 2.0 * av.abs()));
            }
        }
    }
    if scratch.is_empty() {
        return f64::INFINITY;
    }
    scratch.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut g = g0;
    let mut prev = 0.0f64;
    let mut cum = 0.0f64;
    for &(lam, decr) in scratch.iter() {
        g += (gp0 - cum) * (lam - prev);
        cum += decr;
        prev = lam;
        if cum >= gp0 {
            return g;
        }
    }
    f64::INFINITY
}

/// Reusable buffers for the exact per-row split score. Reserving the full
/// breakpoint capacity makes the streaming builder's memory bound explicit.
pub(crate) struct SplitScoreScratch {
    doff: Vec<f64>,
    don: Vec<f64>,
    neg_a: Vec<f64>,
    s1: Vec<(f64, f64)>,
    s2: Vec<(f64, f64)>,
}

impl SplitScoreScratch {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            doff: vec![0.0; n],
            don: vec![0.0; n],
            neg_a: vec![0.0; n],
            s1: Vec::with_capacity(n),
            s2: Vec::with_capacity(n),
        }
    }
}

/// The existing exact box + single-halfspace score, shared by the full and
/// streaming builders. `base` is computed once from the unchanged objective.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_score(
    a: &[f64],
    d: &[f64],
    c0: f64,
    base: f64,
    lam: f64,
    mu: f64,
    cin: f64,
    enc: usize,
    scratch: &mut SplitScoreScratch,
) -> f64 {
    let ew = d[enc] / mu;
    let d_enc = d[enc] - ew * mu;
    for j in 0..d.len() {
        scratch.doff[j] = d[j] - ew * lam * a[j];
        scratch.don[j] = d[j] + ew * (1.0 - lam) * a[j];
    }
    scratch.doff[enc] = d_enc;
    scratch.don[enc] = d_enc;
    let c0off = c0 - ew * (lam * cin + mu);
    let c0on = c0 + ew * ((1.0 - lam) * cin - mu);
    let lb_off = lmin(a, &scratch.doff, c0off, -cin, &mut scratch.s1);
    for (dst, &v) in scratch.neg_a.iter_mut().zip(a) {
        *dst = -v;
    }
    let lb_on = lmin(&scratch.neg_a, &scratch.don, c0on, cin, &mut scratch.s2);
    let sc = lb_off.min(lb_on) - base;
    if sc.is_nan() {
        f64::NEG_INFINITY
    } else {
        sc
    }
}

/// Per-row score `min(LB_off, LB_on) - (c_obj - sum|d|)` (higher = better).
#[must_use]
pub fn split_scores(state: &FrozenDualState) -> Vec<f64> {
    let d = &state.d;
    let c0 = state.c_obj;
    let base = c0 - d.iter().map(|v| v.abs()).sum::<f64>();
    (0..state.num_unstable())
        .into_par_iter()
        .map_init(
            || SplitScoreScratch::new(state.n_gens),
            |scratch, k| {
                split_score(
                    state.row(k),
                    d,
                    c0,
                    base,
                    state.lam[k],
                    state.mu[k],
                    state.c_in[k],
                    state.enc[k],
                    scratch,
                )
            },
        )
        .collect()
}

/// Rows by descending [`split_scores`], ties by row index (a permutation of
/// `0..S`).
#[must_use]
pub fn static_order(state: &FrozenDualState) -> Vec<usize> {
    let scores = split_scores(state);
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&x, &y| scores[y].total_cmp(&scores[x]).then(x.cmp(&y)));
    order
}

#[cfg(test)]
pub(crate) fn lmin_for_test(a: &[f64], d: &[f64], c0: f64, beta: f64) -> f64 {
    lmin(a, d, c0, beta, &mut Vec::new())
}
