// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lowdim-bundle (2) — per-leaf BOUNDARY ALPHA for the lowdim lane's
//! objective bound (study_5 `#lowdim-boundary-alpha`, acasxu 3_3 prop_2).
//!
//! # Why
//!
//! The lane's ReLU lower relaxation used a fixed slope in `{0, 1}` (the
//! CROWN-adaptive rule). Vibecheck's acasxu config optimises the lower slopes
//! of leaves whose bound lies within a WIDE band (`eps: 10`) below the
//! threshold, while they are still splittable: 3_3 prop_2 went from 1.88M
//! leaves / unknown to 431k leaves / verified.
//!
//! # Soundness
//!
//! * For every `alpha in [0, 1]` and every real `z`, `relu(z) >= alpha z`
//!   (`z >= 0`: `z >= alpha z`; `z < 0`: `0 >= alpha z`). An alpha pass is
//!   the lane's own backward pass with those slopes (`super::backward_alpha`),
//!   every multiplier product charged exactly as the upper slope's, so its
//!   bound is rigorous whatever the slopes are.
//! * The slopes are chosen by a sign-projected heuristic step and are
//!   CLAMPED to `[0, 1]`; the backward pass independently refuses any slope
//!   outside `[0, 1]` (it falls back to the historical rule).
//! * The published bound of a domain is the MAXIMUM of rigorous lower bounds
//!   of the same quantity (the lane's passes and every alpha pass); the plane
//!   kept for the clip is ONE alpha pass's own plane on that domain's box.
//! * Only the OBJECTIVE pass takes alpha. Intermediate bounds (the forward
//!   forms, the refinement passes, the joint intercept pass) are unchanged.
//!
//! MEMORY: per chunk, the slope table and one pass's recorded multipliers are
//! one f64 each per ReLU neuron and domain — less than the chunk's forward
//! forms (`2 (d + 1)` f64 per neuron and domain), which the caller's
//! `CHUNK_BYTES` clamp already bounds; nothing outlives the chunk.

use super::{backward_alpha, relu_lower_is_identity, Chunk, Fam, LowDimStats, NodeVal, Op, Pass};

/// Band below the threshold inside which an open domain gets alpha steps
/// (Vibecheck's acasxu `eps: 10`). Only decides where compute is spent.
pub(crate) const ALPHA_BAND: f64 = 10.0;

/// Largest iteration count honoured (a typo must not stall the lane).
pub(crate) const ALPHA_MAX_ITERS: usize = 200;

/// First sign-step size on a slope, then multiplied by [`ALPHA_DECAY`].
const ALPHA_STEP0: f64 = 0.5;
const ALPHA_DECAY: f64 = 0.8;

/// Per node: `Some([n][m])` lower slopes for a ReLU node (NaN where the
/// neuron is not unstable on that domain), `None` for every other node.
pub(crate) struct AlphaTable {
    pub(crate) slopes: Vec<Option<Vec<f64>>>,
}

/// One objective's alpha refinement on one chunk.
pub(super) struct AlphaResult {
    /// Per domain, the best alpha pass's lower bound, input multiplier and
    /// plane constant (`lb = -inf` where no alpha pass beat `-inf`). Every
    /// component of domain `b` comes from ONE pass, so domain `b`'s plane
    /// `grad[b] . x + pconst[b] <= objective . y(x)` holds on its box.
    pub(super) pass: Pass,
    /// Per domain: it was in the band (alpha steps were taken for it).
    pub(super) ran: Vec<bool>,
}

/// Run `iters` alpha passes of `objective` over a chunk whose forward pass is
/// `vals`. `base_lb[b]` is the domain's bound without alpha (it decides the
/// band only). `None` when no domain is in the band, no neuron is unstable,
/// or `iters == 0`: nothing ran.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    seed: usize,
    objective: &[&[f64]],
    poison: &[bool],
    base_lb: &[f64],
    thr: f64,
    iters: usize,
) -> Option<AlphaResult> {
    let prog = fam.base();
    let (m, d) = (ch.m, ch.d);
    let d1 = d + 1;
    let iters = iters.min(ALPHA_MAX_ITERS);
    if iters == 0 || base_lb.len() != m || poison.len() != m || !thr.is_finite() {
        return None;
    }
    let ran: Vec<bool> = (0..m)
        .map(|b| !poison[b] && base_lb[b] <= thr && base_lb[b] >= thr - ALPHA_BAND)
        .collect();
    if !ran.iter().any(|&r| r) {
        return None;
    }
    // Start from the historical CROWN-adaptive slope of every unstable neuron
    // (so the first pass reproduces the lane's bound up to the extra charge).
    let mut table = AlphaTable {
        slopes: vec![None; prog.ops.len()],
    };
    let mut any_unstable = false;
    for (idx, op) in prog.ops.iter().enumerate() {
        let Op::Relu { src } = op else { continue };
        let Some(s) = vals.get(*src).and_then(Option::as_ref) else {
            continue;
        };
        let n = prog.sizes[idx];
        if s.lo.len() != n * m || s.hi.len() != n * m {
            continue;
        }
        let mut sl = vec![f64::NAN; n * m];
        for (i, slot) in sl.iter_mut().enumerate() {
            let (lo, hi) = (s.lo[i], s.hi[i]);
            if lo < 0.0 && hi > 0.0 {
                *slot = if relu_lower_is_identity(lo, hi) {
                    1.0
                } else {
                    0.0
                };
                any_unstable = true;
            }
        }
        table.slopes[idx] = Some(sl);
    }
    if !any_unstable {
        return None;
    }
    let mut best = Pass {
        lb: vec![f64::NEG_INFINITY; m],
        grad: vec![0.0; m * d],
        pconst: vec![f64::NEG_INFINITY; m],
        hit_site: false,
        relu_mult: Vec::new(),
    };
    let mut step = ALPHA_STEP0;
    for _ in 0..iters {
        let pass = backward_alpha(fam, vals, ch, seed, objective, poison, false, Some(&table));
        for b in 0..m {
            if ran[b] && pass.lb[b] > best.lb[b] {
                best.lb[b] = pass.lb[b];
                best.grad[b * d..(b + 1) * d].copy_from_slice(&pass.grad[b * d..(b + 1) * d]);
                best.pconst[b] = pass.pconst[b];
            }
        }
        // Sign step on every recorded slope of an in-band domain: the bound's
        // local slope in alpha_j is `l_j * L_j(x*)`, `L_j` the source's
        // forward LOWER form and `x*` the box corner minimising this pass's
        // input term. A heuristic direction only.
        for (idx, rec) in pass.relu_mult.iter().enumerate() {
            let Some(rec) = rec else { continue };
            let Op::Relu { src } = &prog.ops[idx] else {
                continue;
            };
            let Some(s) = vals.get(*src).and_then(Option::as_ref) else {
                continue;
            };
            let Some(Some(sl)) = table.slopes.get_mut(idx) else {
                continue;
            };
            let n = prog.sizes[idx];
            if rec.len() != n * m || sl.len() != n * m {
                continue;
            }
            for j in 0..n {
                for b in 0..m {
                    let i = j * m + b;
                    if !ran[b]
                        || rec[i].partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
                        || !sl[i].is_finite()
                    {
                        continue;
                    }
                    let r = i * d1;
                    let g_row = &pass.grad[b * d..(b + 1) * d];
                    let (bl, bh) = (&ch.lo[b * d..(b + 1) * d], &ch.hi[b * d..(b + 1) * d]);
                    let mut z = s.lo_form[r + d];
                    for k in 0..d {
                        let x = if g_row[k] >= 0.0 { bl[k] } else { bh[k] };
                        z += s.lo_form[r + k] * x;
                    }
                    let g = rec[i] * z;
                    if g > 0.0 {
                        sl[i] = (sl[i] + step).min(1.0);
                    } else if g < 0.0 {
                        sl[i] = (sl[i] - step).max(0.0);
                    }
                }
            }
        }
        step *= ALPHA_DECAY;
    }
    Some(AlphaResult { pass: best, ran })
}

/// The `[lowdim-alpha] done` line, printed unconditionally at the end of every
/// lane run with the boundary alpha armed (fixed schema). An armed run with
/// `domains=0` never had a leaf in the band; `improved=0` with `domains>0` is
/// a measured null for this row, not an engagement failure.
pub(super) fn emit_line(stats: &LowDimStats, iters: usize) {
    eprintln!(
        "[lowdim-alpha] done iters={} band={ALPHA_BAND:e} domains={} improved={} rescued={} \
         lane_domains={} closed={} lane_secs={:.3}",
        iters.min(ALPHA_MAX_ITERS),
        stats.alpha_domains,
        stats.alpha_improved,
        stats.alpha_rescued,
        stats.domains,
        stats.closed,
        stats.secs
    );
}
