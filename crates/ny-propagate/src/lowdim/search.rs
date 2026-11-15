// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane stage 3 — the lane's per-domain SEARCH step: relaxed
//! Clip-and-Verify, the smart-branching split rules, and the wave runner that
//! decides every domain of a wave in parallel.
//!
//! # Why
//!
//! The numpy decision gate (`scratchpad/lsncproto`, variant `allmax`,
//! 2026-09-24) needs 2.0e5-1.4e6 domains per `lsnc_relu` unsat row even with
//! the joint quadratic bound (stage 2). Two of its three remaining factors are
//! here: the relaxed clip of each open box by its own conjunct planes, and the
//! aspect-guarded smart-branching split (`sbwr`). The third is THROUGHPUT: a
//! row fits the official 25 s only if the lane sustains ~1e5 domains/s or
//! more, so a wave's whole per-domain pipeline — bound, close test, clip,
//! violation-candidate test, split — can run inside the rayon chunk tasks
//! ([`LowDimParallel::Full`]).
//!
//! # Soundness
//!
//! * A PLANE is `a . x + c <= objective_i . y(x)` for every `x` of the box it
//!   was computed on: the input multiplier and plane constant of ONE backward
//!   pass over THIS box's forward pass (`super::Pass`, every rounding of the
//!   pass charged into its constant). A plane is never rebuilt from a lower
//!   bound (the published bound is a MAXIMUM over passes and the seed's forms
//!   — the shape abc's `deconstruct_bias` gets wrong), and it lives only in
//!   its own domain's [`DomainBound`]: a child is bounded afresh and clips
//!   with its OWN planes, never its parent's (the 807bf511 false-unsat
//!   mechanism was clipping with reused, stacked planes).
//! * A counterexample of a clause satisfies EVERY conjunct `objective_i . y
//!   <= t_i`, hence every plane's halfspace `a . x + c <= t_i`. [`clip_box`]
//!   removes only points that violate some halfspace, each removal certified
//!   by one plane over the UNCLIPPED box, with the numerator raised by a
//!   rigorous rounding charge plus a `1e-12` relative guard and the new face
//!   rounded outward — the numpy gate measured abc's raw float clip failing
//!   753 of 1525 exact-rational checks; with this guard, 0.
//! * A split only chooses WHERE to bisect: both children are the two halves
//!   of the (clipped) box, so every rule covers the box ([`choose_split`]).
//! * The wave runner is a pure map from a domain to its [`Fate`] followed by
//!   an in-order merge, so where the map runs cannot change the outcome.

use rayon::prelude::*;

use super::{
    alpha, candidate_margin, first_argmax, forward, gslack, next_down, next_up,
    objective_bounds_passes, point_objective, Chunk, Fam, LowDimConfig, LowDimParallel,
    LowDimSplit, Pass, Program, MAX_INPUT_DIM,
};

/// Relative guard on a clip numerator, on top of its rigorous rounding
/// charge: the numpy gate's value (`gate_engine.clip_boxes`, `guard=1e-12`),
/// whose exact-rational falsifier found 0 violations where abc's raw clip
/// found 753 of 1525.
pub(super) const CLIP_REL_GUARD: f64 = 1e-12;

/// `sb_sum`'s coefficient floor (abc `sb_coeff_thresh`; the lsnc yaml's
/// `sb_coeff_thresh: 1.0e-2`).
pub(super) const SB_COEFF: f64 = 1e-2;

/// `sb_sum`'s bonus for a coordinate whose interval touches zero (abc
/// `touch_zero_score`; the lsnc yaml's `touch_zero_score: 0.1`).
pub(super) const SB_TOUCH_ZERO: f64 = 0.1;

/// The aspect guard: a coordinate is eligible for a smart-branching split
/// only when its width relative to the ROOT box is at least `1 / SB_ASPECT`
/// of the largest relative width (the numpy gate's `sbwr` / `sbar` value).
pub(super) const SB_ASPECT: f64 = 8.0;

// ---------------------------------------------------------------- clip

/// One clip halfspace `a . x + c <= t`: a plane of the box's own backward
/// pass (`a . x + c <= objective . y(x)` on the box) and the clause's
/// conjunct `objective . y(x) <= t`, which every counterexample satisfies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClipPlane<'a> {
    pub(crate) a: &'a [f64],
    pub(crate) c: f64,
    pub(crate) t: f64,
}

/// One candidate face of a plane: every point of the box with `x[dim] >
/// value` (`upper`) or `x[dim] < value` (`!upper`) violates the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Face {
    pub(crate) dim: usize,
    pub(crate) upper: bool,
    pub(crate) value: f64,
}

/// Every face ONE plane certifies over the box `[lo, hi]`, reported through
/// `face`; returns `true` when the plane alone proves the box holds no point
/// of its halfspace (its minimum over the box exceeds `t`).
///
/// For a coordinate `n` with `a_n != 0`, every point of the box in the
/// halfspace satisfies
///
/// ```text
/// a_n x_n <= num_n := t - c - sum_{m != n} min_{x_m in [lo_m, hi_m]} a_m x_m
/// ```
///
/// so `x_n <= num_n / a_n` when `a_n > 0` and `x_n >= num_n / a_n` when
/// `a_n < 0` (abc `clip.py:_clip_main_fn`, one iteration).
///
/// ROUNDING. `num_n` is computed from `d` rounded products and `d + 1` rounded
/// sums of terms whose magnitudes add to at most `scale = |t| + |c| + sum_m
/// |a_m| max(|lo_m|, |hi_m|)`, so the exact value is at most the computed one
/// plus `gslack(d + 4, scale)` (Higham, any summation order, with the
/// subnormal term). The numerator is raised by that charge PLUS `1e-12 *
/// scale` and rounded up (`num_up >= num_n`), which is conservative for both
/// signs of `a_n` (the bound `num / a_n` grows with `num` when `a_n > 0` and
/// shrinks when `a_n < 0`, and each side keeps the points that satisfy it).
/// The quotient is then rounded OUTWARD by one ulp: `next_up` for an upper
/// face, `next_down` for a lower one (round-to-nearest is within half an ulp
/// of the exact quotient, also in the subnormal range). A non-finite quotient
/// (overflow) or a non-finite input yields no face.
pub(crate) fn plane_faces(
    lo: &[f64],
    hi: &[f64],
    p: &ClipPlane<'_>,
    mut face: impl FnMut(Face),
) -> bool {
    let d = lo.len();
    if d > MAX_INPUT_DIM || hi.len() != d || p.a.len() != d {
        return false;
    }
    if !(p.c.is_finite() && p.t.is_finite()) || p.a.iter().any(|v| !v.is_finite()) {
        return false;
    }
    // e[m] = min over [lo_m, hi_m] of a_m x_m (attained at an end, exactly).
    let mut e = [0.0f64; MAX_INPUT_DIM];
    let mut scale = p.t.abs() + p.c.abs();
    for m in 0..d {
        let a = p.a[m];
        e[m] = if a >= 0.0 { a * lo[m] } else { a * hi[m] };
        scale += a.abs() * lo[m].abs().max(hi[m].abs());
    }
    if !scale.is_finite() || e[..d].iter().any(|v| !v.is_finite()) {
        return false;
    }
    let guard = next_up(CLIP_REL_GUARD * scale + gslack(d + 4, scale));
    // The whole plane: `t - c - sum_m e_m < 0` rigorously means the plane's
    // minimum over the box exceeds `t` — no point of the box is in the
    // halfspace (the same fact as the pass's own `lb > t`, re-derived here so
    // a clip never depends on the caller having tested it).
    let mut all = p.t - p.c;
    for &v in &e[..d] {
        all -= v;
    }
    if next_up(all + guard) < 0.0 {
        return true;
    }
    for n in 0..d {
        let a = p.a[n];
        if a == 0.0 {
            continue;
        }
        let mut num = p.t - p.c;
        for (m, &v) in e[..d].iter().enumerate() {
            if m != n {
                num -= v;
            }
        }
        let q = next_up(num + guard) / a;
        if !q.is_finite() {
            continue;
        }
        if a > 0.0 {
            face(Face {
                dim: n,
                upper: true,
                value: next_up(q),
            });
        } else {
            face(Face {
                dim: n,
                upper: false,
                value: next_down(q),
            });
        }
    }
    false
}

/// RELAXED CLIP-AND-VERIFY of one box, in place: `[lo, hi]` becomes the
/// intersection of the box with every face every plane certifies over the
/// ORIGINAL box ([`plane_faces`]; all planes are evaluated on the same box,
/// then intersected — one iteration, the numpy gate's `clip_iterations=1`).
///
/// Returns `None` when the box holds no point satisfying every plane — a
/// plane certified that by itself, or two faces crossed (every point of the
/// box then lies beyond one of them) — and the box is left unchanged.
/// Otherwise `Some(shrunk)`, with every point of the original box that
/// satisfies every plane still inside the new box, and the box never grown.
pub(crate) fn clip_box(lo: &mut [f64], hi: &mut [f64], planes: &[ClipPlane<'_>]) -> Option<bool> {
    let d = lo.len();
    if d > MAX_INPUT_DIM || hi.len() != d {
        return Some(false);
    }
    let (mut nl, mut nh) = ([0.0f64; MAX_INPUT_DIM], [0.0f64; MAX_INPUT_DIM]);
    nl[..d].copy_from_slice(lo);
    nh[..d].copy_from_slice(hi);
    for p in planes {
        let empty = plane_faces(lo, hi, p, |f| {
            if f.upper {
                if f.value < nh[f.dim] {
                    nh[f.dim] = f.value;
                }
            } else if f.value > nl[f.dim] {
                nl[f.dim] = f.value;
            }
        });
        if empty {
            return None;
        }
    }
    if (0..d).any(|n| {
        !matches!(
            nl[n].partial_cmp(&nh[n]),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        )
    }) {
        return None;
    }
    let shrunk = (0..d).any(|n| nl[n] > lo[n] || nh[n] < hi[n]);
    lo.copy_from_slice(&nl[..d]);
    hi.copy_from_slice(&nh[..d]);
    Some(shrunk)
}

// ---------------------------------------------------------------- split

/// The coordinate to bisect, under `split`.
///
/// * `Widest` / `Gradient`: the historical loops, bit for bit (first maximum
///   of `w_k`, or of `w_k max(|g_k|, 1e-12)` for the nearest objective).
/// * `SbSum` / `SbWr`: see [`LowDimSplit`]; a coordinate is eligible only
///   under the aspect guard (relative width `>= max / SB_ASPECT`, relative to
///   the root widths `w0`); a NaN score is never chosen, and when no eligible
///   coordinate has a number for a score, the coordinate of largest relative
///   width is.
///
/// `grads` is `[k][d]`, one input multiplier per objective; `nearest` indexes
/// the objective closest to closing. Any choice is sound (both halves are
/// kept); the choice only shapes the tree.
pub(super) fn choose_split(
    split: LowDimSplit,
    l: &[f64],
    h: &[f64],
    w0: &[f64],
    grads: &[f64],
    nearest: usize,
) -> usize {
    let d = l.len();
    match split {
        LowDimSplit::Widest | LowDimSplit::Gradient => {
            let grad = grads.get(nearest * d..(nearest + 1) * d);
            let (mut best, mut score) = (0usize, f64::NEG_INFINITY);
            for k in 0..d {
                let w = h[k] - l[k];
                let sc = if split == LowDimSplit::Gradient {
                    w * grad.map_or(0.0, |g| g[k].abs()).max(1e-12)
                } else {
                    w
                };
                if sc > score {
                    score = sc;
                    best = k;
                }
            }
            best
        }
        LowDimSplit::SbSum | LowDimSplit::SbWr => smart_split(split, l, h, w0, grads, nearest),
    }
}

/// `SbSum` / `SbWr` (see [`choose_split`]).
fn smart_split(
    split: LowDimSplit,
    l: &[f64],
    h: &[f64],
    w0: &[f64],
    grads: &[f64],
    nearest: usize,
) -> usize {
    let d = l.len().min(MAX_INPUT_DIM);
    let mut w = [0.0f64; MAX_INPUT_DIM];
    let mut rw = [0.0f64; MAX_INPUT_DIM];
    // `rel_widest`: the first coordinate of largest RELATIVE width — always
    // eligible under the guard, and the fallback when no eligible score is a
    // number.
    let (mut rel_widest, mut rw_max) = (0usize, 0.0f64);
    for k in 0..d {
        w[k] = h[k] - l[k];
        let base = w0.get(k).copied().unwrap_or(0.0);
        rw[k] = if base > 0.0 && w[k].is_finite() && w[k] > 0.0 {
            w[k] / base
        } else {
            0.0
        };
        if rw[k] > rw_max {
            rw_max = rw[k];
            rel_widest = k;
        }
    }
    let mut score = [f64::NEG_INFINITY; MAX_INPUT_DIM];
    if split == LowDimSplit::SbSum {
        // abc branching_heuristics.py sb_sum: sum over the clause's rows of
        // clamp(|lA|, min=thresh) * w / 2, plus touch_zero_score * w when the
        // interval touches 0.
        let n_obj = grads.len().checked_div(d).unwrap_or(0);
        for k in 0..d {
            let mut s = 0.0f64;
            for i in 0..n_obj {
                s += grads[i * d + k].abs().max(SB_COEFF) * (w[k] / 2.0);
            }
            if l[k] == 0.0 || h[k] == 0.0 {
                s += SB_TOUCH_ZERO * w[k];
            }
            score[k] = s;
        }
    } else {
        // The numpy gate's `sbwr`: |lA| w of the nearest objective,
        // max-normalised, plus the max-normalised relative width.
        let g = grads.get(nearest * d..(nearest + 1) * d);
        let mut a = [0.0f64; MAX_INPUT_DIM];
        let mut amax = 0.0f64;
        for k in 0..d {
            a[k] = g.map_or(0.0, |g| g[k].abs()) * w[k];
            amax = amax.max(a[k]);
        }
        for k in 0..d {
            let rel = if rw_max > 0.0 { rw[k] / rw_max } else { 0.0 };
            score[k] = a[k] / amax.max(1e-300) + rel;
        }
    }
    let (mut best, mut best_score) = (rel_widest, f64::NEG_INFINITY);
    for k in 0..d {
        if rw[k] >= rw_max / SB_ASPECT && score[k] > best_score {
            best = k;
            best_score = score[k];
        }
    }
    best
}

// ---------------------------------------------------------------- waves

/// One domain's bounds, from its chunk's forward pass: the lower bound and
/// the split gradient of every objective and — when the clip is armed — the
/// PLANE of every backward pass that bounded it.
pub(super) struct DomainBound {
    /// Rigorous lower bound per objective.
    pub(super) lbs: Vec<f64>,
    /// Input multiplier per objective, `[k][d]` (the best pass's).
    pub(super) grads: Vec<f64>,
    /// `(objective index, a, c)` with `a . x + c <= objectives[index] . y(x)`
    /// on this domain's box: each backward pass's OWN plane (never derived
    /// from `lbs`), finite, not poisoned. Empty unless requested.
    pub(super) planes: Vec<(usize, Vec<f64>, f64)>,
    /// Joint sites whose joint pass bounded this domain.
    pub(super) joint_hits: u32,
    /// #lowdim-bundle (2): what the boundary alpha did on this domain.
    pub(super) alpha: AlphaNote,
}

/// #lowdim-bundle (2): one domain's boundary-alpha engagement (counters only;
/// nothing reads it back into a bound).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct AlphaNote {
    /// In the band for at least one objective (alpha passes ran for it).
    pub(super) ran: bool,
    /// Objectives whose bound an alpha pass raised.
    pub(super) improved: u32,
    /// No conjunct closed on the bounds without alpha; one does with it.
    pub(super) rescued: bool,
}

/// #lowdim-bundle (2): the boundary alpha's per-wave arming: iteration count
/// and each objective's threshold (the band is measured against it).
#[derive(Debug, Clone, Copy)]
pub(super) struct AlphaSpec<'a> {
    pub(super) iters: usize,
    pub(super) thrs: &'a [f64],
}

/// Bound one chunk of domains: one forward pass, then every objective's
/// backward passes (`objective_bounds_passes`). With `planes`, each pass's
/// plane is kept per domain. The bounds are exactly those
/// `bound_batch_multi` publishes (it calls this with `planes = false` and
/// `alpha = None`).
///
/// #lowdim-bundle (2): with `alpha`, every objective whose threshold is known
/// also runs `alpha::refine`; a domain's bound becomes the maximum of its
/// bound and the best alpha pass's (both rigorous), its split gradient the
/// winner's, and — with `planes` — the best alpha pass's own plane joins its
/// clip planes.
pub(super) fn bound_chunk(
    prog: &Program,
    ch: &Chunk<'_>,
    objectives: &[Vec<f64>],
    planes: bool,
    alpha_spec: Option<&AlphaSpec<'_>>,
) -> Vec<DomainBound> {
    let (m, d) = (ch.m, ch.d);
    let one = [prog];
    let fam = Fam::single(&one);
    let (vals, poison) = forward(fam, ch);
    let mut per: Vec<(Vec<f64>, Vec<f64>, Vec<Pass>)> = objectives
        .iter()
        .map(|objective| {
            objective_bounds_passes(
                fam,
                &vals,
                ch,
                prog.seed,
                &vec![&objective[..]; m],
                &poison,
                planes,
            )
        })
        .collect();
    let mut notes = vec![AlphaNote::default(); m];
    if let Some(spec) = alpha_spec.filter(|s| s.iters > 0) {
        let base_closes: Vec<bool> = (0..m)
            .map(|b| {
                per.iter()
                    .zip(spec.thrs)
                    .any(|((lb, _, _), thr)| lb[b] > *thr)
            })
            .collect();
        for (i, (lb, grad, passes)) in per.iter_mut().enumerate() {
            let Some(&thr) = spec.thrs.get(i) else {
                continue;
            };
            let rows = vec![&objectives[i][..]; m];
            let Some(res) = alpha::refine(
                fam, &vals, ch, prog.seed, &rows, &poison, lb, thr, spec.iters,
            ) else {
                continue;
            };
            for b in 0..m {
                notes[b].ran |= res.ran[b];
                if res.pass.lb[b] > lb[b] {
                    lb[b] = res.pass.lb[b];
                    grad[b * d..(b + 1) * d].copy_from_slice(&res.pass.grad[b * d..(b + 1) * d]);
                    notes[b].improved += 1;
                }
            }
            if planes {
                passes.push(res.pass);
            }
        }
        for (b, note) in notes.iter_mut().enumerate() {
            let closes = per
                .iter()
                .zip(spec.thrs)
                .any(|((lb, _, _), thr)| lb[b] > *thr);
            note.rescued = closes && !base_closes[b];
        }
    }
    (0..m)
        .map(|b| {
            let lbs = per.iter().map(|(lb, _, _)| lb[b]).collect();
            let grads = per
                .iter()
                .flat_map(|(_, g, _)| g[b * d..(b + 1) * d].iter().copied())
                .collect();
            let mut kept = Vec::new();
            if planes {
                for (i, (_, _, passes)) in per.iter().enumerate() {
                    for pass in passes {
                        if pass.plane_ok(b, d) {
                            kept.push((i, pass.grad[b * d..(b + 1) * d].to_vec(), pass.pconst[b]));
                        }
                    }
                }
            }
            let joint_hits = prog
                .sites
                .iter()
                .filter(|site| {
                    vals.get(site.node)
                        .and_then(Option::as_ref)
                        .is_some_and(|v| v.joint_ok.get(b).copied().unwrap_or(false))
                })
                .count();
            DomainBound {
                lbs,
                grads,
                planes: kept,
                joint_hits: u32::try_from(joint_hits).unwrap_or(u32::MAX),
                alpha: notes[b],
            }
        })
        .collect()
}

/// What one domain of a wave became.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Fate {
    /// A bound came back NaN or with the wrong length (the lane stops).
    NonFinite,
    /// Some conjunct's rigorous lower bound exceeds its threshold.
    Closed,
    /// The relaxed clip, with the domain's own planes, emptied the box.
    ClipClosed,
    /// The (clipped) box's centre violates every conjunct by the candidate
    /// margin on the lowered program: a likely genuine violation.
    Candidate(Vec<f64>),
    /// The split coordinate is narrower than `min_width`, or its midpoint is
    /// not strictly inside (adjacent floats).
    Resolution,
    /// The clip shrank the box so far that it cannot be bisected: bound the
    /// clipped box `[lo, hi]` again as it is (its bounds tighten with the
    /// box). Only a box the clip STRICTLY shrank is requeued, and each
    /// requeue moves a face by at least one ulp, so this terminates; an
    /// unshrunk unsplittable box is `Resolution`, as before.
    Requeue { lo: Vec<f64>, hi: Vec<f64> },
    /// Bisect the (clipped) box `[lo, hi]` at `mid` along `dim`.
    Split {
        lo: Vec<f64>,
        hi: Vec<f64>,
        dim: usize,
        mid: f64,
        /// The clip shrank the box.
        shrunk: bool,
    },
}

/// Everything a wave's domains share.
pub(super) struct Wave<'a> {
    pub(super) prog: &'a Program,
    pub(super) objs: &'a [Vec<f64>],
    pub(super) thrs: &'a [f64],
    pub(super) cfg: &'a LowDimConfig,
    /// Root widths (the split rules' aspect guard).
    pub(super) w0: &'a [f64],
    /// Domains per chunk (already clamped by the caller's memory cap).
    pub(super) chunk: usize,
}

/// What a wave reports about one domain besides its [`Fate`] (counters and
/// the leaf-attack ranking only; nothing reads it back into a bound).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DomainNote {
    /// Joint sites whose joint pass bounded the domain.
    pub(super) joint_hits: u32,
    /// #lowdim-bundle (2): the boundary alpha's engagement.
    pub(super) alpha: AlphaNote,
    /// #lowdim-bundle (3): `max_i (lb_i - t_i)`, how far the conjunct
    /// closest to closing still is (NaN for a malformed bound). The leaf
    /// attack ranks open leaves by it, most negative first.
    pub(super) margin: f64,
    /// M8a: how the untrusted wave search treated the domain (`None` on the
    /// legacy bounder). Counters only.
    pub(super) claim: super::bbe::ClaimMark,
}

impl Wave<'_> {
    /// Decide every domain of a wave (`lo` / `hi` are `[count][d]`): per
    /// domain its [`Fate`] and its [`DomainNote`], in domain order.
    ///
    /// Every mode cuts the wave into the SAME chunks (a domain's bounds may
    /// depend on its chunk-mates through the chunk-wide refinement choice,
    /// so the partition is part of the result) and decides each domain with
    /// [`fate`]; rayon's ordered collect keeps domain order. Memory per
    /// in-flight chunk is one chunk's forward values (bounded by the caller's
    /// `CHUNK_BYTES` clamp); the wave's result is at most two `d`-vectors per
    /// domain.
    ///
    /// #lowdim-bundle (2): with `cfg.alpha_iters > 0` every chunk also runs
    /// the boundary alpha (`bound_chunk`'s `alpha_spec`), a deterministic
    /// function of the chunk, so the modes still agree.
    pub(super) fn run(&self, lo: &[f64], hi: &[f64], count: usize) -> Vec<(Fate, DomainNote)> {
        let d = self.prog.d;
        let chunk = self.chunk.max(1);
        let starts: Vec<usize> = (0..count).step_by(chunk).collect();
        let spec = AlphaSpec {
            iters: self.cfg.alpha_iters,
            thrs: self.thrs,
        };
        let alpha_spec = (self.cfg.alpha_iters > 0).then_some(&spec);
        let bound = |s: usize| -> Vec<DomainBound> {
            let m = chunk.min(count - s);
            let ch = Chunk::new(m, d, &lo[s * d..(s + m) * d], &hi[s * d..(s + m) * d]);
            bound_chunk(self.prog, &ch, self.objs, self.cfg.clip, alpha_spec)
        };
        let decide = |s: usize, bounds: Vec<DomainBound>| -> Vec<(Fate, DomainNote)> {
            bounds
                .into_iter()
                .enumerate()
                .map(|(j, db)| {
                    let b = s + j;
                    let f = fate(self, &lo[b * d..(b + 1) * d], &hi[b * d..(b + 1) * d], &db);
                    let margin = if db.lbs.len() == self.thrs.len() {
                        db.lbs
                            .iter()
                            .zip(self.thrs)
                            .map(|(lb, t)| lb - t)
                            .fold(f64::NEG_INFINITY, f64::max)
                    } else {
                        f64::NAN
                    };
                    let note = DomainNote {
                        joint_hits: db.joint_hits,
                        alpha: db.alpha,
                        margin,
                        claim: super::bbe::ClaimMark::None,
                    };
                    (f, note)
                })
                .collect()
        };
        match self.cfg.parallel {
            LowDimParallel::Bound => {
                let per_chunk: Vec<Vec<DomainBound>> =
                    starts.par_iter().map(|&s| bound(s)).collect();
                starts
                    .iter()
                    .zip(per_chunk)
                    .flat_map(|(&s, bounds)| decide(s, bounds))
                    .collect()
            }
            LowDimParallel::Full => starts
                .par_iter()
                .flat_map_iter(|&s| decide(s, bound(s)))
                .collect(),
            LowDimParallel::Serial => starts.iter().flat_map(|&s| decide(s, bound(s))).collect(),
        }
    }
}

/// One domain's fate, from its own box `[l, h]` and its own bounds `db`
/// alone, in the historical order: a malformed bound stops the lane; a
/// conjunct whose lower bound exceeds its threshold closes the domain; the
/// armed clip shrinks the box with the domain's own planes (and closes it
/// when empty); a centre that violates every conjunct is a candidate; else
/// the split rule bisects the (clipped) box — or, when the clip left it
/// unsplittable, requeues it. Without the clip this is the historical
/// decision exactly (`shrunk` is false, so there is no `Requeue`).
pub(super) fn fate(w: &Wave<'_>, l: &[f64], h: &[f64], db: &DomainBound) -> Fate {
    let (thrs, cfg) = (w.thrs, w.cfg);
    if db.lbs.len() != thrs.len() || db.lbs.iter().any(|lb| lb.is_nan()) {
        return Fate::NonFinite;
    }
    // Closed iff SOME conjunct is proved impossible on this leaf. The
    // comparison is the strict `lb > thr` itself, never a subtracted margin.
    if db.lbs.iter().zip(thrs).any(|(lb, thr)| lb > thr) {
        return Fate::Closed;
    }
    let (mut bl, mut bh) = (l.to_vec(), h.to_vec());
    let mut shrunk = false;
    if cfg.clip && !db.planes.is_empty() {
        let planes: Vec<ClipPlane<'_>> = db
            .planes
            .iter()
            .filter_map(|(i, a, c)| {
                thrs.get(*i).map(|&t| ClipPlane {
                    a: a.as_slice(),
                    c: *c,
                    t,
                })
            })
            .collect();
        match clip_box(&mut bl, &mut bh, &planes) {
            None => return Fate::ClipClosed,
            Some(s) => shrunk = s,
        }
    }
    open_fate(w, bl, bh, shrunk, &db.grads, &db.lbs)
}

/// The OPEN part of [`fate`] (extracted verbatim, M8a): the (clipped) box
/// `[bl, bh]` is not closed; its centre is a violation candidate when it
/// violates every conjunct by the candidate margin on the lowered program;
/// else the split rule bisects it along the conjunct closest to closing
/// (largest `lbs[i] - thr_i`, steered by `grads`) — or, when unsplittable,
/// it is requeued (`shrunk`) or a Resolution stop. It NEVER closes: `lbs` and
/// `grads` only choose where to split (any split of a box covers it), so the
/// M8a search may pass its untrusted hints here.
pub(super) fn open_fate(
    w: &Wave<'_>,
    bl: Vec<f64>,
    bh: Vec<f64>,
    shrunk: bool,
    grads: &[f64],
    lbs: &[f64],
) -> Fate {
    let (prog, objs, thrs, cfg) = (w.prog, w.objs, w.thrs, w.cfg);
    let centre: Vec<f64> = bl.iter().zip(&bh).map(|(a, c)| 0.5 * (a + c)).collect();
    if objs
        .iter()
        .zip(thrs)
        .all(|(o, thr)| point_objective(prog, &centre, o) <= thr - candidate_margin(*thr))
    {
        return Fate::Candidate(centre);
    }
    // Split along the conjunct closest to closing (largest lb - thr).
    let nearest = first_argmax(lbs.iter().zip(thrs).map(|(lb, thr)| lb - thr));
    let dim = choose_split(cfg.split, &bl, &bh, w.w0, grads, nearest);
    let mid = centre[dim];
    if bh[dim] - bl[dim] < cfg.min_width || !(mid > bl[dim] && mid < bh[dim]) {
        return if shrunk {
            Fate::Requeue { lo: bl, hi: bh }
        } else {
            Fate::Resolution
        };
    }
    Fate::Split {
        lo: bl,
        hi: bh,
        dim,
        mid,
        shrunk,
    }
}
