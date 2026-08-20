// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #clip-and-verify for the margin-row lane.
//!
//! # Why this exists
//!
//! cifar100 rows fail by FRONTIER EXPLOSION, not by bound quality. Measured
//! 2026-08-18 at the official 100 s budget:
//!
//! ```text
//! idx_6659  PROVES in 189 expansions      idx_8600  TIMES OUT at 729
//! exp 151  -0.03261 depth 12 open  26     exp 121  -0.03718 depth 12 open  18
//! exp 185  -0.00432 depth 14 open   4     exp 359  -0.02269 depth 24 open 156
//!                                 DRAINS  exp 719  -0.01288 depth 30 open 415
//! ```
//!
//! Both rows improve their bound at the SAME rate, and the failing one even
//! STARTS BETTER (-0.2324 against -0.4198). The difference is that children
//! CLOSE on one and not the other. The margin-row lane had no split-derived
//! tightening at all: a split fixes ONE neuron's gate and nothing else in the
//! domain moves, so a child barely improves on its parent, the queue grows
//! faster than it drains, and the row times out 0.0129 from the threshold with
//! 415 domains outstanding.
//!
//! Clip-and-Verify (Zhou et al., NeurIPS 2025) is the mechanism that makes one
//! split tighten the WHOLE domain. alpha-beta-CROWN enables it on cifar100 and
//! tinyimagenet and on no other benchmark it ships.
//!
//! # The mechanism, and why it is sound
//!
//! Let `B` be the root input box and `S ⊆ B` the region carved out by a domain's
//! split history. For a split "neuron `i` of layer `k` is INACTIVE"
//! (`z_k,i(x) <= 0` for all `x ∈ S`), the ROOT relaxation gives
//! `lA_k,i · x + lbias_k,i <= z_k,i(x)` for all `x ∈ B`, hence
//!
//! ```text
//!     x ∈ S  ==>  lA_k,i · x + lbias_k,i <= 0
//! ```
//!
//! and symmetrically an ACTIVE split gives `uA_k,i · x + ubias_k,i >= 0`. Each
//! emitted halfspace is a NECESSARY condition of the split, never a sufficient
//! one, so the feasible set `F = B ∩ {halfspaces}` satisfies `S ⊆ F`. Minimising
//! over the LARGER set `F` therefore lower-bounds the minimum over `S` — the
//! safe direction. Over-approximating the domain is what keeps this sound; an
//! under-approximation would be a false-`unsat` generator.
//!
//! The LP is solved by its Lagrangian dual in closed form. By WEAK DUALITY every
//! `beta >= 0` yields a valid lower bound, so a suboptimal multiplier costs
//! tightness and never validity — there is no "solver correctness" obligation
//! here beyond `beta >= 0`.
//!
//! # Frame
//!
//! The rows consumed here are INPUT-relative (`n_in` columns), retained by
//! `root.rs` as `LayerGates::clip_rows` at the exact point `concretize_box`
//! consumes them. They are NOT the tier-0 capture, which is incoming
//! coefficients on the relu OUTPUT. Reading one as the other is "wrong in both
//! axes at once" and is precisely what quarantined the sequential clip
//! (`beta_crown/engine/domain/clip.rs`).

use super::root::{LayerGates, RootGates};

/// One halfspace `a · x + d <= 0` over the network input.
#[derive(Debug, Clone)]
pub struct Halfspace {
    /// Coefficients over the input, length `n_in`.
    pub a: Vec<f64>,
    /// Constant term.
    pub d: f64,
}

/// Build the halfspaces implied by a domain's trunk split history.
///
/// `splits` are `(layer_index, position_in_unst, sign)` exactly as
/// `DomainEntry.trunk` carries them. `sign < 0` means the neuron was fixed
/// INACTIVE, `sign > 0` ACTIVE.
///
/// Returns an empty vector when no layer has retained rows, which makes every
/// downstream step a no-op rather than an error.
///
/// # Soundness
///
/// Each row is a NECESSARY condition of its split (see the module docs), so the
/// constrained set over-approximates the true subdomain. Adding FEWER halfspaces
/// than available is always safe — it only loosens.
#[must_use]
pub fn halfspaces_for_splits(root: &RootGates, splits: &[(usize, usize, i8)]) -> Vec<Halfspace> {
    let mut out = Vec::new();
    for &(li, pos, sign) in splits {
        let Some(layer) = root.layers.get(li) else {
            continue;
        };
        let Some(rows) = layer.clip_rows.as_ref() else {
            continue;
        };
        if pos >= rows.m.nrows() || rows.m.ncols() != root.mid.len() {
            continue;
        }
        // THE LINE, and the slack it must pay.
        //
        // `m -/+ d` ARE the DeepPoly lower/upper lines — the lane retains
        // exactly `m - d` / `m + d` as `RetainedLayer::a_l` / `a_u` a few lines
        // from where these rows are kept, and `concretize_box` minimises
        // `(m - d) . x + (bm - bd)` over the box to produce `l`.
        //
        // But it does not stop there: it subtracts a certified slack
        // (`gam * (tabs + |bl| + |bu|)`, plus a per-neuron f32 error term on the
        // f32 fast path) that GROWS WITH DEPTH. So the line bounds `z`
        // pointwise only up to that slack, and a halfspace that ignores it cuts
        // into the true subdomain. `sl_lo` / `sl_up` carry it, recovered at
        // retention as the gap between the line's own box-minimum and the bound
        // the lane published — exact by construction, no re-derivation.
        let n_in = root.mid.len();
        let mut a = Vec::with_capacity(n_in);
        let (bm, bd) = (rows.bm[pos], rows.bd[pos].abs());
        // Sign convention MUST match `domain_gates`, which reads `d > 0` as
        // ACTIVE and everything else as INACTIVE.
        let d = if sign <= 0 {
            // INACTIVE: `lower_line(x) - sl_lo <= z(x) <= 0`.
            for j in 0..n_in {
                a.push(rows.m[[pos, j]] - rows.d[[pos, j]]);
            }
            bm - bd - rows.sl_lo[pos]
        } else {
            // ACTIVE: `upper_line(x) + sl_up >= z(x) >= 0`; negate the whole
            // thing into `a . x + d <= 0` form.
            for j in 0..n_in {
                a.push(-(rows.m[[pos, j]] + rows.d[[pos, j]]));
            }
            -(bm + bd + rows.sl_up[pos])
        };
        if !d.is_finite() || a.iter().any(|v| !v.is_finite()) {
            continue;
        }
        out.push(Halfspace { a, d });
    }
    out
}

/// Lagrangian value `L(beta)` for a single halfspace: the minimum of
/// `obj · x + beta * (a · x + d)` over the box `|x - x0| <= eps`.
///
/// Valid lower bound on the CONSTRAINED minimum for every `beta >= 0` by weak
/// duality, which is the whole soundness argument.
fn lagrangian(obj: &[f64], a: &[f64], d: f64, x0: &[f64], eps: &[f64], beta: f64) -> f64 {
    let mut v = beta * d;
    for j in 0..obj.len() {
        let c = beta.mul_add(a[j], obj[j]);
        v += c.mul_add(x0[j], -(c.abs() * eps[j]));
    }
    v
}

/// Exact closed-form dual for ONE halfspace.
///
/// `L(beta)` is piecewise linear and concave in `beta`, with breakpoints where
/// some `obj_j + beta * a_j` changes sign, i.e. at `beta = -obj_j / a_j`. The
/// maximum is attained at a breakpoint or at `beta = 0`, so evaluating the
/// candidate set is exact — no iteration, no tolerance.
///
/// Returns the unconstrained box minimum when the constraint cannot help.
#[must_use]
pub fn dual_lower_one(obj: &[f64], hs: &Halfspace, x0: &[f64], eps: &[f64]) -> f64 {
    let mut best = lagrangian(obj, &hs.a, hs.d, x0, eps, 0.0);
    for j in 0..obj.len() {
        if hs.a[j] == 0.0 {
            continue;
        }
        let beta = -obj[j] / hs.a[j];
        if beta.is_finite() && beta > 0.0 {
            let v = lagrangian(obj, &hs.a, hs.d, x0, eps, beta);
            if v > best {
                best = v;
            }
        }
    }
    best
}

/// Exact maximiser of the single-constraint Lagrangian, by a sorted sweep.
///
/// `L(beta) = beta*d + sum_j f_j(beta)` with `f_j(beta) = c_j*x0_j - |c_j|*eps_j`
/// and `c_j = work_j + beta*a_j`, is piecewise linear and CONCAVE in `beta`,
/// with one kink per coordinate where `c_j` changes sign — at
/// `beta_j = -work_j / a_j`. Crossing that kink changes the total slope by
/// exactly `-2*|a_j|*eps_j` (both signs of `a_j` give the same expression), so
/// walking the kinks in increasing `beta` while carrying the running value and
/// slope finds the maximum EXACTLY.
///
/// # Why not the obvious version
///
/// Evaluating `L` at every candidate breakpoint is `O(n^2)` per constraint. On
/// cifar100 `n_in = 3072`, so that is ~9.4M flops per constraint per output row;
/// with ~30 splits and ~99 rows it would be ~28 GFLOP per expansion against a
/// lane that currently spends ~500 ms on the whole thing. The sweep is
/// `O(n log n)` — the difference between this path being affordable inside a
/// 100 s budget and not being usable at all.
///
/// Returns the maximising `beta >= 0`. `0.0` means the constraint cannot help,
/// which the caller treats as "skip this fold".
fn dual_argmax(
    work: &[f64],
    h: &Halfspace,
    x0: &[f64],
    eps: &[f64],
    scratch: &mut Vec<(f64, f64)>,
) -> f64 {
    let n = work.len();
    // Value and right-slope at beta = 0.
    let mut val = 0.0f64;
    let mut slope = h.d;
    scratch.clear();
    for j in 0..n {
        let (w, a, xa, ea) = (work[j], h.a[j], x0[j], eps[j]);
        val += w.mul_add(xa, -(w.abs() * ea));
        // Side of `c_j` just above beta = 0: the sign of `work_j`, falling back
        // to the sign of `a_j` when `work_j` is exactly zero (kink at 0).
        let positive = if w > 0.0 {
            true
        } else if w < 0.0 {
            false
        } else {
            a > 0.0
        };
        slope += a * if positive { xa - ea } else { xa + ea };
        if a != 0.0 {
            let beta = -w / a;
            if beta.is_finite() && beta > 0.0 {
                // Slope decrement on crossing this kink: `-2|a_j|*eps_j`.
                scratch.push((beta, -2.0 * a.abs() * ea));
            }
        }
    }
    if !val.is_finite() || !slope.is_finite() || slope <= 0.0 {
        return 0.0; // already at the maximum, or degenerate: keep beta = 0
    }
    scratch.sort_unstable_by(|p, q| p.0.total_cmp(&q.0));
    let (mut cur_beta, mut cur_val) = (0.0f64, val);
    let (mut best_beta, mut best_val) = (0.0f64, val);
    for &(beta, dslope) in scratch.iter() {
        if slope <= 0.0 {
            break; // concave: past the peak
        }
        cur_val += slope * (beta - cur_beta);
        cur_beta = beta;
        slope += dslope;
        if cur_val > best_val && cur_val.is_finite() {
            best_val = cur_val;
            best_beta = cur_beta;
        }
    }
    // If the slope never turns over, `L` is unbounded above, which means the
    // primal is INFEASIBLE (the subdomain is empty). That is a real prunable
    // condition, but exploiting it needs its own oracle; refusing here is the
    // conservative reading and costs only tightness.
    if slope > 0.0 {
        return 0.0;
    }
    best_beta
}

/// Lower bound of `obj · x` over the box intersected with ALL halfspaces.
///
/// Greedy coordinate ascent, one constraint at a time, folding each solved
/// multiplier into the objective — the same shape alpha-beta-CROWN uses. Each
/// fold keeps a valid dual point, so the running value stays a valid lower bound
/// throughout; stopping early is always safe.
///
/// Returns `None` on a shape mismatch or a non-finite intermediate, so callers
/// fail closed to their existing bound.
#[must_use]
pub fn dual_lower(obj: &[f64], hs: &[Halfspace], x0: &[f64], eps: &[f64]) -> Option<f64> {
    if obj.len() != x0.len() || obj.len() != eps.len() {
        return None;
    }
    let mut work = obj.to_vec();
    let mut acc_d = 0.0f64;
    let mut scratch: Vec<(f64, f64)> = Vec::with_capacity(obj.len());
    for h in hs {
        if h.a.len() != work.len() {
            return None;
        }
        // Solve for this constraint against the accumulated objective.
        let best_beta = dual_argmax(&work, h, x0, eps, &mut scratch);
        if best_beta > 0.0 {
            for j in 0..work.len() {
                work[j] = best_beta.mul_add(h.a[j], work[j]);
            }
            acc_d += best_beta * h.d;
        }
    }
    let mut v = acc_d;
    for j in 0..work.len() {
        v += work[j].mul_add(x0[j], -(work[j].abs() * eps[j]));
    }
    v.is_finite().then_some(v)
}

/// Range of `a . x + d` over the root box, as `(lo, hi)`.
///
/// THE DIAGNOSTIC that separates "the mechanism does not pay" from "the
/// constraint is vacuous". If `hi <= 0` the entire box already satisfies the
/// halfspace, so it cuts nothing and the dual correctly returns `beta = 0` —
/// a zero result would then say nothing about Clip-and-Verify, only about the
/// line that was emitted for it.
#[must_use]
pub fn halfspace_range(h: &Halfspace, x0: &[f64], eps: &[f64]) -> (f64, f64) {
    let mut ctr = h.d;
    let mut hw = 0.0f64;
    for j in 0..h.a.len().min(x0.len()) {
        ctr += h.a[j] * x0[j];
        hw += h.a[j].abs() * eps[j];
    }
    (ctr - hw, ctr + hw)
}

/// Is any layer carrying retained rows? Cheap guard so callers can skip the
/// whole path when retention is off.
#[must_use]
pub fn any_rows_retained(root: &RootGates) -> bool {
    root.layers
        .iter()
        .any(|l: &LayerGates| l.clip_rows.is_some())
}

/// Non-negative tightening deltas for a set of concretized rows.
///
/// For each column `ri` of `a` (shape `(n_in, r)`), returns
/// `dual_lower(a[:,ri]) - box_min(a[:,ri])` — the improvement the halfspaces buy
/// on the LOWER side, and symmetrically on the upper.
///
/// # Why a DELTA rather than a replacement
///
/// The lane's `concretize` subtracts certified error and penalty terms from the
/// box minimum. Those terms bound the gap between the true function and its
/// linear form and are INDEPENDENT of which `x` attains the minimum, so they
/// remain valid verbatim when the feasible set shrinks. Returning a delta lets
/// the caller keep its existing error accounting untouched:
///
/// ```text
///   ly  = box_min(linear) - err            (existing, sound)
///   ly' = ly + delta = min_F(linear) - err  <=  min_F(true)     (still sound)
/// ```
///
/// # Why the delta is non-negative by construction
///
/// `dual_lower` always evaluates `beta = 0`, whose Lagrangian IS the
/// unconstrained box minimum, and takes the maximum over candidates. So it can
/// never fall below `box_min`, and the delta can never LOOSEN a bound. That is a
/// structural guarantee, not a numerical hope.
///
/// Returns `None` on any shape problem or non-finite value, so the caller keeps
/// its existing box.
#[must_use]
pub fn ybox_deltas(
    a: &ndarray::Array2<f64>,
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
    lower: bool,
) -> Option<Vec<f64>> {
    if hs.is_empty() {
        return None;
    }
    let n_in = a.nrows();
    let r = a.ncols();
    if n_in != x0.len() || n_in != eps.len() {
        return None;
    }
    let mut out = Vec::with_capacity(r);
    let mut obj = vec![0.0f64; n_in];
    for ri in 0..r {
        for (i, o) in obj.iter_mut().enumerate() {
            // Upper side: max c.x = -min (-c).x, so negate and reuse one solver.
            *o = if lower { a[[i, ri]] } else { -a[[i, ri]] };
        }
        let base = {
            let mut v = 0.0f64;
            for i in 0..n_in {
                v += obj[i].mul_add(x0[i], -(obj[i].abs() * eps[i]));
            }
            v
        };
        let tightened = dual_lower(&obj, hs, x0, eps)?;
        // OUTWARD MARGIN. `dual_lower` and `base` accumulate O(n_in) products in
        // plain f64 while the lane concretizes with `next_down`/`certify_up`.
        // The relative error of that accumulation is bounded by ~n*eps
        // (3072 * 2.2e-16 ~= 7e-13), so shaving 1e-9 relative plus an absolute
        // floor keeps the delta a strict UNDER-estimate of the true gap. It
        // costs a part per billion of tightness and removes the last way this
        // path could overstate a bound.
        let raw = tightened - base;
        if !raw.is_finite() || raw < 0.0 {
            // Cannot happen (beta = 0 is always a candidate), but a negative
            // delta would LOOSEN a bound, so refuse the whole set rather than
            // silently widen.
            return None;
        }
        out.push((raw * (1.0 - 1e-9) - 1e-12).max(0.0));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_min(obj: &[f64], x0: &[f64], eps: &[f64]) -> f64 {
        let mut v = 0.0;
        for j in 0..obj.len() {
            v += obj[j].mul_add(x0[j], -(obj[j].abs() * eps[j]));
        }
        v
    }

    #[test]
    fn no_constraints_reproduces_the_box_minimum() {
        let obj = [1.0, -2.0, 0.5];
        let x0 = [0.0, 1.0, -1.0];
        let eps = [1.0, 0.5, 2.0];
        let got = dual_lower(&obj, &[], &x0, &eps).expect("finite");
        assert!((got - box_min(&obj, &x0, &eps)).abs() < 1e-12);
    }

    #[test]
    fn a_constraint_never_loosens_the_bound() {
        // Weak duality: adding a halfspace can only RAISE the minimum, since it
        // shrinks the feasible set. A constraint that made the bound worse would
        // mean the dual is being read with the wrong sign.
        let obj = [1.0, -2.0, 0.5];
        let x0 = [0.0, 1.0, -1.0];
        let eps = [1.0, 0.5, 2.0];
        let base = box_min(&obj, &x0, &eps);
        for a in [
            vec![1.0, 0.0, 0.0],
            vec![-1.0, 1.0, 0.0],
            vec![0.3, -0.7, 1.2],
        ] {
            let hs = [Halfspace { a, d: 0.25 }];
            let got = dual_lower(&obj, &hs, &x0, &eps).expect("finite");
            assert!(
                got >= base - 1e-9,
                "constraint loosened the bound: {got} < {base}"
            );
        }
    }

    #[test]
    fn dual_is_a_valid_lower_bound_against_brute_force() {
        // The dual must never EXCEED the true constrained minimum, or it is a
        // false-`unsat` generator. Checked against a dense scan of the box.
        let obj = [1.0, -1.5];
        let x0 = [0.0, 0.0];
        let eps = [1.0, 1.0];
        let hs = [Halfspace {
            a: vec![1.0, 1.0],
            d: -0.5,
        }];
        let dual = dual_lower(&obj, &hs, &x0, &eps).expect("finite");
        let n = 400;
        let mut true_min = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                #[allow(clippy::cast_precision_loss)]
                let x = [
                    (i as f64 / n as f64).mul_add(2.0, -1.0),
                    (j as f64 / n as f64).mul_add(2.0, -1.0),
                ];
                if hs[0].a[0].mul_add(x[0], hs[0].a[1] * x[1]) + hs[0].d <= 0.0 {
                    let v = obj[0].mul_add(x[0], obj[1] * x[1]);
                    if v < true_min {
                        true_min = v;
                    }
                }
            }
        }
        assert!(
            dual <= true_min + 1e-6,
            "dual {dual} EXCEEDS the true constrained minimum {true_min} — unsound"
        );
    }

    #[test]
    fn single_constraint_face_matches_the_greedy_fold() {
        let obj = [0.7, -1.3, 2.0];
        let x0 = [0.1, -0.2, 0.3];
        let eps = [0.5, 1.0, 0.25];
        let hs = Halfspace {
            a: vec![1.0, -0.5, 0.25],
            d: 0.1,
        };
        let one = dual_lower_one(&obj, &hs, &x0, &eps);
        let many = dual_lower(&obj, std::slice::from_ref(&hs), &x0, &eps).expect("finite");
        assert!((one - many).abs() < 1e-12, "{one} vs {many}");
    }

    /// The `O(n log n)` sweep must agree with the `O(n^2)` reference EXACTLY.
    ///
    /// `dual_lower_one` evaluates `L` at every candidate breakpoint, which is
    /// obviously correct and obviously too slow for `n_in = 3072`.
    /// `dual_lower`'s sweep replaces it by carrying the running value and slope
    /// across sorted kinks. That rewrite is where an off-by-one in the slope
    /// decrement, or a mishandled `work_j == 0` kink, would hide — and it would
    /// hide as a bound that is too HIGH, which is unsound rather than merely
    /// loose. 3 dimensions cannot exercise it; this walks 200 pseudo-random
    /// problems at 64 dimensions, including exact-zero coefficients.
    #[test]
    fn the_sweep_matches_the_exhaustive_reference() {
        let mut seed = 0x2026_0818_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let v = ((seed >> 11) as f64) / ((1u64 << 53) as f64);
            v.mul_add(2.0, -1.0)
        };
        let n = 64;
        for case in 0..200 {
            let obj: Vec<f64> = (0..n)
                .map(|j| {
                    let v = next();
                    // Exact zeros in both the objective and the constraint: the
                    // kink-at-zero and skip-this-coordinate branches.
                    if (j + case) % 7 == 0 {
                        0.0
                    } else {
                        v
                    }
                })
                .collect();
            let a: Vec<f64> = (0..n)
                .map(|j| if (j + case) % 5 == 0 { 0.0 } else { next() })
                .collect();
            let x0: Vec<f64> = (0..n).map(|_| next()).collect();
            let eps: Vec<f64> = (0..n).map(|_| next().abs()).collect();
            let hs = Halfspace { a, d: next() };
            let reference = dual_lower_one(&obj, &hs, &x0, &eps);
            let sweep = dual_lower(&obj, std::slice::from_ref(&hs), &x0, &eps).expect("finite");
            let scale = reference.abs().max(1.0);
            assert!(
                (reference - sweep).abs() <= 1e-9 * scale,
                "case {case}: reference {reference} vs sweep {sweep}"
            );
        }
    }

    #[test]
    fn shape_mismatch_fails_closed() {
        assert!(dual_lower(&[1.0, 2.0], &[], &[0.0], &[1.0, 1.0]).is_none());
        let hs = [Halfspace {
            a: vec![1.0],
            d: 0.0,
        }];
        assert!(dual_lower(&[1.0, 2.0], &hs, &[0.0, 0.0], &[1.0, 1.0]).is_none());
    }
}

/// One intermediate neuron's bounds, tightened over `box ∩ halfspaces`.
#[derive(Debug, Clone, Copy)]
pub struct Tightened {
    /// Position within the layer's `unst` list.
    pub pos: usize,
    /// Neuron index within the layer.
    pub idx: usize,
    /// Tightened lower bound (never below the stored one).
    pub l: f64,
    /// Tightened upper bound (never above the stored one).
    pub u: f64,
}

/// Re-concretize a layer's unstable neurons over `box ∩ halfspaces`.
///
/// THE ACTUAL MECHANISM. Tightening the OUTPUT y-box changes nothing that
/// propagates — measured: 96 of 100 rows moved, `sum_delta = 0.36`, and the
/// search trajectory was bit-identical. What makes one split tighten the WHOLE
/// domain is narrowing the INTERMEDIATE neurons: a tighter `[l, u]` shrinks that
/// neuron's ReLU relaxation triangle, which changes its `(alpha, s, c)` gates,
/// which tightens every backward pass through it. alpha-beta-CROWN names this
/// exactly — `bab.clip.interm_domain`, `interm_topk: 20`.
///
/// Only the `topk` widest unstable neurons are touched, since the cost is two
/// dual solves each and the relaxation gap `-l*u/(u-l)` is what the tightening
/// actually buys back.
///
/// # Soundness
///
/// Each returned bound is `max`/`min`-ed against the stored one, so this can
/// only SHRINK an interval. The dual is a valid bound for any `beta >= 0`
/// (weak duality), and the certified slack `sl_lo`/`sl_up` is paid exactly as
/// the root paid it, so a tightened bound is valid wherever the stored one was.
/// Returning FEWER neurons is always safe.
#[must_use]
pub fn tighten_intermediate(
    layer: &LayerGates,
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
    topk: usize,
) -> (Vec<Tightened>, usize) {
    let Some(rows) = layer.clip_rows.as_ref() else {
        return (Vec::new(), 0);
    };
    if hs.is_empty() || topk == 0 || rows.m.ncols() != x0.len() {
        return (Vec::new(), 0);
    }
    // Widest-relaxation first: `-l*u/(u-l)` is the triangle's height, i.e. the
    // slack a tightening here removes from every path through this neuron.
    let mut order: Vec<(f64, usize)> = layer
        .unst
        .iter()
        .enumerate()
        .filter_map(|(pos, &i)| {
            let (l, u) = (layer.l[i], layer.u[i]);
            let w = u - l;
            (w > 0.0 && l < 0.0 && u > 0.0).then(|| (-l * u / w, pos))
        })
        .collect();
    order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    order.truncate(topk);

    let n_in = x0.len();
    let mut out = Vec::with_capacity(order.len());
    let mut crossed = 0usize;
    let mut lo_obj = vec![0.0f64; n_in];
    let mut up_obj = vec![0.0f64; n_in];
    for (_, pos) in order {
        let idx = layer.unst[pos];
        for j in 0..n_in {
            let (m, d) = (rows.m[[pos, j]], rows.d[[pos, j]]);
            lo_obj[j] = m - d;
            up_obj[j] = -(m + d); // max of the upper line = -min of its negation
        }
        let (bm, bd) = (rows.bm[pos], rows.bd[pos].abs());
        // Same construction the halfspaces use: the line's constrained extremum,
        // then the certified slack the root already paid for this neuron.
        let l_new = dual_lower(&lo_obj, hs, x0, eps)
            .map(|v| v + (bm - bd) - rows.sl_lo[pos])
            .filter(|v| v.is_finite())
            .map_or(layer.l[idx], |v| v.max(layer.l[idx]));
        let u_new = dual_lower(&up_obj, hs, x0, eps)
            .map(|v| -(v - (bm + bd)) + rows.sl_up[pos])
            .filter(|v| v.is_finite())
            .map_or(layer.u[idx], |v| v.min(layer.u[idx]));
        // Shrink-only. CROSSED bounds are not an error to discard — they are
        // the VERIFY half of Clip-and-Verify: `l_new` under-estimates the
        // constrained min and `u_new` over-estimates the constrained max, so
        // `l_new > u_new` is a Farkas certificate that `box ∩ halfspaces` is
        // EMPTY, and the caller can discharge the whole domain.
        if l_new > u_new {
            crossed += 1;
        } else if l_new > layer.l[idx] || u_new < layer.u[idx] {
            out.push(Tightened {
                pos,
                idx,
                l: l_new,
                u: u_new,
            });
        }
    }
    (out, crossed)
}

/// #clip-and-verify VERIFY telemetry: domains discharged by emptiness
/// certificates. Non-empty output is the precondition for believing any
/// negative measurement of the verify half.
pub fn report_infeasible(layer: usize, crossed: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(256) {
        eprintln!("[clip-verify] infeasible domain #{n}: layer={layer} crossed_rows={crossed}");
    }
}

/// #clip-and-verify intermediate-tightening telemetry. Non-empty output is the
/// precondition for believing any negative result from this path.
pub fn report_interm(layer: usize, touched: usize, found: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static TOTAL: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let t = TOTAL.fetch_add(touched, Ordering::Relaxed) + touched;
    if n < 6 || n.is_multiple_of(256) {
        eprintln!("[clip-interm] call={n} layer={layer} touched={touched} found={found} total={t}");
    }
}
