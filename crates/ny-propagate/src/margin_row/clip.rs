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
//! split tighten the WHOLE domain. alpha-beta-CROWN enables it on cifar100,
//! tinyimagenet, `cora.yaml` (the `clip_in_alpha_crown` sub-key) and
//! `relusplitter.yaml`: `grep -rl clip_interm_domain exp_configs/vnncomp25/`
//! returns FOUR configs.
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

use super::root::{ClipRows, LayerGates, RootGates};
use super::rounding::{gamma_n, next_down, next_up};

#[path = "clip_cooperative.rs"]
mod cooperative;
pub(in crate::margin_row) use cooperative::{
    dual_lower_outward_request, dual_lower_outward_with_checkpoint,
};

#[cfg(test)]
#[path = "clip_dual_rounding_tests.rs"]
mod dual_rounding_tests;

/// Error-free addition in the qualified binary64 environment. Refusing a
/// non-finite intermediate also covers exceptional TwoSum reconstruction.
#[inline]
fn clip_two_sum(a: f64, b: f64) -> Option<(f64, f64)> {
    let s = a + b;
    let bv = s - a;
    let av = s - bv;
    let br = b - bv;
    let ar = a - av;
    let e = ar + br;
    [a, b, s, bv, av, br, ar, e]
        .iter()
        .all(|v| v.is_finite())
        .then_some((s, e))
}

#[inline]
fn clip_add_down(a: f64, b: f64) -> Option<f64> {
    let (s, e) = clip_two_sum(a, b)?;
    let value = if e < 0.0 { next_down(s) } else { s };
    value.is_finite().then_some(value)
}

#[inline]
fn clip_add_up(a: f64, b: f64) -> Option<f64> {
    let (s, e) = clip_two_sum(a, b)?;
    let value = if e > 0.0 { next_up(s) } else { s };
    value.is_finite().then_some(value)
}

/// A product needs an absolute gradual-underflow allowance as well as relative
/// rounding. One adjacent float encloses every finite nearest product, even
/// when its exact value is half the smallest subnormal. An FMA residual of zero
/// is insufficient to identify that case. Only algebraically exact zero/one
/// products skip widening.
#[inline]
fn clip_mul_bounds(a: f64, b: f64) -> Option<(f64, f64)> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let p = a * b;
    if !p.is_finite() {
        return None;
    }
    if a == 0.0 || b == 0.0 || a.abs() == 1.0 || b.abs() == 1.0 {
        return Some((p, p));
    }
    let (mut lo, mut hi) = (next_down(p), next_up(p));
    if a.is_sign_negative() == b.is_sign_negative() {
        lo = lo.max(0.0);
    } else {
        hi = hi.min(0.0);
    }
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

fn legacy_rows_shape(rows: &ClipRows, count: usize, n_in: usize) -> bool {
    rows.m.dim() == (count, n_in)
        && rows.d.dim() == rows.m.dim()
        && rows.bm.len() == count
        && rows.bd.len() == count
        && rows.sl_lo.len() == count
        && rows.sl_up.len() == count
}

/// Reconstruct a legacy line without losing the signed addition residual.
/// `objective` is the lower line, or the NEGATION of the upper line. Its
/// returned bias remains in the original line's sign convention. The existing
/// certified producer slack is kept in full, then construction dust is added
/// pointwise over the original declared input magnitude.
fn legacy_row_objective(
    rows: &ClipRows,
    pos: usize,
    lower: bool,
    objective: &mut [f64],
    mut input_magnitude: impl FnMut(usize) -> Option<f64>,
) -> Option<(f64, f64)> {
    if pos >= rows.m.nrows() || !legacy_rows_shape(rows, rows.m.nrows(), objective.len()) {
        return None;
    }
    let slack = if lower {
        rows.sl_lo[pos]
    } else {
        rows.sl_up[pos]
    };
    if !slack.is_finite() || slack < 0.0 {
        return None;
    }
    let (bias, bias_error) = clip_two_sum(
        rows.bm[pos],
        if lower {
            -rows.bd[pos].abs()
        } else {
            rows.bd[pos].abs()
        },
    )?;
    let mut error = clip_add_up(slack, bias_error.abs())?;
    for (j, coefficient) in objective.iter_mut().enumerate() {
        let (center, residual) = clip_two_sum(
            rows.m[[pos, j]],
            if lower {
                -rows.d[[pos, j]]
            } else {
                rows.d[[pos, j]]
            },
        )?;
        *coefficient = if lower { center } else { -center };
        let magnitude = input_magnitude(j)?;
        if !magnitude.is_finite() || magnitude < 0.0 {
            return None;
        }
        error = clip_add_up(error, clip_mul_bounds(residual.abs(), magnitude)?.1)?;
    }
    Some((bias, error))
}

/// One halfspace `a · x + d <= 0` over the network input.
#[derive(Debug, Clone)]
pub struct Halfspace {
    /// Coefficients over the input, length `n_in`.
    pub a: Vec<f64>,
    /// Constant term.
    pub d: f64,
}

/// #slack-magnitude: how much of each Clip-and-Verify halfspace is surrendered
/// to the certified slack.
///
/// The slack includes the line's supplied certification error (rounding +
/// interval widening), retained by `root.rs::retain_clip_rows`, and the
/// additional reconstruction error paid by this consumer. It
/// is NOT a tunable margin: shrinking it directly would make the halfspace claim
/// the line bounds `z` more tightly than certified — a false-`unsat` generator.
/// It shrinks only if the LINES carry less error, i.e. under higher-precision
/// arithmetic.
///
/// So `slack / |d|` is the fraction of each constraint that exact arithmetic
/// would hand back: bound gain at ZERO search cost, the one profile that breaks
/// the measured gain-vs-wall coupling
/// (`docs/GAIN_AND_WALL_ARE_COUPLED_2026-08-31.md`). This probe measures whether
/// that fraction is worth chasing. Verdicts are byte-identical.
fn note_slack(slack: f64, d: f64) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static SUM_PPM: AtomicUsize = AtomicUsize::new(0);
    static MAX_PPM: AtomicUsize = AtomicUsize::new(0);
    static NONZERO: AtomicUsize = AtomicUsize::new(0);
    if !slack.is_finite() || !d.is_finite() {
        return;
    }
    let ratio = (slack / d.abs().max(1e-12)).clamp(0.0, 1.0);
    let ppm = (ratio * 1.0e6) as usize;
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    SUM_PPM.fetch_add(ppm, Ordering::Relaxed);
    MAX_PPM.fetch_max(ppm, Ordering::Relaxed);
    if slack > 0.0 {
        NONZERO.fetch_add(1, Ordering::Relaxed);
    }
    // A run builds ~90 halfspaces, not thousands: report the first few and then
    // periodically. (The original 20000 threshold meant this probe ran silently
    // through an entire session — an absence of output that looked like an
    // absence of the code path.)
    if n <= 3 || n.is_multiple_of(50) {
        let mean = SUM_PPM.load(Ordering::Relaxed) as f64 / n as f64 / 1.0e6;
        let mx = MAX_PPM.load(Ordering::Relaxed) as f64 / 1.0e6;
        let nz = NONZERO.load(Ordering::Relaxed);
        eprintln!(
            "[slack-magnitude] halfspaces={n} nonzero={nz} mean_frac={mean:.6} max_frac={mx:.6}"
        );
    }
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
    if ny_core::require_f64_interval_proof_environment().is_err()
        || root.lo.len() != root.mid.len()
        || root.hi.len() != root.mid.len()
        || root
            .lo
            .iter()
            .zip(&root.hi)
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &(li, pos, sign) in splits {
        let Some(layer) = root.layers.get(li) else {
            continue;
        };
        let Some(rows) = layer.clip_rows.as_ref() else {
            continue;
        };
        if pos >= layer.unst.len() || !legacy_rows_shape(rows, layer.unst.len(), root.mid.len()) {
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
        // into the true subdomain. `sl_lo` / `sl_up` supply that producer charge.
        // Preserve it completely, and pay reconstruction residuals and final
        // constant rounding separately below. This consumer never recalibrates
        // source slack against subsequently intersected scalar bounds.
        let n_in = root.mid.len();
        let mut a = vec![0.0; n_in];
        // Sign convention MUST match `domain_gates`, which reads `d > 0` as
        // ACTIVE and everything else as INACTIVE.
        let Some((bias, error)) = legacy_row_objective(rows, pos, sign <= 0, &mut a, |j| {
            Some(root.lo[j].abs().max(root.hi[j].abs()))
        }) else {
            continue;
        };
        let d = if sign <= 0 {
            // INACTIVE: `lower_line(x) - error <= z(x) <= 0`.
            clip_add_down(bias, -error)
        } else {
            // ACTIVE: `upper_line(x) + error >= z(x) >= 0`; the helper
            // already negated its coefficients. Round the constant DOWN so
            // the emitted necessary condition can only admit extra points.
            clip_add_down(-bias, -error)
        };
        let Some(d) = d else {
            continue;
        };
        // #slack-magnitude probe: how much of this halfspace is GIVEN AWAY to
        // the certified slack? Reporting only; the halfspace is unchanged.
        note_slack(error, d);
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

/// Nominal dual proposal for `obj · x` over the box and ALL halfspaces.
///
/// Greedy coordinate ascent, one constraint at a time, folding each solved
/// multiplier into the objective — the same shape alpha-beta-CROWN uses. Each
/// fold keeps a valid mathematical dual point. Its plain binary64 evaluation
/// still needs a rounding enclosure before establishing a bound; the actual
/// crossing consumer uses [`dual_lower_outward`]. This API preserves the
/// historical nominal proposal used by existing diagnostic/search callers.
///
/// Returns `None` on a shape mismatch or a non-finite intermediate, so callers
/// fail closed to their existing bound.
#[must_use]
pub fn dual_lower(obj: &[f64], hs: &[Halfspace], x0: &[f64], eps: &[f64]) -> Option<f64> {
    dual_lower_certified(obj, hs, x0, eps).map(|(v, _)| v)
}

/// [`dual_lower`] plus the OPERAND-MAGNITUDE SUM its floating-point error is
/// relative to, so a caller can take a CERTIFIED outward margin.
///
/// Returns `(value, magnitude)` with
/// `magnitude = Σ_k β_k·|d_k| + Σ_j (|obj_j| + Σ_k β_k·|a_kj|) · (|x0_j| + eps_j)`.
/// Every partial sum that produced `value` — the β-accumulated objective, the
/// bias accumulator, and the final box-minimum sum — is bounded in absolute
/// value by `magnitude`, so by the standard Higham bound the accumulated
/// rounding error of `value` is at most `γ_{n_in + K + c} · magnitude` for a
/// small constant `c`. The margin a consumer must pay is therefore RELATIVE TO
/// `magnitude`, not to the delta it goes on to compute; a delta-relative shave
/// is not a theorem in exactly the small-delta regime this lever operates in.
#[must_use]
pub fn dual_lower_certified(
    obj: &[f64],
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
) -> Option<(f64, f64)> {
    if obj.len() != x0.len() || obj.len() != eps.len() {
        return None;
    }
    let mut work = obj.to_vec();
    // Magnitude twins of `work` and `acc_d`: monotone upper bounds on every
    // partial value either accumulator ever held.
    let mut mag_work: Vec<f64> = obj.iter().map(|o| o.abs()).collect();
    let mut acc_d = 0.0f64;
    let mut mag_d = 0.0f64;
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
                mag_work[j] = best_beta.mul_add(h.a[j].abs(), mag_work[j]);
            }
            acc_d += best_beta * h.d;
            mag_d += best_beta * h.d.abs();
        }
    }
    let mut v = acc_d;
    let mut mag = mag_d;
    for j in 0..work.len() {
        v += work[j].mul_add(x0[j], -(work[j].abs() * eps[j]));
        mag += mag_work[j] * (x0[j].abs() + eps[j]);
    }
    (v.is_finite() && mag.is_finite()).then_some((v, mag))
}

/// A directed lower bound for a particular nonnegative dual point. The
/// nominal result and upward whole-operand magnitude are diagnostics; only
/// `lower` may establish a crossing. No source authority is minted here.
#[derive(Debug, Clone, Copy)]
pub(super) struct DualOutwardBound {
    pub(super) lower: f64,
    pub(super) nominal: f64,
    pub(super) magnitude: f64,
}

/// Numerical observations for a directed dual candidate over the supplied
/// halfspaces and box. Only `directed_lower` is an outward lower bound;
/// `nominal` and `operand_magnitude` describe its floating-point proposal.
/// This data does not authenticate the constraints or carry proof authority.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualOutwardObservation {
    pub directed_lower: f64,
    pub nominal: f64,
    pub operand_magnitude: f64,
}

/// Observe the existing checked dual calculation without installing callbacks
/// in its numerical path. This uses the same arithmetic, allocations and
/// finite-input/FP-environment refusals as the clipping consumer. The caller
/// owns complete resource admission and the source of its numerical premises:
/// this is an unmetered arithmetic helper, not a proof or publication token.
#[must_use]
pub fn dual_lower_outward_observation(
    obj: &[f64],
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
) -> Option<DualOutwardObservation> {
    let bound = dual_lower_outward(obj, hs, x0, eps)?;
    Some(DualOutwardObservation {
        directed_lower: bound.lower,
        nominal: bound.nominal,
        operand_magnitude: bound.magnitude,
    })
}

/// Select multipliers with the existing nominal search, then enclose the
/// ENTIRE resulting Lagrangian. The exact accumulated coefficients lie in
/// `[lo_work, hi_work]` even when folding cancels large terms. Directed box
/// endpoints and all four endpoint products enclose its exact box minimum.
/// Gradual underflow is paid by `clip_mul_bounds`, rather than by a relative
/// allowance that disappears when the nominal sum rounds to zero.
///
/// The search need not find the true optimum: any finite nonnegative
/// multipliers give a valid weak-duality bound. Exceptional arithmetic refuses
/// this candidate, leaving the caller's independently certified bound intact.
#[must_use]
pub(super) fn dual_lower_outward(
    obj: &[f64],
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
) -> Option<DualOutwardBound> {
    ny_core::require_f64_interval_proof_environment().ok()?;
    if obj.len() != x0.len()
        || obj.len() != eps.len()
        || obj.iter().chain(x0).any(|v| !v.is_finite())
        || eps.iter().any(|&v| !v.is_finite() || v < 0.0)
        || hs.iter().any(|h| {
            h.a.len() != obj.len() || !h.d.is_finite() || h.a.iter().any(|v| !v.is_finite())
        })
    {
        return None;
    }
    let mut work = obj.to_vec();
    let mut lo_work = obj.to_vec();
    let mut hi_work = obj.to_vec();
    let mut mag_work: Vec<f64> = obj.iter().map(|v| v.abs()).collect();
    let mut scratch: Vec<(f64, f64)> = Vec::with_capacity(obj.len());
    let (mut nominal_bias, mut lo_bias, mut mag_bias) = (0.0, 0.0, 0.0);
    for h in hs {
        let beta = dual_argmax(&work, h, x0, eps, &mut scratch);
        if !beta.is_finite() || beta < 0.0 {
            return None;
        }
        if beta == 0.0 {
            continue;
        }
        for j in 0..obj.len() {
            work[j] = beta.mul_add(h.a[j], work[j]);
            if !work[j].is_finite() {
                return None;
            }
            let (product_lo, product_hi) = clip_mul_bounds(beta, h.a[j])?;
            lo_work[j] = clip_add_down(lo_work[j], product_lo)?;
            hi_work[j] = clip_add_up(hi_work[j], product_hi)?;
            mag_work[j] = clip_add_up(mag_work[j], clip_mul_bounds(beta, h.a[j].abs())?.1)?;
        }
        nominal_bias += beta * h.d;
        if !nominal_bias.is_finite() {
            return None;
        }
        lo_bias = clip_add_down(lo_bias, clip_mul_bounds(beta, h.d)?.0)?;
        mag_bias = clip_add_up(mag_bias, clip_mul_bounds(beta, h.d.abs())?.1)?;
    }
    let (mut lower, mut nominal, mut magnitude) = (lo_bias, nominal_bias, mag_bias);
    for j in 0..obj.len() {
        let xl = clip_add_down(x0[j], -eps[j])?;
        let xu = clip_add_up(x0[j], eps[j])?;
        let products = [
            clip_mul_bounds(lo_work[j], xl)?.0,
            clip_mul_bounds(lo_work[j], xu)?.0,
            clip_mul_bounds(hi_work[j], xl)?.0,
            clip_mul_bounds(hi_work[j], xu)?.0,
        ];
        lower = clip_add_down(lower, products.into_iter().fold(f64::INFINITY, f64::min))?;
        nominal += work[j].mul_add(x0[j], -(work[j].abs() * eps[j]));
        magnitude = clip_add_up(
            magnitude,
            clip_mul_bounds(mag_work[j], clip_add_up(x0[j].abs(), eps[j])?)?.1,
        )?;
    }
    nominal.is_finite().then_some(DualOutwardBound {
        lower,
        nominal,
        magnitude,
    })
}

/// CERTIFIED outward margin for a delta `raw = tightened - base`.
///
/// `tightened` and `base` are plain-f64 accumulations over `n_terms` products
/// whose partial sums are bounded by `dual_mag` and `base_mag` respectively, so
/// each carries at most `γ_{n_terms + 16} · mag` of rounding error and their
/// difference at most the sum of the two — which `2·γ·max(base_mag, dual_mag)`
/// dominates. Rounded up, plus an absolute floor. The result can only be
/// SMALLER than the true gap, never larger: paying too much margin costs
/// tightness, paying too little would let a tightened bound overstate itself
/// and feed an emptiness certificate downstream.
#[inline]
#[must_use]
fn certified_delta(raw: f64, base_mag: f64, dual_mag: f64, n_terms: usize) -> f64 {
    let gam = gamma_n(n_terms + 16);
    let margin = next_up(2.0 * gam * base_mag.max(dual_mag)) + 1e-12;
    (raw - margin).max(0.0)
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

/// THE FAIL-CLOSED FRAME GATE, shared by every Clip-and-Verify consumer.
///
/// A retained row in the wrong frame (transposed, output-relative, or missing
/// the augmented bias) still produces plausible numbers, and the halfspace built
/// from it cuts away part of the TRUE subdomain — a false-`unsat` generator, and
/// exactly how the sequential clip earned its quarantine. So every consumer must
/// pass this before it may tighten anything.
///
/// WHY IT LIVES HERE. It used to be a private `OnceLock` inside
/// `bab::clip_tighten_pack`, which meant the y-box consumer was guarded and the
/// INTERMEDIATE-gate consumer (`engine::clip_tighten_gates`) was not — and the
/// intermediate consumer is the more dangerous of the two, because it is the one
/// that can mark a domain `infeasible` and CLOSE it outright without any
/// backward pass ever disagreeing. One latch, checked once per process, shared
/// by both; a failure disables the whole path for the run rather than degrading
/// it quietly.
///
/// TOLERANCE 0.5, preserved verbatim from the y-box consumer's own note:
/// `concretize_box` widens both sides by an additive certified slack this check
/// does not re-derive (measured on cifar100 idx_8600 at ~1e-4 relative), while a
/// REAL frame error is wrong by factors, not by one slack term.
#[must_use]
pub fn frame_ok(root: &RootGates) -> bool {
    static FRAME_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FRAME_OK.get_or_init(|| {
        let ok = super::root::clip_rows_frame_holds(root, 0.5);
        if !ok {
            eprintln!("[clip] FRAME CHECK FAILED - clip-and-verify disabled for this run");
        } else {
            // R9: the gate is a precondition of every clip measurement, so a
            // run must be able to prove it PASSED rather than never ran.
            eprintln!("[clip] frame check PASSED (tol 0.5) - clip consumers admitted");
        }
        ok
    })
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
        let (base, base_mag) = {
            let (mut v, mut m) = (0.0f64, 0.0f64);
            for i in 0..n_in {
                v += obj[i].mul_add(x0[i], -(obj[i].abs() * eps[i]));
                m += obj[i].abs() * (x0[i].abs() + eps[i]);
            }
            (v, m)
        };
        let (tightened, dual_mag) = dual_lower_certified(&obj, hs, x0, eps)?;
        // OUTWARD MARGIN, CERTIFIED. `dual_lower_certified` and `base` are
        // plain-f64 accumulations over O(n_in) products; their rounding error
        // is relative to the OPERAND MAGNITUDES (Σ|obj_j|(|x0_j|+eps_j) and the
        // β-inflated twin), NOT to the delta. The earlier shave
        // (`raw*(1-1e-9) - 1e-12`) was delta-relative and therefore not a bound
        // when the delta is small against large operands — exactly the regime a
        // clip lever lives in. `certified_delta` pays 2·γ_{n+16} times the larger
        // magnitude sum, rounded up, as `concretize` does for the lane's own
        // bounds. This also certifies the previously dark trunk consumer.
        let raw = tightened - base;
        if !raw.is_finite() || raw < 0.0 {
            // Cannot happen (beta = 0 is always a candidate), but a negative
            // delta would LOOSEN a bound, so refuse the whole set rather than
            // silently widen.
            return None;
        }
        out.push(certified_delta(raw, base_mag, dual_mag, n_in + hs.len()));
    }
    Some(out)
}

// ===================== #clip-head: HEAD-SPLIT HALFSPACES =====================
//
// WHY THIS EXISTS, and why it is not the trunk path with a different index.
//
// `halfspaces_for_splits` above reads `root.layers[li].clip_rows`, and
// `RootGates.layers` holds TRUNK relus only (`root.rs` pushes one `LayerGates`
// per trunk relu and breaks at `last_relu`). The head layer has no entry, so a
// head split contributes NO halfspace today — it moves exactly one interval
// (`YBox::clamp`) and nothing else in the domain tightens. On the traced
// alpha-beta-CROWN rows that is the DOMINANT term: 10 of 12 splits land at the
// 100-wide head layer and `clip_interm_domain` then re-concretizes ~13-17 OTHER
// head neurons per domain (per-domain sum 1.29-1.94, i.e. ~0.07-0.11 each),
// after which 15/16 and 16/16 children verify at iteration <= 1.
//
// The certified lines for those head neurons are ALREADY IN HAND: the y-pack's
// identity-seeded passes give column `i` of `al.a` / `au.a` (shape `(n_in, R)`,
// one column per head neuron) with bias `al.b[i]` / `au.b[i]`, and the lane
// published `ly0/uy0` from exactly those columns. So the halfspace and its
// certified slack are both recoverable with NO new backward work, NO `clip_rows`
// retention, and no dependence on the process-wide trunk frame gate — which is
// what lets tinyimagenet (which arms no clip key at all) carry this.
//
// SOUNDNESS, the same three legs as the trunk port:
//  1. NECESSARY CONDITION. `al[:,i]·x + al.b[i] - sl_lo_i <= y_i(x)` pointwise
//     on the pack's validity region, so `y_i(x) <= 0` (an INACTIVE head fix)
//     implies `al[:,i]·x + (al.b[i] - sl_lo_i) <= 0`. Symmetrically an ACTIVE
//     fix gives `-(au[:,i]·x + au.b[i] + sl_up_i) <= 0`. Necessary, never
//     sufficient: the feasible set OVER-approximates the true subdomain.
//  2. THE SLACK IS RECOVERED, NOT ASSUMED. `concretize` subtracts a certified
//     penalty and rounds outward twice, so the published `ly_fwd[i]` sits BELOW
//     the line's exact box-minimum by exactly the pointwise error the line
//     carries. `sl_lo_i = box_min_i - ly_fwd[i]` recovers it; a rounding pad is
//     ADDED (an over-recovered slack only LOOSENS the halfspace, which is the
//     safe direction). A NEGATIVE recovered slack means the calibration base was
//     not the forward concretization — the `#backward-interm` ordering trap —
//     and is REFUSED rather than clamped.
//  3. WEAK DUALITY + SHRINK-ONLY COMMIT. The deltas come from the untouched
//     `ybox_deltas`/`dual_lower` pair (non-negative by construction, `None` on
//     any refusal) and are committed with `max`/`min` against the forward base.

/// Recovered certified slack for one y-pack's head lines.
///
/// `sl_lo[i]` is how far the lane's published lower bound for head neuron `i`
/// sits BELOW the exact box-minimum of that neuron's certified lower line, and
/// `sl_up[i]` the mirror on the upper side. Both are >= 0 by construction here;
/// a negative value refuses the whole pack (see [`head_slack`]).
#[derive(Debug, Clone)]
pub struct HeadSlack {
    /// Per head neuron: certified slack on the LOWER line.
    pub sl_lo: Vec<f64>,
    /// Per head neuron: certified slack on the UPPER line.
    pub sl_up: Vec<f64>,
}

impl HeadSlack {
    /// Number of head columns this slack covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sl_lo.len()
    }

    /// No columns at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sl_lo.is_empty()
    }
}

/// Rate-limited frame telemetry for the head path.
fn head_frame_report(ok: bool, n_in: usize, n_y: usize, lo_max: f64, up_max: f64, frac: f64) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static FAILED: AtomicUsize = AtomicUsize::new(0);
    if !ok {
        // A failure must be VISIBLE: a head path that never tightened and a head
        // path that refused its own calibration look identical from the outside,
        // and only one of them is a measurement. Rate-limited like every other
        // probe in this lane so a per-pack refusal cannot flood the log.
        if FAILED.fetch_add(1, Ordering::Relaxed) < 3 {
            eprintln!(
                "[clip-head] FRAME CHECK FAILED n_in={n_in} n_y={n_y} sl_lo_max={lo_max:.6e} \
                 sl_up_max={up_max:.6e} sl_max_frac={frac:.6e} - head clip disabled for this pack"
            );
        }
        return;
    }
    if N.fetch_add(1, Ordering::Relaxed) < 3 {
        eprintln!(
            "[clip-head] frame n_in={n_in} n_y={n_y} ok=true sl_lo_max={lo_max:.6e} \
             sl_up_max={up_max:.6e} sl_max_frac={frac:.6e}"
        );
    }
}

/// Recover the certified slack on every head line of one y-pack.
///
/// `ly_fwd`/`uy_fwd` MUST be the UNTIGHTENED concretization of `al`/`au` — the
/// values `YBox::from_rows` produced inside `pack_from_rows`, before
/// `clip_tighten_pack` may have added trunk deltas into `ly0/uy0`. Calibrating
/// against a tightened bound under-recovers the slack, and a halfspace built
/// from an under-recovered slack cuts into the TRUE subdomain: a false-`unsat`
/// generator. That is the ordering trap `root.rs`'s CALIBRATION INVARIANT
/// records for the trunk rows, arriving here through a different door, and it is
/// pinned by `head_slack_calibration_must_use_fresh_concretization`.
///
/// # The rounding pad
///
/// The box-minimum below is re-derived in plain f64 with a different summation
/// order than `concretize` used, so it can land a few ulps BELOW the value the
/// lane actually minimised. That would under-recover the slack. `pad` covers it
/// with the same `gamma_n(n_in + 16)` envelope `concretize` itself uses, doubled.
/// Over-recovery is free: a larger slack only makes the halfspace LOOSER.
///
/// # The frame gate
///
/// Fails closed (returns `None`, head path disabled for this pack) on: a shape
/// that is not `(n_in, n_y)`, any non-finite value, ANY negative recovered slack
/// (the calibration trap), or a slack larger than `tol` of the published width
/// plus a rounding floor. A transposed or bias-dropped read shows up as slack of
/// the order of the interval WIDTH, which is what the last check catches; a
/// genuine certified slack is orders of magnitude smaller.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn head_slack(
    al: &super::engine::PassOut,
    au: &super::engine::PassOut,
    ly_fwd: &[f64],
    uy_fwd: &[f64],
    mid: &[f64],
    rad: &[f64],
    xabs: &[f64],
    tol: f64,
) -> Option<HeadSlack> {
    let n_in = mid.len();
    let n_y = ly_fwd.len();
    if al.a.nrows() != n_in
        || au.a.nrows() != n_in
        || al.a.ncols() != n_y
        || au.a.ncols() != n_y
        || uy_fwd.len() != n_y
        || rad.len() != n_in
        || xabs.len() != n_in
        || al.b.len() != n_y
        || au.b.len() != n_y
        || n_y == 0
    {
        head_frame_report(false, n_in, n_y, f64::NAN, f64::NAN, f64::NAN);
        return None;
    }
    let gam = gamma_n(n_in + 16);
    let mut sl_lo = Vec::with_capacity(n_y);
    let mut sl_up = Vec::with_capacity(n_y);
    let (mut lo_max, mut up_max, mut frac_max) = (0.0f64, 0.0f64, 0.0f64);
    let mut ok = true;
    for i in 0..n_y {
        let (mut bmin, mut bmax) = (al.b[i], au.b[i]);
        let (mut tabs_l, mut tabs_u) = (al.b[i].abs(), au.b[i].abs());
        for j in 0..n_in {
            let cl = al.a[[j, i]];
            let cu = au.a[[j, i]];
            bmin += cl.mul_add(mid[j], -(cl.abs() * rad[j]));
            bmax += cu.mul_add(mid[j], cu.abs() * rad[j]);
            tabs_l += cl.abs() * xabs[j];
            tabs_u += cu.abs() * xabs[j];
        }
        let pad_l = 2.0 * gam * tabs_l;
        let pad_u = 2.0 * gam * tabs_u;
        let s_lo = bmin - ly_fwd[i] + pad_l;
        let s_up = uy_fwd[i] - bmax + pad_u;
        // Factor-level frame discriminant: half the published width, plus a
        // rounding floor so a degenerate (near-constant) head neuron cannot
        // refuse the whole pack over a few ulps.
        let width = (uy_fwd[i] - ly_fwd[i]).abs();
        let floor = 16.0 * gam * tabs_l.max(tabs_u) + 1e-12;
        let cap = tol.mul_add(width, floor);
        if !s_lo.is_finite()
            || !s_up.is_finite()
            || s_lo < 0.0
            || s_up < 0.0
            || s_lo > cap
            || s_up > cap
        {
            ok = false;
        }
        lo_max = lo_max.max(s_lo);
        up_max = up_max.max(s_up);
        if width > 0.0 {
            frac_max = frac_max.max((s_lo / width).max(s_up / width));
        }
        sl_lo.push(s_lo);
        sl_up.push(s_up);
    }
    head_frame_report(ok, n_in, n_y, lo_max, up_max, frac_max);
    if ok {
        // The invariant every consumer relies on: a `HeadSlack` this function
        // hands out is finite and non-negative in every column. `ok` is what
        // enforces it; the assertion is here so a future edit that relaxes one
        // branch of the gate fails a debug build instead of shipping a
        // halfspace that cuts into the true subdomain.
        debug_assert!(
            sl_lo
                .iter()
                .chain(sl_up.iter())
                .all(|s| s.is_finite() && *s >= 0.0),
            "head_slack admitted a negative or non-finite slack"
        );
    }
    ok.then_some(HeadSlack { sl_lo, sl_up })
}

/// Structural cap on how many of a domain's head fixes become halfspaces.
///
/// Documented here rather than exposed as a key: each halfspace costs one dual
/// fold per objective column, and a SUBSET of necessary conditions is always
/// sound (it only loosens). abc itself drops to a single constraint per domain
/// after round 2. The LAST fixes are kept because they are the deepest and the
/// most recently informative.
pub const HEAD_H_CAP: usize = 8;

/// Structural cap on how many head columns are re-concretized per eval.
///
/// abc's `topk_objective` is 20 on the traced rows; the head layer is 100 wide,
/// so this is the same order of work and keeps the per-eval cost bounded
/// independently of head width.
pub const HEAD_C_CAP: usize = 20;

/// Halfspaces implied by a domain's HEAD sign fixes, from the domain's own
/// y-pack rows plus their recovered certified slack.
///
/// `heads` are `(head_neuron, dir)` exactly as `DomainEntry.heads` carries them;
/// `dir > 0` means the neuron was fixed ACTIVE (`y_i >= 0`), `dir <= 0` INACTIVE
/// (`y_i <= 0`) — the `YBox::clamp` / `domain_gates` convention.
///
/// Returns `None` (fail closed) on any index or finiteness problem. Emits at
/// most `h_cap` halfspaces, taken from the END of `heads`.
#[must_use]
pub fn halfspaces_for_heads(
    al: &super::engine::PassOut,
    au: &super::engine::PassOut,
    hs: &HeadSlack,
    heads: &[(usize, i8)],
    h_cap: usize,
) -> Option<Vec<Halfspace>> {
    if heads.is_empty() || h_cap == 0 {
        return Some(Vec::new());
    }
    let n_in = al.a.nrows();
    let n_y = al.a.ncols();
    if au.a.nrows() != n_in || au.a.ncols() != n_y || hs.sl_lo.len() != n_y || hs.sl_up.len() != n_y
    {
        return None;
    }
    let take = heads.len().saturating_sub(h_cap);
    let mut out = Vec::with_capacity(heads.len() - take);
    for &(i, dir) in &heads[take..] {
        if i >= n_y {
            return None;
        }
        let mut a = Vec::with_capacity(n_in);
        let d = if dir <= 0 {
            // INACTIVE: `al[:,i]·x + al.b[i] - sl_lo_i <= y_i(x) <= 0`.
            for j in 0..n_in {
                a.push(al.a[[j, i]]);
            }
            al.b[i] - hs.sl_lo[i]
        } else {
            // ACTIVE: `0 <= y_i(x) <= au[:,i]·x + au.b[i] + sl_up_i`; negate
            // into `a·x + d <= 0` form.
            for j in 0..n_in {
                a.push(-au.a[[j, i]]);
            }
            -(au.b[i] + hs.sl_up[i])
        };
        if !d.is_finite() || a.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Released-mode refusal at the CONSUMER, independent of the pack-time
        // frame gate: a slack that is negative or non-finite would make this
        // halfspace claim more than the line certifies, so the whole build
        // refuses (counted as `refused_hs` in the `[clip-head]` observable)
        // rather than clamping to zero.
        let slack = if dir <= 0 { hs.sl_lo[i] } else { hs.sl_up[i] };
        if !(slack.is_finite() && slack >= 0.0) {
            return None;
        }
        note_slack(slack, d);
        out.push(Halfspace { a, d });
    }
    Some(out)
}

/// [`ybox_deltas`] restricted to a subset of columns.
///
/// Identical arithmetic, identical refusal semantics; the returned vector is
/// indexed by POSITION IN `cols`, not by column. Pinned bit-for-bit against
/// [`ybox_deltas`] for `cols = 0..r` by `ybox_deltas_cols_matches_ybox_deltas`.
#[must_use]
pub fn ybox_deltas_cols(
    a: &ndarray::Array2<f64>,
    cols: &[usize],
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
    lower: bool,
) -> Option<Vec<f64>> {
    if hs.is_empty() || cols.is_empty() {
        return None;
    }
    let n_in = a.nrows();
    let r = a.ncols();
    if n_in != x0.len() || n_in != eps.len() {
        return None;
    }
    let mut out = Vec::with_capacity(cols.len());
    let mut obj = vec![0.0f64; n_in];
    for &ri in cols {
        if ri >= r {
            return None;
        }
        for (i, o) in obj.iter_mut().enumerate() {
            *o = if lower { a[[i, ri]] } else { -a[[i, ri]] };
        }
        let (base, base_mag) = {
            let (mut v, mut m) = (0.0f64, 0.0f64);
            for i in 0..n_in {
                v += obj[i].mul_add(x0[i], -(obj[i].abs() * eps[i]));
                m += obj[i].abs() * (x0[i].abs() + eps[i]);
            }
            (v, m)
        };
        let (tightened, dual_mag) = dual_lower_certified(&obj, hs, x0, eps)?;
        let raw = tightened - base;
        if !raw.is_finite() || raw < 0.0 {
            return None;
        }
        out.push(certified_delta(raw, base_mag, dual_mag, n_in + hs.len()));
    }
    Some(out)
}

/// Which head columns are worth re-concretizing: the still-UNSTABLE ones
/// (`ly < 0 < uy`), ranked by alpha-beta-CROWN's own clip score
/// `(-ly * uy) / (uy - ly)` (`domain_clipper.py::get_clip_decisions_static`),
/// ties broken by index so the selection is deterministic across workers.
#[must_use]
pub fn head_columns(ly: &[f64], uy: &[f64], c_cap: usize) -> Vec<usize> {
    if c_cap == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f64)> = (0..ly.len().min(uy.len()))
        .filter(|&i| ly[i] < 0.0 && uy[i] > 0.0)
        .filter_map(|i| {
            let w = uy[i] - ly[i];
            let s = (-ly[i] * uy[i]) / w;
            s.is_finite().then_some((i, s))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(c_cap);
    let mut cols: Vec<usize> = scored.into_iter().map(|(i, _)| i).collect();
    cols.sort_unstable();
    cols
}

/// What one head-clip step did to a domain's y-box.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadClipOut {
    /// Halfspaces built from the domain's head fixes.
    pub hs: usize,
    /// Head columns offered to the dual.
    pub cols: usize,
    /// Columns whose interval strictly shrank.
    pub moved: usize,
    /// Total shrink across those columns (both sides).
    pub sum_delta: f64,
    /// Columns that became STABLE (`ly >= 0` or `uy <= 0`).
    pub closes: usize,
    /// Columns whose tightened interval would have crossed: REFUSED and left at
    /// the incoming values.
    pub refused_cross: usize,
    /// The halfspace build itself refused (index/finiteness).
    pub refused_hs: bool,
}

/// THE HEAD-CLIP STEP: re-concretize a domain's OTHER head columns against the
/// halfspaces its own head sign fixes imply.
///
/// `ly`/`uy` are the domain's CURRENT y-box (already clamped by the head fixes
/// and intersected with the inherited box); they are tightened in place,
/// shrink-only. `ly_fwd`/`uy_fwd` are the pack's UNTIGHTENED concretization and
/// are the base the dual's delta is added to — never the current values, because
/// a delta from one dual solve may not be stacked on a bound another already
/// tightened.
///
/// Pure: no globals, no I/O, deterministic in its arguments, so a domain's
/// result is identical whichever rayon worker evaluates it.
///
/// # Soundness
///
/// * every halfspace is a NECESSARY condition of a head fix the domain has
///   already made, built from a certified line MINUS its recovered slack, so the
///   feasible set OVER-approximates the true subdomain;
/// * [`ybox_deltas_cols`] returns a non-negative delta by construction and
///   refuses the whole set on anything else;
/// * the commit is `max`/`min` against the incoming box, which cannot widen;
/// * an IN-STEP crossing is REFUSED and counted — this function never reads
///   one as an emptiness certificate. But the tightened box it commits IS read
///   downstream: a column driven to `uy <= 0` (or `ly >= 0`) closes the opposite
///   head child at the split step, and the inherited box can make a child's
///   `intersect`+`is_empty` fire `Infeasible`. Those are sound iff these bounds
///   are sound, which is why the delta's margin is certified against operand
///   magnitudes and why the moat protocol carries known-SAT sentinels.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn tighten_head_box(
    al: &super::engine::PassOut,
    au: &super::engine::PassOut,
    slack: &HeadSlack,
    ly_fwd: &[f64],
    uy_fwd: &[f64],
    heads: &[(usize, i8)],
    mid: &[f64],
    rad: &[f64],
    ly: &mut [f64],
    uy: &mut [f64],
    h_cap: usize,
    c_cap: usize,
) -> HeadClipOut {
    let mut out = HeadClipOut::default();
    if heads.is_empty()
        || ly.len() != uy.len()
        || ly_fwd.len() != ly.len()
        || uy_fwd.len() != ly.len()
    {
        return out;
    }
    let Some(hs) = halfspaces_for_heads(al, au, slack, heads, h_cap) else {
        out.refused_hs = true;
        return out;
    };
    if hs.is_empty() {
        return out;
    }
    let cols = head_columns(ly, uy, c_cap);
    out.hs = hs.len();
    out.cols = cols.len();
    if cols.is_empty() {
        return out;
    }
    let dl = ybox_deltas_cols(&al.a, &cols, &hs, mid, rad, true);
    let du = ybox_deltas_cols(&au.a, &cols, &hs, mid, rad, false);
    for (k, &j) in cols.iter().enumerate() {
        let (l0, u0) = (ly[j], uy[j]);
        let mut l = l0;
        let mut u = u0;
        if let Some(dl) = dl.as_ref() {
            l = l.max(ly_fwd[j] + dl[k]);
        }
        if let Some(du) = du.as_ref() {
            u = u.min(uy_fwd[j] - du[k]);
        }
        if !(l.is_finite() && u.is_finite()) || l > u + 1e-12 {
            out.refused_cross += 1;
            continue;
        }
        if l > l0 || u < u0 {
            out.moved += 1;
            out.sum_delta += (l - l0) + (u0 - u);
            if l >= 0.0 || u <= 0.0 {
                out.closes += 1;
            }
        }
        ly[j] = l;
        uy[j] = u;
    }
    out
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

    // ==================== #clip-head unit oracles ====================

    /// THE pin: a delta that is positive but SMALLER than
    /// the rounding envelope of the operands must be refused to zero. Built from
    /// powers of two so every product and sum is EXACT in f64 and `raw` is
    /// exactly 2^-20; the old delta-relative shave published ~9.5e-7 here.
    #[test]
    fn a_tiny_delta_under_large_magnitudes_is_refused_to_zero() {
        let n_in = 4096usize;
        let big = (1u64 << 20) as f64; // 2^20
        let e = 2f64.powi(-10);
        let x0 = vec![1.0f64; n_in];
        let eps = vec![e; n_in];
        // One column: obj_0 = -2^20 (wants x_0 at its box max), obj_j = 2^20.
        let mut a = ndarray::Array2::<f64>::from_elem((n_in, 1), big);
        a[[0, 0]] = -big;
        // Halfspace on coordinate 0 that cuts the maximiser by exactly 2^-40:
        // 2^20 x_0 - 2^20 (1 + 2^-10) + 2^-20 <= 0  <=>  x_0 <= 1 + 2^-10 - 2^-40.
        let mut ha = vec![0.0f64; n_in];
        ha[0] = big;
        let hs = vec![Halfspace {
            a: ha,
            d: -big * (1.0 + e) + 2f64.powi(-20),
        }];
        let out = ybox_deltas(&a, &hs, &x0, &eps, true).expect("shapes are valid");
        assert_eq!(out.len(), 1);
        // The true gap is exactly 2^-20 ~= 9.5e-7; the operand magnitudes are
        // ~4096 * 2^20 ~= 4.3e9, whose gamma envelope is ~4e-3. The delta MUST be
        // refused to zero, never published.
        assert_eq!(
            out[0], 0.0,
            "a sub-envelope delta must be refused, got {}",
            out[0]
        );
        let cols = ybox_deltas_cols(&a, &[0], &hs, &x0, &eps, true).expect("shapes are valid");
        assert_eq!(cols[0], 0.0);
    }

    /// The margin actually paid is at least the analytic envelope and the delta
    /// is never negative or larger than the raw gap — on a wide fixture whose
    /// gap is LARGE (2^10), so the delta survives and can be inspected.
    #[test]
    fn ybox_delta_margin_is_the_gamma_envelope_over_operand_magnitudes() {
        let n_in = 4096usize;
        let big = (1u64 << 20) as f64;
        let e = 2f64.powi(-10);
        let x0 = vec![1.0f64; n_in];
        let eps = vec![e; n_in];
        let mut a = ndarray::Array2::<f64>::from_elem((n_in, 1), big);
        a[[0, 0]] = -big;
        let mut ha = vec![0.0f64; n_in];
        ha[0] = big;
        let shift = 1024.0f64; // exact gap of 2^10
        let hs = vec![Halfspace {
            a: ha,
            d: -big * (1.0 + e) + shift,
        }];
        let out = ybox_deltas(&a, &hs, &x0, &eps, true).expect("shapes are valid");
        let delta = out[0];
        // Analytic envelope: base_mag = n_in * 2^20 * (1 + 2^-10); the beta=1
        // dual adds |d| and one more 2^20 (1 + 2^-10) on coordinate 0.
        let base_mag = (n_in as f64) * big * (1.0 + e);
        let dual_mag = (big * (1.0 + e) - shift) + base_mag + big * (1.0 + e);
        let gam = gamma_n(n_in + hs.len() + 16);
        let envelope = 2.0 * gam * base_mag.max(dual_mag);
        let paid = shift - delta;
        assert!(
            delta >= 0.0 && delta <= shift,
            "delta {delta} outside [0, {shift}]"
        );
        assert!(
            paid >= envelope,
            "margin paid {paid:e} is below the gamma envelope {envelope:e}"
        );
        assert!(
            paid <= 4.0 * envelope + 1e-9,
            "margin paid {paid:e} is far beyond the envelope {envelope:e}: tightness is being thrown away"
        );
    }

    #[test]
    fn ybox_deltas_cols_matches_ybox_deltas() {
        // The subset solver must be the SAME arithmetic, not a re-derivation:
        // over the full column range it is pinned bit-for-bit.
        let n_in = 5usize;
        let r = 4usize;
        let mut a = ndarray::Array2::<f64>::zeros((n_in, r));
        for i in 0..n_in {
            for k in 0..r {
                a[[i, k]] = ((i * 7 + k * 3) as f64).mul_add(0.031, -0.4);
            }
        }
        let x0 = vec![0.05, -0.2, 0.3, 0.0, -0.1];
        let eps = vec![0.5, 0.4, 0.6, 0.3, 0.45];
        let hs = vec![
            Halfspace {
                a: vec![1.0, 0.5, -0.25, 0.0, 0.75],
                d: -0.1,
            },
            Halfspace {
                a: vec![-0.3, 1.0, 0.2, 0.4, -0.6],
                d: 0.05,
            },
        ];
        for lower in [true, false] {
            let full = ybox_deltas(&a, &hs, &x0, &eps, lower).expect("full");
            let cols: Vec<usize> = (0..r).collect();
            let sub = ybox_deltas_cols(&a, &cols, &hs, &x0, &eps, lower).expect("cols");
            assert_eq!(full.len(), sub.len());
            for k in 0..r {
                assert_eq!(
                    full[k].to_bits(),
                    sub[k].to_bits(),
                    "lower={lower} column {k}: {} vs {}",
                    full[k],
                    sub[k]
                );
            }
            // A strict subset returns the SAME values for the columns it covers.
            let pick = vec![1usize, 3usize];
            let part = ybox_deltas_cols(&a, &pick, &hs, &x0, &eps, lower).expect("subset");
            assert_eq!(part[0].to_bits(), full[1].to_bits());
            assert_eq!(part[1].to_bits(), full[3].to_bits());
        }
        assert!(
            ybox_deltas_cols(&a, &[r], &hs, &x0, &eps, true).is_none(),
            "an out-of-range column must fail closed"
        );
    }

    /// Deterministic LCG: the exact-line oracle must not depend on a crate's
    /// sampler staying byte-stable across versions.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
    }

    fn exact_pass(a: &ndarray::Array2<f64>, b: &[f64]) -> super::super::engine::PassOut {
        super::super::engine::PassOut {
            a: a.clone(),
            e: None,
            b: b.to_vec(),
            eb: vec![0.0; b.len()],
            coll: None,
            coll_rows: None,
            unst_rows: None,
        }
    }

    #[test]
    fn head_halfspace_on_exact_affine_head_does_not_cut_the_true_subdomain() {
        // THE EXACT-LINE FIXTURE. When the certified lower and upper lines
        // COINCIDE with the true function, the emitted halfspace coincides with
        // the split boundary `y_i(x) = 0` up to the recovered slack — which is
        // exactly where a missing or under-recovered slack would exclude true
        // points, and where an over-wide one would show as a constraint that no
        // longer cuts. Built ALGEBRAICALLY rather than from a net, so "the line
        // is exact" is a fact about the fixture instead of a hope about which
        // ReLUs happened to be stable on some random seed.
        //
        // `ly_fwd`/`uy_fwd` are the box concretization of those same lines, i.e.
        // what `YBox::from_rows` publishes for a pack whose error lane is empty,
        // so `head_slack` recovers only the rounding pad.
        //
        // Column 0 is the split neuron and is deliberately NARROW with a large
        // offset, so `y_0 <= 0` removes most of the box along its own direction;
        // columns 1, 2 and 4 are deliberately CORRELATED with it, so the dual has
        // something to find. A fixture where the constraint barely bites makes
        // every enclosure assertion pass trivially, which is the failure mode the
        // net-based sibling test measured on eight seeds.
        let (n_in, n_y) = (6usize, 5usize);
        let mut st = 0x11EA_D0CE_u64 ^ 0x9E37_79B9_7F4A_7C15;
        let split_col = [0.20f64, -0.15, 0.25, 0.10, -0.20, 0.10]; // sum |.| = 1.0
        let mut a = ndarray::Array2::<f64>::zeros((n_in, n_y));
        let mix = [1.0f64, 2.0, -1.5, 0.0, 1.0];
        for i in 0..n_y {
            for j in 0..n_in {
                a[[j, i]] = mix[i].mul_add(split_col[j], 0.05 * lcg(&mut st));
            }
        }
        let b = vec![0.6f64, 0.0, 0.1, 0.0, -0.2];
        let mid = vec![0.0f64; n_in];
        let rad = vec![1.0f64; n_in];
        let xabs = vec![1.0f64; n_in];
        let mut ly_fwd = vec![0.0f64; n_y];
        let mut uy_fwd = vec![0.0f64; n_y];
        for i in 0..n_y {
            let (mut lo, mut hi) = (b[i], b[i]);
            for j in 0..n_in {
                lo += a[[j, i]].mul_add(mid[j], -(a[[j, i]].abs() * rad[j]));
                hi += a[[j, i]].mul_add(mid[j], a[[j, i]].abs() * rad[j]);
            }
            ly_fwd[i] = lo;
            uy_fwd[i] = hi;
        }
        let al = exact_pass(&a, &b);
        let au = exact_pass(&a, &b);
        let slack = head_slack(&al, &au, &ly_fwd, &uy_fwd, &mid, &rad, &xabs, 0.5)
            .expect("exact lines must pass the frame gate");
        for i in 0..n_y {
            assert!(
                slack.sl_lo[i] >= 0.0 && slack.sl_lo[i] < 1e-9,
                "column {i}: an exact line must recover a NON-NEGATIVE, rounding-sized \
                 slack, got {}",
                slack.sl_lo[i]
            );
        }
        let i = 0usize;
        assert!(
            ly_fwd[i] < -0.2 && uy_fwd[i] > 0.2,
            "the split column must straddle zero: [{}, {}]",
            ly_fwd[i],
            uy_fwd[i]
        );

        // Sample the box and evaluate the TRUE function, which here IS the line.
        let nb = 40_000usize;
        let mut x = ndarray::Array2::<f64>::zeros((n_in, nb));
        for j in 0..n_in {
            for k in 0..nb {
                x[[j, k]] = mid[j] + lcg(&mut st) * rad[j];
            }
        }
        let yv = |i: usize, k: usize| -> f64 {
            let mut v = b[i];
            for j in 0..n_in {
                v += a[[j, i]] * x[[j, k]];
            }
            v
        };

        let mut any_moved = false;
        for dir in [-1i8, 1i8] {
            let hs = halfspaces_for_heads(&al, &au, &slack, &[(i, dir)], 8).expect("build");
            assert_eq!(hs.len(), 1);
            let h = &hs[0];
            let mut worst = f64::NEG_INFINITY;
            // How close the closest in-child sample gets to the split boundary
            // `y_i = 0`, signed so it is <= 0. The tightness leg is stated
            // against THIS rather than against a fixed epsilon, because how near
            // the boundary a finite sample lands is a fact about the sampler,
            // not about the halfspace.
            let mut edge = f64::NEG_INFINITY;
            let mut inside = Vec::new();
            for k in 0..nb {
                let y = yv(i, k);
                let in_child = if dir > 0 { y >= 0.0 } else { y <= 0.0 };
                if !in_child {
                    continue;
                }
                inside.push(k);
                edge = edge.max(if dir > 0 { -y } else { y });
                let mut v = h.d;
                for j in 0..n_in {
                    v += h.a[j] * x[[j, k]];
                }
                assert!(
                    v <= 1e-9,
                    "dir={dir}: the head halfspace EXCLUDES a true-subdomain point \
                     (a.x + d = {v}, y_i = {y}) — false-unsat generator"
                );
                worst = worst.max(v);
            }
            assert!(
                inside.len() >= 100,
                "dir={dir}: only {} in-child samples — vacuous",
                inside.len()
            );
            // TIGHTNESS: the slack must not be hiding a LOOSE halfspace. On an
            // exact line the constraint IS the split boundary shifted by exactly
            // the recovered slack, so at every in-child sample
            // `a.x + d == +/-y_i(x) - slack` and the most binding one must sit
            // exactly `slack` inside the closest sampled approach to `y_i = 0`.
            // Anything looser means the halfspace is not the boundary; anything
            // tighter means the slack was not paid.
            let sl = if dir > 0 {
                slack.sl_up[i]
            } else {
                slack.sl_lo[i]
            };
            assert!(
                (worst - (edge - sl)).abs() <= 1e-12,
                "dir={dir}: the most binding sample is {worst}, but the closest sampled \
                 approach to the boundary is {edge} and the recovered slack is {sl}, so it \
                 should be {} — the halfspace is not the split boundary minus its slack",
                edge - sl
            );
            // ENCLOSURE after the step, on the true values of every column.
            let mut ly = ly_fwd.clone();
            let mut uy = uy_fwd.clone();
            if dir > 0 {
                ly[i] = ly[i].max(0.0);
            } else {
                uy[i] = uy[i].min(0.0);
            }
            let out = tighten_head_box(
                &al,
                &au,
                &slack,
                &ly_fwd,
                &uy_fwd,
                &[(i, dir)],
                &mid,
                &rad,
                &mut ly,
                &mut uy,
                8,
                20,
            );
            assert_eq!(out.refused_cross, 0);
            any_moved |= out.moved >= 1;
            eprintln!(
                "[test clip-head exact] dir={dir} hs={} cols={} moved={} sum_delta={:.6e} \
                 inside={}",
                out.hs,
                out.cols,
                out.moved,
                out.sum_delta,
                inside.len()
            );
            for j in 0..n_y {
                assert!(
                    ly[j] >= ly_fwd[j] - 1e-12 && uy[j] <= uy_fwd[j] + 1e-12,
                    "dir={dir} column {j}: not a shrink"
                );
                for &k in &inside {
                    let v = yv(j, k);
                    assert!(
                        ly[j] <= v + 1e-9 && uy[j] >= v - 1e-9,
                        "dir={dir} column {j}: clipped [{}, {}] EXCLUDES the true value {v}",
                        ly[j],
                        uy[j]
                    );
                }
            }
        }
        // R9. Only the direction that EXCLUDES the box centre can bite (the
        // other's halfspace is satisfied by the whole half it keeps), so this is
        // asserted once across the pair rather than per direction.
        assert!(
            any_moved,
            "neither direction tightened anything, so both enclosure legs above \
             passed trivially"
        );
    }

    #[test]
    fn dropping_the_head_slack_cuts_into_the_true_subdomain() {
        // THE DEFECT, DEMONSTRATED — not merely avoided. A 2-input oracle where
        // the certified line `L` over-estimates the true pre-activation by a
        // known slack `s`:
        //
        //     L(x) = a.x + b          (the certified LOWER line)
        //     y(x) = L(x) - s         (the truth it under-estimates by s)
        //
        // The domain fixes `y <= 0`, i.e. the TRUE region is `{L <= s}`. The
        // SLACKED halfspace `{L - s <= 0}` is exactly that region, so the dual
        // over it lower-bounds the true minimum. The UNSLACKED halfspace
        // `{L <= 0}` is a STRICT SUBSET of the true region, so minimising over
        // it can report a value ABOVE the true minimum — a bound that is not a
        // bound, which is how a false `unsat` is manufactured.
        let x0 = [0.0f64, 0.0];
        let eps = [1.0f64, 1.0];
        let a = [1.0f64, 0.5];
        let b = 0.1f64;
        let s = 0.05f64;
        // Objective: the direction the constraint actually binds.
        let obj = [-1.0f64, -0.4];

        // Brute-force minimum of obj.x over box INTERSECT {a.x + b - s <= 0}.
        let n = 1200;
        let mut truth = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                let x = [
                    (i as f64 / n as f64).mul_add(2.0, -1.0),
                    (j as f64 / n as f64).mul_add(2.0, -1.0),
                ];
                if a[0].mul_add(x[0], a[1] * x[1]) + b - s > 0.0 {
                    continue;
                }
                truth = truth.min(obj[0].mul_add(x[0], obj[1] * x[1]));
            }
        }
        assert!(truth.is_finite(), "the true region came out empty");

        let slacked = Halfspace {
            a: a.to_vec(),
            d: b - s,
        };
        let unslacked = Halfspace {
            a: a.to_vec(),
            d: b,
        };
        let with = dual_lower(&obj, std::slice::from_ref(&slacked), &x0, &eps).expect("dual");
        let without = dual_lower(&obj, std::slice::from_ref(&unslacked), &x0, &eps).expect("dual");

        assert!(
            with <= truth + 1e-9,
            "the SLACKED halfspace must lower-bound the true minimum: {with} vs {truth}"
        );
        assert!(
            without > truth + 1e-6,
            "the unslacked halfspace was supposed to OVERSTATE the minimum on this \
             fixture ({without} vs truth {truth}); if it no longer does, the test no \
             longer demonstrates the defect it guards against"
        );
    }

    #[test]
    fn head_columns_picks_unstable_neurons_by_the_clip_score() {
        // Stable columns can gain nothing (their gate is already exact) and are
        // never offered; the ranking is alpha-beta-CROWN's own
        // `(-ly * uy) / (uy - ly)`, ties by index, and the returned order is
        // ascending so the delta vector lines up with the column list.
        let ly = [-1.0, 0.5, -2.0, -0.1, -3.0];
        let uy = [1.0, 2.0, 2.0, 0.1, 0.05];
        let all = head_columns(&ly, &uy, 10);
        assert_eq!(all, vec![0, 2, 3, 4], "stable column 1 must be excluded");
        // scores: 0 -> 0.5, 2 -> 1.0, 3 -> 0.05, 4 -> 0.0491...
        let top2 = head_columns(&ly, &uy, 2);
        assert_eq!(top2, vec![0, 2]);
        assert_eq!((head_columns(&ly, &uy, 0)).len(), 0);
    }

    #[test]
    fn head_halfspace_signs_follow_the_ybox_clamp_convention() {
        // `YBox::clamp` reads `dir > 0` as ACTIVE (`ly = max(ly, 0)`) and
        // everything else as INACTIVE (`uy = min(uy, 0)`). Getting this backwards
        // builds a necessary condition of the WRONG child, which is the second
        // false-unsat generator in this family.
        let n_in = 3usize;
        let n_y = 2usize;
        let mut a_lo = ndarray::Array2::<f64>::zeros((n_in, n_y));
        let mut a_up = ndarray::Array2::<f64>::zeros((n_in, n_y));
        for j in 0..n_in {
            a_lo[[j, 0]] = 1.0 + j as f64;
            a_up[[j, 0]] = 2.0 + j as f64;
            a_lo[[j, 1]] = -0.5;
            a_up[[j, 1]] = 0.5;
        }
        let mk = |a: ndarray::Array2<f64>, b: Vec<f64>| super::super::engine::PassOut {
            a,
            e: None,
            b,
            eb: vec![0.0; n_y],
            coll: None,
            coll_rows: None,
            unst_rows: None,
        };
        let al = mk(a_lo, vec![0.25, -0.5]);
        let au = mk(a_up, vec![0.75, 0.5]);
        let hs = HeadSlack {
            sl_lo: vec![0.01, 0.02],
            sl_up: vec![0.03, 0.04],
        };
        let inactive = halfspaces_for_heads(&al, &au, &hs, &[(0, -1)], 8).expect("build");
        assert_eq!(inactive.len(), 1);
        assert_eq!(
            inactive[0].a,
            vec![1.0, 2.0, 3.0],
            "INACTIVE uses the LOWER line"
        );
        assert!((inactive[0].d - (0.25 - 0.01)).abs() < 1e-15);
        let active = halfspaces_for_heads(&al, &au, &hs, &[(0, 1)], 8).expect("build");
        assert_eq!(
            active[0].a,
            vec![-2.0, -3.0, -4.0],
            "ACTIVE negates the UPPER line"
        );
        assert!((active[0].d - (-(0.75 + 0.03))).abs() < 1e-15);
        // The cap keeps the LAST fixes and a subset is always sound.
        let many =
            halfspaces_for_heads(&al, &au, &hs, &[(0, -1), (1, 1), (0, 1)], 2).expect("build");
        assert_eq!(many.len(), 2);
        assert_eq!(many[1].a, vec![-2.0, -3.0, -4.0]);
        // Out-of-range index fails closed.
        assert!(halfspaces_for_heads(&al, &au, &hs, &[(9, 1)], 8).is_none());
        // A NEGATIVE slack refuses the build in released mode too (the pack-time
        // frame gate is the first line of defence; this is the second, and it
        // is the one a hand-built or corrupted `HeadSlack` would reach).
        let bad_lo = HeadSlack {
            sl_lo: vec![-1e-9, 0.02],
            sl_up: vec![0.03, 0.04],
        };
        assert!(
            halfspaces_for_heads(&al, &au, &bad_lo, &[(0, -1)], 8).is_none(),
            "negative sl_lo must refuse the INACTIVE halfspace"
        );
        assert!(
            halfspaces_for_heads(&al, &au, &bad_lo, &[(0, 1)], 8).is_some(),
            "the ACTIVE halfspace reads sl_up, which is fine here"
        );
        let bad_up = HeadSlack {
            sl_lo: vec![0.01, 0.02],
            sl_up: vec![f64::NAN, 0.04],
        };
        assert!(
            halfspaces_for_heads(&al, &au, &bad_up, &[(0, 1)], 8).is_none(),
            "non-finite sl_up must refuse the ACTIVE halfspace"
        );
    }

    #[test]
    fn negative_head_slack_is_refused_by_the_consumer_and_counted() {
        // End to end through `tighten_head_box`: a corrupted slack must leave
        // the box UNTOUCHED and report `refused_hs`, never tighten from it.
        let n_in = 3usize;
        let n_y = 2usize;
        let mut a_lo = ndarray::Array2::<f64>::zeros((n_in, n_y));
        let mut a_up = ndarray::Array2::<f64>::zeros((n_in, n_y));
        for j in 0..n_in {
            a_lo[[j, 0]] = 1.0 + j as f64;
            a_up[[j, 0]] = 2.0 + j as f64;
            a_lo[[j, 1]] = -0.5;
            a_up[[j, 1]] = 0.5;
        }
        let mk = |a: ndarray::Array2<f64>, b: Vec<f64>| super::super::engine::PassOut {
            a,
            e: None,
            b,
            eb: vec![0.0; n_y],
            coll: None,
            coll_rows: None,
            unst_rows: None,
        };
        let al = mk(a_lo, vec![0.25, -0.5]);
        let au = mk(a_up, vec![0.75, 0.5]);
        let bad = HeadSlack {
            sl_lo: vec![-0.5, 0.0],
            sl_up: vec![0.0, 0.0],
        };
        let mid = vec![0.0; n_in];
        let rad = vec![1.0; n_in];
        let ly_fwd = vec![-7.0, -2.0];
        let uy_fwd = vec![10.0, 2.0];
        let mut ly = ly_fwd.clone();
        let mut uy = uy_fwd.clone();
        let out = tighten_head_box(
            &al,
            &au,
            &bad,
            &ly_fwd,
            &uy_fwd,
            &[(0, -1)],
            &mid,
            &rad,
            &mut ly,
            &mut uy,
            8,
            20,
        );
        assert!(
            out.refused_hs,
            "a negative slack must be REFUSED and counted"
        );
        assert_eq!(out.hs, 0);
        assert_eq!(out.moved, 0);
        assert_eq!(ly, ly_fwd, "refusal must leave the lower box untouched");
        assert_eq!(uy, uy_fwd, "refusal must leave the upper box untouched");
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

/// #clip-stateful: a domain's ACCUMULATED intermediate tightening.
///
/// # The gap this closes
///
/// alpha-beta-CROWN's 2024 -> 2025 cifar100 diff is exactly two lines: it adds
/// `bab.clip_n_verify.clip_interm_domain` and it REMOVES `interm_transfer:
/// False`. The second line is the tell. `interm_transfer: False` means "do not
/// carry per-domain intermediate bounds down the tree", and it is only valid
/// while those bounds never change. Clip-and-Verify makes them change, so abc
/// had to start transferring them: it intersects each round's tightening
/// against the PARENT's stored per-domain interval and writes the result back
/// (`state/intermediate_bounds.py:589-593`), and children inherit the
/// accumulated object.
///
/// ny's port was STATELESS. `engine::clip_tighten_gates` recomputed from the
/// root frame on every call and dropped the result when it returned, so the
/// measured dose-response was flat: 2.1x more tightening produced an IDENTICAL
/// bound and an IDENTICAL frontier. One node's shrink is 1.6e-4 at the split
/// layer and 8.4e-16 elsewhere — immaterial ONCE, a different object when it is
/// accumulated over 13-30 levels at 13-30 different layers.
///
/// # What actually compounds, and what does not
///
/// Be precise about this, because the naive story is wrong. The retained rows
/// (`LayerGates::clip_rows`) are ROOT-frame lines and are never rebuilt, so the
/// dual value for a FIXED neuron against a FIXED halfspace set does not improve
/// just because it is recomputed. Two things do compound:
///
/// * COVERAGE. Each visit tightens at most `topk` neurons per layer. The
///   accumulated set is the UNION over the whole ancestor chain, so a depth-30
///   domain can carry up to 30x the neurons a single stateless call ever
///   touches.
/// * SELECTION (D2). The top-k is rescored against the CURRENT interval rather
///   than the root one, so a neuron that is already narrow drops out of the
///   ranking and a fresh one takes its place. Without that, every level would
///   re-pick the same 20 neurons and coverage would not grow at all.
///
/// A third effect is real but second order: deeper domains carry MORE
/// halfspaces, so a re-tightened neuron can improve. The `merge` below keeps
/// whichever is tighter.
///
/// # Representation
///
/// SPARSE, sorted by `(layer, idx)`. The frontier reaches 400+ live domains and
/// a dense per-domain copy of every layer's `l`/`u` would be hundreds of MB;
/// this is 24 bytes per TIGHTENED neuron and nothing for the rest. Domains
/// share the parent's `Arc` until one of them extends it.
///
/// # Soundness
///
/// Every stored interval ENCLOSES the true range of its neuron over the
/// domain's region:
///
/// * a fresh entry comes from `dual_lower` on the root line plus the root's own
///   certified slack, which is the same construction the stateless path already
///   uses and is valid over any set containing the region;
/// * an inherited entry was valid over the PARENT's region, and the child's
///   region is a subset, so it stays valid verbatim;
/// * [`ClipAcc::merge`] intersects, and an intersection of two valid enclosures
///   is a valid enclosure.
///
/// The induction bottoms out at the root, where the set is empty. Nothing here
/// can widen an interval: `merge` is max/min against what is already stored.
#[derive(Debug, Clone, Default)]
pub struct ClipAcc {
    /// `(layer, neuron index within the layer, lo, hi)`, sorted by the first
    /// two fields. INVARIANT: `lo <= hi` for every entry (see `merge`).
    ent: Vec<(u32, u32, f64, f64)>,
    /// How many ancestor evaluations contributed to this set (R9: separates
    /// "inherited and grew" from "rebuilt at this node").
    generations: u32,
    /// Neurons this node's own visit newly tightened or improved (R9).
    fresh: u32,
    /// Tightenings REFUSED because the merge would have crossed (R9, and the
    /// count that must stay 0 for the fail-closed argument to be uninteresting).
    refused: u32,
}

/// Hard cap on accumulated entries per domain.
///
/// Bounds worst-case memory at `400 domains * 4096 * 24 B ~= 39 MB` on the
/// widest frontier measured. Dropping a tightening past the cap only LOOSENS,
/// so the cap is sound by construction and needs no other justification.
pub const CLIP_ACC_MAX: usize = 4096;

impl ClipAcc {
    /// Entries carried.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ent.len()
    }

    /// Nothing accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ent.is_empty()
    }

    /// Generations of inheritance behind this set.
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generations
    }

    /// Neurons this node's own visit contributed.
    #[must_use]
    pub const fn fresh(&self) -> u32 {
        self.fresh
    }

    /// Merges refused for a crossed interval (fail-closed; see `merge`).
    #[must_use]
    pub const fn refused(&self) -> u32 {
        self.refused
    }

    /// Mark a fresh generation. Called once per node visit, before its merges.
    pub fn open_generation(&mut self) {
        self.generations = self.generations.saturating_add(1);
        self.fresh = 0;
        self.refused = 0;
    }

    /// Record `n` tightenings dropped by the caller rather than by `merge`
    /// (currently: the crossed neurons `tighten_intermediate_acc` already
    /// filtered out, which the stateful arm refuses instead of closing on).
    pub fn note_refused(&mut self, n: usize) {
        self.refused = self
            .refused
            .saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
    }

    fn slot(&self, layer: u32, idx: u32) -> Result<usize, usize> {
        self.ent.binary_search_by(|e| (e.0, e.1).cmp(&(layer, idx)))
    }

    /// Currently stored interval for a neuron, if any.
    #[must_use]
    pub fn get(&self, layer: usize, idx: usize) -> Option<(f64, f64)> {
        let (l, i) = (u32::try_from(layer).ok()?, u32::try_from(idx).ok()?);
        self.slot(l, i).ok().map(|p| (self.ent[p].2, self.ent[p].3))
    }

    /// Every entry for one layer, as `(idx, lo, hi)`.
    pub fn layer_entries(&self, layer: usize) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        let l = u32::try_from(layer).unwrap_or(u32::MAX);
        let start = self.ent.partition_point(|e| e.0 < l);
        self.ent[start..]
            .iter()
            .take_while(move |e| e.0 == l)
            .map(|e| (e.1 as usize, e.2, e.3))
    }

    /// SHRINK-ONLY intersect of `(lo, hi)` into the stored interval.
    ///
    /// Returns `true` when the set actually moved. Refuses — leaving the stored
    /// interval untouched and bumping `refused` — when:
    ///
    /// * either endpoint is non-finite, or `lo > hi` on its own;
    /// * the intersection would CROSS (`max(lo) > min(hi)`);
    /// * the cap is reached and this is a new neuron.
    ///
    /// # Why a crossing REFUSES here instead of closing the domain
    ///
    /// The stateless path treats a crossed interval as a Farkas certificate
    /// that `box ∩ halfspaces` is empty, and that argument is valid there: both
    /// endpoints are derived in one call from the same root frame, so
    /// `l_new > u_new` really does mean no point satisfies all the constraints.
    ///
    /// The accumulated path introduces a NEW failure surface that argument does
    /// not cover: the inherited endpoint's validity rests on the inheritance
    /// actually being parent -> child. It is (see the type docs), but a
    /// crossing here would be the FIRST place an inheritance bug became visible,
    /// and the thing it would do is CLOSE a domain that is not empty — a
    /// false-`unsat`, the one failure this project cannot absorb. So this path
    /// refuses: it keeps the parent interval, loses the tightening, and counts
    /// the event. Cost is tightness; the alternative cost is the moat.
    pub fn merge(&mut self, layer: usize, idx: usize, lo: f64, hi: f64) -> bool {
        let (Ok(l), Ok(i)) = (u32::try_from(layer), u32::try_from(idx)) else {
            return false;
        };
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            self.refused = self.refused.saturating_add(1);
            return false;
        }
        match self.slot(l, i) {
            Ok(p) => {
                let (clo, chi) = (self.ent[p].2, self.ent[p].3);
                let (nlo, nhi) = (clo.max(lo), chi.min(hi));
                if nlo > nhi {
                    self.refused = self.refused.saturating_add(1);
                    return false;
                }
                if nlo > clo || nhi < chi {
                    self.ent[p].2 = nlo;
                    self.ent[p].3 = nhi;
                    self.fresh = self.fresh.saturating_add(1);
                    return true;
                }
                false
            }
            Err(p) => {
                if self.ent.len() >= CLIP_ACC_MAX {
                    return false;
                }
                self.ent.insert(p, (l, i, lo, hi));
                self.fresh = self.fresh.saturating_add(1);
                true
            }
        }
    }
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
    tighten_intermediate_acc(layer, 0, hs, x0, eps, topk, None)
}

/// [`tighten_intermediate`], scored and intersected against a domain's
/// ACCUMULATED intervals instead of the root ones (#clip-stateful, D1+D2).
///
/// `acc = None` reproduces [`tighten_intermediate`] EXACTLY — same ranking, same
/// baseline, same output — which is what keeps the dark arm byte-identical.
///
/// With `acc = Some(..)`:
///
/// * D2, SELECTION: the top-k ranking reads the CURRENT interval, so already
///   narrowed neurons fall out of the ranking and fresh ones enter. This is the
///   half that makes coverage grow with depth rather than re-picking the same
///   `topk` every level.
/// * D1, BASELINE: `max`/`min` is taken against the CURRENT interval, so the
///   returned `Tightened` is the intersection of everything the ancestor chain
///   has proved with what this visit proves.
///
/// # THE ORDERING TRAP, and why this does not re-open it
///
/// `backward_interm.rs:37` and the `NY_MARGIN_ROW_BACKWARD_INTERM` declaration
/// document a real false-`unsat` generator: `LayerGates::clip_rows` carries a
/// slack (`sl_lo`/`sl_up`) recovered as the gap between the retained LINE's own
/// box-minimum and the bound the lane PUBLISHED. Calibrate that gap against an
/// already-TIGHTENED published `l` and it under-recovers, clamps to zero, and
/// every halfspace built from the line then cuts into the TRUE subdomain.
/// `root.rs` avoids this by calibrating against the forward-only bounds captured
/// BEFORE the backward-intermediate phase.
///
/// This function CANNOT re-open it, and the reason is structural rather than
/// careful: the accumulated intervals are never fed back into row construction.
/// `rows.m`, `rows.d`, `rows.bm`, `rows.bd`, `rows.sl_lo`, `rows.sl_up` are read
/// verbatim from the frozen root record, and `halfspaces_for_splits` — the only
/// other consumer of those rows — takes `&RootGates` and a split list and never
/// sees a `ClipAcc` at all. The accumulated numbers enter at exactly two places,
/// both AFTER the dual has been solved: the `order` ranking (which cannot affect
/// soundness, only which neurons get looked at) and the final `max`/`min`
/// (an intersection of two independently valid enclosures). If a future change
/// wants the tightened bounds to influence the LINES, it must re-derive the
/// slack against the untightened frame first.
#[must_use]
pub fn tighten_intermediate_acc(
    layer: &LayerGates,
    li: usize,
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
    topk: usize,
    acc: Option<&ClipAcc>,
) -> (Vec<Tightened>, usize) {
    let (out, crossed, _) = tighten_intermediate_acc_crossing(layer, li, hs, x0, eps, topk, acc);
    (out, crossed)
}

/// [`tighten_intermediate_acc`] plus the largest CROSSING `l_new - u_new` over
/// the crossed neurons (`f64::NEG_INFINITY` when none crossed).
///
/// #mr-close-telemetry: that crossing is the certified slack of the VERIFY
/// half's emptiness certificate — how far past `l_new == u_new` the
/// certificate sits — which is what the proof-integrity receipt reports for a
/// domain closed by clip-and-verify. Pure telemetry: the tightening, the
/// ranking and `crossed` are computed exactly as before; the extra value is a
/// `max` over numbers already computed and read by no decision.
#[must_use]
pub fn tighten_intermediate_acc_crossing(
    layer: &LayerGates,
    li: usize,
    hs: &[Halfspace],
    x0: &[f64],
    eps: &[f64],
    topk: usize,
    acc: Option<&ClipAcc>,
) -> (Vec<Tightened>, usize, f64) {
    let Some(rows) = layer.clip_rows.as_ref() else {
        return (Vec::new(), 0, f64::NEG_INFINITY);
    };
    if ny_core::require_f64_interval_proof_environment().is_err()
        || hs.is_empty()
        || topk == 0
        || x0.len() != eps.len()
        || x0.iter().any(|v| !v.is_finite())
        || eps.iter().any(|&v| !v.is_finite() || v < 0.0)
        || !legacy_rows_shape(rows, layer.unst.len(), x0.len())
        || layer
            .unst
            .iter()
            .any(|&i| i >= layer.l.len() || i >= layer.u.len())
    {
        return (Vec::new(), 0, f64::NEG_INFINITY);
    }
    // Current interval for a neuron: the accumulated one where the chain has
    // narrowed it, the frozen root one otherwise. Both enclose the true range
    // over this domain's region.
    let cur = |idx: usize| -> (f64, f64) {
        acc.and_then(|a| a.get(li, idx))
            .unwrap_or((layer.l[idx], layer.u[idx]))
    };
    // Widest-relaxation first: `-l*u/(u-l)` is the triangle's height, i.e. the
    // slack a tightening here removes from every path through this neuron.
    // D2: scored off `cur`, so the ranking FOLLOWS the shrink.
    let mut order: Vec<(f64, usize)> = layer
        .unst
        .iter()
        .enumerate()
        .filter_map(|(pos, &i)| {
            let (l, u) = cur(i);
            let w = u - l;
            (w > 0.0 && l < 0.0 && u > 0.0).then(|| (-l * u / w, pos))
        })
        .collect();
    order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    order.truncate(topk);

    let n_in = x0.len();
    let mut out = Vec::with_capacity(order.len());
    let mut crossed = 0usize;
    // #mr-close-telemetry: the largest crossing seen (telemetry only).
    let mut max_crossing = f64::NEG_INFINITY;
    let mut lo_obj = vec![0.0f64; n_in];
    let mut up_obj = vec![0.0f64; n_in];
    for (_, pos) in order {
        let idx = layer.unst[pos];
        let (cl, cu) = cur(idx);
        // Source slack, line reconstruction, dual folding, box products and
        // final constants are independent obligations. Keep every charge;
        // neither the nominal dual nor a relative-to-delta allowance may
        // establish the emptiness certificate.
        let lower_line = legacy_row_objective(rows, pos, true, &mut lo_obj, |j| {
            clip_add_up(x0[j].abs(), eps[j])
        });
        let upper_line = legacy_row_objective(rows, pos, false, &mut up_obj, |j| {
            clip_add_up(x0[j].abs(), eps[j])
        });
        let l_new = lower_line
            .and_then(|(bias, error)| {
                let dual = dual_lower_outward(&lo_obj, hs, x0, eps)?;
                clip_add_down(clip_add_down(dual.lower, bias)?, -error)
            })
            .map_or(cl, |v| v.max(cl));
        let u_new = upper_line
            .and_then(|(bias, error)| {
                let dual = dual_lower_outward(&up_obj, hs, x0, eps)?;
                clip_add_up(clip_add_up(-dual.lower, bias)?, error)
            })
            .map_or(cu, |v| v.min(cu));
        // Shrink-only. CROSSED bounds are not an error to discard — they are
        // the VERIFY half of Clip-and-Verify: `l_new` under-estimates the
        // constrained min and `u_new` over-estimates the constrained max, so
        // `l_new > u_new` is a Farkas certificate that `box ∩ halfspaces` is
        // EMPTY, and the caller can discharge the whole domain.
        //
        // The STATEFUL caller does not exploit that: `ClipAcc::merge` refuses a
        // crossing instead, because on the accumulated path one endpoint may be
        // inherited and a crossing would then be the first symptom of an
        // inheritance bug — see the argument on `merge`.
        if l_new > u_new {
            crossed += 1;
            // Report a directed lower bound on the crossing, rather than
            // overstating the certificate's margin through subtraction.
            max_crossing =
                max_crossing.max(clip_add_down(l_new, -u_new).unwrap_or(f64::NEG_INFINITY));
        } else if l_new > cl || u_new < cu {
            out.push(Tightened {
                pos,
                idx,
                l: l_new,
                u: u_new,
            });
        }
    }
    (out, crossed, max_crossing)
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

/// #clip-stateful R9 observable.
///
/// The assignment this shipped under is explicit that a mechanism without
/// telemetry separating "did not run" / "ran and did nothing" / "ran and
/// compounded" is unfalsifiable and its null is VACUOUS — this lane already
/// shipped one lever with no observable, measured it, got a bit-identical
/// result, and had to VOID the whole A/B. The three states here:
///
/// * NO `[clip-acc]` line at all — the gate is dark or the frame check failed.
/// * `gen=N carried=0` or `carried>0 fresh=0 shrink=0.000e0` — ran, no effect.
/// * `carried` growing with `depth` and a non-zero `shrink` — ran and compounded.
///
/// `shrink` is measured against the ROOT interval, which is the number that
/// distinguishes accumulation from re-deriving the same 1.6e-4 every visit:
/// `mean(1 - width_acc / width_root)` over the carried entries.
pub fn report_acc(
    depth: usize,
    generations: u32,
    carried: usize,
    fresh: u32,
    refused: u32,
    mean_shrink: f64,
    max_shrink: f64,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static MAXCARRY: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let peak = MAXCARRY.fetch_max(carried, Ordering::Relaxed).max(carried);
    if n < 12 || n.is_multiple_of(128) {
        eprintln!(
            "[clip-acc] call={n} depth={depth} gen={generations} carried={carried} peak={peak} \
             fresh={fresh} refused={refused} shrink_mean={mean_shrink:.4e} \
             shrink_max={max_shrink:.4e}"
        );
    }
}

/// #clip-stateful R9: the gate is armed but a PREREQUISITE is not, so the path
/// returns without doing anything.
///
/// The distinction this exists for: a silent `None` here is indistinguishable
/// from a dark build, and "measured a lever that was never actually armed" is
/// the single most repeated failure in this campaign's ledger. One line, once.
pub fn report_acc_unarmed(why: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!("[clip-acc] ARMED BUT INERT: {why} — any measurement from this run is VACUOUS");
    });
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

#[cfg(test)]
mod acc_tests {
    use super::{ClipAcc, CLIP_ACC_MAX};

    #[test]
    fn merge_is_shrink_only_and_never_widens() {
        let mut a = ClipAcc::default();
        assert!(a.merge(3, 7, -1.0, 2.0));
        // A LOOSER proposal must leave the stored interval untouched — this is
        // the whole inheritance contract: a child may only narrow.
        assert!(!a.merge(3, 7, -5.0, 9.0));
        assert_eq!(a.get(3, 7), Some((-1.0, 2.0)));
        // A tighter one on one side only moves that side.
        assert!(a.merge(3, 7, -0.25, 9.0));
        assert_eq!(a.get(3, 7), Some((-0.25, 2.0)));
        assert!(a.merge(3, 7, -5.0, 0.5));
        assert_eq!(a.get(3, 7), Some((-0.25, 0.5)));
    }

    #[test]
    fn a_crossed_merge_refuses_and_keeps_the_parent_interval() {
        // THE MOAT CASE. On the accumulated path one endpoint may be inherited,
        // so a crossing is not accepted as an emptiness certificate: the stored
        // interval must survive verbatim and the event must be COUNTED, because
        // a silent drop is indistinguishable from "the mechanism did nothing".
        let mut a = ClipAcc::default();
        assert!(a.merge(0, 1, -1.0, 1.0));
        a.open_generation();
        assert!(!a.merge(0, 1, 2.0, 3.0), "crossed merge must be refused");
        assert_eq!(a.get(0, 1), Some((-1.0, 1.0)));
        assert_eq!(a.refused(), 1);
        // Self-inconsistent input is refused on the same path.
        assert!(!a.merge(0, 2, 1.0, -1.0));
        assert!(!a.merge(0, 3, f64::NAN, 1.0));
        assert_eq!(a.get(0, 2), None);
        assert_eq!(a.refused(), 3);
    }

    #[test]
    fn entries_stay_sorted_and_layer_scoped() {
        let mut a = ClipAcc::default();
        for (l, i) in [(2usize, 5usize), (0, 9), (2, 1), (1, 0), (0, 3)] {
            #[allow(clippy::cast_precision_loss)]
            let (lo, hi) = (-(i as f64), i as f64 + 1.0);
            a.merge(l, i, lo, hi);
        }
        assert_eq!(a.len(), 5);
        let l0: Vec<usize> = a.layer_entries(0).map(|(i, _, _)| i).collect();
        let l2: Vec<usize> = a.layer_entries(2).map(|(i, _, _)| i).collect();
        assert_eq!(l0, vec![3, 9], "layer 0 entries, ascending");
        assert_eq!(l2, vec![1, 5], "layer 2 entries, ascending");
        assert_eq!(a.layer_entries(7).count(), 0);
    }

    #[test]
    fn the_cap_refuses_new_neurons_but_still_tightens_known_ones() {
        // Dropping past the cap only LOOSENS, which is why the cap needs no
        // soundness argument beyond this: known neurons must keep narrowing.
        let mut a = ClipAcc::default();
        for i in 0..CLIP_ACC_MAX {
            assert!(a.merge(0, i, -1.0, 1.0));
        }
        assert_eq!(a.len(), CLIP_ACC_MAX);
        assert!(!a.merge(0, CLIP_ACC_MAX, -1.0, 1.0));
        assert_eq!(a.len(), CLIP_ACC_MAX);
        assert!(a.merge(0, 5, -0.5, 1.0));
        assert_eq!(a.get(0, 5), Some((-0.5, 1.0)));
    }

    #[test]
    fn generations_count_inheritance_not_entries() {
        // R9: `gen` must separate "inherited and grew" from "rebuilt here".
        let mut a = ClipAcc::default();
        a.open_generation();
        a.merge(0, 0, -1.0, 1.0);
        assert_eq!((a.generation(), a.fresh()), (1, 1));
        let mut child = a.clone();
        child.open_generation();
        assert_eq!(
            (child.generation(), child.fresh(), child.len()),
            (2, 0, 1),
            "a child inherits the entries and resets only the per-visit counters"
        );
        child.merge(0, 1, -2.0, 2.0);
        assert_eq!((child.fresh(), child.len()), (1, 2));
        assert_eq!(a.len(), 1, "the parent set must not alias the child's");
    }
}
