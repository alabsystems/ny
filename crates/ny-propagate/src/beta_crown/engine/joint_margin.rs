// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-domain JOINT-MARGIN closer for same-LHS conjunctive (max-diff) verification.
//!
//! # Why this exists (acasxu prop_2 divergence, task #40)
//!
//! Same-LHS conjunctions (ACAS Xu prop_2/3/4) reduce to proving
//! `min_x max_j signed_diff_j(x) > 0` over an input box, encoded as an augmented
//! net `[orig] -> Linear(k signed diffs) -> Reshape -> MaxPool(1,k) -> Reshape`.
//! CROWN's per-domain lower bound on that scalar routes the MaxPool *lower*
//! relaxation through a SINGLE input `i* = argmax_j l_j` (see
//! `layers/pooling/max.rs`): the lower bound of `max(z_1..z_k)` is taken as one
//! `z_{i*}`. That cannot express "different conjuncts dominate different
//! sub-regions of the box" — so when every conjunct is individually falsifiable
//! somewhere in the box (measured on 1_5/prop_2: `Y_1-Y_0 = -3.7e-4 < 0`), no
//! single-conjunct choice stays positive and the plain-CROWN input-split BaB
//! DIVERGES: verified-count plateaus while the queue explodes (the "hard shell",
//! diag c7126554).
//!
//! # What this does
//!
//! For a domain box `[lo, hi]`, take the k per-conjunct CROWN *linear-in-input*
//! LOWER bounds `signed_diff_j(x) >= a_j·x + b_j` (certified over the box after
//! folding coefficient error into the bias), and certify a lower bound on the
//! JOINT margin
//!
//! ```text
//!   m* = min_{x in box} max_j (a_j·x + b_j)  <=  min_{x in box} max_j signed_diff_j(x).
//! ```
//!
//! `m*` is what the single-conjunct MaxPool relaxation cannot see: it lets the k
//! conjuncts *jointly* cover the box. If `m* > 0` the whole box is safe in ONE
//! bound with no further splitting.
//!
//! # Soundness (independent of the internal search)
//!
//! For ANY simplex weights `λ` (`λ_j >= 0`, `Σ λ_j = 1`),
//! `max_j g_j(x) >= Σ_j λ_j g_j(x)`, so
//! `m* >= min_{x in box} Σ_j λ_j (a_j·x + b_j)`, and the right-hand side is an
//! affine function of `x` whose minimum over the box is computed exactly (per
//! dim: `min(c_i·lo_i, c_i·hi_i)`), in f64, then rounded DOWN with a conservative
//! slack. That yields a rigorous lower bound on `m*` for *whatever* `λ` we pass —
//! the multiplicative-weights search below only chooses a *good* `λ` (tighter
//! bound); it can never make the result unsound. Returned value is only ever used
//! to RAISE a domain's lower bound (`max(crown_lb, joint_lb)`), so a loose or
//! missed bound loses precision, never soundness.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_core::GemmEngine;
use ny_tensor::{next_down_f32, BoundedTensor};

use crate::Network;

/// Number of multiplicative-weights (exponentiated-gradient) iterations used to
/// search the conjunct-mixture simplex. Each iteration is O(k·d) (k conjuncts,
/// d input dims) — negligible for the tiny same-LHS nets (k <= ~4, d = 5).
const MW_ITERS: usize = 64;
/// Exponentiated-gradient step. Applied to the per-conjunct sub-gradient after
/// normalizing its spread to `[-1, 0]`, so it is scale-invariant across the wide
/// range of joint-margin magnitudes seen across BaB depth (~1e0 down to ~1e-5).
const MW_ETA: f64 = 4.0;

/// Per-domain joint-margin closer for same-LHS conjunctive (max-diff) specs.
///
/// Holds the max-diff augmented network with its final
/// `[Reshape, MaxPool, Reshape]` tail STRIPPED — i.e. `[orig] + Linear(signed
/// diffs)`. Its `k` outputs ARE the per-conjunct signed differences
/// `g_j(x) = signed_diff_j(x)` whose pointwise max is the max-diff objective, so
/// a single CROWN-with-linear pass on it yields all `k` per-conjunct linear
/// lower bounds at once.
pub struct JointMarginCloser {
    truncated: Arc<Network>,
    num_conjuncts: usize,
}

impl std::fmt::Debug for JointMarginCloser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JointMarginCloser")
            .field("num_conjuncts", &self.num_conjuncts)
            .field("truncated_layers", &self.truncated.num_layers())
            .finish()
    }
}

impl JointMarginCloser {
    /// Build a closer from the maxpool-stripped signed-diff network and its
    /// conjunct count `k`.
    pub fn new(truncated: Arc<Network>, num_conjuncts: usize) -> Self {
        Self {
            truncated,
            num_conjuncts,
        }
    }

    /// Certify a lower bound on `min_{x in box} max_j signed_diff_j(x)`, or
    /// `None` if it cannot be computed / cannot beat the trivial case.
    ///
    /// The returned value is a rigorous lower bound on the joint max-diff margin
    /// over `input_bounds` (see module docs: sound for any internal `λ`).
    /// `engine` is intentionally not forwarded to the inner CROWN pass — the
    /// same-LHS nets are tiny (6x50), so the pure-CPU fast path avoids per-domain
    /// GPU launch latency.
    pub(crate) fn certified_joint_lower_bound(
        &self,
        input_bounds: &BoundedTensor,
        _engine: Option<&dyn GemmEngine>,
        crown_lb: f32,
        threshold: f32,
    ) -> Option<f32> {
        if self.num_conjuncts < 2 {
            // A single conjunct's max IS that conjunct — the MaxPool relaxation is
            // already exact, nothing to gain.
            diag_record(DiagKind::SkippedTrivial, crown_lb, threshold);
            return None;
        }

        // 1. CROWN linear-in-input bounds for the k signed diffs (no maxpool tail).
        let (_out, mut lin) = match self.truncated.propagate_crown_with_linear(input_bounds) {
            Ok(v) => v,
            Err(_) => {
                diag_record(DiagKind::CrownFailed, crown_lb, threshold);
                return None;
            }
        };

        let k = lin.lower_a.nrows();
        let d = lin.lower_a.ncols();
        if k != self.num_conjuncts || k < 2 || d == 0 {
            diag_record(DiagKind::ShapeMismatch, crown_lb, threshold);
            return None;
        }

        // 2. Fold certified coefficient error into the bias over THIS box, giving
        //    error-free affine lower bounds  signed_diff_j(x) >= a_j·x + b_j.
        let flat = input_bounds.flatten();
        let lo_f: Vec<f32> = flat.lower().iter().copied().collect();
        let hi_f: Vec<f32> = flat.upper().iter().copied().collect();
        if lo_f.len() != d || hi_f.len() != d {
            return None;
        }
        lin.fold_coeff_err_into_bias(&lo_f, &hi_f);

        let a = &lin.lower_a; // (k, d), certified lower-bound coefficients
        let b = &lin.lower_b; // (k,)
                              // Any non-finite affine piece (e.g. a conservative -inf bias fallback)
                              // makes the joint search meaningless; bail to the CROWN bound.
        if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
            diag_record(DiagKind::NonFinite, crown_lb, threshold);
            return None;
        }

        let lo: Vec<f64> = lo_f.iter().map(|&v| v as f64).collect();
        let hi: Vec<f64> = hi_f.iter().map(|&v| v as f64).collect();

        // 3a. Seed with all k pure-conjunct vertices e_j. The single-conjunct
        //     MaxPool relaxation CROWN already applies is exactly `min_x g_{i*}`
        //     for one i*, so `max_j (min_x g_j)` — the best vertex — is >= the
        //     CROWN bound. Seeding guarantees the returned joint bound dominates
        //     CROWN regardless of where the multiplicative-weights search lands.
        let mut best_val = f64::NEG_INFINITY;
        let mut best_lam = vec![1.0f64 / k as f64; k];
        for j in 0..k {
            let mut vtx = vec![0.0f64; k];
            vtx[j] = 1.0;
            let (val, _) = eval_lambda(a, b, &lo, &hi, &vtx);
            if val > best_val {
                best_val = val;
                best_lam.copy_from_slice(&vtx);
            }
        }

        // 3b. Multiplicative-weights search for a good conjunct mixture λ. Every
        //     evaluated λ yields a valid lower bound; we return the best certified
        //     one, so the search only ever tightens past the vertex seed.
        let mut lam = vec![1.0f64 / k as f64; k];

        for _ in 0..MW_ITERS {
            let (val, sub) = eval_lambda(a, b, &lo, &hi, &lam);
            if val > best_val {
                best_val = val;
                best_lam.copy_from_slice(&lam);
            }
            // Exponentiated gradient ascent on the concave objective. Normalize
            // the sub-gradient spread so a fixed η adapts to the margin scale.
            let smax = sub.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let smin = sub.iter().copied().fold(f64::INFINITY, f64::min);
            let spread = smax - smin;
            if !spread.is_finite() || spread <= 1e-30 {
                // All conjuncts tie at the aggregate minimizer — converged.
                break;
            }
            let mut sum = 0.0f64;
            for j in 0..k {
                lam[j] *= (MW_ETA * (sub[j] - smax) / spread).exp();
                sum += lam[j];
            }
            if !sum.is_finite() || sum <= 0.0 {
                break;
            }
            let inv = 1.0 / sum;
            for l in lam.iter_mut() {
                *l *= inv;
            }
        }

        if !best_val.is_finite() {
            diag_record(DiagKind::NonFinite, crown_lb, threshold);
            return None;
        }

        // 4. Rigorous certificate for the best λ: recompute the aggregate-affine
        //    box minimum with a conservative slack, then round DOWN to f32.
        let cert = certified_box_min(a, b, &lo, &hi, &best_lam);
        diag_record(DiagKind::Computed(cert), crown_lb, threshold);
        Some(cert)
    }

    /// #bab-frontier v2 (b): the per-row MINIMIZER corners of the k certified
    /// affine lower bounds over `input_bounds` — for conjunct row j, the box
    /// corner minimizing `a_j·x` (`x_d = lo_d` if `a[j,d] > 0` else `hi_d`).
    /// These are exactly the corners where conjunct j's certified lower bound
    /// bottoms out, i.e. the most violation-likely corners of the subbox.
    ///
    /// ATTACK-SIDE GUIDANCE ONLY (see `bab_frontier_export`): the corners are
    /// consumed as APGD restart seeds; every candidate still passes the
    /// unchanged trusted-ORT + zero-tolerance acceptance gate, so a wrong
    /// corner can only spend otherwise-dead leftover budget. Returns `None`
    /// when the inner CROWN pass fails, shapes mismatch, or any coefficient is
    /// non-finite (fail-closed to "no corners" — the consumer falls back to
    /// the subbox's own extreme corners).
    pub(crate) fn per_row_minimizer_corners(
        &self,
        input_bounds: &BoundedTensor,
    ) -> Option<Vec<Vec<f32>>> {
        let (_out, mut lin) = self
            .truncated
            .propagate_crown_with_linear(input_bounds)
            .ok()?;
        let k = lin.lower_a.nrows();
        let d = lin.lower_a.ncols();
        if k == 0 || d == 0 {
            return None;
        }
        let flat = input_bounds.flatten();
        let lo: Vec<f32> = flat.lower().iter().copied().collect();
        let hi: Vec<f32> = flat.upper().iter().copied().collect();
        if lo.len() != d || hi.len() != d {
            return None;
        }
        // Same coefficient-error fold as the certified path: the corner picks
        // below then read the error-free coefficient signs over THIS box.
        lin.fold_coeff_err_into_bias(&lo, &hi);
        if lin.lower_a.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut corners: Vec<Vec<f32>> = Vec::with_capacity(k);
        for j in 0..k {
            let row: Vec<f32> = lin.lower_a.row(j).iter().copied().collect();
            let corner = row_minimizer_corner(&row, &lo, &hi);
            if !corners.contains(&corner) {
                corners.push(corner);
            }
        }
        Some(corners)
    }
}

/// The box corner minimizing the affine form `a·x` over `[lo, hi]`:
/// `x_d = lo_d` if `a_d > 0` else `hi_d` (a zero coefficient contributes
/// nothing, so either end is a minimizer there; `hi_d` is picked to match the
/// documented rule exactly). Pure for the #bab-frontier corner-seed oracle.
pub(crate) fn row_minimizer_corner(a_row: &[f32], lo: &[f32], hi: &[f32]) -> Vec<f32> {
    a_row
        .iter()
        .zip(lo.iter().zip(hi))
        .map(|(&a, (&l, &h))| if a > 0.0 { l } else { h })
        .collect()
}

/// Diagnostic outcomes for the env-gated (`NY_JOINT_MARGIN_DIAG`) accounting.
enum DiagKind {
    /// Skipped: fewer than 2 conjuncts.
    SkippedTrivial,
    /// The inner CROWN-with-linear pass errored.
    CrownFailed,
    /// The truncated net produced a row count != num_conjuncts.
    ShapeMismatch,
    /// A per-conjunct affine piece was non-finite (or the search diverged).
    NonFinite,
    /// A certified joint bound was produced (its value).
    Computed(f32),
}

/// Env-gated per-domain accounting that distinguishes *why* the joint-margin
/// closer does or does not help: how often it actually produced a bound vs
/// bailed, how that bound compared to the CROWN bound it must beat (raw signed
/// delta, so a *looser* joint reads as negative), and how often it crossed the
/// verification threshold (the only outcome that changes BaB exploration).
/// Zero overhead unless `NY_JOINT_MARGIN_DIAG` is set.
fn diag_record(kind: DiagKind, crown_lb: f32, threshold: f32) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ENABLED: AtomicU64 = AtomicU64::new(u64::MAX);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static COMPUTED: AtomicU64 = AtomicU64::new(0);
    static BAILED: AtomicU64 = AtomicU64::new(0);
    static RAW_IMPROVED: AtomicU64 = AtomicU64::new(0); // cert > crown_lb
    static FLIPPED: AtomicU64 = AtomicU64::new(0); // crown<=thr<cert
    static MAX_DELTA_BITS: AtomicU64 = AtomicU64::new((f64::NEG_INFINITY).to_bits());

    let mut on = ENABLED.load(Ordering::Relaxed);
    if on == u64::MAX {
        on = u64::from(std::env::var_os("NY_JOINT_MARGIN_DIAG").is_some());
        ENABLED.store(on, Ordering::Relaxed);
    }
    if on == 0 {
        return;
    }

    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    match kind {
        DiagKind::Computed(cert) => {
            COMPUTED.fetch_add(1, Ordering::Relaxed);
            let delta = (cert - crown_lb) as f64; // signed: negative = looser
            let mut cur = MAX_DELTA_BITS.load(Ordering::Relaxed);
            // Lock-free max update: terminates via CAS success/exhaustion, not float
            // stepping, and `<` deliberately rejects NaN deltas (never stored as max).
            #[allow(clippy::while_float)]
            while f64::from_bits(cur) < delta {
                match MAX_DELTA_BITS.compare_exchange_weak(
                    cur,
                    delta.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(prev) => cur = prev,
                }
            }
            if cert > crown_lb {
                RAW_IMPROVED.fetch_add(1, Ordering::Relaxed);
            }
            if crown_lb <= threshold && cert > threshold {
                FLIPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
        _ => {
            BAILED.fetch_add(1, Ordering::Relaxed);
        }
    }
    if calls.is_multiple_of(20_000) {
        eprintln!(
            "[NY_JOINT_MARGIN_DIAG] calls={calls} computed={} bailed={} raw_improved={} flipped={} max_signed_delta={:.3e}",
            COMPUTED.load(Ordering::Relaxed),
            BAILED.load(Ordering::Relaxed),
            RAW_IMPROVED.load(Ordering::Relaxed),
            FLIPPED.load(Ordering::Relaxed),
            f64::from_bits(MAX_DELTA_BITS.load(Ordering::Relaxed)),
        );
    }
}

/// Evaluate the aggregate objective `min_{x in box} Σ_j λ_j (a_j·x + b_j)` and
/// its super-gradient w.r.t. λ (the concave objective's sub-gradient for ascent).
///
/// Returns `(value, subgrad)` where `subgrad[j] = a_j·x* + b_j` at the aggregate
/// minimizer `x*` — i.e. conjunct j's own affine value at `x*`.
fn eval_lambda(
    a: &Array2<f32>,
    b: &Array1<f32>,
    lo: &[f64],
    hi: &[f64],
    lam: &[f64],
) -> (f64, Vec<f64>) {
    let k = lam.len();
    let d = lo.len();

    // Aggregate coefficients c_i = Σ_j λ_j a[j,i].
    let mut c = vec![0.0f64; d];
    for j in 0..k {
        let lj = lam[j];
        for i in 0..d {
            c[i] += lj * a[[j, i]] as f64;
        }
    }

    // Per-dim box minimizer and the affine box minimum.
    let mut xstar = vec![0.0f64; d];
    let mut val = 0.0f64;
    for i in 0..d {
        let t_lo = c[i] * lo[i];
        let t_hi = c[i] * hi[i];
        if t_lo <= t_hi {
            xstar[i] = lo[i];
            val += t_lo;
        } else {
            xstar[i] = hi[i];
            val += t_hi;
        }
    }
    for j in 0..k {
        val += lam[j] * b[j] as f64;
    }

    // Sub-gradient s_j = g_j(x*) = a_j·x* + b_j.
    let mut sub = vec![0.0f64; k];
    for j in 0..k {
        let mut g = b[j] as f64;
        for i in 0..d {
            g += a[[j, i]] as f64 * xstar[i];
        }
        sub[j] = g;
    }

    (val, sub)
}

/// Rigorous lower bound on `min_{x in box} Σ_j λ_j (a_j·x + b_j)` for a fixed λ,
/// as an `f32` rounded strictly downward past a conservative f64-accumulation
/// slack. Sound for any `λ` on the simplex (see module docs).
fn certified_box_min(a: &Array2<f32>, b: &Array1<f32>, lo: &[f64], hi: &[f64], lam: &[f64]) -> f32 {
    let k = lam.len();
    let d = lo.len();

    let mut c = vec![0.0f64; d];
    for j in 0..k {
        let lj = lam[j];
        for i in 0..d {
            c[i] += lj * a[[j, i]] as f64;
        }
    }

    // Aggregate affine box minimum + running magnitude for the error slack.
    let mut val = 0.0f64;
    let mut mag = 0.0f64;
    for i in 0..d {
        let t_lo = c[i] * lo[i];
        let t_hi = c[i] * hi[i];
        let t = t_lo.min(t_hi);
        val += t;
        mag += t_lo.abs().max(t_hi.abs());
    }
    for j in 0..k {
        let term = lam[j] * b[j] as f64;
        val += term;
        mag += term.abs();
    }

    // Conservative f64-accumulation slack: the aggregate uses O(k·d) additions,
    // each with relative error <= 2^-53. `1e-12·(|val| + mag)` dominates that by
    // several orders of magnitude while staying astronomically below the f32
    // joint-margin scale (~1e-4..1e-5) it must not falsely clear.
    let slack = 1e-12 * (val.abs() + mag) + 1e-30;
    next_down_f32((val - slack) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Two conjuncts that each dip below zero in one corner but whose pointwise
    /// max stays strictly positive everywhere: the joint bound must be positive,
    /// while any single conjunct's box minimum is negative.
    #[test]
    fn joint_bound_beats_single_conjunct_on_crossing_pair() {
        // g_0(x) = x        (min over [-1,1] = -1)
        // g_1(x) = -x       (min over [-1,1] = -1)
        // max(g_0,g_1) = |x| >= 0 everywhere; min = 0 at x=0.
        let a = array![[1.0f32], [-1.0f32]];
        let b = array![0.0f32, 0.0f32];
        let lo = vec![-1.0f64];
        let hi = vec![1.0f64];

        // Uniform λ = (0.5, 0.5): aggregate = 0·x + 0, box min = 0.
        let v_uniform = certified_box_min(&a, &b, &lo, &hi, &[0.5, 0.5]);
        assert!(v_uniform <= 0.0 && v_uniform > -1e-3, "got {v_uniform}");

        // Any pure conjunct is strictly negative over the box.
        let v0 = certified_box_min(&a, &b, &lo, &hi, &[1.0, 0.0]);
        let v1 = certified_box_min(&a, &b, &lo, &hi, &[0.0, 1.0]);
        assert!(v0 < -0.9 && v1 < -0.9, "v0={v0} v1={v1}");
        // The mixture is strictly better than either pure choice.
        assert!(v_uniform > v0 && v_uniform > v1);
    }

    /// A strictly-safe crossing pair: max stays >= +m > 0, joint bound certifies it.
    #[test]
    fn joint_bound_certifies_strictly_positive_margin() {
        // g_0(x) = x + 0.25, g_1(x) = -x + 0.25 over [-1,1].
        // max = |x| + 0.25 >= 0.25 > 0 everywhere. Uniform λ aggregate = 0.25.
        let a = array![[1.0f32], [-1.0f32]];
        let b = array![0.25f32, 0.25f32];
        let lo = vec![-1.0f64];
        let hi = vec![1.0f64];
        let v = certified_box_min(&a, &b, &lo, &hi, &[0.5, 0.5]);
        assert!(v > 0.0, "expected certified positive joint margin, got {v}");
        assert!(
            v <= 0.25 + 1e-6,
            "must not exceed the true minimum, got {v}"
        );
    }

    /// #bab-frontier v2 (b) corner oracle: the per-row minimizer corner is a
    /// TRUE corner of the box (every coordinate at lo or hi) and it minimizes
    /// the row's affine form over the box.
    #[test]
    fn row_minimizer_corner_is_true_corner_and_minimizes_row() {
        let lo = [-1.0f32, 0.0, -2.0, 0.5];
        let hi = [1.0f32, 3.0, -0.5, 0.5];
        for row in [
            [1.0f32, -2.0, 0.0, 4.0],
            [-1.0f32, -1.0, -1.0, -1.0],
            [0.0f32, 0.0, 0.0, 0.0],
            [3.0f32, 2.0, 1.0, 0.5],
        ] {
            let c = row_minimizer_corner(&row, &lo, &hi);
            assert_eq!(c.len(), 4);
            let val =
                |x: &[f32]| -> f64 { row.iter().zip(x).map(|(&a, &v)| a as f64 * v as f64).sum() };
            for d in 0..4 {
                assert!(
                    c[d] == lo[d] || c[d] == hi[d],
                    "corner coord {d} = {} is not an endpoint of [{}, {}]",
                    c[d],
                    lo[d],
                    hi[d]
                );
                // Documented rule: lo on positive coefficients, hi otherwise.
                let expect = if row[d] > 0.0 { lo[d] } else { hi[d] };
                assert_eq!(c[d], expect, "coord {d} violates the minimizer rule");
            }
            // Exhaustive corner check: no other corner is strictly smaller.
            for mask in 0..16u32 {
                let other: Vec<f32> = (0..4)
                    .map(|d| if mask & (1 << d) != 0 { hi[d] } else { lo[d] })
                    .collect();
                assert!(
                    val(&c) <= val(&other) + 1e-12,
                    "corner {c:?} is not a minimizer (beaten by {other:?})"
                );
            }
        }
    }

    /// The certified value must never exceed the true aggregate box minimum
    /// (soundness: it is a *lower* bound).
    #[test]
    fn certified_box_min_is_a_lower_bound() {
        let a = array![[2.0f32, -1.0f32], [-1.0f32, 3.0f32]];
        let b = array![0.5f32, -0.25f32];
        let lo = vec![-0.5f64, -1.0f64];
        let hi = vec![1.0f64, 0.5f64];
        let lam = [0.3f64, 0.7f64];
        // Brute-force the true min over a fine grid.
        let (_v, _sub) = eval_lambda(&a, &b, &lo, &hi, &lam);
        let mut true_min = f64::INFINITY;
        let n = 200;
        for ix in 0..=n {
            for iy in 0..=n {
                let x = lo[0] + (hi[0] - lo[0]) * ix as f64 / n as f64;
                let y = lo[1] + (hi[1] - lo[1]) * iy as f64 / n as f64;
                let g0 = a[[0, 0]] as f64 * x + a[[0, 1]] as f64 * y + b[0] as f64;
                let g1 = a[[1, 0]] as f64 * x + a[[1, 1]] as f64 * y + b[1] as f64;
                let agg = lam[0] * g0 + lam[1] * g1;
                true_min = true_min.min(agg);
            }
        }
        let cert = certified_box_min(&a, &b, &lo, &hi, &lam) as f64;
        assert!(
            cert <= true_min + 1e-6,
            "certified {cert} exceeded true min {true_min}"
        );
    }
}
