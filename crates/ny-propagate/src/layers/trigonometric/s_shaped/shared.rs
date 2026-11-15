// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for S-shaped activation relaxations.
//!
//! Contains precomputed tangent-point tables, the finalization helper, and the
//! generic linear relaxation kernels used by Tanh, Sigmoid, and Arctan.

use crate::bounds::MonotoneSShapedPathAlpha;
use ndarray::Array2;
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_core::{f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::compose;
use crate::layers::linear::bias::{add_f64_down, add_f64_up, f32_to_f64_exact};
use crate::LinearBounds;

pub(super) const S_SHAPED_RELAX_EPS: f32 = 1e-6;

const S_SHAPED_STEP_PRE: f32 = 0.01;
const S_SHAPED_X_LIMIT: f32 = 500.0;

pub(super) struct SShapedPrecomputeTables {
    step_pre: f32,
    pub(super) d_lower: Vec<f32>,
    pub(super) d_upper: Vec<f32>,
}

impl SShapedPrecomputeTables {
    pub(super) fn new(func: fn(f64) -> f64, dfunc: fn(f64) -> f64) -> Self {
        let step_pre = S_SHAPED_STEP_PRE;
        let x_limit = S_SHAPED_X_LIMIT;
        // SAFETY: x_limit=500.0 / step_pre=0.01 = 50000.0 — compile-time constants,
        // always positive, finite, and well within usize range.
        let num_points = (x_limit / step_pre) as usize + 5;
        let max_iter = 100;

        let mut d_lower = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let upper = step_pre as f64 * i as f64;
            let mut r = 0.0_f64;
            let mut l = -1.0_f64;

            loop {
                if dfunc(l) * (upper - l) + func(l) <= func(upper) {
                    break;
                }
                l *= 2.0;
            }

            for _ in 0..max_iter {
                // Kept verbatim: the unbounded doubling loop above puts no provable cap
                // on `l`, and f64::midpoint rounds differently past |x| > f64::MAX/2.
                #[allow(clippy::manual_midpoint)]
                let m = 0.5 * (l + r);
                if dfunc(m) * (upper - m) + func(m) <= func(upper) {
                    l = m;
                } else {
                    r = m;
                }
            }

            // The loop invariant is `l` valid, `r` invalid.  A nearest-f32
            // cast can round the final valid root toward `r` and therefore
            // publish an invalid tangent point.  This is not hypothetical:
            // tanh table entry 49_089 used to miss its defining endpoint
            // inequality by about 4.75e-7.  Store toward the valid side.
            d_lower.push(next_down_f32(l as f32));
        }

        let mut d_upper = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let lower = -(step_pre as f64 * i as f64);
            let mut l = 0.0_f64;
            let mut r = 1.0_f64;

            loop {
                if dfunc(r) * (lower - r) + func(r) >= func(lower) {
                    break;
                }
                r *= 2.0;
            }

            for _ in 0..max_iter {
                // Kept verbatim: the unbounded doubling loop above puts no provable cap
                // on `r`, and f64::midpoint rounds differently past |x| > f64::MAX/2.
                #[allow(clippy::manual_midpoint)]
                let m = 0.5 * (l + r);
                if dfunc(m) * (lower - m) + func(m) >= func(lower) {
                    r = m;
                } else {
                    l = m;
                }
            }

            // Symmetric invariant: `l` is invalid and `r` is valid, so round
            // the stored root upward, away from the invalid side.
            d_upper.push(next_up_f32(r as f32));
        }

        Self {
            step_pre,
            d_lower,
            d_upper,
        }
    }

    pub(super) fn retrieve(&self, table: &[f32], bound: f32, default_d: f32) -> f32 {
        if !bound.is_finite() {
            return default_d;
        }
        // Match alpha-beta-CROWN's retrieve_from_precompute behavior:
        // out-of-range precompute indices must fall back to default_d.
        // Keep index arithmetic in float space until bounds checks so large
        // finite values cannot overflow integer addition in debug/release.
        let raw_idx = (bound / self.step_pre).floor() + 1.0;
        let idx = if raw_idx <= 0.0 {
            0
        } else if !raw_idx.is_finite() || raw_idx >= table.len() as f32 {
            return default_d;
        } else {
            // SAFETY: raw_idx is finite (checked above), > 0.0, and < table.len().
            // Cast is guaranteed non-negative and in-bounds.
            raw_idx as usize
        };
        table[idx]
    }

    pub(super) fn lower_tangent(&self, upper: f32, default_d: f32) -> f32 {
        self.retrieve(&self.d_lower, upper.max(0.0), default_d)
    }

    pub(super) fn upper_tangent(&self, lower: f32, default_d: f32) -> f32 {
        self.retrieve(&self.d_upper, (-lower).max(0.0), default_d)
    }
}

pub(super) fn s_shaped_finalize(
    l: f32,
    u: f32,
    lower_slope: f64,
    lower_intercept: f64,
    upper_slope: f64,
    upper_intercept: f64,
) -> LinearRelaxation {
    if !l.is_finite()
        || !u.is_finite()
        || l > u
        || !lower_slope.is_finite()
        || !lower_intercept.is_finite()
        || !upper_slope.is_finite()
        || !upper_intercept.is_finite()
    {
        return LinearRelaxation::nan_fallback();
    }

    // Publish each f64 source line as one f32 slope plus a directionally
    // corrected f32 intercept.  If L(x)=s*x+b and R(x)=s_r*x+b_r, then
    //
    //   R(x) <= L(x)  iff  b_r <= (s-s_r)*x+b,
    //   R(x) >= L(x)  iff  b_r >= (s-s_r)*x+b.
    //
    // The right-hand side is affine, so its extrema on [l,u] occur at the two
    // endpoints.  This is both tighter and actually complete for slope-cast
    // error.  The former `min(max_abs_x, 1e20)` radius was not: on an extreme
    // finite interval it could undercharge the cast error by eighteen orders
    // of magnitude and fail to preserve the supplied affine line.
    let lower_slope_f = lower_slope as f32;
    let upper_slope_f = upper_slope as f32;
    if !lower_slope_f.is_finite() || !upper_slope_f.is_finite() {
        return LinearRelaxation::nan_fallback();
    }

    let l64 = f64::from(l);
    let u64 = f64::from(u);
    let lower_delta = lower_slope - f64::from(lower_slope_f);
    let upper_delta = upper_slope - f64::from(upper_slope_f);

    // Direct every f64 multiply and add outward as well.  This matters when
    // `(s-s_r)*x` nearly cancels `b`: a plain f64 expression can lose far more
    // than one f32 ULP of the small result before the final cast sees it.
    let endpoint_down =
        |delta: f64, x: f64, intercept: f64| next_down_f64(next_down_f64(delta * x) + intercept);
    let endpoint_up =
        |delta: f64, x: f64, intercept: f64| next_up_f64(next_up_f64(delta * x) + intercept);

    let lower_at_l = endpoint_down(lower_delta, l64, lower_intercept);
    let lower_at_u = endpoint_down(lower_delta, u64, lower_intercept);
    let upper_at_l = endpoint_up(upper_delta, l64, upper_intercept);
    let upper_at_u = endpoint_up(upper_delta, u64, upper_intercept);
    let eps = f64::from(S_SHAPED_RELAX_EPS);
    let lower_intercept_f = f64_to_f32_down(next_down_f64(lower_at_l.min(lower_at_u) - eps));
    let upper_intercept_f = f64_to_f32_up(next_up_f64(upper_at_l.max(upper_at_u) + eps));

    LinearRelaxation::new(
        lower_slope_f,
        lower_intercept_f,
        upper_slope_f,
        upper_intercept_f,
    )
}

pub(super) fn s_shaped_linear_relaxation(
    l: f32,
    u: f32,
    func: fn(f64) -> f64,
    dfunc: fn(f64) -> f64,
    tables: &SShapedPrecomputeTables,
    constant_relaxation: fn() -> LinearRelaxation,
) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return constant_relaxation();
    }

    if (u - l).abs() < 1e-8 {
        let slope = dfunc(l as f64);
        let intercept = func(l as f64) - slope * l as f64;
        return s_shaped_finalize(l, u, slope, intercept, slope, intercept);
    }

    let l64 = l as f64;
    let u64 = u as f64;
    let y_l = func(l64);
    let y_u = func(u64);
    let k_direct = (y_u - y_l) / (u64 - l64);
    let b_direct = y_l - k_direct * l64;

    // Bit-identical to `0.5 * (l64 + u64)`: finite f32-cast operands stay on
    // f64::midpoint's non-overflow `(a + b) * 0.5` path.
    let m = f64::midpoint(l64, u64);
    let k_mid = dfunc(m);
    let b_mid = func(m) - k_mid * m;

    // Single-convexity tangent line, PARALLEL TO THE CHORD.
    //
    // In the all-negative (u<=0) regime f is convex, and in the all-positive
    // (l>=0) regime f is concave (true for sigmoid, tanh, arctan: f'' has the
    // opposite sign of x). Every tangent of a convex f is a global LOWER bound,
    // and every tangent of a concave f is a global UPPER bound — so the tangent
    // point only affects tightness, never soundness. The single tangent with the
    // smallest worst-case gap to the chord is the one whose slope equals the
    // chord slope k_direct: tangent at the point d where f'(d) = k_direct.
    //
    // By the mean value theorem there exists d in (l, u) with f'(d) = k_direct,
    // and since f' is strictly monotone on a single-convexity interval, d is
    // unique and found by binary search. If rounding leaves k_direct outside the
    // closed endpoint-derivative interval (degenerate / near-flat case), we fall
    // back to the midpoint tangent (k_mid, b_mid) — still a valid tangent, hence
    // still sound. This mirrors the softplus parallel-to-chord tightening.
    let parallel_tangent = || {
        let d_l = dfunc(l64);
        let d_u = dfunc(u64);
        // k_direct must lie between the two endpoint derivatives (in either
        // order, depending on whether f' is increasing (convex) or decreasing
        // (concave)). If not, the binary search cannot bracket d — fall back.
        let (lo_d, hi_d) = if d_l <= d_u { (d_l, d_u) } else { (d_u, d_l) };
        if !(k_direct > lo_d && k_direct < hi_d) {
            return (k_mid, b_mid);
        }
        // Bisection on f'(x) - k_direct, which is strictly monotone on [l, u].
        // `f_increasing` records the direction of f' so the same loop works for
        // both the convex (f' increasing) and concave (f' decreasing) regimes.
        let f_increasing = d_l < d_u;
        let mut lo = l64;
        let mut hi = u64;
        for _ in 0..60 {
            // Bit-identical: the bracket stays inside the f32-cast [l64, u64].
            let mid_pt = f64::midpoint(lo, hi);
            let below = dfunc(mid_pt) < k_direct;
            // Move the endpoint that keeps d bracketed: when f' increases,
            // f'(mid) < target means d is to the right; when f' decreases, the
            // logic inverts.
            if below == f_increasing {
                lo = mid_pt;
            } else {
                hi = mid_pt;
            }
        }
        let d = f64::midpoint(lo, hi);
        let k_d = dfunc(d);
        (k_d, func(d) - k_d * d)
    };

    if u <= 0.0 {
        // CONVEX regime: parallel-to-chord tangent is the LOWER bound; chord is
        // the UPPER bound (unchanged).
        let (k_lo, b_lo) = parallel_tangent();
        return s_shaped_finalize(l, u, k_lo, b_lo, k_direct, b_direct);
    }
    if l >= 0.0 {
        // CONCAVE regime: chord is the LOWER bound (unchanged); parallel-to-chord
        // tangent is the UPPER bound.
        let (k_hi, b_hi) = parallel_tangent();
        return s_shaped_finalize(l, u, k_direct, b_direct, k_hi, b_hi);
    }

    let d_lower = tables.lower_tangent(u, l);
    let d_upper = tables.upper_tangent(l, u);
    let k_lower = dfunc(d_lower as f64);
    let b_lower = func(d_lower as f64) - k_lower * d_lower as f64;
    let k_upper = dfunc(d_upper as f64);
    let b_upper = func(d_upper as f64) - k_upper * d_upper as f64;

    let d_l = dfunc(l64);
    let d_u = dfunc(u64);
    let lower_line = if k_direct < d_l {
        (k_direct, b_direct)
    } else {
        (k_lower, b_lower)
    };
    let upper_line = if k_direct < d_u {
        (k_direct, b_direct)
    } else {
        (k_upper, b_upper)
    };

    s_shaped_finalize(l, u, lower_line.0, lower_line.1, upper_line.0, upper_line.1)
}

pub(super) fn s_shaped_linear_relaxation_with_alpha(
    l: f32,
    u: f32,
    func: fn(f64) -> f64,
    dfunc: fn(f64) -> f64,
    constant_relaxation: fn() -> LinearRelaxation,
    alpha: MonotoneSShapedPathAlpha,
) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return constant_relaxation();
    }

    if (u - l).abs() < 1e-8 {
        let slope = dfunc(l as f64);
        let intercept = func(l as f64) - slope * l as f64;
        return s_shaped_finalize(l, u, slope, intercept, slope, intercept);
    }

    let l64 = l as f64;
    let u64 = u as f64;
    let y_l = func(l64);
    let y_u = func(u64);
    let k_direct = (y_u - y_l) / (u64 - l64);
    let b_direct = y_l - k_direct * l64;

    if u <= 0.0 {
        let tp_neg = alpha.tp_neg.clamp(l, u);
        let k_lower = dfunc(tp_neg as f64);
        let b_lower = func(tp_neg as f64) - k_lower * tp_neg as f64;
        return s_shaped_finalize(l, u, k_lower, b_lower, k_direct, b_direct);
    }
    if l >= 0.0 {
        let tp_pos = alpha.tp_pos.clamp(l, u);
        let k_upper = dfunc(tp_pos as f64);
        let b_upper = func(tp_pos as f64) - k_upper * tp_pos as f64;
        return s_shaped_finalize(l, u, k_direct, b_direct, k_upper, b_upper);
    }

    let tp_both_lower = alpha.tp_both_lower.min(alpha.d_lower);
    let tp_both_upper = alpha.tp_both_upper.max(alpha.d_upper);
    let k_lower = dfunc(tp_both_lower as f64);
    let b_lower = func(tp_both_lower as f64) - k_lower * tp_both_lower as f64;
    let k_upper = dfunc(tp_both_upper as f64);
    let b_upper = func(tp_both_upper as f64) - k_upper * tp_both_upper as f64;

    // Reference: alpha-beta-CROWN `auto_LiRPA/operators/s_shaped.py:300-377`.
    let use_direct_lower = k_direct < dfunc(l64);
    let use_direct_upper = k_direct < dfunc(u64);
    let lower_line = if use_direct_lower {
        (k_direct, b_direct)
    } else {
        (k_lower, b_lower)
    };
    let upper_line = if use_direct_upper {
        (k_direct, b_direct)
    } else {
        (k_upper, b_upper)
    };

    s_shaped_finalize(l, u, lower_line.0, lower_line.1, upper_line.0, upper_line.1)
}

/// The lower-direction slope actually used by [`compose::compose_lower`] for a
/// coefficient of the given sign (`lower_slope` for `a > 0`, `upper_slope` for
/// `a < 0`, and no line at all for `a == 0`).
///
/// Byte-identical twin of `layers::common::crown_dense::lr_slope`; kept under
/// the same name so a grep for the certificate finds both arms.
#[inline]
fn lr_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.lower_slope as f64
    } else if a < 0.0 {
        relax.upper_slope as f64
    } else {
        0.0
    }
}

/// A sound binary64 upper bound on `|a| + |b|` for two binary32 magnitudes —
/// the two-line cover used when the incoming error may flip a coefficient's
/// sign and select the OTHER envelope.
///
/// Forming that sum with an **f32** add and widening afterwards (what this arm
/// and its twin used to do) lets nearest-rounding place the cover up to `2^-24`
/// RELATIVE **below** the true sum. The publication step
/// `next_up_f32(v as f32)` supplies only `≥ 0.5·ulp ≈ 2^-25·v` of slack, so it
/// does not reliably repay that under-charge — and `int_sum` feeds the BIAS, so
/// the shortfall lands in the published constant as well as the coefficient.
/// Widen first, add in f64 (still inexact when the exponents are far apart),
/// then take one outward f64 step.
///
/// An exactly-zero sum is published as exact zero rather than the minimum
/// subnormal: both magnitudes were zero, so there is nothing to round, and
/// inventing a subnormal cover would put a spurious nonzero charge on a row
/// whose bias must stay exactly zero (the zero-row invariant a stacked-backward
/// injection guard refuses a pass over).
#[inline]
fn cover_sum_up(a: f32, b: f32) -> f64 {
    let sum = f64::from(a.abs()) + f64::from(b.abs());
    if sum == 0.0 {
        0.0
    } else {
        next_up_f64(sum)
    }
}

/// The upper-direction slope actually used by [`compose::compose_upper`].
///
/// Twin of `layers::common::crown_dense::ur_slope`.
#[inline]
fn ur_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.upper_slope as f64
    } else if a < 0.0 {
        relax.lower_slope as f64
    } else {
        0.0
    }
}

/// Dense CROWN backward for an element-wise activation whose LOWER and UPPER
/// envelopes come from **different** per-neuron relaxations — the alpha arm of
/// Sigmoid/Tanh, where `lower_path_alpha(i)` and `upper_path_alpha(i)` are
/// optimized independently.
///
/// # Soundness: the certified coefficient error (#vnncomp-aw-soundness)
///
/// This is the alpha twin of
/// [`crown_elementwise_backward_indexed`](crate::layers::common::crown_elementwise_backward_indexed)
/// (`layers/common/crown_dense.rs`) and it carries the SAME certificate. It
/// used to publish through `LinearBounds::new_or_conservative`, i.e. with NO
/// error matrices — which asserts that its f32 coefficients are EXACT. That
/// dropped two distinct terms, both in the false-proof direction:
///
///  1. The **always-charged** directed-rounding gap
///     `|a_f64·slope_f64 − stored_f32|`. `compose_*` rounds the composed f32
///     coefficient outward, which makes it sound only for concretizing at THIS
///     layer's input box; once a further backward layer reads the stored f32 it
///     treats it as exact (absent a certified error) and under-counts the true
///     error. Charged even when the incoming bounds carry no error at all.
///  2. The **incoming** certified error `e ≥ |a_true − a_stored|`, with the same
///     sign-stability refinement as the twin: while `|a| > e` the true
///     coefficient keeps `a`'s sign, so the envelope line `compose_*` selected
///     is a valid substitute and the charge is `e·|slope_chosen| + gap`, with
///     `e·|intercept_chosen|` folded OUTWARD into the bias. Once `|a| ≤ e` the
///     sign may flip and select the OTHER envelope, so the charge falls back to
///     the two-line cover `e·(|lower_slope| + |upper_slope|)` with
///     `e·(|lower_intercept| + |upper_intercept|)` into the bias. `a == 0` with
///     `e > 0` takes the cover branch, so incoming error is never silently
///     dropped on an exact-zero coefficient.
///
/// Because the two directions use DIFFERENT relaxations here, each direction's
/// terms are formed from ITS OWN relaxation: `compose_lower` only ever reads
/// `lower_relaxations[i]` (its `lower_slope` for `a > 0`, its `upper_slope` for
/// `a < 0`) and `compose_upper` only ever reads `upper_relaxations[i]`. So the
/// two-line sign-flip cover for the lower direction is
/// `|lower_relaxations[i].lower_slope| + |lower_relaxations[i].upper_slope|`,
/// and the mirror for the upper direction — never a mix of the two bundles.
/// Everything else is the twin's formula verbatim; nothing here is invented.
///
/// # Soundness: the certified CONSTANT (#vnncomp-aw-soundness)
///
/// The coefficient half above was certified while the constant half stayed
/// BARE, which is the same defect class one construct over. Each per-neuron
/// term `intercept_contrib = coeff_f32 · intercept_f32` is EXACT in f64 (24+24
/// significand bits fit in 53, and the f32×f32 magnitude ceiling ~1.2e77 cannot
/// overflow binary64), so the *only* error in the published constant is the
/// ADDITION CHAIN — and a plain `+=` chain rounds to NEAREST, i.e. it may round
/// the running sum INWARD on every one of `num_neurons` steps.
///
/// The publication step `next_down_f32`/`next_up_f32` is worth ~1 ulp OF THE
/// RESULT (`≈2^-23·|b|`). The accumulation residual is Higham Thm 3.1's
/// `γ_n^f64 · Σ|term|` — scaled by the summand MAGNITUDES, not by the result.
/// Whenever the terms cancel, `Σ|term| / |b|` grows without bound while the
/// allowance collapses with `|b|`, so the published constant becomes TIGHTER
/// THAN TRUE = false proof. The canonical witness: `+2^60`, then 4096 × `-1`,
/// then `-2^60` sums to EXACTLY 0.0 in binary64 (`ulp(2^60) = 256` swallows
/// every `-1`) where the real sum is `-4096`; one ulp of 0.0f32 is 1.4e-45
/// against a real error of 4096. That shape is reachable here whenever the two
/// path bundles put large opposite-sign intercepts on neighbouring neurons —
/// which the DIVERGENT-ALPHA arm makes *more* likely than the shared-bundle
/// twin, since `lower_relaxations[i]` and `upper_relaxations[i]` are optimized
/// independently and need not agree in sign or scale.
///
/// Fix (idiom (b), the strictly-stronger per-step directed accumulation used by
/// `layers/linear/bias.rs` and by the already-certified `pooling/max.rs`
/// sibling): take a DIRECTED f64 step after EVERY addition — `add_f64_down` on
/// the lower accumulator, `add_f64_up` on the upper. The accumulator then
/// encloses the exact real sum however many additions round the same way, at a
/// cost of ~`n·2^-52` RELATIVE (far under the `2^-24` f32 publication ulp for
/// any reachable `n`, so non-cancelling rows are unchanged after the f32 cast).
/// It also fails closed: a NaN sum publishes `∓inf` instead of poisoning the
/// bias, and the `term == 0.0` short-circuit keeps an all-zero row EXACTLY
/// zero. The same directed step is applied to the per-row intercept-error
/// accumulators and to the end-of-row fold of those into the bias, since both
/// are `+=` chains feeding the same published constant.
///
/// This matches the twin `crown_elementwise_backward_indexed`
/// (`layers/common/crown_dense.rs`) construct for construct — same helpers,
/// same directions, same `f32_to_f64_exact` seed decode — so the two arms stay
/// diffable and neither can drift into charging less than the other.
///
/// One more inward-rounding hole in the same certificate: the two-line cover
/// sums `|lower_slope| + |upper_slope|` and `|lower_intercept| +
/// |upper_intercept|` were formed by an **f32** add and only then widened to
/// f64, so nearest-rounding could put the cover BELOW the true sum by up to
/// `2^-24` relative. The publication `next_up_f32(v as f32)` supplies only
/// `≥ 0.5·ulp ≈ 2^-25·v` of slack, which does not cover it. Both sums now go
/// through [`cover_sum_up`]; `int_sum` in particular feeds the BIAS, so this is
/// a constant-half defect too, not only a coefficient one. **This is the one
/// place this arm is currently AHEAD of its twin**: `crown_dense.rs` and the
/// other f32-cover-sum sites still form both sums in f32. Propagating
/// [`cover_sum_up`] there is outstanding work, not a deliberate asymmetry.
///
/// # Local discharge
///
/// The certified error is then folded into the BIAS over this layer's
/// pre-activation box and cleared, exactly as the ReLU alpha twin does
/// (`activations/relu/mod.rs`, `b.fold_coeff_err_into_bias(pre_lower,
/// pre_upper)`). Sound: the columns multiply the pre-activation values
/// `x ∈ [pre_l, pre_u]`, so the uncertainty `|Σ_i δ_i·x_i|` is bounded by
/// `Σ_i err_i·max(|pre_l_i|, |pre_u_i|)` — precisely the penalty
/// `fold_coeff_err_into_bias` applies, and precisely the penalty a later
/// `concretize_sound` would charge for the carried error anyway. Discharging
/// here makes the published object sound ON ITS OWN: no downstream consumer
/// has to honour an error field, which matters because the α-CROWN graph
/// backward (`network/graph_alpha/backward/nonlinear.rs::handle_monotone_node`)
/// hands this result straight to `accumulate_dense_result` with no
/// `propagates_coeff_err` gate in between.
///
/// That fold runs AFTER the constant is published, on the already-cast f32
/// bias — so it can never repair the accumulation above, and the accumulation
/// had to be fixed at its source. The fold itself is NOT a bare half: it sums
/// `err_ij · max(|pre_l_j|, |pre_u_j|)` through `certified_affine_sum_f32`
/// (idiom (c): `dd_fma` + `γ_n^dd`, so the magnitude-scaled accumulation charge
/// is carried explicitly) and then applies its own directed `next_down_f64` +
/// `next_down_f32` on the lower side and `next_up_f64` + `next_up_f32` on the
/// upper, with a non-finite penalty degrading the row to `∓inf`. Verified
/// against `bounds/linear.rs::fold_coeff_err_into_bias`.
pub(super) fn crown_elementwise_backward_dual_indexed<F, G>(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    lower_path_relaxation: F,
    upper_path_relaxation: G,
) -> Result<LinearBounds>
where
    F: Fn(f32, f32, usize) -> LinearRelaxation,
    G: Fn(f32, f32, usize) -> LinearRelaxation,
{
    let pre_flat = pre_activation.flatten();
    let pre_lower = pre_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.lower().shape().to_vec(),
        })?;
    let pre_upper = pre_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.upper().shape().to_vec(),
        })?;

    let num_neurons = pre_lower.len();
    if bounds.num_inputs() != num_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_neurons],
            got: vec![bounds.num_inputs()],
        });
    }

    let pre_lower_slice = pre_lower
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_lower array".into()))?;
    let pre_upper_slice = pre_upper
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_upper array".into()))?;
    let lower_relaxations = pre_lower_slice
        .iter()
        .zip(pre_upper_slice.iter())
        .enumerate()
        .map(|(i, (&l, &u))| lower_path_relaxation(l, u, i))
        .collect::<Vec<_>>();
    let upper_relaxations = pre_lower_slice
        .iter()
        .zip(pre_upper_slice.iter())
        .enumerate()
        .map(|(i, (&l, &u))| upper_path_relaxation(l, u, i))
        .collect::<Vec<_>>();

    let num_outputs = bounds.num_outputs();
    // Seeds are decoded with `f32_to_f64_exact` rather than `as f64` — matching
    // the twin. A plain hardware conversion may observe a binary32-subnormal
    // seed as signed zero on a DAZ host, which for a lower bias reads
    // `-1.4e-45` as `0.0` and moves the bound INWARD.
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(f32_to_f64_exact);
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(f32_to_f64_exact);
    let mut lower_nonfinite_rows = vec![false; num_outputs];
    let mut upper_nonfinite_rows = vec![false; num_outputs];

    // Certified coefficient-error accumulators — the block this arm used to be
    // missing entirely. See the function doc for the derivation; the formulas
    // are lifted verbatim from `crown_elementwise_backward_indexed`.
    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    // Per-row certified intercept (bias) error: subtracted from lower, added to upper.
    let mut lower_b_err = vec![0.0f64; num_outputs];
    let mut upper_b_err = vec![0.0f64; num_outputs];

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];

            // The lower direction composes ONLY against the lower-path bundle,
            // the upper direction ONLY against the upper-path bundle. Every
            // certificate term below is formed from the same bundle its
            // direction composed with.
            let lower_relax = &lower_relaxations[i];
            let upper_relax = &upper_relaxations[i];

            // CONSTANT (#vnncomp-aw-soundness): each `intercept_contrib` is an
            // EXACT f64 product, so the whole error of the published constant
            // lives in this addition chain. A plain `+=` charges nothing for it
            // and the final `next_*_f32` charges only ~1 ulp OF THE RESULT,
            // which cancellation drives to zero while the real residual
            // (`γ_n^f64·Σ|term|`) does not. Take the directed f64 step here so
            // the accumulator ENCLOSES the exact real sum. See the fn doc.
            let lower_compose = compose::compose_lower(la, lower_relax);
            new_lower_a[[j, i]] = lower_compose.new_coeff;
            new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], lower_compose.intercept_contrib);
            lower_nonfinite_rows[j] |= lower_compose.nonfinite;

            let upper_compose = compose::compose_upper(ua, upper_relax);
            new_upper_a[[j, i]] = upper_compose.new_coeff;
            new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], upper_compose.intercept_contrib);
            upper_nonfinite_rows[j] |= upper_compose.nonfinite;

            // Lower direction.
            {
                // Two-line covers, summed in f64 and rounded OUTWARD; see
                // `cover_sum_up` for why the old f32 add under-charged both the
                // coefficient and (through `int_sum`) the bias.
                let slope_sum = cover_sum_up(lower_relax.lower_slope, lower_relax.upper_slope);
                let int_sum =
                    cover_sum_up(lower_relax.lower_intercept, lower_relax.upper_intercept);
                let gap = if la != 0.0 {
                    (la as f64 * lr_slope(la, lower_relax) - lower_compose.new_coeff as f64).abs()
                } else {
                    0.0
                };
                let ea = in_lower_err.map_or(0.0, |e| e[[j, i]] as f64);
                if gap != 0.0 || ea != 0.0 {
                    let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                        // Sign-stable: the chosen line covers the true coefficient.
                        let slope = lr_slope(la, lower_relax).abs();
                        let intercept = if la > 0.0 {
                            lower_relax.lower_intercept.abs() as f64
                        } else {
                            lower_relax.upper_intercept.abs() as f64
                        };
                        (slope, intercept)
                    } else {
                        (slope_sum, int_sum)
                    };
                    // `next_up_f32(v as f32) >= v` for every finite `v`: a
                    // nearest cast lands within half the step it is about to
                    // take, so this repays both f64 roundings above.
                    new_lower_a_err[[j, i]] = next_up_f32((ea * slope_cover + gap) as f32);
                    if ea != 0.0 {
                        // A NON-NEGATIVE charge accumulator, so it is summed
                        // UPWARD: a round-to-nearest `+=` here UNDER-counts the
                        // charge, and an under-counted widening is a false
                        // proof exactly like an over-tight bias. One outward
                        // step on the sum also repays the product's own
                        // rounding, since `0.5·ulp(p) + 0.5·ulp(acc+p) ≤
                        // ulp(acc+p)` for a non-negative `acc`. `term == 0.0`
                        // short-circuits, so a zero-intercept relaxation leaves
                        // the accumulator exactly zero; a NaN from `inf · 0`
                        // fails closed to `+inf`, which the subtraction below
                        // turns into a `-inf` lower bias.
                        lower_b_err[j] = add_f64_up(lower_b_err[j], ea * int_cover);
                    }
                }
            }

            // Upper direction. Mirror of the lower arm, term for term, and
            // formed from the UPPER bundle only.
            {
                let slope_sum = cover_sum_up(upper_relax.lower_slope, upper_relax.upper_slope);
                let int_sum =
                    cover_sum_up(upper_relax.lower_intercept, upper_relax.upper_intercept);
                let gap = if ua != 0.0 {
                    (ua as f64 * ur_slope(ua, upper_relax) - upper_compose.new_coeff as f64).abs()
                } else {
                    0.0
                };
                let ea = in_upper_err.map_or(0.0, |e| e[[j, i]] as f64);
                if gap != 0.0 || ea != 0.0 {
                    let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                        // Sign-stable: the chosen line covers the true coefficient.
                        let slope = ur_slope(ua, upper_relax).abs();
                        let intercept = if ua > 0.0 {
                            upper_relax.upper_intercept.abs() as f64
                        } else {
                            upper_relax.lower_intercept.abs() as f64
                        };
                        (slope, intercept)
                    } else {
                        (slope_sum, int_sum)
                    };
                    new_upper_a_err[[j, i]] = next_up_f32((ea * slope_cover + gap) as f32);
                    if ea != 0.0 {
                        upper_b_err[j] = add_f64_up(upper_b_err[j], ea * int_cover);
                    }
                }
            }
        }
        // Fold the certified intercept error into the bias accumulators BEFORE
        // the directed cast: lower decreases, upper increases. Directed here
        // too — this is the last `+=` on the published constant, and a
        // nearest-rounded fold could give back part of the very charge it is
        // applying.
        new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], -lower_b_err[j]);
        new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], upper_b_err[j]);
    }

    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback(
        "Monotone S-shaped activation",
        lower_affected,
        upper_affected,
        num_outputs,
    );

    let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));
    for j in 0..num_outputs {
        if lower_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_lower_a[[j, i]] = 0.0;
                new_lower_a_err[[j, i]] = 0.0;
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_upper_a[[j, i]] = 0.0;
                new_upper_a_err[[j, i]] = 0.0;
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    let mut out = LinearBounds::new_or_conservative_with_err(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_lower_a_err,
        new_upper_a_err,
    )?;
    // LOCAL DISCHARGE over this layer's pre-activation box (the box the columns
    // multiply) — same identity and same policy as the ReLU alpha twin.
    out.fold_coeff_err_into_bias(pre_lower_slice, pre_upper_slice);
    Ok(out)
}

#[cfg(test)]
mod dual_indexed_coeff_err_tests {
    //! WIRING tests for the alpha arm's certified coefficient error.
    //!
    //! The pre-existing s-shaped tests exercise the RELAXATION KERNEL with
    //! self-consistent inputs, which is exactly why they could not see this
    //! defect: the kernel was correct and the wiring dropped its certificate.
    //! Every assertion below is on the OUTPUT of
    //! `crown_elementwise_backward_dual_indexed`, and the first, second and
    //! fourth FAIL on the pre-fix code (which published through
    //! `new_or_conservative`, i.e. with no error term and an unwidened bias).

    use super::crown_elementwise_backward_dual_indexed;
    use crate::layers::activations::LinearRelaxation;
    use crate::LinearBounds;
    use ndarray::{array, Array1, Array2};
    use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

    /// The lower-path bundle. Its four numbers are pairwise distinct from the
    /// upper-path bundle's, so a direction that read the WRONG bundle would
    /// change the published value.
    fn lower_bundle() -> LinearRelaxation {
        LinearRelaxation::new(0.25, 0.0, 0.5, 0.125)
    }

    /// The upper-path bundle (the alpha arm optimizes the two independently).
    fn upper_bundle() -> LinearRelaxation {
        LinearRelaxation::new(0.125, -0.0625, 0.375, 0.25)
    }

    fn pre_box(l: f32, u: f32) -> BoundedTensor {
        BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())
            .expect("invariant: matching 1-element bound shapes")
    }

    fn incoming(la: f32, ua: f32, err: Option<f32>) -> LinearBounds {
        let mut b = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![la]).expect("1x1 lower_a"),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![ua]).expect("1x1 upper_a"),
            Array1::zeros(1),
        )
        .expect("invariant: 1x1 coefficient matrices with matching 1-element biases");
        if let Some(e) = err {
            b.set_coeff_err(
                Array2::from_shape_vec((1, 1), vec![e]).expect("1x1 lower err"),
                Array2::from_shape_vec((1, 1), vec![e]).expect("1x1 upper err"),
            );
        }
        b
    }

    fn run(bounds: &LinearBounds, l: f32, u: f32) -> LinearBounds {
        let pre = pre_box(l, u);
        crown_elementwise_backward_dual_indexed(
            bounds,
            &pre,
            |_l, _u, _i| lower_bundle(),
            |_l, _u, _i| upper_bundle(),
        )
        .expect("dual-indexed backward must accept 1x1 bounds over a 1-element box")
    }

    /// FAILS ON PRE-FIX CODE: the alpha arm never read `lower_a_err`/`upper_a_err`,
    /// so an incoming certified error made NO difference to the published bounds.
    #[test]
    fn dual_indexed_alpha_arm_charges_incoming_coeff_err() {
        let clean = run(&incoming(1.0, 1.0, None), -8.0, 8.0);
        let dirty = run(&incoming(1.0, 1.0, Some(0.25)), -8.0, 8.0);

        assert!(
            dirty.lower_b[0] < clean.lower_b[0],
            "incoming lower coefficient error must LOWER the published lower bias \
             (clean={}, with_err={})",
            clean.lower_b[0],
            dirty.lower_b[0]
        );
        assert!(
            dirty.upper_b[0] > clean.upper_b[0],
            "incoming upper coefficient error must RAISE the published upper bias \
             (clean={}, with_err={})",
            clean.upper_b[0],
            dirty.upper_b[0]
        );
    }

    /// FAILS ON PRE-FIX CODE: with NO incoming error the arm published its
    /// directed-rounded f32 coefficients as if they were the exact real product.
    /// `la * lower_slope = 1.0 * 0.25` is exact in f32, so `compose_lower`'s
    /// unconditional `next_down_f32` moves the stored coefficient a full ulp off
    /// the true product — a gap that must be charged.
    #[test]
    fn dual_indexed_alpha_arm_charges_fresh_rounding_gap_without_incoming_err() {
        let (l, u) = (-100.0f32, 100.0f32);
        let out = run(&incoming(1.0, 1.0, None), l, u);

        let lower_relax = lower_bundle();
        let upper_relax = upper_bundle();
        // What the pre-fix code published: the composed f64 bias cast outward,
        // with no coefficient-error term and therefore no fold. With `la = 1`
        // the only bias contribution is the selected line's intercept.
        let uncharged_lower = next_down_f32(lower_relax.lower_intercept);
        let uncharged_upper = next_up_f32(upper_relax.upper_intercept);

        let mag = f64::from(l.abs().max(u.abs()));
        let lower_gap =
            f64::from(lower_relax.lower_slope) - f64::from(next_down_f32(lower_relax.lower_slope));
        let upper_gap =
            f64::from(next_up_f32(upper_relax.upper_slope)) - f64::from(upper_relax.upper_slope);
        assert!(
            lower_gap > 0.0 && upper_gap > 0.0,
            "test setup: both directed-rounding gaps must be nonzero \
             (lower={lower_gap}, upper={upper_gap})"
        );

        assert!(
            out.lower_b[0] < uncharged_lower,
            "the always-charged directed-rounding gap must widen the lower bias \
             (published={}, uncharged={})",
            out.lower_b[0],
            uncharged_lower
        );
        assert!(
            out.upper_b[0] > uncharged_upper,
            "the always-charged directed-rounding gap must widen the upper bias \
             (published={}, uncharged={})",
            out.upper_b[0],
            uncharged_upper
        );

        // The widening is the fold penalty `gap * max(|l|,|u|)`, up to the
        // outward rounding the fold itself applies.
        let lower_widening = f64::from(uncharged_lower) - f64::from(out.lower_b[0]);
        assert!(
            lower_widening >= lower_gap * mag * 0.9,
            "lower widening {lower_widening} is below the certified penalty {} — \
             the gap was under-charged",
            lower_gap * mag
        );
        let upper_widening = f64::from(out.upper_b[0]) - f64::from(uncharged_upper);
        assert!(
            upper_widening >= upper_gap * mag * 0.9,
            "upper widening {upper_widening} is below the certified penalty {} — \
             the gap was under-charged",
            upper_gap * mag
        );
    }

    /// The certificate is DISCHARGED here, not carried: the published object is
    /// sound on its own, so no downstream consumer can drop it.
    #[test]
    fn dual_indexed_alpha_arm_publishes_a_discharged_object() {
        let out = run(&incoming(1.0, -1.0, Some(0.125)), -3.0, 3.0);
        assert!(
            !out.has_coeff_err(),
            "the alpha arm must fold its certified error into the bias over the \
             pre-activation box, leaving no error field for a downstream layer to drop"
        );
    }

    /// A zero stored coefficient with a nonzero incoming error must still be
    /// charged: the true coefficient can take either sign and select either
    /// envelope. FAILS ON PRE-FIX CODE for the same reason as the first test.
    #[test]
    fn dual_indexed_alpha_arm_charges_err_on_exact_zero_coefficient() {
        let clean = run(&incoming(0.0, 0.0, None), -8.0, 8.0);
        let dirty = run(&incoming(0.0, 0.0, Some(0.5)), -8.0, 8.0);
        assert!(
            dirty.lower_b[0] < clean.lower_b[0] && dirty.upper_b[0] > clean.upper_b[0],
            "incoming error on an exact-zero coefficient must still widen the bias \
             (lower {} -> {}, upper {} -> {})",
            clean.lower_b[0],
            dirty.lower_b[0],
            clean.upper_b[0],
            dirty.upper_b[0]
        );
        // And the widening must be the certified two-line cover, not a token ulp.
        assert!(
            f64::from(clean.lower_b[0]) - f64::from(dirty.lower_b[0]) > 1.0,
            "the |a| <= e branch must apply the two-line cover \
             e*(|lower_slope|+|upper_slope|) over the box (lower {} -> {})",
            clean.lower_b[0],
            dirty.lower_b[0]
        );
    }

    /// FAILS ON PRE-FIX CODE — **both** assertions, in the **same call**.
    ///
    /// The COEFFICIENT half of this arm was certified while the CONSTANT half
    /// stayed bare: `new_*_b_f64[j] += intercept_contrib` is a plain
    /// nearest-rounded binary64 chain, and the only charge for it was the
    /// `next_down_f32` / `next_up_f32` at publication — worth ~1 ulp OF THE
    /// RESULT. Higham Thm 3.1 bounds that chain's error by `γ_n · Σ|term|`,
    /// scaled by the SUMMAND magnitudes; under cancellation the result — and
    /// with it the allowance — collapses toward zero while the true error does
    /// not.
    ///
    /// Witness, the class's canonical shape: `+2^60`, then 4096 × `-1`, then
    /// `-2^60`. Every `-1` is swallowed by `ulp(2^60) = 256`, so the naive f64
    /// chain lands on EXACTLY 0.0 where the real sum is `-4096`, and the
    /// published lower constant was `next_down_f32(0.0) = -1.4e-45` — 4096
    /// TIGHTER THAN TRUE, which is the false-`unsat` direction. The upper
    /// direction carries the mirrored shape in the SAME call, so a fix that
    /// certified only one accumulator still fails here.
    ///
    /// The pre-activation box is pinned to zero so `fold_coeff_err_into_bias`
    /// contributes exactly nothing (`mag = 0`, and `certified_affine_sum_f32`
    /// short-circuits an all-zero-magnitude term list to an exact `0.0`).
    /// Whatever the coefficient certificate does, it can neither mask nor
    /// manufacture the constant's widening measured below.
    ///
    /// This is also the DIVERGENT-ALPHA case the arm exists for: the two path
    /// bundles disagree in slope AND in intercept sign at every index, checked
    /// explicitly below, so an implementation that read one bundle for both
    /// directions would not produce these two numbers either.
    #[test]
    fn dual_indexed_alpha_arm_certifies_a_cancelling_constant_in_both_directions() {
        const UNIT_TERMS: usize = 4096;
        const N: usize = UNIT_TERMS + 2;

        // 2^60: `ulp(2^60) = 2^8 = 256` in binary64, so a `±1` term added to it
        // rounds away entirely.
        let big = 2.0f32.powi(60);

        // Lower-path bundle. The incoming `la = 1 > 0`, so `compose_lower`
        // reads `lower_slope` / `lower_intercept`.
        let lower_at = move |i: usize| -> LinearRelaxation {
            let intercept = if i == 0 {
                big
            } else if i == N - 1 {
                -big
            } else {
                -1.0
            };
            LinearRelaxation::new(0.25, intercept, 0.5, 0.125)
        };
        // Upper-path bundle — independently optimized, mirrored intercepts and
        // different slopes. The incoming `ua = 1 > 0`, so `compose_upper` reads
        // `upper_slope` / `upper_intercept`.
        let upper_at = move |i: usize| -> LinearRelaxation {
            let intercept = if i == 0 {
                -big
            } else if i == N - 1 {
                big
            } else {
                1.0
            };
            LinearRelaxation::new(0.125, -0.0625, 0.375, intercept)
        };

        for i in 0..N {
            let lo = lower_at(i);
            let up = upper_at(i);
            assert!(
                lo.lower_slope != up.upper_slope && lo.lower_intercept != up.upper_intercept,
                "test setup: the two path bundles must diverge at index {i} \
                 (alpha_lower != alpha_upper is the whole point of this arm)"
            );
        }

        // The witness only bites if the NAIVE chain really does cancel; assert
        // the shape is live rather than trusting the arithmetic story.
        let mut naive_lower = 0.0f64;
        let mut naive_upper = 0.0f64;
        for i in 0..N {
            naive_lower += f64::from(lower_at(i).lower_intercept);
            naive_upper += f64::from(upper_at(i).upper_intercept);
        }
        assert_eq!(
            naive_lower, 0.0,
            "test setup: the naive f64 lower chain must cancel to EXACTLY 0.0"
        );
        assert_eq!(
            naive_upper, 0.0,
            "test setup: the naive f64 upper chain must cancel to EXACTLY 0.0"
        );

        // Exact real sums, by construction: the two `±2^60` terms annihilate
        // and the unit terms survive.
        let true_lower = -(UNIT_TERMS as f64);
        let true_upper = UNIT_TERMS as f64;

        let ones = Array2::from_elem((1, N), 1.0f32);
        let bounds = LinearBounds::new(ones.clone(), Array1::zeros(1), ones, Array1::zeros(1))
            .expect("invariant: 1xN coefficient matrices with matching 1-element biases");
        let zero_box = BoundedTensor::new(
            Array1::<f32>::zeros(N).into_dyn(),
            Array1::<f32>::zeros(N).into_dyn(),
        )
        .expect("invariant: a degenerate all-zero box is a valid BoundedTensor");

        let out = crown_elementwise_backward_dual_indexed(
            &bounds,
            &zero_box,
            |_l, _u, i| lower_at(i),
            |_l, _u, i| upper_at(i),
        )
        .expect("dual-indexed backward must accept 1xN bounds over an N-element box");

        assert!(
            f64::from(out.lower_b[0]) <= true_lower,
            "LOWER constant {} is TIGHTER THAN TRUE (exact real sum {true_lower}). \
             The bare `+=` chain cancels to 0.0 and publishes \
             next_down_f32(0.0) = {}; the Higham charge gamma_n*sum|term| is \
             what must be paid instead.",
            out.lower_b[0],
            next_down_f32(0.0)
        );
        assert!(
            f64::from(out.upper_b[0]) >= true_upper,
            "UPPER constant {} is TIGHTER THAN TRUE (exact real sum {true_upper}). \
             Same cancellation, mirrored: the bare chain publishes \
             next_up_f32(0.0) = {}.",
            out.upper_b[0],
            next_up_f32(0.0)
        );

        // ...and the charge is the Higham bound, not an unbounded degrade:
        // gamma_n * sum|term| ~= 4098 * 2^-53 * 2^61 ~= 1.05e6 here, which the
        // per-step directed accumulation reproduces to within a small factor.
        assert!(
            out.lower_b[0].is_finite() && out.upper_b[0].is_finite(),
            "the certified constant must stay finite on a finite witness \
             (lower={}, upper={})",
            out.lower_b[0],
            out.upper_b[0]
        );
        assert!(
            f64::from(out.lower_b[0]) >= -4.0e6 && f64::from(out.upper_b[0]) <= 4.0e6,
            "the certified constant must stay within a small factor of the \
             Higham charge ~1.05e6, not degrade without bound \
             (lower={}, upper={})",
            out.lower_b[0],
            out.upper_b[0]
        );
    }
}
