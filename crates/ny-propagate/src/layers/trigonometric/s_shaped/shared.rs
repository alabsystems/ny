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
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);
    let mut lower_nonfinite_rows = vec![false; num_outputs];
    let mut upper_nonfinite_rows = vec![false; num_outputs];

    for j in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[j, i]];
            let ua = bounds.upper_a()[[j, i]];

            let lower_compose = compose::compose_lower(la, &lower_relaxations[i]);
            new_lower_a[[j, i]] = lower_compose.new_coeff;
            new_lower_b_f64[j] += lower_compose.intercept_contrib;
            lower_nonfinite_rows[j] |= lower_compose.nonfinite;

            let upper_compose = compose::compose_upper(ua, &upper_relaxations[i]);
            new_upper_a[[j, i]] = upper_compose.new_coeff;
            new_upper_b_f64[j] += upper_compose.intercept_contrib;
            upper_nonfinite_rows[j] |= upper_compose.nonfinite;
        }
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
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_upper_a[[j, i]] = 0.0;
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}
