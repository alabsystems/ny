// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_tensor::{next_down_f32, next_up_f32};

use crate::layers::activations::LinearRelaxation;

/// Convex x^2 relaxation with directed rounding.
pub fn pow2_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    let l64 = l as f64;
    let u64 = u as f64;
    if l == u {
        let y = (l64 * l64) as f32;
        return LinearRelaxation::new(0.0, next_down_f32(y), 0.0, next_up_f32(y));
    }

    let upper_slope = (l64 + u64) as f32;
    let s64 = upper_slope as f64;
    let needed_upper = (l64 * l64 - s64 * l64).max(u64 * u64 - s64 * u64);
    let eps = f64::from(f32::EPSILON);
    let max_sq = (l64 * l64).max(u64 * u64);
    let upper_intercept = next_up_f32((needed_upper + 4.0 * eps * max_sq) as f32);
    let upper_intercept = if upper_intercept.is_finite() {
        upper_intercept
    } else {
        next_up_f32(needed_upper as f32)
    };
    if l < 0.0 && u > 0.0 || max_sq < f64::from(f32::MIN_POSITIVE) {
        return LinearRelaxation::new(0.0, 0.0, upper_slope, upper_intercept);
    }

    let lower_slope = (l64 + u64) as f32;
    let ls64 = lower_slope as f64;
    let vertex_x = ls64 / 2.0;
    let allowed_lower = (l64 * l64 - ls64 * l64)
        .min(u64 * u64 - ls64 * u64)
        .min(vertex_x * vertex_x - ls64 * vertex_x);
    let lower_intercept = next_down_f32((allowed_lower - 4.0 * eps * max_sq) as f32);
    let lower_intercept = if lower_intercept.is_finite() {
        lower_intercept
    } else {
        next_down_f32(allowed_lower as f32)
    };
    LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
}

/// Convex 1/x relaxation on x > 0 with directed rounding.
pub fn pow_neg1_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l <= 0.0 || u <= 0.0 {
        return LinearRelaxation::new(0.0, 0.0, 0.0, f32::INFINITY);
    }
    let l64 = l as f64;
    let u64 = u as f64;
    if l == u {
        let y = (1.0 / l64) as f32;
        return LinearRelaxation::new(0.0, next_down_f32(y), 0.0, next_up_f32(y));
    }

    let lu64 = l64 * u64;
    let upper_slope = (-1.0 / lu64) as f32;
    let s64 = upper_slope as f64;
    let needed_upper = (1.0 / l64 - s64 * l64).max(1.0 / u64 - s64 * u64);
    let eps = f64::from(f32::EPSILON);
    let max_recip = (1.0 / l64).max(1.0 / u64);
    let upper_intercept = next_up_f32((needed_upper + 4.0 * eps * max_recip) as f32);
    let upper_intercept = if upper_intercept.is_finite() {
        upper_intercept
    } else {
        next_up_f32(needed_upper as f32)
    };

    // Bit-identical to `0.5 * (l64 + u64)`: finite f32-cast operands stay on
    // f64::midpoint's non-overflow `(a + b) * 0.5` path.
    let m64 = f64::midpoint(l64, u64);
    let lower_slope = (-1.0 / (m64 * m64)) as f32;
    let ls64 = lower_slope as f64;
    // For the convex lower line `ls*x + c <= 1/x`, the tightest sound intercept is
    // `c = min_{[l,u]} g(x)` with `g(x) = 1/x - ls*x`. Since `ls < 0`, g is strictly
    // convex on x>0 with its minimum at the INTERIOR point `x* = sqrt(-1/ls)` (here
    // ~= m). The endpoint-only min `min(g(l), g(u))` is strictly GREATER than g(x*),
    // which would lift the lower line ABOVE 1/x near x* — a false-proof (certified
    // lower bound above the true value). Include the interior critical point, exactly
    // as the sibling relaxations do (pow2 uses `vertex_x`, pow_positive_integer uses
    // `x_star`). #soundness-pow-neg1-interior-tangent.
    let mut allowed_lower = (1.0 / l64 - ls64 * l64).min(1.0 / u64 - ls64 * u64);
    if ls64 < 0.0 {
        let x_star = (-1.0 / ls64).sqrt();
        if x_star.is_finite() && x_star >= l64 && x_star <= u64 {
            allowed_lower = allowed_lower.min(1.0 / x_star - ls64 * x_star);
        }
    }
    let lower_intercept = next_down_f32((allowed_lower - 4.0 * eps * max_recip) as f32);
    let lower_intercept = if lower_intercept.is_finite() {
        lower_intercept
    } else {
        next_down_f32(allowed_lower as f32)
    };
    LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
}

/// Convex x^p relaxation for integer p >= 2 over nonnegative intervals.
pub(crate) fn pow_positive_integer_nonnegative_linear_relaxation(
    exponent: i32,
    l: f32,
    u: f32,
) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l < 0.0 || u < 0.0 || exponent < 2 {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    let p64 = f64::from(exponent);
    let l64 = l as f64;
    let u64 = u as f64;
    if l == u {
        let y = l64.powi(exponent) as f32;
        return LinearRelaxation::new(0.0, next_down_f32(y), 0.0, next_up_f32(y));
    }

    let y_l = l64.powi(exponent);
    let y_u = u64.powi(exponent);
    let eps = f64::from(f32::EPSILON);
    let max_y = y_l.max(y_u);
    let upper_slope = ((y_u - y_l) / (u64 - l64)) as f32;
    let s64 = upper_slope as f64;
    let needed_upper = (y_l - s64 * l64).max(y_u - s64 * u64);
    let upper_intercept = next_up_f32((needed_upper + 4.0 * eps * max_y) as f32);
    let upper_intercept = if upper_intercept.is_finite() {
        upper_intercept
    } else {
        next_up_f32(needed_upper as f32)
    };
    if max_y < f64::from(f32::MIN_POSITIVE) {
        return LinearRelaxation::new(0.0, 0.0, upper_slope, upper_intercept);
    }

    // Bit-identical to `0.5 * (l64 + u64)`: finite f32-cast operands stay on
    // f64::midpoint's non-overflow `(a + b) * 0.5` path.
    let m64 = f64::midpoint(l64, u64);
    let lower_slope = (p64 * m64.powi(exponent - 1)) as f32;
    let ls64 = lower_slope as f64;
    let mut allowed_lower = (y_l - ls64 * l64).min(y_u - ls64 * u64);
    let x_star = (ls64 / p64).powf(1.0 / f64::from(exponent - 1));
    if x_star.is_finite() && x_star >= l64 && x_star <= u64 {
        allowed_lower = allowed_lower.min(x_star.powi(exponent) - ls64 * x_star);
    }
    let lower_intercept = next_down_f32((allowed_lower - 4.0 * eps * max_y) as f32);
    let lower_intercept = if lower_intercept.is_finite() {
        lower_intercept
    } else {
        next_down_f32(allowed_lower as f32)
    };
    LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
}

#[cfg(test)]
mod soundness_tests {
    use super::*;

    /// Fail-before / pass-after repro for #soundness-pow-neg1-interior-tangent.
    ///
    /// The LinearRelaxation contract requires
    /// `lower_slope*x + lower_intercept <= 1/x <= upper_slope*x + upper_intercept`
    /// for ALL x in [l,u]. Before the interior-tangent fix, the pow_neg1 lower line
    /// used a slope-tangent at the midpoint but an intercept taken over the ENDPOINTS
    /// only; since `g(x)=1/x-ls*x` is convex with its minimum at the interior point
    /// `x*=sqrt(-1/ls)`, the endpoint-min lifted the lower line ABOVE 1/x near x*
    /// (e.g. [1,3]: lower line = 0.583 at x=2 while 1/2 = 0.5) — a false VERIFIED for
    /// any `output >= threshold` property routed through a reciprocal layer.
    #[test]
    fn pow_neg1_lower_relaxation_encloses_reciprocal() {
        let intervals = [
            (1.0_f32, 3.0),
            (0.5, 4.0),
            (0.01, 0.02),
            (0.1, 10.0),
            (2.0, 2.5),
            (0.25, 16.0),
            (1e-3, 1e-2),
            (5.0, 5.5),
        ];
        for &(l, u) in &intervals {
            let r = pow_neg1_linear_relaxation(l, u);
            let n = 257usize;
            for i in 0..=n {
                let t = i as f64 / n as f64;
                let x = (l as f64 + (u as f64 - l as f64) * t) as f32;
                if x <= 0.0 {
                    continue;
                }
                let f = 1.0_f32 / x;
                let lo = r.lower_slope * x + r.lower_intercept;
                let hi = r.upper_slope * x + r.upper_intercept;
                // Small tolerance absorbs f32 sampling/representation noise only; the
                // pre-fix overshoot (>= 0.08 here, up to ~5 on tight intervals) dwarfs it.
                let tol = 1e-5 * (1.0 + f.abs());
                assert!(
                    lo <= f + tol,
                    "[{l},{u}] LOWER line {lo} ABOVE 1/x={f} at x={x} \
                     (slope={}, intercept={})",
                    r.lower_slope,
                    r.lower_intercept
                );
                assert!(
                    hi >= f - tol,
                    "[{l},{u}] UPPER line {hi} BELOW 1/x={f} at x={x}"
                );
            }
            // Explicit midpoint check — the worst-case point for the pre-fix bug.
            let m = f32::midpoint(l, u);
            let fm = 1.0_f32 / m;
            let lom = r.lower_slope * m + r.lower_intercept;
            assert!(
                lom <= fm + 1e-5 * (1.0 + fm.abs()),
                "[{l},{u}] midpoint LOWER {lom} ABOVE 1/m={fm}"
            );
        }
    }

    /// Exact convex-envelope soundness for positive-integer powers on the
    /// one-sided domain `x >= 0` (the pensieve fractional-head cubic): on
    /// random `[l, u] ⊆ [0, u]` boxes the relaxation must satisfy
    /// `lower_line(x) <= x^k <= upper_line(x)` at 1000 sampled points, and
    /// the lower (tangent) line must never exceed the upper (secant) line
    /// anywhere in the box.
    #[test]
    fn pow_positive_integer_envelope_encloses_cubic_on_random_boxes() {
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / f64::from(u32::MAX)
        };
        for case in 0..64 {
            let exponent = 2 + (case % 4); // 2..=5, cubic included
                                           // Boxes across the pensieve-relevant scales, including l = 0.
            let scale = [0.5_f64, 5.0, 50.0, 400.0][case % 4];
            let l = if case % 3 == 0 { 0.0 } else { next() * scale };
            let u = l + next() * scale + 1e-6;
            let r = pow_positive_integer_nonnegative_linear_relaxation(
                exponent as i32,
                l as f32,
                u as f32,
            );
            let n = 1000usize;
            for i in 0..=n {
                let x64 = l + (u - l) * (i as f64) / (n as f64);
                let x = x64 as f32;
                if !(x >= l as f32 && x <= u as f32) {
                    continue;
                }
                let f = (f64::from(x)).powi(exponent as i32);
                let lo = f64::from(r.lower_slope) * f64::from(x) + f64::from(r.lower_intercept);
                let hi = f64::from(r.upper_slope) * f64::from(x) + f64::from(r.upper_intercept);
                let tol = 1e-5 * (1.0 + f.abs());
                assert!(
                    lo <= f + tol,
                    "k={exponent} [{l},{u}] LOWER (tangent) line {lo} ABOVE x^k={f} at x={x}"
                );
                assert!(
                    hi >= f - tol,
                    "k={exponent} [{l},{u}] UPPER (secant) line {hi} BELOW x^k={f} at x={x}"
                );
                assert!(
                    lo <= hi + tol,
                    "k={exponent} [{l},{u}] tangent {lo} above secant {hi} at x={x}"
                );
            }
        }
    }
}
