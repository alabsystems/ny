// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Log linear relaxation.

use crate::rounding::{next_down_f32, next_up_f32};
use crate::types::LinearRelaxation;
use ny_core::{f32_affine_eval_error, f64_to_f32_down, f64_to_f32_up, nan_propagating_max};

/// Relaxation for ln(x) when upper bound is infinite.
fn log_relaxation_infinite_upper(l: f32) -> LinearRelaxation {
    if l <= 0.0 || !l.is_finite() {
        return LinearRelaxation::nan_fallback();
    }

    let log_l = (l as f64).ln();
    let upper_slope = 1.0 / (l as f64);
    let upper_intercept = log_l - 1.0;
    // A finite intercept cannot compensate a downward-rounded slope for every
    // x on an unbounded interval. Round the slope upward; if that overflows,
    // use a constant +infinity upper bound.
    let upper_slope_f32 = next_up_f32(upper_slope as f32);
    if !upper_slope_f32.is_finite() {
        return LinearRelaxation::new(0.0, next_down_f32(log_l as f32), 0.0, f32::INFINITY);
    }
    LinearRelaxation::new(
        0.0,
        next_down_f32(log_l as f32),
        upper_slope_f32,
        next_up_f32(upper_intercept as f32),
    )
}

/// Linear relaxation for ln(x) on interval [l, u].
pub fn log_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    if u.is_infinite() {
        if u.is_sign_negative() {
            return LinearRelaxation::nan_fallback();
        }
        return log_relaxation_infinite_upper(l);
    }
    if l.is_infinite() {
        return LinearRelaxation::nan_fallback();
    }

    // ln is defined for every positive f32. Clamping tiny positive bounds to
    // an epsilon would raise the claimed lower bound and be unsound.
    if l <= 0.0 || u <= 0.0 || l > u {
        return LinearRelaxation::nan_fallback();
    }

    // Tiny `l`: the ENDPOINT-CONSTANT BAND, not a sloped line.
    //
    // Ported from production (`ny-propagate` `layers/activations/log.rs`) to close
    // a drift caught by `tests/drift.rs`: at l = 1e-45, u = 1e-30 the mirror
    // returned the honest chord with slope 3.42e31 while production returns
    // slope 0. Both are SOUND — the chord really is that steep, because that is
    // how fast ln moves there — but every downstream certified error term scales
    // with |coeff|, so a 1e31 coefficient trades a valid bound for an error
    // blow-up. Production deliberately takes the looser constant band, and the
    // mirror must match it or the Kani proofs certify a function production does
    // not run.
    //
    // The band is sound for any 0 < l <= u: ln is increasing, so
    // ln(l) <= ln(x) <= ln(u) on [l, u], and both ends are rounded outward.
    const EPSILON: f32 = 1e-10;
    if l < EPSILON {
        return LinearRelaxation::new(
            0.0,
            next_down_f32(f64::from(l).ln() as f32),
            0.0,
            next_up_f32(f64::from(u).ln() as f32),
        );
    }

    let l64 = l as f64;
    let u64 = u as f64;

    // Narrow intervals, guarded RELATIVE to l (#drift-log-narrow-tiny): the
    // former ABSOLUTE guard (|u - l| < 1e-8) returned the tangent at l for
    // BOTH lines, but a tangent lies ABOVE concave ln and is never a valid
    // LOWER line — at x = u it overshoots ln(u) by r - ln(1 + r) with
    // r = (u - l)/l (+45.09 on [1e-10, 5e-9]). For r < 1e-7 the output range
    // ln(u) - ln(l) <= r is below f32 resolution, so the endpoint-constant
    // band [ln(l) rounded down, ln(u) rounded up] (sqrt.rs-style) is sound
    // and near-tight. No slope means no mul_err correction, so this branch
    // is kept TEXTUALLY IDENTICAL to production and tests/drift.rs holds it
    // bit-exact.
    const NARROW_REL: f64 = 1e-7;
    if u64 - l64 < l64 * NARROW_REL {
        return LinearRelaxation::new(
            0.0,
            next_down_f32(l64.ln() as f32),
            0.0,
            next_up_f32(u64.ln() as f32),
        );
    }

    let log_l = l64.ln();
    let log_u = u64.ln();

    let lower_slope = (log_u - log_l) / (u64 - l64);
    let lower_intercept = log_l - lower_slope * l64;

    // Upper bound: tangent at the parallel-to-chord point d = 1/chord_slope
    // (tightest single tangent line; sound for concave log at any tangent
    // point), falling back to the midpoint tangent when d is out of range.
    // Matches the production branch structure in
    // `ny_propagate::layers::activations::log` so tests/drift.rs can assert
    // slope equality; the intercept correction below intentionally keeps the
    // caller-side multiplication-rounding correction below.
    let chord_slope = lower_slope;
    let (upper_slope, upper_intercept) = {
        let d = 1.0 / chord_slope;
        if l64 > 0.0 && chord_slope.is_finite() && chord_slope > 0.0 && d >= l64 && d <= u64 {
            (chord_slope, d.ln() - 1.0)
        } else {
            let m = f64::midpoint(l64, u64);
            (1.0 / m, m.ln() - 1.0)
        }
    };

    let max_abs_x = nan_propagating_max(l.abs(), u.abs());
    let lower_slope_f32 = lower_slope as f32;
    let upper_slope_f32 = upper_slope as f32;
    let lower_eval_err =
        f32_affine_eval_error(lower_slope, lower_slope_f32, lower_intercept, max_abs_x);
    let upper_eval_err =
        f32_affine_eval_error(upper_slope, upper_slope_f32, upper_intercept, max_abs_x);
    let lower_intercept_f32 = next_down_f32(f64_to_f32_down(lower_intercept - lower_eval_err));
    let upper_intercept_f32 = next_up_f32(f64_to_f32_up(upper_intercept + upper_eval_err));

    // Extreme positive f32s can require an affine slope or intercept outside
    // f32. A monotonic endpoint band is looser but always representable and
    // avoids `inf * 0` during downstream concretization.
    if !lower_slope_f32.is_finite()
        || !upper_slope_f32.is_finite()
        || !lower_intercept_f32.is_finite()
        || !upper_intercept_f32.is_finite()
    {
        return LinearRelaxation::new(
            0.0,
            next_down_f32(log_l as f32),
            0.0,
            next_up_f32(log_u as f32),
        );
    }

    LinearRelaxation::new(
        lower_slope_f32,
        lower_intercept_f32,
        upper_slope_f32,
        upper_intercept_f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Narrow-interval regression tests (#drift-log-narrow-tiny) ──────
    //
    // Mirror of the regression tests in
    // `ny_propagate::layers::activations::log`: the old ABSOLUTE narrow
    // guard (|u - l| < 1e-8) returned the tangent at l for BOTH lines; a
    // tangent lies ABOVE concave ln, so the certified LOWER line overshot
    // ln(u) by r - ln(1 + r), r = (u - l)/l — up to +45.09 on [1e-10, 5e-9].

    /// Strict f64 envelope check with the certified f32 coefficients on the
    /// complete positive binary32 domain.
    fn assert_log_envelope_strict(l: f32, u: f32) {
        assert!(l > 0.0 && l <= u && u.is_finite());
        let r = log_linear_relaxation(l, u);
        let lc = l as f64;
        let uc = u as f64;
        let (ls, li) = (r.lower_slope as f64, r.lower_intercept as f64);
        let (us, ui) = (r.upper_slope as f64, r.upper_intercept as f64);
        let n = 256;
        for i in 0..=n {
            let x = lc + (uc - lc) * (i as f64) / (n as f64);
            let fx = x.ln();
            let slack = 1e-12 * (1.0 + fx.abs());
            let lower = ls * x + li;
            let upper = us * x + ui;
            assert!(
                lower <= fx + slack,
                "log LOWER line above ln on [{l:e}, {u:e}] at x={x:e}: \
                 line={lower:e} ln={fx:e} relax={r:?}"
            );
            assert!(
                upper >= fx - slack,
                "log UPPER line below ln on [{l:e}, {u:e}] at x={x:e}: \
                 line={upper:e} ln={fx:e} relax={r:?}"
            );
            assert!(
                lower <= upper + slack,
                "log lines cross on [{l:e}, {u:e}] at x={x:e}: relax={r:?}"
            );
        }
    }

    #[test]
    fn log_relaxation_narrow_tiny_magnitude_sound() {
        // Historical unsoundness: on [1e-10, 5e-9] the old narrow path's
        // LOWER line evaluated to +25.974148 at x = u vs ln(u) = -19.113828
        // (unsound by 45.09).
        assert_log_envelope_strict(1e-10, 5e-9);
        let r = log_linear_relaxation(1e-10, 5e-9);
        let at_u = (r.lower_slope as f64) * 5e-9 + (r.lower_intercept as f64);
        let ln_u = 5e-9_f64.ln();
        assert!(
            at_u <= ln_u,
            "lower line at u must not exceed ln(u): {at_u} > {ln_u}"
        );

        // Unsound by 1.1e-5 under the old absolute guard.
        let u = 1e-6_f32 + 5e-9_f32;
        assert_log_envelope_strict(1e-6, u);

        // Tiny positive inputs retain their authored domain; no epsilon clamp.
        assert_log_envelope_strict(1e-12, 5e-9);
        assert_log_envelope_strict(1e-12, 2e-12);
    }

    #[test]
    fn log_relaxation_point_interval_constant_band() {
        // l == u exactly: constant band [ln(l) rounded down, ln(l) rounded up].
        for x in [
            f32::from_bits(1),
            1e-12_f32,
            1e-10,
            5e-9,
            1e-6,
            0.5,
            1.0,
            2.0,
            1e8,
        ] {
            let r = log_linear_relaxation(x, x);
            assert_eq!(r.lower_slope, 0.0, "point interval lower slope at x={x:e}");
            assert_eq!(r.upper_slope, 0.0, "point interval upper slope at x={x:e}");
            let fx = (x as f64).ln();
            assert!(
                (r.lower_intercept as f64) <= fx,
                "point band lower above ln at x={x:e}: {} > {fx}",
                r.lower_intercept
            );
            assert!(
                (r.upper_intercept as f64) >= fx,
                "point band upper below ln at x={x:e}: {} < {fx}",
                r.upper_intercept
            );
            // Outward rounding costs at most ~2 f32 ulps of ln(x).
            let width = (r.upper_intercept - r.lower_intercept) as f64;
            assert!(
                width <= 4.0 * (f32::EPSILON as f64) * (1.0 + fx.abs()),
                "point band too wide at x={x:e}: {width}"
            );
        }
    }

    /// Broad soundness sweep over a log-spaced grid of (l, width) pairs:
    /// subnormal-adjacent magnitudes (preserved without an epsilon clamp),
    /// relative widths straddling the 1e-7 narrow threshold, absolute widths
    /// straddling the old 1e-8 guard, and 1-ulp intervals in every binade.
    #[test]
    fn log_relaxation_narrow_sweep_sound() {
        let anchors: [f32; 14] = [
            f32::from_bits(1),           // smallest positive subnormal
            f32::from_bits(0x0040_0000), // mid subnormal
            f32::MIN_POSITIVE,
            1e-38,
            1e-30,
            1e-12,
            1e-10,
            1e-8,
            1e-6,
            1e-3,
            1.0,
            2.0,
            1e4,
            1e8,
        ];
        let rel_widths: [f64; 8] = [0.0, 1e-9, 5e-8, 1e-7, 2e-7, 1e-3, 5e-2, 49.0];
        let abs_widths: [f64; 6] = [0.0, 1e-12, 5e-9, 1e-8, 2e-8, 1e-4];
        for &l in &anchors {
            for &w in &rel_widths {
                let u = ((l as f64) * (1.0 + w)) as f32;
                if u >= l && u.is_finite() {
                    assert_log_envelope_strict(l, u);
                }
            }
            for &w in &abs_widths {
                let u = ((l as f64) + w) as f32;
                if u >= l && u.is_finite() {
                    assert_log_envelope_strict(l, u);
                }
            }
            let u = next_up_f32(l);
            if u.is_finite() {
                assert_log_envelope_strict(l, u);
            }
        }
    }
}
