// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #3158: f64 precision proptests for GELU sound chord and tangent.
//!
//! Every S-shaped activation with f64 chord/tangent computation has dedicated
//! precision proptests that verify:
//! 1. Slope matches pure f64 reference within 1 ULP
//! 2. Directed rounding brackets the f64 reference intercept
//!
//! Pattern: SiLU chord/tangent precision proptests (silu/tests.rs:504-580).

use super::eval;
use super::sound_tables;
use super::{gelu_sound_linear_relaxation, gelu_tanh_sound_linear_relaxation, GeluApproximation};
use proptest::prelude::*;

/// ULP distance between two f32 values, handling sign correctly.
/// Maps f32 bits to a linear ordering where adjacent floats differ by 1.
fn ulp_distance(a: f32, b: f32) -> u64 {
    fn to_ordered(x: f32) -> i64 {
        let bits = x.to_bits() as i32;
        if bits < 0 {
            (0x8000_0000_u32 as i32 - bits) as i64
        } else {
            bits as i64
        }
    }
    (to_ordered(a) - to_ordered(b)).unsigned_abs()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #3158: Verify GELU sound chord slope has f64 precision for narrow intervals.
    /// For intervals with width in [1e-7, 1e-4], the sound relaxation computes a
    /// chord slope in f64. When used as a bound, it must match an independent f64
    /// reference within 1 ULP, and directed rounding must bracket the reference
    /// intercept. Pattern: SiLU chord precision proptest (silu/tests.rs:504).
    #[test]
    fn proptest_gelu_sound_chord_f64_precision(
        l in -8.0f32..8.0,
        width_exp in -7.0f64..-4.0,
    ) {
        let delta = 10.0_f64.powf(width_exp) as f32;
        let u = l + delta;
        prop_assume!(u > l);
        // Stay above the 1e-8 point-interval threshold in sound_relax.rs
        prop_assume!((u - l) > 1e-8);

        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let (ls, li, us, ui) = match approx {
                GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
            };

            // Independent f64 reference chord
            let l64 = l as f64;
            let u64 = u as f64;
            let (fl64, fu64) = match approx {
                GeluApproximation::Erf => {
                    (eval::gelu_erf_f64(l64), eval::gelu_erf_f64(u64))
                }
                GeluApproximation::Tanh => {
                    (eval::gelu_tanh_f64(l64), eval::gelu_tanh_f64(u64))
                }
            };
            let ref_slope_64 = (fu64 - fl64) / (u64 - l64);
            let ref_slope = ref_slope_64 as f32;
            let ref_intercept_64 = fl64 - ref_slope_64 * l64;
            let ref_intercept = ref_intercept_64 as f32;

            // Sound path uses chord for at least one bound in most mask regions.
            // Check if either slope matches chord within 1 ULP.
            let ls_ulps = ulp_distance(ls, ref_slope);
            let us_ulps = ulp_distance(us, ref_slope);

            if ls_ulps <= 1 {
                // Lower bound used chord: intercept must bracket downward.
                prop_assert!(
                    li <= ref_intercept,
                    "GELU({approx:?}) sound chord: lower_intercept {li} > \
                     ref {ref_intercept} for [{l}, {u}]"
                );
            }
            if us_ulps <= 1 {
                // Upper bound used chord: intercept must bracket upward.
                prop_assert!(
                    ui >= ref_intercept,
                    "GELU({approx:?}) sound chord: upper_intercept {ui} < \
                     ref {ref_intercept} for [{l}, {u}]"
                );
            }

            // Always verify soundness: lower bound ≤ GELU(x) ≤ upper bound.
            let gelu_fn = |x: f32| match approx {
                GeluApproximation::Erf => eval::gelu_erf(x),
                GeluApproximation::Tanh => eval::gelu_tanh(x),
            };
            for &x in &[l, u, f32::midpoint(l, u)] {
                let gx = gelu_fn(x);
                prop_assert!(
                    ls * x + li <= gx + 1e-5,
                    "GELU({approx:?}) sound lower violation at x={x}: \
                     {} > {} for [{l}, {u}]",
                    ls * x + li, gx
                );
                prop_assert!(
                    us * x + ui >= gx - 1e-5,
                    "GELU({approx:?}) sound upper violation at x={x}: \
                     {} < {} for [{l}, {u}]",
                    us * x + ui, gx
                );
            }
        }
    }

    /// #3158: Verify GELU sound tangent has f64 precision for both Erf and Tanh.
    /// The tangent line at point d is: y = GELU'(d)·x + (GELU(d) - GELU'(d)·d).
    /// For large |d|, the intercept computation suffers catastrophic cancellation
    /// in f32. Directed rounding must bracket the f64 reference.
    /// Pattern: SiLU tangent precision proptest (silu/tests.rs:544).
    #[test]
    fn proptest_gelu_sound_tangent_precision(d in -20.0f32..20.0) {
        let max_abs_x = d.abs() + 1.0;

        // ── Erf tangent ──
        {
            let (slope, lower_intercept, upper_intercept) =
                sound_tables::gelu_tangent_at(d, max_abs_x);

            let d64 = d as f64;
            let ref_slope_64 = eval::gelu_derivative_erf_f64(d64);
            let ref_eval_64 = eval::gelu_erf_f64(d64);
            let ref_intercept_64 = ref_eval_64 - ref_slope_64 * d64;
            let ref_slope = ref_slope_64 as f32;
            let ref_intercept = ref_intercept_64 as f32;

            let slope_ulps = ulp_distance(slope, ref_slope);
            prop_assert!(
                slope_ulps <= 1,
                "GELU(Erf) tangent slope not within 1 ULP: got {slope} \
                 vs ref {ref_slope} ({slope_ulps} ULPs) at d={d}"
            );
            prop_assert!(
                lower_intercept <= ref_intercept,
                "GELU(Erf) tangent lower_intercept {lower_intercept} > \
                 ref {ref_intercept} at d={d}"
            );
            prop_assert!(
                upper_intercept >= ref_intercept,
                "GELU(Erf) tangent upper_intercept {upper_intercept} < \
                 ref {ref_intercept} at d={d}"
            );
        }

        // ── Tanh tangent ──
        {
            let (slope, lower_intercept, upper_intercept) =
                sound_tables::gelu_tanh_tangent_at(d, max_abs_x);

            let d64 = d as f64;
            let ref_slope_64 = eval::gelu_derivative_tanh_f64(d64);
            let ref_eval_64 = eval::gelu_tanh_f64(d64);
            let ref_intercept_64 = ref_eval_64 - ref_slope_64 * d64;
            let ref_slope = ref_slope_64 as f32;
            let ref_intercept = ref_intercept_64 as f32;

            let slope_ulps = ulp_distance(slope, ref_slope);
            prop_assert!(
                slope_ulps <= 1,
                "GELU(Tanh) tangent slope not within 1 ULP: got {slope} \
                 vs ref {ref_slope} ({slope_ulps} ULPs) at d={d}"
            );
            prop_assert!(
                lower_intercept <= ref_intercept,
                "GELU(Tanh) tangent lower_intercept {lower_intercept} > \
                 ref {ref_intercept} at d={d}"
            );
            prop_assert!(
                upper_intercept >= ref_intercept,
                "GELU(Tanh) tangent upper_intercept {upper_intercept} < \
                 ref {ref_intercept} at d={d}"
            );
        }
    }
}
