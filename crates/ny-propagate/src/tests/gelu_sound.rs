// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::activations::LinearRelaxation;
use crate::layers::{
    gelu_eval, gelu_sound_linear_relaxation, gelu_tanh_inflection_point,
    gelu_tanh_sound_linear_relaxation, pow2_linear_relaxation, GeluApproximation,
};
use proptest::prelude::*;

/// Independent f64 GELU (erf) reference for strict proptest. (#3292)
/// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
fn gelu_erf_f64_reference(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let inv_sqrt2: f64 = 1.0 / 2.0_f64.sqrt();
    0.5 * x * (1.0 + libm::erf(x * inv_sqrt2))
}

/// Independent f64 GELU (tanh approximation) reference for strict proptest. (#3292)
/// GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_tanh_f64_reference(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

fn assert_sound_interval(l: f32, u: f32) {
    let (ls, li, us, ui) = gelu_sound_linear_relaxation(l, u);

    // Probe a dense-ish grid; the implementation is intended to be provably sound,
    // but this test is a pragmatic regression check.
    let n = 2000;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = l + (u - l) * t;
        let y = gelu_eval(x, GeluApproximation::Erf);
        let lo = ls * x + li;
        let up = us * x + ui;
        assert!(
            lo <= y + 1e-5,
            "lower violated at x={x}: lo={lo} > y={y} on [{l}, {u}]"
        );
        assert!(
            y <= up + 1e-5,
            "upper violated at x={x}: y={y} > up={up} on [{l}, {u}]"
        );
    }
}

fn assert_sound_interval_tanh(l: f32, u: f32) {
    let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(l, u);

    let n = 2000;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = l + (u - l) * t;
        let y = gelu_eval(x, GeluApproximation::Tanh);
        let lo = ls * x + li;
        let up = us * x + ui;
        assert!(
            lo <= y + 1e-5,
            "tanh lower violated at x={x}: lo={lo} > y={y} on [{l}, {u}]"
        );
        assert!(
            y <= up + 1e-5,
            "tanh upper violated at x={x}: y={y} > up={up} on [{l}, {u}]"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_sound_relaxation_smoke() {
    let intervals = [
        // Entirely negative intervals
        (-5.0, -3.0),
        (-2.0, -1.0),
        (-1.0, -0.5),
        // Entirely positive intervals
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        // Cross-zero intervals (all sub-regions)
        (-1.0, 1.0),
        (-3.0, 0.5),
        (-0.5, 3.0),
        (-3.0, 3.0),
        (-2.0, 1.2),
    ];

    for (l, u) in intervals {
        assert_sound_interval(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_sound_relaxation_edge_cases() {
    let edge_cases = [
        // Near sqrt(2)
        (1.41, 1.42),
        (1.414, 1.415),
        // Near -sqrt(2)
        (-1.42, -1.41),
        (-1.415, -1.414),
        // Near zero
        (-0.1, 0.1),
        (-0.01, 0.01),
        // Very small intervals
        (0.5, 0.501),
        (-0.5, -0.499),
        // Around the GELU minimizer
        (-0.8, -0.7),
        (-1.0, -0.5),
    ];

    for (l, u) in edge_cases {
        assert_sound_interval(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_sound_relaxation_smoke() {
    let intervals = [
        (-5.0, -3.0),
        (-2.0, -1.0),
        (-1.0, -0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (-1.0, 1.0),
        (-3.0, 0.5),
        (-0.5, 3.0),
        (-3.0, 3.0),
        (-2.0, 1.2),
    ];

    for (l, u) in intervals {
        assert_sound_interval_tanh(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_sound_relaxation_edge_cases() {
    let split = gelu_tanh_inflection_point();
    let delta = 5e-4;
    let edge_cases = [
        // Near tanh-approx inflection point
        (split - 2.0 * delta, split - delta),
        (split - delta, split + delta),
        (-split - delta, -split + delta),
        (-split + delta, -split + 2.0 * delta),
        // Near zero
        (-0.1, 0.1),
        (-0.01, 0.01),
        // Very small intervals
        (0.5, 0.501),
        (-0.5, -0.499),
        // Around the GELU minimizer
        (-0.8, -0.7),
        (-1.0, -0.5),
    ];

    for (l, u) in edge_cases {
        assert_sound_interval_tanh(l, u);
    }
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_inflection_point_sane() {
    let split = gelu_tanh_inflection_point();
    assert!(split.is_finite());
    assert!(
        (1.0..2.0).contains(&split),
        "inflection point out of expected range: {split}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn gelu_sound_relaxation_degenerate_and_nonfinite() {
    // Point interval should produce identical lower/upper lines.
    let (ls, li, us, ui) = gelu_sound_linear_relaxation(1.0, 1.0);
    assert!((ls - us).abs() < 1e-6);
    assert!((li - ui).abs() < 1e-6);

    // Infinite/NaN bounds return maximally loose (NOT identity — identity is unsound, #1837).
    assert_eq!(
        gelu_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
    assert_eq!(
        gelu_sound_linear_relaxation(f32::NAN, 1.0),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
}

#[ntest::timeout(10000)]
#[test]
fn gelu_tanh_sound_relaxation_degenerate_and_nonfinite() {
    let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(1.0, 1.0);
    assert!((ls - us).abs() < 1e-6);
    assert!((li - ui).abs() < 1e-6);

    // Infinite/NaN bounds return maximally loose (NOT identity — #1837).
    assert_eq!(
        gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
    assert_eq!(
        gelu_tanh_sound_linear_relaxation(f32::NAN, 1.0),
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
    );
}

#[ntest::timeout(10000)]
#[test]
fn pow2_subnormal_kani_counterexample_regression_1795() {
    // Values from issue #1795 Kani counterexample.
    let l = -3.743_392e-23f32;
    let u = -1.175_494e-38f32;
    let x = -1.871_839e-23f32;

    assert!(l < u, "counterexample interval must satisfy l < u");
    assert!(
        (l as f64) * (l as f64) < f64::from(f32::MIN_POSITIVE),
        "counterexample should be in subnormal x^2 regime"
    );

    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = pow2_linear_relaxation(l, u);

    // Guard behavior: subnormal intervals use a constant zero lower bound.
    assert_eq!(
        lower_slope, 0.0,
        "expected zero lower slope, got {lower_slope:e}"
    );
    assert_eq!(
        lower_intercept, 0.0,
        "expected zero lower intercept, got {lower_intercept:e}"
    );

    let fx = x * x; // underflows to 0.0 in f32 for this counterexample
    let lb = lower_slope * x + lower_intercept;
    let ub = upper_slope * x + upper_intercept;
    assert!(
        lb <= fx,
        "subnormal lower bound must be sound: lb={lb:e}, x2={fx:e}, slope={lower_slope:e}, intercept={lower_intercept:e}"
    );
    assert!(
        ub >= fx,
        "subnormal upper bound must be sound: ub={ub:e}, x2={fx:e}, slope={upper_slope:e}, intercept={upper_intercept:e}"
    );
}

// ── Strict zero-tolerance CROWN relaxation proptests (#3292) ─────────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.
// GELU (erf) reference: eval::gelu_erf_f64, GELU (tanh) reference: eval::gelu_tanh_f64.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for GELU (erf) CROWN relaxation.
    /// Uses f64 reference (gelu_erf_f64) with zero tolerance on 200-point grid.
    /// Ref: alpha-beta-CROWN auto_LiRPA GELU relaxation, #3292.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_gelu_erf_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let (ls, li, us, ui) = gelu_sound_linear_relaxation(l, u);

        // Skip NaN fallback (infinite bounds).
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = gelu_erf_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "GELU(erf) lower bound UNSOUND at x={x}: {lower_val} > GELU({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "GELU(erf) upper bound UNSOUND at x={x}: {upper_val} < GELU({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }

    /// Strict soundness proptest for GELU (tanh) CROWN relaxation.
    /// Uses f64 reference (gelu_tanh_f64) with zero tolerance on 200-point grid.
    /// Ref: alpha-beta-CROWN auto_LiRPA GELU tanh approximation, #3292.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_gelu_tanh_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(l, u);

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = gelu_tanh_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "GELU(tanh) lower bound UNSOUND at x={x}: {lower_val} > GELU_tanh({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "GELU(tanh) upper bound UNSOUND at x={x}: {upper_val} < GELU_tanh({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }
}
