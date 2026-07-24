// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::TRIG_RELAX_EPS;
use super::cos::cos_linear_relaxation;
use super::sin::sin_linear_relaxation;
use super::tan::tan_linear_relaxation;
use crate::tests::assert_relaxation_sound;
use proptest::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_sin_linear_relaxation_concave() {
    let relaxation = sin_linear_relaxation(0.1, 1.0);
    assert_relaxation_sound(0.1, 1.0, relaxation, f32::sin, 1e-4, "sin");
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_linear_relaxation_convex() {
    let relaxation = sin_linear_relaxation(3.4, 5.0);
    assert_relaxation_sound(3.4, 5.0, relaxation, f32::sin, 1e-4, "sin");
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_linear_relaxation_cross_inflection_falls_back() {
    let r = sin_linear_relaxation(2.5, 4.0);
    assert!(r.lower_slope.abs() < 1e-6 && r.upper_slope.abs() < 1e-6);
    assert!(
        r.lower_intercept <= -1.0 - TRIG_RELAX_EPS * 0.5
            && r.upper_intercept >= 1.0 + TRIG_RELAX_EPS * 0.5
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_linear_relaxation_concave() {
    let relaxation = cos_linear_relaxation(0.1, 1.0);
    assert_relaxation_sound(0.1, 1.0, relaxation, f32::cos, 1e-4, "cos");
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_linear_relaxation_convex() {
    let relaxation = cos_linear_relaxation(2.0, 3.0);
    assert_relaxation_sound(2.0, 3.0, relaxation, f32::cos, 1e-4, "cos");
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_linear_relaxation_cross_inflection_falls_back() {
    let r = cos_linear_relaxation(1.0, 2.0);
    assert!(r.lower_slope.abs() < 1e-6 && r.upper_slope.abs() < 1e-6);
    assert!(
        r.lower_intercept <= -1.0 - TRIG_RELAX_EPS * 0.5
            && r.upper_intercept >= 1.0 + TRIG_RELAX_EPS * 0.5
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_linear_relaxation_sound() {
    for (l, u) in [(-1.2, -0.2), (-0.5, 0.3), (0.2, 1.0)] {
        let relaxation = tan_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::tan, 1e-4, "tan");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_linear_relaxation_asymptote_fallback() {
    let r = tan_linear_relaxation(1.4, 1.8);
    assert!(r.lower_slope.abs() < 1e-6 && r.upper_slope.abs() < 1e-6);
    assert!(r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative());
    assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    #[ntest::timeout(10000)]
    #[test]
    fn prop_sin_linear_relaxation_sound(l in -20.0f32..20.0, u in -20.0f32..20.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = sin_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::sin, 1e-4, "sin");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn prop_cos_linear_relaxation_sound(l in -20.0f32..20.0, u in -20.0f32..20.0) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = cos_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::cos, 1e-4, "cos");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn prop_tan_linear_relaxation_sound(l in -1.4f32..1.4, u in -1.4f32..1.4) {
        let (l, u) = if l <= u { (l, u) } else { (u, l) };
        let relaxation = tan_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, f32::tan, 1e-4, "tan");
    }
}

// ========================================================================
// CROWN backward soundness tests (#2292)
//
// These test propagate_linear_with_bounds (the full CROWN backward path),
// not just the per-element relaxation functions. They exercise:
//   - coefficient matrix multiplication via crown_elementwise_backward
//   - sign-dependent slope/intercept swapping for negative A coefficients
//   - BoundedTensor → LinearBounds extraction
//
// Interval choices:
//   Sin: concave region [0.1, 1.0], convex region [3.4, 5.0],
//        crossing inflection [-1.0, 1.0] (falls back to constant bounds)
//   Cos: concave region [0.1, 1.0], convex region [2.0, 3.0],
//        crossing [-0.5, 1.5]
//   Tan: within half-period [-1.2, -0.2], crossing zero [-0.5, 0.3],
//        positive [0.2, 1.0]; all avoid asymptotes
// ========================================================================

// assert_crown_backward_sound extracted to crate::tests::assert_crown_backward_sound (#2307)
use crate::tests::assert_crown_backward_sound;

#[ntest::timeout(10000)]
#[test]
fn test_sin_crown_backward_soundness() {
    let layer = super::SinLayer::new();
    // Concave region, convex region, and crossing-inflection (constant fallback)
    let intervals = [(0.1, 1.0), (3.4, 5.0), (-1.0, 1.0)];
    assert_crown_backward_sound(&layer, &intervals, f32::sin);
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_crown_backward_soundness() {
    let layer = super::CosLayer::new();
    // Concave region, convex region, and crossing
    let intervals = [(0.1, 1.0), (2.0, 3.0), (-0.5, 1.5)];
    assert_crown_backward_sound(&layer, &intervals, f32::cos);
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_crown_backward_soundness() {
    let layer = super::TanLayer::new();
    // All within (-π/2, π/2) to avoid asymptotes
    let intervals = [(-1.2, -0.2), (-0.5, 0.3), (0.2, 1.0)];
    assert_crown_backward_sound(&layer, &intervals, f32::tan);
}

/// Regression for #2287: sound-mode Sin path uses constant bounds built from
/// `concretize_sound()`, which now repairs non-finite/inverted elements internally.
#[test]
fn test_sin_sound_mode_constant_bounds_repairs_invalid_concretization() {
    use crate::LinearBounds;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    let layer = super::SinLayer::new().with_sound_mode(true);
    let pre = BoundedTensor::new(arr1(&[5.0]).into_dyn(), arr1(&[5.0]).into_dyn())
        .expect("invariant: valid singleton pre-activation interval");

    // Asymmetric coefficients can induce concretization inversions.
    let bounds = LinearBounds::new(
        arr2(&[[-10.0]]),
        arr1(&[100.0]),
        arr2(&[[10.0]]),
        arr1(&[-100.0]),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre)
        .expect("sound-mode Sin should return conservative constant bounds");

    assert!(
        result.lower_a.iter().all(|&v| v == 0.0),
        "sound-mode constant bounds must zero lower slopes"
    );
    assert!(
        result.upper_a.iter().all(|&v| v == 0.0),
        "sound-mode constant bounds must zero upper slopes"
    );
    assert!(
        result.lower_b[0] <= result.upper_b[0],
        "sound-mode constant bounds must not invert: {} > {}",
        result.lower_b[0],
        result.upper_b[0]
    );
}
