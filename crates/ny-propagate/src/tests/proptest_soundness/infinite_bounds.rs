// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for infinite-bound soundness bugs.
//!
//! #1836: Activation eval functions produce NaN at -inf via 0*inf, corrupting IBP bounds.
//! #1837: GELU CROWN relaxation returns identity for infinite bounds — lower bound unsound.
//!
//! These are deterministic unit tests, not property tests, since they target
//! specific infinite-input corner cases that proptest's finite ranges never reach.

use crate::layers::common::BoundPropagation;
use crate::layers::{
    gelu_eval, gelu_linear_relaxation, gelu_sound_linear_relaxation,
    gelu_tanh_sound_linear_relaxation, GELULayer, GeluApproximation, HardSwishLayer, MishLayer,
    SoftsignLayer,
};
use crate::LinearBounds;
use ndarray::arr1;
use ntest::timeout;
use ny_tensor::BoundedTensor;

// =========================================================================
// #1836: Activation eval NaN at -inf (IBP soundness)
// =========================================================================

/// Regression #1836: gelu_eval(NEG_INFINITY, Erf) must not be NaN.
/// GELU(-inf) = 0.5 * (-inf) * (1 + erf(-inf/sqrt(2))) = 0.5 * (-inf) * (1 + (-1)) = 0.5 * (-inf) * 0 = NaN
/// Correct value: 0.0 (limit as x -> -inf)
#[timeout(10000)]
#[test]
fn regression_1836_gelu_erf_eval_neg_inf() {
    let val = gelu_eval(f32::NEG_INFINITY, GeluApproximation::Erf);
    assert!(
        !val.is_nan(),
        "gelu_eval(NEG_INFINITY, Erf) = NaN — must be 0.0. Bug #1836."
    );
    assert_eq!(
        val, 0.0,
        "gelu_eval(NEG_INFINITY, Erf) should be 0.0, got {val}"
    );
}

/// Regression #1836: gelu_eval(NEG_INFINITY, Tanh) must not be NaN.
#[timeout(10000)]
#[test]
fn regression_1836_gelu_tanh_eval_neg_inf() {
    let val = gelu_eval(f32::NEG_INFINITY, GeluApproximation::Tanh);
    assert!(
        !val.is_nan(),
        "gelu_eval(NEG_INFINITY, Tanh) = NaN — must be 0.0. Bug #1836."
    );
    assert_eq!(
        val, 0.0,
        "gelu_eval(NEG_INFINITY, Tanh) should be 0.0, got {val}"
    );
}

/// Regression #1836: gelu_eval(+INFINITY) must be +inf, not NaN.
#[timeout(10000)]
#[test]
fn regression_1836_gelu_eval_pos_inf() {
    let val_erf = gelu_eval(f32::INFINITY, GeluApproximation::Erf);
    let val_tanh = gelu_eval(f32::INFINITY, GeluApproximation::Tanh);
    assert!(
        val_erf == f32::INFINITY,
        "gelu_eval(INFINITY, Erf) should be INFINITY, got {val_erf}"
    );
    assert!(
        val_tanh == f32::INFINITY,
        "gelu_eval(INFINITY, Tanh) should be INFINITY, got {val_tanh}"
    );
}

/// Regression #1836: Mish eval at NEG_INFINITY produces NaN.
/// mish(-inf) = (-inf) * tanh(softplus(-inf)) = (-inf) * tanh(exp(-inf))
///            = (-inf) * tanh(0) = (-inf) * 0 = NaN
/// Correct value: 0.0
///
/// Note: Rust f32::min/max absorb NaN (IEEE 754-2008), so NaN from mish_eval(-inf)
/// may be masked by min/max with non-NaN values. But the bounds are still WRONG:
/// for [-inf, -5], the upper bound should include Mish(-inf)=0.0, but the NaN gets
/// absorbed and the upper bound becomes Mish(-5) ≈ -0.034, missing the true maximum.
#[timeout(10000)]
#[test]
fn regression_1836_mish_ibp_neg_inf_soundness() {
    // Test with [-inf, -5] where the critical point (-0.31) is NOT in the interval.
    // This means the NaN from mish_eval(-inf) flows through min/max without the
    // critical-point backup, producing wrong bounds.
    //
    // True Mish values on [-inf, -5]:
    //   Mish(-inf) = 0.0 (limit)
    //   Mish(-5) ≈ -0.0340
    //
    // So the true range is approximately [-0.034, 0.0]
    // Upper bound must be >= 0.0 for soundness.
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[-5.0]).into_dyn(),
    )
    .unwrap();

    let mish = MishLayer::new();
    let output = mish.propagate_ibp(&input).unwrap();

    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];
    assert!(
        !lower.is_nan(),
        "Mish IBP lower bound is NaN for [-inf, -5]. Bug #1836."
    );
    assert!(
        !upper.is_nan(),
        "Mish IBP upper bound is NaN for [-inf, -5]. Bug #1836."
    );
    // Mish(-inf) approaches 0.0, so the upper bound must be >= 0.0 - epsilon.
    // With the fix, mish_eval(-inf) returns 0.0 exactly, so upper = max(0.0, -0.034) = 0.0.
    assert!(
        upper >= -1e-6,
        "Mish IBP upper bound for [-inf, -5] is {upper}, but Mish(-inf) approaches 0.0, \
         so upper must be >= 0.0. The NaN from mish_eval(-inf) was absorbed by \
         f32::max, producing an unsound upper bound. Bug #1836."
    );
}

/// Regression #1836: HardSwish eval at NEG_INFINITY produces NaN.
/// hardswish(-inf) = (-inf) * clamp((-inf+3)/6, 0, 1) = (-inf) * 0 = NaN
/// Correct value: 0.0
///
/// For interval [-inf, -2]: HardSwish(-inf) = 0, HardSwish(-2) = -2*(1/6) = -0.333
/// The NaN from eval(-inf) gets absorbed by f32::max, so the upper bound becomes
/// HardSwish(-2) ≈ -0.333 instead of max(0, -0.333) = 0. This is unsound.
#[timeout(10000)]
#[test]
fn regression_1836_hardswish_ibp_neg_inf_soundness() {
    // Test with [-inf, -2] where HardSwish(-inf) = 0, HardSwish(-2) = -1/3
    // Upper bound should be >= 0, but NaN absorption gives -1/3 (unsound)
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[-2.0]).into_dyn(),
    )
    .unwrap();

    let hardswish = HardSwishLayer::new();
    let output = hardswish.propagate_ibp(&input).unwrap();

    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];
    assert!(
        !lower.is_nan(),
        "HardSwish IBP lower bound is NaN for [-inf, -2]. Bug #1836."
    );
    assert!(
        !upper.is_nan(),
        "HardSwish IBP upper bound is NaN for [-inf, -2]. Bug #1836."
    );
    // HardSwish(-inf) approaches 0.0, so upper must be >= 0.
    // With the fix, hardswish_eval(-inf) returns 0.0 exactly, so upper = max(0.0, -0.333) = 0.0.
    assert!(
        upper >= -1e-6,
        "HardSwish IBP upper for [-inf, -2] is {upper}, but HardSwish(-inf) approaches 0, \
         so upper must be >= 0. NaN from eval(-inf) was absorbed by f32::max. Bug #1836."
    );
}

/// Regression #1836: Softsign eval at +/-INFINITY produces NaN.
/// softsign(±inf) = ±inf / (1 + |±inf|) = ±inf / inf = NaN
/// Correct values: softsign(-inf) = -1.0, softsign(+inf) = +1.0
///
/// Since softsign_scalar produces NaN for ±inf, BoundedTensor::new rejects the output.
/// After fix: softsign_scalar should handle ±inf, producing ±1.0.
#[timeout(10000)]
#[test]
fn regression_1836_softsign_eval_inf() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();

    let softsign = SoftsignLayer::new();
    let result = softsign.propagate_ibp(&input);

    match result {
        Ok(output) => {
            let lower = output.lower()[[0]];
            let upper = output.upper()[[0]];
            assert!(
                !lower.is_nan() && !upper.is_nan(),
                "Softsign IBP produced NaN bounds for [-inf, +inf]. Bug #1836."
            );
            assert!(
                lower <= upper,
                "Softsign IBP bounds inverted for [-inf, +inf]: lower={lower} > upper={upper}"
            );
            assert!(
                lower <= -1.0 + 1e-5,
                "Softsign lower for [-inf, +inf] should be <= -1, got {lower}"
            );
            assert!(
                upper >= 1.0 - 1e-5,
                "Softsign upper for [-inf, +inf] should be >= 1, got {upper}"
            );
        }
        Err(e) => {
            let msg = format!("{e}");
            // The bug: softsign_scalar(±inf) = NaN (inf/inf), BoundedTensor::new rejects NaN
            assert!(
                msg.contains("NaN"),
                "Softsign IBP for [-inf, +inf] failed with NaN in output. Bug #1836. Error: {msg}"
            );
            panic!(
                "Bug #1836: Softsign propagate_ibp returns Err for [-inf, +inf] because \
                 softsign_scalar(inf) = inf/(1+inf) = NaN. Fix: add is_finite() guard."
            );
        }
    }
}

/// Regression #1836: GELU IBP with NEG_INFINITY lower bound must not produce NaN.
#[timeout(10000)]
#[test]
fn regression_1836_gelu_ibp_neg_inf() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let input = BoundedTensor::new_unchecked(
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            arr1(&[1.0]).into_dyn(),
        )
        .unwrap();

        let gelu = GELULayer::new(approx);
        let output = gelu.propagate_ibp(&input).unwrap();

        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        assert!(
            !lower.is_nan(),
            "GELU({approx:?}) IBP lower bound is NaN for [-inf, 1]. Bug #1836."
        );
        assert!(
            !upper.is_nan(),
            "GELU({approx:?}) IBP upper bound is NaN for [-inf, 1]. Bug #1836."
        );
        assert!(
            lower <= upper,
            "GELU({approx:?}) IBP bounds inverted for [-inf, 1]: lower={lower} > upper={upper}"
        );
        // GELU(-inf) = 0, GELU(1) ≈ 0.841
        // GELU minimum ≈ -0.17 at x ≈ -0.752 (GELU_MINIMIZER_X)
        // Lower bound must contain the minimum, upper must contain GELU(1)
        assert!(
            lower <= -0.1699,
            "GELU({approx:?}) IBP lower for [-inf, 1] must contain minimum ~-0.17, got {lower}"
        );
        assert!(
            upper >= 0.84,
            "GELU({approx:?}) IBP upper for [-inf, 1] must contain GELU(1) ~0.841, got {upper}"
        );
    }
}

// =========================================================================
// #1837: GELU CROWN relaxation identity for infinite bounds (unsound lower)
// =========================================================================

/// Regression #1837: gelu_linear_relaxation returns identity (1,0,1,0) for infinite bounds.
/// The lower bound y=x is unsound: GELU(x) < x for x near -0.5.
/// At x = -0.5: GELU(-0.5) ≈ -0.154, but identity says lower = -0.5 < -0.154.
/// Wait — that means identity lower IS sound (lower <= f(x)).
/// The actual issue: identity UPPER says GELU(x) <= x, but GELU(-0.5) ≈ -0.154 > -0.5.
/// So GELU(x) > x near x = -0.5, violating the upper bound.
///
/// Let me re-analyze per the issue:
/// - Lower bound (slope=1, intercept=0): claims GELU(x) >= x.
///   GELU(-0.5) ≈ -0.154, x = -0.5, so -0.154 >= -0.5 ✓
///   GELU(0.3) ≈ 0.185, x = 0.3, so 0.185 >= 0.3 ✗ — UNSOUND
///
/// So the lower bound GELU(x) >= x fails for 0 < x < ~1 where Φ(x) < 1.
/// GELU(x) = x * Φ(x), so GELU(x) >= x requires Φ(x) >= 1, which only holds at x = +inf.
#[timeout(10000)]
#[test]
fn regression_1837_gelu_relaxation_identity_unsound() {
    // Test with [-inf, 2.0] — this triggers the infinite guard
    let (ls, li, us, ui) = gelu_linear_relaxation(f32::NEG_INFINITY, 2.0, GeluApproximation::Erf);

    // If we get identity, the lower bound claims GELU(x) >= x, which is false
    // at x = 0.3: GELU(0.3) ≈ 0.185 < 0.3
    if ls == 1.0 && li == 0.0 {
        // Verify it IS unsound
        let x = 0.3_f32;
        let gelu_x = gelu_eval(x, GeluApproximation::Erf);
        let lower = ls * x + li; // = x = 0.3
        assert!(
            gelu_x < lower,
            "Expected identity lower to be unsound at x=0.3: \
             GELU(0.3) = {gelu_x} should be < {lower}. \
             If this passes, the identity IS unsound. Bug #1837 confirmed."
        );
        panic!(
            "Bug #1837: gelu_linear_relaxation returns identity (1,0,1,0) for infinite bounds. \
             Lower bound GELU(x) >= x is unsound: GELU(0.3)={gelu_x} < 0.3"
        );
    }

    // If we DON'T get identity, the bug is fixed. Verify the new relaxation is sound.
    // Include GELU minimum at x ≈ -0.752 (GELU_MINIMIZER_X) and transition region.
    let test_points = [
        -1e6_f32, -1000.0, -100.0, -10.0, -1.0, -0.752, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0,
    ];
    for &x in &test_points {
        let gelu_x = gelu_eval(x, GeluApproximation::Erf);
        let lower = ls * x + li;
        let upper = us * x + ui;
        let tol = 1e-5 * gelu_x.abs().max(1.0);
        assert!(
            lower <= gelu_x + tol,
            "GELU relaxation lower unsound at x={x}: lower={lower} > GELU(x)={gelu_x}"
        );
        assert!(
            upper + tol >= gelu_x,
            "GELU relaxation upper unsound at x={x}: upper={upper} < GELU(x)={gelu_x}"
        );
    }
}

/// Regression #1837: gelu_sound_linear_relaxation identity for infinite bounds.
#[timeout(10000)]
#[test]
fn regression_1837_gelu_sound_relaxation_identity_unsound() {
    let (ls, li, us, ui) = gelu_sound_linear_relaxation(f32::NEG_INFINITY, 2.0);

    if ls == 1.0 && li == 0.0 {
        let x = 0.3_f32;
        let gelu_x = gelu_eval(x, GeluApproximation::Erf);
        let lower = ls * x + li;
        assert!(gelu_x < lower, "Identity lower should be unsound at x=0.3");
        panic!(
            "Bug #1837: gelu_sound_linear_relaxation returns identity for infinite bounds. \
             Lower bound unsound: GELU(0.3)={gelu_x} < 0.3"
        );
    }

    // Fixed: verify soundness on sample points including GELU minimum and large negatives.
    let test_points = [
        -1e6_f32, -1000.0, -100.0, -10.0, -1.0, -0.752, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0,
    ];
    for &x in &test_points {
        let gelu_x = gelu_eval(x, GeluApproximation::Erf);
        let lower = ls * x + li;
        let upper = us * x + ui;
        let tol = 1e-5 * gelu_x.abs().max(1.0);
        assert!(
            lower <= gelu_x + tol,
            "gelu_sound_linear_relaxation lower unsound at x={x}: lower={lower} > GELU(x)={gelu_x}"
        );
        assert!(
            upper + tol >= gelu_x,
            "gelu_sound_linear_relaxation upper unsound at x={x}: upper={upper} < GELU(x)={gelu_x}"
        );
    }
}

/// Regression #1837: gelu_tanh_sound_linear_relaxation identity for infinite bounds.
#[timeout(10000)]
#[test]
fn regression_1837_gelu_tanh_sound_relaxation_identity_unsound() {
    let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, 2.0);

    if ls == 1.0 && li == 0.0 {
        let x = 0.3_f32;
        let gelu_x = gelu_eval(x, GeluApproximation::Tanh);
        let lower = ls * x + li;
        assert!(gelu_x < lower, "Identity lower should be unsound at x=0.3");
        panic!(
            "Bug #1837: gelu_tanh_sound_linear_relaxation returns identity for infinite bounds. \
             Lower bound unsound: GELU_tanh(0.3)={gelu_x} < 0.3"
        );
    }

    // Fixed: verify soundness on sample points including GELU minimum and large negatives.
    let test_points = [
        -1e6_f32, -1000.0, -100.0, -10.0, -1.0, -0.752, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0,
    ];
    for &x in &test_points {
        let gelu_x = gelu_eval(x, GeluApproximation::Tanh);
        let lower = ls * x + li;
        let upper = us * x + ui;
        let tol = 1e-5 * gelu_x.abs().max(1.0);
        assert!(
            lower <= gelu_x + tol,
            "gelu_tanh_sound_linear_relaxation lower unsound at x={x}: lower={lower} > GELU_tanh(x)={gelu_x}"
        );
        assert!(
            upper + tol >= gelu_x,
            "gelu_tanh_sound_linear_relaxation upper unsound at x={x}: upper={upper} < GELU_tanh(x)={gelu_x}"
        );
    }
}

/// Regression #1837: GELU CROWN backward with infinite input bounds
/// must not produce unsound linear relaxation via the identity fallback.
#[timeout(10000)]
#[test]
fn regression_1837_gelu_crown_backward_infinite_bounds() {
    for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let gelu = GELULayer::new(approx);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new_unchecked(
            arr1(&[f32::NEG_INFINITY]).into_dyn(),
            arr1(&[2.0]).into_dyn(),
        )
        .unwrap();

        let result = gelu.propagate_linear_with_bounds(&identity, &pre_activation);

        // If this errors, that's acceptable (refusing to propagate infinite bounds)
        if let Ok(result) = result {
            let ls = result.lower_a[[0, 0]];
            let li = result.lower_b[0];
            let us = result.upper_a[[0, 0]];
            let ui = result.upper_b[0];

            // NaN check first
            assert!(
                !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
                "GELU({approx:?}) CROWN produced NaN coefficients for [-inf, 2]: \
                 ls={ls}, li={li}, us={us}, ui={ui}"
            );

            // Verify soundness on sample points including GELU minimum
            let test_points = [
                -1e6_f32, -1000.0, -100.0, -10.0, -1.0, -0.752, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0,
            ];
            for &x in &test_points {
                let gelu_x = gelu_eval(x, approx);
                let lower = ls * x + li;
                let upper = us * x + ui;
                let tol = 1e-5 * gelu_x.abs().max(1.0);

                assert!(
                    lower <= gelu_x + tol,
                    "GELU({approx:?}) CROWN lower unsound at x={x} with [-inf, 2]: \
                     lower={lower} > GELU(x)={gelu_x}. Slopes: ls={ls}, li={li}. Bug #1837."
                );
                assert!(
                    upper + tol >= gelu_x,
                    "GELU({approx:?}) CROWN upper unsound at x={x} with [-inf, 2]: \
                     upper={upper} < GELU(x)={gelu_x}. Slopes: us={us}, ui={ui}. Bug #1837."
                );
            }
        }
    }
}
