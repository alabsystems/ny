// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use proptest::prelude::*;

/// Check that linear bounds contain Softsign(x) at all grid points in [l, u].
fn assert_softsign_relaxation_soundness(l: f32, u: f32, tol: f32) {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = softsign_linear_relaxation(l, u);
    let n = 101;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = (l + t * (u - l)).clamp(l, u);
        let y = softsign_scalar(x);
        let lb = ls * x + li;
        let ub = us * x + ui;
        assert!(
            lb <= y + tol,
            "Softsign lower bound violated at x={}: lb={} > y={} (l={}, u={})",
            x,
            lb,
            y,
            l,
            u
        );
        assert!(
            ub >= y - tol,
            "Softsign upper bound violated at x={}: ub={} < y={} (l={}, u={})",
            x,
            ub,
            y,
            l,
            u
        );
    }
}

// ========== Soundness grid tests for CROWN linear relaxation ==========

#[test]
fn softsign_relaxation_positive_soundness() {
    // Entirely concave (x > 0)
    assert_softsign_relaxation_soundness(0.1, 5.0, 1e-5);
    assert_softsign_relaxation_soundness(1.0, 10.0, 1e-5);
    assert_softsign_relaxation_soundness(0.0, 1.0, 1e-5);
}

#[test]
fn softsign_relaxation_negative_soundness() {
    // Entirely convex (x < 0)
    assert_softsign_relaxation_soundness(-5.0, -0.1, 1e-5);
    assert_softsign_relaxation_soundness(-10.0, -1.0, 1e-5);
    assert_softsign_relaxation_soundness(-1.0, 0.0, 1e-5);
}

#[test]
fn softsign_relaxation_crossing_soundness() {
    // Crosses inflection point at x = 0 (convex x<0, concave x>0)
    assert_softsign_relaxation_soundness(-2.0, 2.0, 1e-3);
    assert_softsign_relaxation_soundness(-1.0, 1.0, 1e-3);
    assert_softsign_relaxation_soundness(-5.0, 5.0, 1e-3);
    assert_softsign_relaxation_soundness(-0.5, 0.5, 1e-3);
}

#[test]
fn softsign_relaxation_wide_interval_soundness() {
    // Wide intervals test the S-shaped BoundSShaped crossing case
    assert_softsign_relaxation_soundness(-10.0, 10.0, 1e-2);
    assert_softsign_relaxation_soundness(-50.0, 50.0, 1e-2);
}

#[test]
fn softsign_relaxation_asymmetric_soundness() {
    assert_softsign_relaxation_soundness(-0.01, 10.0, 1e-3);
    assert_softsign_relaxation_soundness(-10.0, 0.01, 1e-3);
    assert_softsign_relaxation_soundness(-0.1, 5.0, 1e-3);
    assert_softsign_relaxation_soundness(-5.0, 0.1, 1e-3);
}

#[test]
fn softsign_relaxation_narrow_soundness() {
    assert_softsign_relaxation_soundness(-0.1, 0.1, 1e-5);
    assert_softsign_relaxation_soundness(0.9, 1.1, 1e-5);
    assert_softsign_relaxation_soundness(-1.1, -0.9, 1e-5);
}

#[test]
fn softsign_relaxation_point_interval_soundness() {
    for x in [-5.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0] {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = softsign_linear_relaxation(x, x);
        let y = softsign_scalar(x);
        let lb = ls * x + li;
        let ub = us * x + ui;
        assert!(
            lb <= y + 1e-5,
            "Point lower violated at x={}: lb={} > y={}",
            x,
            lb,
            y
        );
        assert!(
            ub >= y - 1e-5,
            "Point upper violated at x={}: ub={} < y={}",
            x,
            ub,
            y
        );
    }
}

fn f32_any_with_specials() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(f32::NEG_INFINITY),
        Just(f32::INFINITY),
        Just(0.0_f32),
        Just(f32::NAN),
        prop::num::f32::ANY
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]

    /// #1836 acceptance: exercise Softsign scalar eval on ANY f32 plus explicit IEEE corner values.
    #[test]
    fn proptest_softsign_eval_handles_special_values(x in f32_any_with_specials()) {
        let y = softsign_scalar(x);
        if x.is_nan() {
            prop_assert!(y.is_nan(), "softsign(NaN) should be NaN, got {y}");
        } else if x == f32::NEG_INFINITY {
            prop_assert_eq!(y, -1.0, "softsign(-inf) should be -1.0, got {}", y);
        } else if x == f32::INFINITY {
            prop_assert_eq!(y, 1.0, "softsign(+inf) should be +1.0, got {}", y);
        } else {
            prop_assert!(!y.is_nan(), "softsign({x}) should not be NaN, got {y}");
            prop_assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&y), "softsign({x}) should be in [-1, 1], got {y}");
        }
    }

    /// #1836 acceptance: finite-width IBP intervals must not produce NaN bounds.
    /// #3316: bounds must also respect the mathematical range [-1, 1].
    #[test]
    fn proptest_softsign_ibp_no_nan_for_finite_intervals(a in prop::num::f32::ANY, b in prop::num::f32::ANY) {
        prop_assume!(a.is_finite() && b.is_finite());
        prop_assume!(a.abs() <= 1.0e6 && b.abs() <= 1.0e6);
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let layer = SoftsignLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "Softsign IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "Softsign IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "Softsign IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
        prop_assert!(lower >= -1.0, "Softsign IBP lower {lower} < -1.0 for [{l}, {u}] (#3316)");
        prop_assert!(upper <= 1.0, "Softsign IBP upper {upper} > 1.0 for [{l}, {u}] (#3316)");
    }
}

// ── CROWN backward tests ───────────────────────────────────────────

#[test]
fn test_crown_backward_crossing_soundness() {
    let layer = SoftsignLayer::new();
    let l = -3.0_f32;
    let u = 3.0_f32;
    let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = softsign_scalar(x);
        assert!(
            la * x + lb <= y + 1e-3,
            "Softsign CROWN lb violated at x={x}: {} > {y}",
            la * x + lb
        );
        assert!(
            ua * x + ub >= y - 1e-3,
            "Softsign CROWN ub violated at x={x}: {} < {y}",
            ua * x + ub
        );
    }
}

#[test]
fn test_crown_backward_positive_concave() {
    // Softsign is concave for x > 0
    let layer = SoftsignLayer::new();
    let pre = BoundedTensor::new(arr1(&[0.5_f32]).into_dyn(), arr1(&[5.0_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = 0.5 + 4.5 * (k as f32 / 50.0);
        let y = softsign_scalar(x);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-3,
            "positive concave lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-3,
            "positive concave ub violated at x={x}"
        );
    }
}

#[test]
fn test_crown_backward_negative_convex() {
    // Softsign is convex for x < 0
    let layer = SoftsignLayer::new();
    let pre =
        BoundedTensor::new(arr1(&[-5.0_f32]).into_dyn(), arr1(&[-0.5_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = -5.0 + 4.5 * (k as f32 / 50.0);
        let y = softsign_scalar(x);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-3,
            "negative convex lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-3,
            "negative convex ub violated at x={x}"
        );
    }
}

#[test]
fn test_crown_backward_multi_neuron() {
    let layer = SoftsignLayer::new();
    let pre = BoundedTensor::new(
        arr1(&[-3.0_f32, 0.0]).into_dyn(),
        arr1(&[0.0_f32, 3.0]).into_dyn(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for neuron in 0..2 {
        let la = result.lower_a[[neuron, neuron]];
        let lb = result.lower_b[neuron];
        let ua = result.upper_a[[neuron, neuron]];
        let ub = result.upper_b[neuron];
        let lo = pre.lower()[neuron];
        let hi = pre.upper()[neuron];

        for k in 0..=20 {
            let x = lo + (hi - lo) * (k as f32 / 20.0);
            let y = softsign_scalar(x);
            assert!(
                la * x + lb <= y + 1e-3,
                "neuron {neuron} lb violated at x={x}"
            );
            assert!(
                ua * x + ub >= y - 1e-3,
                "neuron {neuron} ub violated at x={x}"
            );
        }
    }
}

#[test]
fn test_propagate_linear_requires_preact() {
    let layer = SoftsignLayer::new();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "Softsign CROWN without pre-activation bounds should fail"
    );
    assert!(layer.requires_pre_activation_bounds());
}

// ── CROWN backward with non-identity incoming bounds ──────────────

/// Tests that CROWN backward correctly handles negative incoming coefficients.
/// With A = [[-1]], the sign-swap logic in crown_elementwise_backward must use
/// the upper relaxation for the lower bound and vice versa.
#[test]
fn test_crown_backward_negative_coeff_soundness() -> Result<()> {
    use ndarray::Array1;

    let layer = SoftsignLayer::new();
    // S-shaped crossing interval
    let pre = BoundedTensor::new(arr1(&[-3.0_f32]).into_dyn(), arr1(&[3.0_f32]).into_dyn())?;

    let neg_bounds = LinearBounds::new(
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&neg_bounds, &pre)?;

    let l = -3.0_f32;
    let u = 3.0_f32;
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = -(softsign_scalar(x));
        let bound_lo = la.max(0.0) * l + la.min(0.0) * u + lb;
        let bound_hi = ua.max(0.0) * u + ua.min(0.0) * l + ub;
        assert!(
            bound_lo <= y + 1e-3,
            "negative coeff: lower {} > -softsign({}) = {} at x={}",
            bound_lo,
            x,
            y,
            x
        );
        assert!(
            bound_hi >= y - 1e-3,
            "negative coeff: upper {} < -softsign({}) = {} at x={}",
            bound_hi,
            x,
            y,
            x
        );
    }
    Ok(())
}

/// Tests CROWN backward with a 2-neuron, 2-output non-identity coefficient matrix
/// A = [[1, -1], [0.5, 0.5]], verifying that the composed output A @ Softsign(x)
/// is soundly bounded. Exercises both positive and negative coefficient branches
/// simultaneously — neuron 0 in crossing, neuron 1 in concave (positive) region.
#[test]
fn test_crown_backward_non_identity_bounds() -> Result<()> {
    use ndarray::Array1;

    let layer = SoftsignLayer::new();
    // Neuron 0: crossing [-3, 3], Neuron 1: concave positive [0.5, 5]
    let pre = BoundedTensor::new(
        arr1(&[-3.0_f32, 0.5]).into_dyn(),
        arr1(&[3.0_f32, 5.0]).into_dyn(),
    )?;

    let a = ndarray::Array2::from_shape_vec((2, 2), vec![1.0_f32, -1.0, 0.5, 0.5])
        .expect("invariant: static 2x2 shape matches 4 elements");
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(2), a, Array1::zeros(2)).unwrap();
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    let l0 = -3.0_f32;
    let u0 = 3.0_f32;
    let l1 = 0.5_f32;
    let u1 = 5.0_f32;
    let lowers = [l0, l1];
    let uppers = [u0, u1];

    for k0 in 0..=10 {
        for k1 in 0..=10 {
            let x0 = l0 + (u0 - l0) * (k0 as f32 / 10.0);
            let x1 = l1 + (u1 - l1) * (k1 as f32 / 10.0);
            let r0 = softsign_scalar(x0);
            let r1 = softsign_scalar(x1);

            let y0 = r0 - r1;
            let y1 = 0.5 * r0 + 0.5 * r1;

            for i in 0..2 {
                let y = if i == 0 { y0 } else { y1 };
                let mut lo = result.lower_b[i];
                let mut hi = result.upper_b[i];
                for j in 0..2 {
                    let la = result.lower_a[[i, j]];
                    let ua = result.upper_a[[i, j]];
                    lo += la.max(0.0) * lowers[j] + la.min(0.0) * uppers[j];
                    hi += ua.max(0.0) * uppers[j] + ua.min(0.0) * lowers[j];
                }
                assert!(
                    lo <= y + 1e-2,
                    "output {} lower {} > true {} at ({}, {})",
                    i,
                    lo,
                    y,
                    x0,
                    x1
                );
                assert!(
                    hi >= y - 1e-2,
                    "output {} upper {} < true {} at ({}, {})",
                    i,
                    hi,
                    y,
                    x0,
                    x1
                );
            }
        }
    }
    Ok(())
}

// ── f64 chord precision proptest (#2624) ─────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #2624, #3146: Verify softsign_chord_f64 matches f64 reference with directed rounding.
    /// Slope must match exactly; directed rounding bounds must bracket the reference intercept.
    #[test]
    fn proptest_softsign_chord_f64_precision(l in -10.0f32..10.0, width_exp in -8.0f64..-4.0) {
        let delta = 10.0_f64.powf(width_exp) as f32;
        let u = l + delta;
        prop_assume!(u > l);
        let (slope, lower_intercept, upper_intercept) = softsign_chord_f64(l, u);
        // Independent f64 reference: softsign(x) = x / (1 + |x|)
        let (l64, u64) = (l as f64, u as f64);
        let (fl64, fu64) = (l64 / (1.0 + l64.abs()), u64 / (1.0 + u64.abs()));
        let s64 = (fu64 - fl64) / (u64 - l64);
        let ref_s = s64 as f32;
        let ref_i = (fl64 - s64 * l64) as f32;
        prop_assert_eq!(slope, ref_s, "chord slope mismatch for [{}, {}]", l, u);
        // Directed rounding: lower_intercept <= ref <= upper_intercept
        prop_assert!(
            lower_intercept <= ref_i,
            "Softsign chord lower_intercept {lower_intercept} > ref {ref_i} for [{l}, {u}]"
        );
        prop_assert!(
            upper_intercept >= ref_i,
            "Softsign chord upper_intercept {upper_intercept} < ref {ref_i} for [{l}, {u}]"
        );
    }
}

/// Regression test for #1836: softsign_scalar(-inf) must return -1, not NaN.
/// Softsign(x) = x / (1 + |x|); at x = -inf: (-inf)/inf = NaN
/// without the guard. Correct limit: Softsign(-inf) = -1.
#[test]
fn test_softsign_neg_infinity_returns_neg_one() {
    let result = softsign_scalar(f32::NEG_INFINITY);
    assert_eq!(result, -1.0, "softsign(-inf) should be -1.0, got {result}");
}

/// Regression test for #1836: softsign_scalar(+inf) must return +1.
#[test]
fn test_softsign_pos_infinity_returns_pos_one() {
    let result = softsign_scalar(f32::INFINITY);
    assert_eq!(result, 1.0, "softsign(+inf) should be 1.0, got {result}");
}

/// Regression test for #1836: softsign_scalar(NaN) must return NaN.
#[test]
fn test_softsign_nan_returns_nan() {
    let result = softsign_scalar(f32::NAN);
    assert!(result.is_nan(), "softsign(NaN) should be NaN, got {result}");
}

/// Regression test for #3316: softsign IBP with extreme finite inputs must respect
/// the mathematical range (-1, 1). Without the range clamp, next_down_f32(-1.0)
/// pushes the lower bound to -1.0000001 and next_up_f32(1.0) pushes the upper
/// bound to 1.0000001 for inputs where softsign_f64(x) rounds to exactly ±1.0.
#[test]
fn test_softsign_ibp_extreme_range_clamp_3316() {
    let layer = SoftsignLayer::new();
    let input =
        BoundedTensor::new(arr1(&[-1e38_f32]).into_dyn(), arr1(&[1e38_f32]).into_dyn()).unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert!(
        output.lower()[[0]] >= -1.0,
        "Softsign IBP lower must be >= -1.0 for extreme inputs (#3316), got {}",
        output.lower()[[0]]
    );
    assert!(
        output.upper()[[0]] <= 1.0,
        "Softsign IBP upper must be <= 1.0 for extreme inputs (#3316), got {}",
        output.upper()[[0]]
    );
}

/// Regression test for #1836: IBP with infinite input bounds must not produce NaN.
/// Softsign maps ±inf to ±1 (finite), so the IBP output should succeed.
#[test]
fn test_softsign_ibp_infinite_bounds_no_nan() {
    use ndarray::{ArrayD, IxDyn};
    let layer = SoftsignLayer::new();
    // Use new_unchecked since BoundedTensor::new rejects infinite inputs
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -10.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 10.0]).unwrap(),
    )
    .unwrap();
    // Softsign(-inf) = -1, softsign(+inf) = +1, both finite → BoundedTensor::new should succeed
    let result = layer.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!(!v.is_nan(), "Softsign IBP lower must not be NaN, got {v}");
    }
    for &v in result.upper().iter() {
        assert!(!v.is_nan(), "Softsign IBP upper must not be NaN, got {v}");
    }
    // After range clamp (#3316), bounds must stay within [-1, 1] even for infinite inputs.
    assert!(
        result.lower()[0] >= -1.0,
        "softsign lower for [-inf, inf] should be >= -1.0 after range clamp (#3316), got {}",
        result.lower()[0]
    );
    assert!(
        result.upper()[0] <= 1.0,
        "softsign upper for [-inf, inf] should be <= 1.0 after range clamp (#3316), got {}",
        result.upper()[0]
    );
}

// ── CROWN relaxation soundness proptest (#3285) ─────────────────────────

/// Reference Softsign in f64, independent of the crate f32 implementation.
fn softsign_f64_reference(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #3285: Verify softsign_linear_relaxation produces strictly sound bounds.
    /// For random intervals, the lower bound must satisfy
    ///   lower_slope * x + lower_intercept <= Softsign(x)  for all x in [l, u]
    /// and the upper bound must satisfy
    ///   upper_slope * x + upper_intercept >= Softsign(x)  for all x in [l, u]
    /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
    ///
    /// Ref: SiLU proptest_silu_relaxation_strict_soundness (silu/tests.rs:553).
    #[test]
    fn proptest_softsign_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = softsign_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        // Skip NaN fallback (infinite bounds).
        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        // Dense grid: 200 points, evaluated in f64 for mathematical precision.
        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = softsign_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "Softsign lower bound UNSOUND at x={}: {} > Softsign({})={}, \
                 interval=[{}, {}], gap={}", x, lower_val, x, fx, l, u, lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "Softsign upper bound UNSOUND at x={}: {} < Softsign({})={}, \
                 interval=[{}, {}], gap={}", x, upper_val, x, fx, l, u, fx - upper_val
            );
        }
    }
}
