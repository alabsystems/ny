// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use proptest::prelude::*;

/// Check that linear bounds contain HardSwish(x) at all grid points in [l, u].
fn assert_hardswish_relaxation_soundness(l: f32, u: f32, tol: f32) {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = hardswish_linear_relaxation(l, u);
    let n = 101;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = (l + t * (u - l)).clamp(l, u);
        let y = hardswish_eval(x);
        let lb = ls * x + li;
        let ub = us * x + ui;
        assert!(
            lb <= y + tol,
            "HardSwish lower bound violated at x={}: lb={} > y={} (l={}, u={})",
            x,
            lb,
            y,
            l,
            u
        );
        assert!(
            ub >= y - tol,
            "HardSwish upper bound violated at x={}: ub={} < y={} (l={}, u={})",
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
fn hardswish_relaxation_constant_region_soundness() {
    // Entirely in y=0 region (x <= -3)
    assert_hardswish_relaxation_soundness(-10.0, -4.0, 1e-5);
    assert_hardswish_relaxation_soundness(-5.0, -3.0, 1e-5);
}

#[test]
fn hardswish_relaxation_identity_region_soundness() {
    // Entirely in y=x region (x >= 3)
    assert_hardswish_relaxation_soundness(3.0, 10.0, 1e-5);
    assert_hardswish_relaxation_soundness(4.0, 100.0, 1e-2);
}

#[test]
fn hardswish_relaxation_quadratic_region_soundness() {
    // Entirely in quadratic region (-3 < x < 3)
    assert_hardswish_relaxation_soundness(-2.0, 2.0, 1e-5);
    assert_hardswish_relaxation_soundness(-1.0, 1.0, 1e-5);
    assert_hardswish_relaxation_soundness(0.0, 2.5, 1e-5);
    assert_hardswish_relaxation_soundness(-2.9, -0.1, 1e-5);
}

#[test]
fn hardswish_relaxation_crossing_all_regions_soundness() {
    // Spans all three regions
    assert_hardswish_relaxation_soundness(-5.0, 5.0, 1e-3);
    assert_hardswish_relaxation_soundness(-10.0, 10.0, 1e-2);
}

#[test]
fn hardswish_relaxation_crossing_boundaries_soundness() {
    // Crosses constant/quadratic boundary at -3
    assert_hardswish_relaxation_soundness(-4.0, -2.0, 1e-5);
    assert_hardswish_relaxation_soundness(-6.0, 0.0, 1e-3);
    // Crosses quadratic/identity boundary at 3
    assert_hardswish_relaxation_soundness(2.0, 4.0, 1e-5);
    assert_hardswish_relaxation_soundness(0.0, 6.0, 1e-3);
}

#[test]
fn hardswish_relaxation_near_critical_point_soundness() {
    // Quadratic region minimum is at x = -1.5 where y = -0.375
    assert_hardswish_relaxation_soundness(-2.0, -1.0, 1e-5);
    assert_hardswish_relaxation_soundness(-1.6, -1.4, 1e-5);
}

#[test]
fn hardswish_relaxation_asymmetric_soundness() {
    assert_hardswish_relaxation_soundness(-0.01, 10.0, 1e-3);
    assert_hardswish_relaxation_soundness(-10.0, 0.01, 1e-3);
}

#[test]
fn hardswish_relaxation_point_interval_soundness() {
    for x in [-5.0f32, -3.0, -1.5, 0.0, 1.0, 3.0, 5.0] {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = hardswish_linear_relaxation(x, x);
        let y = hardswish_eval(x);
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

    /// #1836 acceptance: exercise HardSwish eval on ANY f32 plus explicit IEEE corner values.
    #[test]
    fn proptest_hardswish_eval_handles_special_values(x in f32_any_with_specials()) {
        let layer = HardSwishLayer::new();
        let y = layer.eval(x);
        if x.is_nan() {
            prop_assert!(y.is_nan(), "HardSwish(NaN) should be NaN, got {y}");
        } else if x == f32::NEG_INFINITY {
            prop_assert_eq!(y, 0.0, "HardSwish(-inf) should be 0.0, got {}", y);
        } else if x == f32::INFINITY {
            prop_assert_eq!(y, f32::INFINITY, "HardSwish(+inf) should be +inf, got {}", y);
        } else {
            prop_assert!(!y.is_nan(), "HardSwish({x}) should not be NaN, got {y}");
        }
    }

    /// #1836 acceptance: finite-width IBP intervals must not produce NaN bounds.
    #[test]
    fn proptest_hardswish_ibp_no_nan_for_finite_intervals(a in prop::num::f32::ANY, b in prop::num::f32::ANY) {
        prop_assume!(a.is_finite() && b.is_finite());
        prop_assume!(a.abs() <= 1.0e6 && b.abs() <= 1.0e6);
        let (l, u) = (a.min(b), a.max(b));
        let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let layer = HardSwishLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        prop_assert!(!lower.is_nan(), "HardSwish IBP lower is NaN for [{l}, {u}]");
        prop_assert!(!upper.is_nan(), "HardSwish IBP upper is NaN for [{l}, {u}]");
        prop_assert!(lower <= upper, "HardSwish IBP bounds inverted for [{l}, {u}]: {lower} > {upper}");
    }
}

#[test]
fn test_hardswish_ibp_non_finite_input_falls_back_to_infinite_bounds() -> Result<()> {
    let input =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let layer = HardSwishLayer::new();
    let output = layer.propagate_ibp(&input)?;

    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];
    assert!(lower.is_infinite() && lower.is_sign_negative());
    assert!(upper.is_infinite() && upper.is_sign_positive());
    Ok(())
}

// ── CROWN backward tests ───────────────────────────────────────────

#[test]
fn test_crown_backward_crossing_soundness() {
    let layer = HardSwishLayer::new();
    let l = -4.0_f32;
    let u = 4.0_f32;
    let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = hardswish_eval(x);
        assert!(
            la * x + lb <= y + 1e-3,
            "HardSwish CROWN lb violated at x={x}: {} > {y}",
            la * x + lb
        );
        assert!(
            ua * x + ub >= y - 1e-3,
            "HardSwish CROWN ub violated at x={x}: {} < {y}",
            ua * x + ub
        );
    }
}

#[test]
fn test_crown_backward_constant_region() {
    // Entirely in y=0 region (x <= -3)
    let layer = HardSwishLayer::new();
    let pre =
        BoundedTensor::new(arr1(&[-6.0_f32]).into_dyn(), arr1(&[-3.5_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    // In zero region, slopes and intercepts should be near zero
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=20 {
        let x = -6.0 + 2.5 * (k as f32 / 20.0);
        let y = hardswish_eval(x);
        assert!(
            la * x + lb <= y + 1e-5,
            "constant region lb violated at x={x}"
        );
        assert!(
            ua * x + ub >= y - 1e-5,
            "constant region ub violated at x={x}"
        );
    }
}

#[test]
fn test_crown_backward_identity_region() {
    // Entirely in y=x region (x >= 3)
    let layer = HardSwishLayer::new();
    let pre = BoundedTensor::new(arr1(&[3.5_f32]).into_dyn(), arr1(&[8.0_f32]).into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    // In identity region, slope should be near 1.0
    let la = result.lower_a[[0, 0]];
    let ua = result.upper_a[[0, 0]];
    // In identity region (x >= 3), hard_swish(x) = x exactly, so slope = 1.
    assert!(
        (la - 1.0).abs() < 1e-5,
        "identity region lower slope should be ~1, got {la}"
    );
    assert!(
        (ua - 1.0).abs() < 1e-5,
        "identity region upper slope should be ~1, got {ua}"
    );
}

#[test]
fn test_crown_backward_multi_neuron() {
    let layer = HardSwishLayer::new();
    let pre = BoundedTensor::new(
        arr1(&[-4.0_f32, 0.0]).into_dyn(),
        arr1(&[0.0_f32, 5.0]).into_dyn(),
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
            let y = hardswish_eval(x);
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
    let layer = HardSwishLayer::new();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "HardSwish CROWN without pre-activation bounds should fail"
    );
    assert!(layer.requires_pre_activation_bounds());
}

/// Regression test for #1836: HardSwish(-inf) must return 0, not NaN.
/// HardSwish(x) = x * clamp((x+3)/6, 0, 1); at x = -inf: (-inf)*0 = NaN
/// without the guard. Correct: HardSwish(x) = 0 for x <= -3.
#[test]
fn test_hardswish_eval_neg_infinity_returns_zero() {
    let layer = HardSwishLayer::new();
    let result = layer.eval(f32::NEG_INFINITY);
    assert_eq!(result, 0.0, "HardSwish(-inf) should be 0.0, got {result}");
}

/// Regression test for #1836: HardSwish(+inf) must return +inf.
#[test]
fn test_hardswish_eval_pos_infinity_returns_pos_infinity() {
    let layer = HardSwishLayer::new();
    let result = layer.eval(f32::INFINITY);
    assert_eq!(
        result,
        f32::INFINITY,
        "HardSwish(+inf) should be +inf, got {result}"
    );
}

/// Regression test for #1836: HardSwish(NaN) must return NaN.
#[test]
fn test_hardswish_eval_nan_returns_nan() {
    let layer = HardSwishLayer::new();
    let result = layer.eval(f32::NAN);
    assert!(
        result.is_nan(),
        "HardSwish(NaN) should be NaN, got {result}"
    );
}

// ── CROWN backward with non-identity incoming bounds ──────────────

/// Tests that CROWN backward correctly handles negative incoming coefficients.
/// With A = [[-1]], the sign-swap logic in crown_elementwise_backward must use
/// the upper relaxation for the lower bound and vice versa.
#[test]
fn test_crown_backward_negative_coeff_soundness() -> Result<()> {
    use ndarray::Array1;

    let layer = HardSwishLayer::new();
    // Crossing interval spanning all three HardSwish regions
    let pre = BoundedTensor::new(arr1(&[-4.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn())?;

    let neg_bounds = LinearBounds::new(
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&neg_bounds, &pre)?;

    let l = -4.0_f32;
    let u = 4.0_f32;
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = -(hardswish_eval(x));
        let bound_lo = la.max(0.0) * l + la.min(0.0) * u + lb;
        let bound_hi = ua.max(0.0) * u + ua.min(0.0) * l + ub;
        assert!(
            bound_lo <= y + 1e-3,
            "negative coeff: lower {} > -hardswish({}) = {} at x={}",
            bound_lo,
            x,
            y,
            x
        );
        assert!(
            bound_hi >= y - 1e-3,
            "negative coeff: upper {} < -hardswish({}) = {} at x={}",
            bound_hi,
            x,
            y,
            x
        );
    }
    Ok(())
}

/// Tests CROWN backward with a 2-neuron, 2-output non-identity coefficient matrix
/// A = [[1, -1], [0.5, 0.5]], verifying that the composed output A @ HardSwish(x)
/// is soundly bounded. Exercises both positive and negative coefficient branches
/// simultaneously on crossing intervals.
#[test]
fn test_crown_backward_non_identity_bounds() -> Result<()> {
    use ndarray::Array1;

    let layer = HardSwishLayer::new();
    // Neuron 0: crosses all regions [-4, 4], Neuron 1: quadratic region [-2, 2]
    let pre = BoundedTensor::new(
        arr1(&[-4.0_f32, -2.0]).into_dyn(),
        arr1(&[4.0_f32, 2.0]).into_dyn(),
    )?;

    let a = ndarray::Array2::from_shape_vec((2, 2), vec![1.0_f32, -1.0, 0.5, 0.5])
        .expect("invariant: static 2x2 shape matches 4 elements");
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(2), a, Array1::zeros(2)).unwrap();
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    let l0 = -4.0_f32;
    let u0 = 4.0_f32;
    let l1 = -2.0_f32;
    let u1 = 2.0_f32;
    let lowers = [l0, l1];
    let uppers = [u0, u1];

    for k0 in 0..=10 {
        for k1 in 0..=10 {
            let x0 = l0 + (u0 - l0) * (k0 as f32 / 10.0);
            let x1 = l1 + (u1 - l1) * (k1 as f32 / 10.0);
            let r0 = hardswish_eval(x0);
            let r1 = hardswish_eval(x1);

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

/// Regression test for #1836: IBP with infinite input bounds must not produce NaN
/// in the computed values. The IBP output constructor (BoundedTensor::new) rejects
/// infinite bounds, so we verify the error is NumericalInstability (not NaN corruption).
#[test]
fn test_hardswish_ibp_infinite_bounds_no_nan() {
    use ndarray::{ArrayD, IxDyn};
    let layer = HardSwishLayer::new();
    // Use new_unchecked since BoundedTensor::new rejects infinite inputs
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -5.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 5.0]).unwrap(),
    )
    .unwrap();
    // IBP output may error because hardswish(+inf) = +inf gets rejected by BoundedTensor::new.
    // The key property: no NaN corruption. Either it succeeds with valid bounds or errors cleanly.
    match layer.propagate_ibp(&input) {
        Ok(result) => {
            for &v in result.lower().iter() {
                assert!(!v.is_nan(), "HardSwish IBP lower must not be NaN, got {v}");
            }
            for &v in result.upper().iter() {
                assert!(!v.is_nan(), "HardSwish IBP upper must not be NaN, got {v}");
            }
        }
        Err(e) => {
            // NumericalInstability from BoundedTensor::new rejecting +inf is acceptable
            let msg = format!("{e}");
            assert!(
                msg.contains("NaN or Inf"),
                "Expected NumericalInstability error for inf bounds, got: {msg}"
            );
        }
    }
}

// ── f64 chord precision proptest (#2846) ─────────────────────────────

/// Reference HardSwish in f64, independent of the crate implementation.
fn hardswish_f64_reference(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

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

    /// #2846, #3146: Verify hardswish_chord produces f64-quality results with directed rounding.
    /// Analogous to proptest_silu_chord_f64_precision in silu/tests.rs.
    /// For intervals with width in [1e-8, 1e-4], the chord slope must match
    /// a pure f64 reference within 1 ULP, and directed rounding bounds must
    /// bracket the reference intercept: lower_intercept <= ref <= upper_intercept.
    #[test]
    fn proptest_hardswish_chord_f64_precision(l in -10.0f32..10.0, width_exp in -8.0f64..-4.0) {
        let delta = 10.0_f64.powf(width_exp) as f32;
        let u = l + delta;
        prop_assume!(u > l); // skip if delta vanishes in f32

        let (slope, lower_intercept, upper_intercept) = hardswish_chord(l, u);

        // Independent f64 reference
        let l64 = l as f64;
        let u64 = u as f64;
        let fl64 = hardswish_f64_reference(l64);
        let fu64 = hardswish_f64_reference(u64);
        let ref_slope64 = (fu64 - fl64) / (u64 - l64);
        let ref_intercept64 = fl64 - ref_slope64 * l64;
        let ref_slope = ref_slope64 as f32;
        let ref_intercept = ref_intercept64 as f32;

        let slope_ulps = ulp_distance(slope, ref_slope);

        prop_assert!(
            slope_ulps <= 1,
            "HardSwish chord slope not within 1 ULP: got {slope} vs ref {ref_slope} \
             ({slope_ulps} ULPs apart) for [{l}, {u}]"
        );
        // Directed rounding: lower_intercept <= ref <= upper_intercept
        prop_assert!(
            lower_intercept <= ref_intercept,
            "HardSwish chord lower_intercept {lower_intercept} > ref {ref_intercept} for [{l}, {u}]"
        );
        prop_assert!(
            upper_intercept >= ref_intercept,
            "HardSwish chord upper_intercept {upper_intercept} < ref {ref_intercept} for [{l}, {u}]"
        );
    }
}

// ── CROWN relaxation soundness proptest (#3285) ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #3285: Verify hardswish_linear_relaxation produces strictly sound bounds.
    /// For random intervals, the lower bound must satisfy
    ///   lower_slope * x + lower_intercept <= HardSwish(x)  for all x in [l, u]
    /// and the upper bound must satisfy
    ///   upper_slope * x + upper_intercept >= HardSwish(x)  for all x in [l, u]
    /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
    ///
    /// Ref: SiLU proptest_silu_relaxation_strict_soundness (silu/tests.rs:553).
    #[test]
    fn proptest_hardswish_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = hardswish_linear_relaxation(l, u);
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
            let fx = hardswish_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "HardSwish lower bound UNSOUND at x={}: {} > HardSwish({})={}, \
                 interval=[{}, {}], gap={}", x, lower_val, x, fx, l, u, lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "HardSwish upper bound UNSOUND at x={}: {} < HardSwish({})={}, \
                 interval=[{}, {}], gap={}", x, upper_val, x, fx, l, u, fx - upper_val
            );
        }
    }
}
