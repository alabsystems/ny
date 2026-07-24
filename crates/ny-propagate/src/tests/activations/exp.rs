// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::activations::exp::exp_linear_relaxation;
use crate::*;
use ndarray::ArrayD;
use ndarray::IxDyn;
use proptest::prelude::*;

// ==================== Exp tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_basic() {
    // Test exp on a simple interval: exp([0, 1]) = [1, e]
    let lower = ArrayD::from_elem(IxDyn(&[3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let output = exp.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - 1.0).abs() < 1e-6,
            "exp(0) should be 1, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - std::f32::consts::E).abs() < 1e-5,
            "exp(1) should be e, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_negative() {
    // exp([-2, -1]) = [exp(-2), exp(-1)]
    let lower = ArrayD::from_elem(IxDyn(&[2]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let output = exp.propagate_ibp(&input).unwrap();

    let exp_neg2 = (-2.0_f32).exp();
    let exp_neg1 = (-1.0_f32).exp();
    for i in 0..2 {
        assert!((output.lower()[[i]] - exp_neg2).abs() < 1e-6);
        assert!((output.upper()[[i]] - exp_neg1).abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_linear_requires_preactivation_bounds() {
    // Without pre-activation bounds, Exp must return an error (nonlinear layer).
    let bounds = LinearBounds::identity(4);
    let exp = ExpLayer::new();
    let result = exp.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "Exp::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_crown_soundness() {
    // Test that CROWN backward bounds contain true exp(x) for all sample points.
    //
    // Pre-activation bounds define the interval for each neuron.
    // We propagate identity LinearBounds backward through exp, then verify
    // that for any x in the interval: lower_a * x + lower_b <= exp(x) <= upper_a * x + upper_b
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, -2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 0.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let exp = ExpLayer::new();

    let result = exp
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points within each neuron's range
    let test_points: [Vec<f32>; 5] = [
        vec![-1.0, 0.0, -2.0], // lower bounds
        vec![1.0, 2.0, 0.0],   // upper bounds
        vec![0.0, 1.0, -1.0],  // midpoints
        vec![-0.5, 0.5, -1.5], // quarter points
        vec![0.5, 1.5, -0.5],  // three-quarter points
    ];

    let dim = 3;
    let tol = 1e-4;

    for point in &test_points {
        let exp_output: Vec<f32> = point.iter().map(|x| x.exp()).collect();

        for (j, &exp_val) in exp_output.iter().enumerate() {
            // Lower bound: lower_a * x + lower_b should be <= exp(point)
            let lb_val: f32 = (0..dim)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            // Upper bound: upper_a * x + upper_b should be >= exp(point)
            let ub_val: f32 = (0..dim)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            assert!(
                lb_val <= exp_val + tol,
                "Exp lower bound violated at point {:?}: lb {} > exp({}) = {}",
                point,
                lb_val,
                point[j],
                exp_val
            );
            assert!(
                ub_val >= exp_val - tol,
                "Exp upper bound violated at point {:?}: ub {} < exp({}) = {}",
                point,
                ub_val,
                point[j],
                exp_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_crown_wide_interval() {
    // Test CROWN soundness on a wider interval where exp grows rapidly.
    // This tests the m-clamping behavior (m <= l + 0.99).
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-5.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![5.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let exp = ExpLayer::new();

    let result = exp
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let tol = 1e-3;
    for i in 0..=20 {
        let x = -5.0 + 10.0 * (i as f32 / 20.0);
        let exp_val = x.exp();
        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];

        assert!(
            lb <= exp_val + tol,
            "Exp lower bound violated at x={x}: lb={lb} > exp(x)={exp_val}"
        );
        assert!(
            ub >= exp_val - tol,
            "Exp upper bound violated at x={x}: ub={ub} < exp(x)={exp_val}"
        );
    }
}

// ==================== Overflow guard tests (#1654) ====================

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_overflow_guard_1654() {
    // exp(89) overflows f32. Guard should reject upper > 88.
    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 100.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let err = exp
        .propagate_ibp(&input)
        .expect_err("exp with upper=100 should return overflow error");
    match err {
        NyError::NumericalInstability(msg) => {
            assert!(
                msg.contains("overflow threshold"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_at_threshold_boundary_1654() {
    // upper=88.0 is exactly at the threshold — should succeed.
    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 88.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let output = exp.propagate_ibp(&input).unwrap();
    assert!(output.upper()[[0]].is_finite(), "exp(88) should be finite");
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_just_above_threshold_1654() {
    // upper=88.1 exceeds threshold — should error.
    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 88.1f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let exp = ExpLayer::new();
    assert!(
        exp.propagate_ibp(&input).is_err(),
        "exp with upper=88.1 should be rejected"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_nan_input_1654() {
    // NaN in bounds should be rejected.
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 0.0]).unwrap();
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let err = exp
        .propagate_ibp(&input)
        .expect_err("NaN input should be rejected");
    match err {
        NyError::NumericalInstability(msg) => {
            assert!(msg.contains("non-finite"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_inf_input_1654() {
    // Infinity in bounds should be rejected.
    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), f32::INFINITY);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let err = exp
        .propagate_ibp(&input)
        .expect_err("infinite input should be rejected");
    match err {
        NyError::NumericalInstability(msg) => {
            assert!(msg.contains("non-finite"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_crown_overflow_guard_1654() {
    // CROWN backward with overflow-risk pre-activation bounds should error.
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 100.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let exp = ExpLayer::new();

    let err = exp
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .expect_err("CROWN with upper=100 should return overflow error");
    match err {
        NyError::NumericalInstability(msg) => {
            assert!(
                msg.contains("overflow threshold"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_crown_nan_preactivation_1654() {
    // CROWN backward with NaN pre-activation should error.
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap();
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let pre_activation = BoundedTensor::new_unchecked(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let exp = ExpLayer::new();

    let err = exp
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .expect_err("CROWN with NaN pre-activation should error");
    match err {
        NyError::NumericalInstability(msg) => {
            assert!(msg.contains("non-finite"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_exp_ibp_negative_large_safe_1654() {
    // Very negative inputs are safe (exp approaches 0).
    let lower = ArrayD::from_elem(IxDyn(&[2]), -1000.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -500.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let exp = ExpLayer::new();
    let output = exp.propagate_ibp(&input).unwrap();
    // exp of very negative numbers is tiny but non-negative
    for i in 0..2 {
        assert!(output.lower()[[i]] >= 0.0);
        assert!(output.upper()[[i]] >= 0.0);
    }
}

// ── Strict zero-tolerance CROWN relaxation proptest (#3292) ──────────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.

/// Independent f64 exp reference for strict proptest. (#3292)
fn exp_f64_reference(x: f64) -> f64 {
    x.exp()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for exp CROWN relaxation.
    /// Uses f64 reference (exp_f64_reference) with zero tolerance on 200-point grid.
    /// exp is convex: lower bound = tangent, upper bound = chord.
    /// Ref: alpha-beta-CROWN auto_LiRPA exp relaxation, #3292.
    #[test]
    fn proptest_exp_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = exp_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = exp_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "exp lower bound UNSOUND at x={x}: {lower_val} > exp({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "exp upper bound UNSOUND at x={x}: {upper_val} < exp({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }
}
