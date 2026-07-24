// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::activations::log::log_linear_relaxation;
use crate::*;
use ndarray::{ArrayD, IxDyn};
use proptest::prelude::*;

// ==================== Log tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_log_ibp_basic() {
    // Test log on positive bounds
    let lower = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), std::f32::consts::E);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let log = LogLayer;
    let output = log.propagate_ibp(&input).unwrap();

    // ln([1, e]) = [0, 1]
    for i in 0..4 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "ln(1) should be 0, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 1.0).abs() < 1e-6,
            "ln(e) should be 1, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_log_ibp_rejects_non_positive_lower() {
    // Test that non-positive lower bound is rejected
    let lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let log = LogLayer;
    let err = log
        .propagate_ibp(&input)
        .expect_err("LogLayer should reject negative input bounds");

    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("strictly positive") || msg.contains("non-positive"),
        "error message should mention positive requirement: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_log_ibp_rejects_zero_lower() {
    // Test that zero lower bound is rejected (log(0) = -inf)
    let lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let log = LogLayer;
    let err = log
        .propagate_ibp(&input)
        .expect_err("LogLayer should reject zero input bounds");

    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("strictly positive") || msg.contains("positive"),
        "error message should mention positive requirement: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_log_ibp_edge_case_small_positive() {
    // Test very small positive values are accepted
    let lower = ArrayD::from_elem(IxDyn(&[2]), 1e-6f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let log = LogLayer;
    let output = log.propagate_ibp(&input).unwrap();

    // ln(1e-6) ≈ -13.8, ln(1) = 0
    assert!(
        output.lower()[[0]] < -10.0,
        "ln(1e-6) should be very negative, got {}",
        output.lower()[[0]]
    );
    assert!(
        output.upper()[[0]].abs() < 1e-6,
        "ln(1) should be 0, got {}",
        output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_log_linear_requires_preactivation_bounds() {
    // Without pre-activation bounds, Log must return an error (nonlinear layer).
    let bounds = LinearBounds::identity(4);
    let log = LogLayer;
    let result = log.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "Log::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_log_crown_soundness() {
    // Test that CROWN backward bounds contain true log(x) for all sample points.
    // Pre-activation bounds must be strictly positive for log.
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.0, 0.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 5.0, 2.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let log = LogLayer;

    let result = log
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points within each neuron's range
    let test_points: [Vec<f32>; 5] = [
        vec![0.5, 1.0, 0.1],   // lower bounds
        vec![3.0, 5.0, 2.0],   // upper bounds
        vec![1.75, 3.0, 1.05], // midpoints
        vec![1.0, 2.0, 0.5],   // quarter
        vec![2.0, 4.0, 1.5],   // three-quarter
    ];

    let dim = 3;
    let tol = 1e-4;

    for point in &test_points {
        let log_output: Vec<f32> = point.iter().map(|x| x.ln()).collect();

        for (j, &log_val) in log_output.iter().enumerate() {
            let lb_val: f32 = (0..dim)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..dim)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            assert!(
                lb_val <= log_val + tol,
                "Log lower bound violated at point {:?}: lb {} > log({}) = {}",
                point,
                lb_val,
                point[j],
                log_val
            );
            assert!(
                ub_val >= log_val - tol,
                "Log upper bound violated at point {:?}: ub {} < log({}) = {}",
                point,
                ub_val,
                point[j],
                log_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_log_crown_wide_interval() {
    // Test CROWN soundness on a wide interval (0.01 to 100).
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.01]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![100.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let log = LogLayer;

    let result = log
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let tol = 1e-3;
    for i in 0..=20 {
        let t = i as f32 / 20.0;
        let x = 0.01 + 99.99 * t;
        let log_val = x.ln();
        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];

        assert!(
            lb <= log_val + tol,
            "Log lower bound violated at x={x}: lb={lb} > log(x)={log_val}"
        );
        assert!(
            ub >= log_val - tol,
            "Log upper bound violated at x={x}: ub={ub} < log(x)={log_val}"
        );
    }
}

// ── Strict zero-tolerance CROWN relaxation proptest (#3292) ──────────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.

/// Independent f64 log reference for strict proptest. (#3292)
fn log_f64_reference(x: f64) -> f64 {
    x.ln()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for log CROWN relaxation.
    /// Uses f64 reference (log_f64_reference) with zero tolerance on 200-point grid.
    /// log is concave on (0, +inf): lower bound = chord, upper bound = tangent.
    /// Domain restricted to positive l (log undefined for x <= 0).
    /// Ref: alpha-beta-CROWN auto_LiRPA log relaxation, #3292.
    #[test]
    fn proptest_log_relaxation_strict_soundness(
        l in 0.01f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = log_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = log_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "log lower bound UNSOUND at x={x}: {lower_val} > log({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "log upper bound UNSOUND at x={x}: {upper_val} < log({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }
}
