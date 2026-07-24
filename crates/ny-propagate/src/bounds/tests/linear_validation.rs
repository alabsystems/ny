// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for LinearBounds NaN/Inf validation (#2977).
//!
//! Split from linear.rs to stay under the 1000-line file limit.
//! Tests validate that `LinearBounds::new()` and factory methods reject
//! NaN coefficients, Inf coefficients, and NaN biases while allowing
//! ±Inf biases (conservative bounds).

use crate::bounds::LinearBounds;
use ndarray::{array, Array1, Array2};
use proptest::prelude::*;

/// LinearBounds::new() rejects NaN in lower_a coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_rejects_nan_lower_a() {
    let err = LinearBounds::new(
        array![[f32::NAN, 1.0]],
        array![0.0],
        array![[1.0, 1.0]],
        array![0.0],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_a") && msg.contains("NaN"),
        "expected lower_a NaN error, got: {msg}"
    );
}

/// LinearBounds::new() rejects NaN in upper_a coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_rejects_nan_upper_a() {
    let err = LinearBounds::new(
        array![[1.0, 1.0]],
        array![0.0],
        array![[1.0, f32::NAN]],
        array![0.0],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("upper_a") && msg.contains("NaN"),
        "expected upper_a NaN error, got: {msg}"
    );
}

/// LinearBounds::new() rejects Inf in lower_a coefficients.
/// Infinite coefficients are not valid linear relaxations.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_rejects_inf_lower_a() {
    let err = LinearBounds::new(
        array![[f32::INFINITY]],
        array![0.0],
        array![[1.0]],
        array![0.0],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_a") && msg.contains("Inf"),
        "expected lower_a Inf error, got: {msg}"
    );
}

/// LinearBounds::new() rejects -Inf in upper_a coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_rejects_neg_inf_upper_a() {
    let err = LinearBounds::new(
        array![[1.0]],
        array![0.0],
        array![[f32::NEG_INFINITY]],
        array![0.0],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("upper_a") && msg.contains("Inf"),
        "expected upper_a Inf error, got: {msg}"
    );
}

/// LinearBounds::new() rejects NaN in lower_b bias.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_rejects_nan_lower_b() {
    let err =
        LinearBounds::new(array![[1.0]], array![f32::NAN], array![[1.0]], array![0.0]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_b") && msg.contains("NaN"),
        "expected lower_b NaN error, got: {msg}"
    );
}

/// LinearBounds::new() rejects NaN in upper_b bias.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_rejects_nan_upper_b() {
    let err =
        LinearBounds::new(array![[1.0]], array![0.0], array![[1.0]], array![f32::NAN]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("upper_b") && msg.contains("NaN"),
        "expected upper_b NaN error, got: {msg}"
    );
}

/// LinearBounds::new() allows ±Inf in bias vectors (conservative bounds).
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_allows_inf_bias() {
    let result = LinearBounds::new(
        array![[1.0]],
        array![f32::NEG_INFINITY],
        array![[1.0]],
        array![f32::INFINITY],
    );
    assert!(
        result.is_ok(),
        "±Inf biases should be allowed for conservative bounds"
    );
}

/// LinearBounds::new() accepts clean finite values.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_new_accepts_clean_values() {
    let result = LinearBounds::new(
        array![[1.0, -0.5], [0.3, 2.0]],
        array![0.1, -0.2],
        array![[1.2, -0.3], [0.5, 2.1]],
        array![0.3, 0.0],
    );
    assert!(result.is_ok(), "clean finite values should be accepted");
}

/// LinearBounds::conservative() produces known-safe bounds.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_conservative() {
    let bounds = LinearBounds::conservative(3, 5);
    assert_eq!(bounds.num_outputs(), 3);
    assert_eq!(bounds.num_inputs(), 5);
    // Coefficients are zero
    assert!(bounds.lower_a.iter().all(|&v| v == 0.0));
    assert!(bounds.upper_a.iter().all(|&v| v == 0.0));
    // Biases are ±Inf
    assert!(bounds.lower_b.iter().all(|&v| v == f32::NEG_INFINITY));
    assert!(bounds.upper_b.iter().all(|&v| v == f32::INFINITY));
}

/// LinearBounds::from_spec_matrix() rejects NaN in spec matrix.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_from_spec_matrix_rejects_nan() {
    let err = LinearBounds::from_spec_matrix(array![[1.0, f32::NAN]]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("from_spec_matrix") && msg.contains("NaN"),
        "expected spec matrix NaN error, got: {msg}"
    );
}

/// LinearBounds::from_spec_matrix() rejects Inf in spec matrix.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_from_spec_matrix_rejects_inf() {
    let err = LinearBounds::from_spec_matrix(array![[f32::INFINITY, 1.0]]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("from_spec_matrix") && msg.contains("Inf"),
        "expected spec matrix Inf error, got: {msg}"
    );
}

/// LinearBounds::from_spec_matrix() accepts clean spec matrix.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_from_spec_matrix_accepts_clean() {
    let result = LinearBounds::from_spec_matrix(array![[1.0, -1.0, 0.0]]);
    assert!(result.is_ok(), "clean spec matrix should be accepted");
    let bounds = result.unwrap();
    assert_eq!(bounds.num_outputs(), 1);
    assert_eq!(bounds.num_inputs(), 3);
}

/// LinearBounds::symmetric() rejects NaN.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_symmetric_rejects_nan() {
    let err = LinearBounds::symmetric(array![[f32::NAN]], array![0.0]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NaN"), "expected NaN error, got: {msg}");
}

/// LinearBounds::from_coefficients() rejects NaN.
#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_from_coefficients_rejects_nan() {
    let err = LinearBounds::from_coefficients(array![[1.0]], array![[f32::NAN]]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NaN"), "expected NaN error, got: {msg}");
}

// =========================================================================
// Property-based tests: NaN injection -> Err(NumericalInstability)
//
// These proptests verify the core invariant of #2977: LinearBounds::new()
// structurally rejects NaN/Inf in coefficients and NaN in biases across
// random shapes and injection positions.
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Property: injecting NaN into lower_a at any position -> Err.
    #[ntest::timeout(10000)]
    #[test]
    fn prop_nan_in_lower_a_rejected(
        rows in 1_usize..=8,
        cols in 1_usize..=8,
        nan_row in 0_usize..8,
        nan_col in 0_usize..8,
    ) {
        let nan_row = nan_row % rows;
        let nan_col = nan_col % cols;
        let mut la = Array2::zeros((rows, cols));
        la[(nan_row, nan_col)] = f32::NAN;
        let result = LinearBounds::new(
            la,
            Array1::zeros(rows),
            Array2::zeros((rows, cols)),
            Array1::zeros(rows),
        );
        prop_assert!(result.is_err(), "NaN in lower_a[{},{}] should be rejected", nan_row, nan_col);
        let msg = result.unwrap_err().to_string();
        prop_assert!(msg.contains("lower_a"), "error should mention lower_a, got: {}", msg);
    }

    /// Property: injecting NaN into upper_a at any position -> Err.
    #[ntest::timeout(10000)]
    #[test]
    fn prop_nan_in_upper_a_rejected(
        rows in 1_usize..=8,
        cols in 1_usize..=8,
        nan_row in 0_usize..8,
        nan_col in 0_usize..8,
    ) {
        let nan_row = nan_row % rows;
        let nan_col = nan_col % cols;
        let mut ua = Array2::zeros((rows, cols));
        ua[(nan_row, nan_col)] = f32::NAN;
        let result = LinearBounds::new(
            Array2::zeros((rows, cols)),
            Array1::zeros(rows),
            ua,
            Array1::zeros(rows),
        );
        prop_assert!(result.is_err(), "NaN in upper_a[{},{}] should be rejected", nan_row, nan_col);
        let msg = result.unwrap_err().to_string();
        prop_assert!(msg.contains("upper_a"), "error should mention upper_a, got: {}", msg);
    }

    /// Property: injecting +/-Inf into coefficient matrices -> Err.
    #[ntest::timeout(10000)]
    #[test]
    fn prop_inf_in_coefficients_rejected(
        rows in 1_usize..=8,
        cols in 1_usize..=8,
        inject_row in 0_usize..8,
        inject_col in 0_usize..8,
        use_lower in proptest::bool::ANY,
        use_neg in proptest::bool::ANY,
    ) {
        let inject_row = inject_row % rows;
        let inject_col = inject_col % cols;
        let inf_val = if use_neg { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut la = Array2::zeros((rows, cols));
        let mut ua = Array2::zeros((rows, cols));
        if use_lower {
            la[(inject_row, inject_col)] = inf_val;
        } else {
            ua[(inject_row, inject_col)] = inf_val;
        }
        let result = LinearBounds::new(
            la,
            Array1::zeros(rows),
            ua,
            Array1::zeros(rows),
        );
        prop_assert!(result.is_err(), "Inf in coefficients should be rejected");
    }

    /// Property: injecting NaN into lower_b at any position -> Err.
    #[ntest::timeout(10000)]
    #[test]
    fn prop_nan_in_lower_b_rejected(
        rows in 1_usize..=8,
        nan_idx in 0_usize..8,
    ) {
        let nan_idx = nan_idx % rows;
        let mut lb = Array1::zeros(rows);
        lb[nan_idx] = f32::NAN;
        let result = LinearBounds::new(
            Array2::zeros((rows, 1)),
            lb,
            Array2::zeros((rows, 1)),
            Array1::zeros(rows),
        );
        prop_assert!(result.is_err(), "NaN in lower_b[{}] should be rejected", nan_idx);
        let msg = result.unwrap_err().to_string();
        prop_assert!(msg.contains("lower_b"), "error should mention lower_b, got: {}", msg);
    }

    /// Property: injecting NaN into upper_b at any position -> Err.
    #[ntest::timeout(10000)]
    #[test]
    fn prop_nan_in_upper_b_rejected(
        rows in 1_usize..=8,
        nan_idx in 0_usize..8,
    ) {
        let nan_idx = nan_idx % rows;
        let mut ub = Array1::zeros(rows);
        ub[nan_idx] = f32::NAN;
        let result = LinearBounds::new(
            Array2::zeros((rows, 1)),
            Array1::zeros(rows),
            Array2::zeros((rows, 1)),
            ub,
        );
        prop_assert!(result.is_err(), "NaN in upper_b[{}] should be rejected", nan_idx);
        let msg = result.unwrap_err().to_string();
        prop_assert!(msg.contains("upper_b"), "error should mention upper_b, got: {}", msg);
    }

    /// Property: +/-Inf in biases is allowed (conservative bounds pattern).
    #[ntest::timeout(10000)]
    #[test]
    fn prop_inf_in_biases_allowed(
        rows in 1_usize..=8,
        cols in 1_usize..=8,
        lb_inf_idx in 0_usize..8,
        ub_inf_idx in 0_usize..8,
    ) {
        let lb_inf_idx = lb_inf_idx % rows;
        let ub_inf_idx = ub_inf_idx % rows;
        let mut lb = Array1::zeros(rows);
        let mut ub = Array1::zeros(rows);
        lb[lb_inf_idx] = f32::NEG_INFINITY;
        ub[ub_inf_idx] = f32::INFINITY;
        let result = LinearBounds::new(
            Array2::zeros((rows, cols)),
            lb,
            Array2::zeros((rows, cols)),
            ub,
        );
        prop_assert!(result.is_ok(), "+/-Inf in biases should be allowed for conservative bounds");
    }
}
