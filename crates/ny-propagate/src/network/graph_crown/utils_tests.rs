// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for graph_crown::utils — safe_add and verify_split_path_bias_zero.
//! Split from utils.rs to keep file under 500-line limit.

use super::GraphNetwork;
use ndarray::{array, Array, IxDyn};

#[test]
fn test_safe_add_2d_nan_inputs_map_to_conservative_infinity_2476() {
    let existing = array![[f32::NAN, 1.0], [2.0, f32::NAN]];
    let new = array![[3.0, f32::NAN], [f32::NAN, 4.0]];

    let lower = GraphNetwork::safe_add(&existing, &new, true);
    assert!(lower.iter().all(|v| *v == f32::NEG_INFINITY));
    assert!(lower.iter().all(|v| !v.is_nan()));

    let upper = GraphNetwork::safe_add(&existing, &new, false);
    assert!(upper.iter().all(|v| *v == f32::INFINITY));
    assert!(upper.iter().all(|v| !v.is_nan()));
}

#[test]
fn test_safe_add_1d_nan_inputs_map_to_conservative_infinity_2476() {
    let existing = array![f32::NAN, 1.0, f32::INFINITY];
    let new = array![2.0, f32::NAN, f32::NEG_INFINITY];

    let lower = GraphNetwork::safe_add(&existing, &new, true);
    assert_eq!(lower[0], f32::NEG_INFINITY);
    assert_eq!(lower[1], f32::NEG_INFINITY);
    assert_eq!(lower[2], f32::NEG_INFINITY);
    assert!(lower.iter().all(|v| !v.is_nan()));

    let upper = GraphNetwork::safe_add(&existing, &new, false);
    assert_eq!(upper[0], f32::INFINITY);
    assert_eq!(upper[1], f32::INFINITY);
    assert_eq!(upper[2], f32::INFINITY);
    assert!(upper.iter().all(|v| !v.is_nan()));
}

#[test]
fn test_safe_add_dynamic_nan_inputs_map_to_conservative_infinity_2476() {
    let existing = Array::from_shape_vec(IxDyn(&[1, 2, 2]), vec![f32::NAN, 1.0, 2.0, f32::NAN])
        .expect("shape should be valid");
    let new = Array::from_shape_vec(IxDyn(&[1, 2, 2]), vec![3.0, f32::NAN, f32::NAN, 4.0])
        .expect("shape should be valid");

    let lower = GraphNetwork::safe_add(&existing, &new, true);
    assert!(lower.iter().all(|v| *v == f32::NEG_INFINITY));
    assert!(lower.iter().all(|v| !v.is_nan()));

    let upper = GraphNetwork::safe_add(&existing, &new, false);
    assert!(upper.iter().all(|v| *v == f32::INFINITY));
    assert!(upper.iter().all(|v| !v.is_nan()));
}

/// #2907: safe_add with mismatched 2D shapes returns conservative bounds
/// instead of panicking.
#[test]
fn test_safe_add_2d_shape_mismatch_returns_conservative_2907() {
    let existing = array![[1.0, 2.0], [3.0, 4.0]]; // 2x2
    let mismatched = array![[5.0, 6.0, 7.0]]; // 1x3

    let lower = GraphNetwork::safe_add(&existing, &mismatched, true);
    assert_eq!(lower.shape(), existing.shape());
    assert!(lower.iter().all(|&v| v == f32::NEG_INFINITY));

    let upper = GraphNetwork::safe_add(&existing, &mismatched, false);
    assert_eq!(upper.shape(), existing.shape());
    assert!(upper.iter().all(|&v| v == f32::INFINITY));
}

/// #2907: safe_add with mismatched 1D lengths returns conservative bounds
/// instead of panicking.
#[test]
fn test_safe_add_1d_length_mismatch_returns_conservative_2907() {
    let existing = array![1.0, 2.0, 3.0]; // len 3
    let mismatched = array![4.0, 5.0]; // len 2

    let lower = GraphNetwork::safe_add(&existing, &mismatched, true);
    assert_eq!(lower.len(), existing.len());
    assert!(lower.iter().all(|&v| v == f32::NEG_INFINITY));

    let upper = GraphNetwork::safe_add(&existing, &mismatched, false);
    assert_eq!(upper.len(), existing.len());
    assert!(upper.iter().all(|&v| v == f32::INFINITY));
}

/// #4243: shape-mismatched arrays with the same element count should reshape
/// instead of widening to conservative infinities.
#[test]
fn test_safe_add_same_count_shape_mismatch_reshapes_4243() {
    let existing = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let reshaped = array![[10.0_f32], [20.0], [30.0], [40.0]];

    let lower = GraphNetwork::safe_add(&existing, &reshaped, true);
    let upper = GraphNetwork::safe_add(&existing, &reshaped, false);

    assert_eq!(lower, array![[11.0_f32, 22.0], [33.0, 44.0]]);
    assert_eq!(upper, array![[11.0_f32, 22.0], [33.0, 44.0]]);
}

/// #2907: safe_add with mismatched dynamic shapes returns conservative bounds
/// instead of panicking.
#[test]
fn test_safe_add_dynamic_shape_mismatch_returns_conservative_2907() {
    let existing = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .expect("shape should be valid");
    let mismatched =
        Array::from_shape_vec(IxDyn(&[3, 1]), vec![5.0, 6.0, 7.0]).expect("shape should be valid");

    let lower = GraphNetwork::safe_add(&existing, &mismatched, true);
    assert_eq!(lower.shape(), existing.shape());
    assert!(lower.iter().all(|&v| v == f32::NEG_INFINITY));

    let upper = GraphNetwork::safe_add(&existing, &mismatched, false);
    assert_eq!(upper.shape(), existing.shape());
    assert!(upper.iter().all(|&v| v == f32::INFINITY));
}

/// #2907: verify_split_path_bias_zero returns Err for non-zero lower bias.
#[test]
fn test_verify_split_path_bias_zero_rejects_nonzero_lower_2907() {
    use crate::bounds::LinearBounds;
    use ndarray::Array2;

    let bounds = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: array![0.0, 0.5], // non-zero!
        upper_a: Array2::zeros((2, 3)),
        upper_b: array![0.0, 0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let result = GraphNetwork::verify_split_path_bias_zero(&bounds, "test");
    assert!(result.is_err(), "should reject non-zero lower_b");
}

/// #2907: verify_split_path_bias_zero returns Err for non-zero upper bias.
#[test]
fn test_verify_split_path_bias_zero_rejects_nonzero_upper_2907() {
    use crate::bounds::LinearBounds;
    use ndarray::Array2;

    let bounds = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: array![0.0, 0.0],
        upper_a: Array2::zeros((2, 3)),
        upper_b: array![0.1, 0.0], // non-zero!
        lower_a_err: None,
        upper_a_err: None,
    };
    let result = GraphNetwork::verify_split_path_bias_zero(&bounds, "test");
    assert!(result.is_err(), "should reject non-zero upper_b");
}

/// #2907: verify_split_path_bias_zero accepts all-zero biases.
#[test]
fn test_verify_split_path_bias_zero_accepts_zero_2907() {
    use crate::bounds::LinearBounds;
    use ndarray::Array2;

    let bounds = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: array![0.0, 0.0],
        upper_a: Array2::zeros((2, 3)),
        upper_b: array![0.0, 0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let result = GraphNetwork::verify_split_path_bias_zero(&bounds, "test");
    assert!(result.is_ok(), "should accept zero biases");
}

/// #2700: NaN in bias produces error message mentioning "NaN", not "non-zero".
#[test]
fn test_verify_split_path_bias_zero_nan_diagnostic_2700() {
    use crate::bounds::LinearBounds;
    use ndarray::Array2;

    let bounds = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: array![0.0, f32::NAN],
        upper_a: Array2::zeros((2, 3)),
        upper_b: array![0.0, 0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let err = GraphNetwork::verify_split_path_bias_zero(&bounds, "test_layer")
        .expect_err("NaN in lower_b must return Err");
    let msg = format!("{err}");
    assert!(
        msg.contains("NaN"),
        "NaN error message must mention 'NaN', got: {msg}"
    );
}

/// #2700: non-zero error includes max absolute value for diagnosis.
#[test]
fn test_verify_split_path_bias_zero_includes_max_abs_2700() {
    use crate::bounds::LinearBounds;
    use ndarray::Array2;

    let bounds = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: array![0.0, 0.0],
        upper_a: Array2::zeros((2, 3)),
        upper_b: array![0.0, 0.5],
        lower_a_err: None,
        upper_a_err: None,
    };
    let err = GraphNetwork::verify_split_path_bias_zero(&bounds, "test_layer")
        .expect_err("non-zero upper_b must return Err");
    let msg = format!("{err}");
    assert!(
        msg.contains("max |v|"),
        "error must include 'max |v|' diagnostic, got: {msg}"
    );
}

/// #3044: IBP spec fallback preserves finite terms when zero-coefficient
/// neurons have Inf bounds. Before the fix, `0.0 * Inf = NaN` poisoned
/// the accumulator, causing the entire row to collapse to `[-Inf, +Inf]`.
/// With `safe_mul_for_bounds`, `0 * Inf = 0`, so finite terms survive.
#[test]
fn test_ibp_spec_fallback_zero_coeff_inf_bound_3044() {
    use ndarray::{arr1, Array2};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    // Build minimal GraphNetwork (function only uses Self::sanitize_bounds_for_fallback).
    let graph = GraphNetwork {
        node_order: vec!["output".to_string()],
        output_node: "output".to_string(),
        ..GraphNetwork::new()
    };

    // 3 neurons: [finite, Inf, finite]
    let lower = arr1(&[-1.0f32, f32::NEG_INFINITY, -2.0]).into_dyn();
    let upper = arr1(&[1.0f32, f32::INFINITY, 2.0]).into_dyn();
    let output_bounds =
        BoundedTensor::new_allow_infinite(lower, upper).expect("invariant: test bounds are valid");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("output".to_string(), output_bounds);

    // Spec: neuron 0 has coeff=1, neuron 1 has coeff=0 (should contribute nothing),
    // neuron 2 has coeff=1.
    let spec_matrix = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 1.0])
        .expect("invariant: spec shape is valid");

    let dummy_input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn())
        .expect("invariant: dummy input is valid");

    let result = graph
        .propagate_crown_with_specs_fallback_ibp(&dummy_input, &spec_matrix, &node_bounds, "output")
        .expect("IBP fallback should succeed");

    // With safe_mul_for_bounds: 0*Inf = 0, so bounds = [-1+0-2, 1+0+2] = [-3, 3].
    // Without fix: 0*Inf = NaN → NaN poisons accumulator → [-Inf, +Inf].
    let restored_lower = result.lower()[[0]];
    let restored_upper = result.upper()[[0]];
    assert!(
        restored_lower.is_finite(),
        "Lower bound should be finite (-3.0), got {}",
        restored_lower
    );
    assert!(
        restored_upper.is_finite(),
        "Upper bound should be finite (3.0), got {}",
        restored_upper
    );
    assert!(
        (restored_lower - (-3.0)).abs() < 1e-6,
        "Expected lower=-3.0, got {}",
        restored_lower
    );
    assert!(
        (restored_upper - 3.0).abs() < 1e-6,
        "Expected upper=3.0, got {}",
        restored_upper
    );
}

/// #3044: Negative coefficient branch coverage for IBP spec fallback.
/// Prover finding: the c < 0 branch (lines 46-49 in utils.rs) swaps
/// `input_l`/`input_u` before safe_mul_for_bounds. This test exercises
/// that swap with a mix of positive, zero, and negative spec coefficients.
#[test]
fn test_ibp_spec_fallback_negative_coeff_3044() {
    use ndarray::{arr1, Array2};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    let graph = GraphNetwork {
        node_order: vec!["output".to_string()],
        output_node: "output".to_string(),
        ..GraphNetwork::new()
    };

    // 3 neurons: [finite, Inf, finite]
    let lower = arr1(&[-1.0f32, f32::NEG_INFINITY, -2.0]).into_dyn();
    let upper = arr1(&[1.0f32, f32::INFINITY, 2.0]).into_dyn();
    let output_bounds =
        BoundedTensor::new_allow_infinite(lower, upper).expect("invariant: test bounds are valid");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("output".to_string(), output_bounds);

    // Spec: neuron 0 coeff=1 (positive), neuron 1 coeff=0 (zero with Inf),
    // neuron 2 coeff=-1 (negative).
    // For c >= 0: l += c * input_l, u += c * input_u.
    // For c < 0: l += c * input_u, u += c * input_l.
    //
    // neuron 0: l += 1*(-1) = -1, u += 1*1 = 1
    // neuron 1: l += safe_mul(0,-Inf) = 0, u += safe_mul(0,Inf) = 0
    // neuron 2: l += (-1)*2 = -2, u += (-1)*(-2) = 2
    // Total: l = -3, u = 3
    let spec_matrix = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, -1.0])
        .expect("invariant: spec shape is valid");

    let dummy_input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn())
        .expect("invariant: dummy input is valid");

    let result = graph
        .propagate_crown_with_specs_fallback_ibp(&dummy_input, &spec_matrix, &node_bounds, "output")
        .expect("IBP fallback should succeed");

    let restored_lower = result.lower()[[0]];
    let restored_upper = result.upper()[[0]];
    assert!(
        restored_lower.is_finite(),
        "Lower bound should be finite (-3.0), got {}",
        restored_lower
    );
    assert!(
        restored_upper.is_finite(),
        "Upper bound should be finite (3.0), got {}",
        restored_upper
    );
    assert!(
        (restored_lower - (-3.0)).abs() < 1e-6,
        "Expected lower=-3.0, got {}",
        restored_lower
    );
    assert!(
        (restored_upper - 3.0).abs() < 1e-6,
        "Expected upper=3.0, got {}",
        restored_upper
    );
}
