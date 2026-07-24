// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for interval arithmetic functions.

use crate::bounds::{
    batched_interval_matvec, batched_interval_matvec_checked, interval_mul_for_bounds,
    safe_add_lower_for_bounds, safe_add_upper_for_bounds, safe_mul_pair_for_bounds,
};
use crate::NyError;
use ndarray::{ArrayD, IxDyn};

// =========================================================================
// interval_mul_for_bounds tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_interval_mul_for_bounds_positive_intervals() {
    // [2, 3] * [4, 5] = [8, 15]. Directed OUTWARD rounding (#concretize-soundness-
    // hardening) widens the endpoints by one ULP, so assert sound enclosure within
    // 1 ULP rather than exact equality.
    let (lower, upper) = interval_mul_for_bounds(2.0, 3.0, 4.0, 5.0);
    assert!(lower <= 8.0, "lower {lower} must enclose true min 8.0");
    assert_eq!(lower, ny_tensor::next_down_f32(8.0));
    assert!(upper >= 15.0, "upper {upper} must enclose true max 15.0");
    assert_eq!(upper, ny_tensor::next_up_f32(15.0));
}

#[ntest::timeout(5000)]
#[test]
fn test_interval_mul_for_bounds_mixed_signs() {
    // [-1, 2] * [3, 4] = [-4, 8]. Outward rounding widens by 1 ULP; assert enclosure.
    let (lower, upper) = interval_mul_for_bounds(-1.0, 2.0, 3.0, 4.0);
    assert!(lower <= -4.0, "lower {lower} must enclose true min -4.0");
    assert_eq!(lower, ny_tensor::next_down_f32(-4.0));
    assert!(upper >= 8.0, "upper {upper} must enclose true max 8.0");
    assert_eq!(upper, ny_tensor::next_up_f32(8.0));
}

#[ntest::timeout(5000)]
#[test]
fn test_interval_mul_for_bounds_zero_handling() {
    // [0, 1] * [inf, inf] should not produce NaN
    let (lower, upper) = interval_mul_for_bounds(0.0, 1.0, f32::INFINITY, f32::INFINITY);
    assert!(!lower.is_nan());
    assert!(!upper.is_nan());
    // 0 * inf = 0, 1 * inf = inf. Directed-outward rounding pushes the 0 lower endpoint
    // down by one ULP (smallest negative subnormal); still a sound enclosure of 0.
    assert!(lower <= 0.0, "lower {lower} must enclose true min 0.0");
    assert_eq!(lower, ny_tensor::next_down_f32(0.0));
    assert!(upper.is_infinite() && upper > 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_interval_mul_for_bounds_nan_input() {
    // NaN input should return conservative (-inf, +inf)
    let (lower, upper) = interval_mul_for_bounds(f32::NAN, 1.0, 2.0, 3.0);
    assert_eq!(lower, f32::NEG_INFINITY);
    assert_eq!(upper, f32::INFINITY);
}

// =========================================================================
// safe_mul_pair_for_bounds tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_safe_mul_pair_for_bounds_zero_times_inf() {
    // 0 * inf should be 0, not NaN
    assert_eq!(safe_mul_pair_for_bounds(0.0, f32::INFINITY), 0.0);
    assert_eq!(safe_mul_pair_for_bounds(f32::NEG_INFINITY, 0.0), 0.0);
}

// =========================================================================
// safe_add_*_for_bounds tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_safe_add_lower_for_bounds_nan_to_neg_inf() {
    // NaN should become -inf for lower bounds (conservative)
    let result = safe_add_lower_for_bounds(f32::INFINITY, f32::NEG_INFINITY);
    assert_eq!(result, f32::NEG_INFINITY);
}

#[ntest::timeout(5000)]
#[test]
fn test_safe_add_upper_for_bounds_nan_to_pos_inf() {
    // NaN should become +inf for upper bounds (conservative)
    let result = safe_add_upper_for_bounds(f32::INFINITY, f32::NEG_INFINITY);
    assert_eq!(result, f32::INFINITY);
}

// =========================================================================
// batched_interval_matvec tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_deterministic() {
    // When lower == upper for coefficients, result should be a sound interval
    // containing the true dot product. Directed rounding (#2391) widens by
    // at most 1 ULP on each side, so lower and upper may differ by up to 2 ULPs.
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]).unwrap();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0_f32]).unwrap();
    let x_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0_f32]).unwrap();

    let (result_lower, result_upper) = batched_interval_matvec(&a, &a, &x_lower, &x_upper)
        .expect("valid deterministic batched interval matvec should succeed");

    // With directed rounding, lower and upper differ by at most 2 ULPs
    assert!((result_lower[[0]] - result_upper[[0]]).abs() < 1e-5);
    assert!((result_lower[[1]] - result_upper[[1]]).abs() < 1e-5);

    // Result should be [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    // Soundness: lower <= true <= upper
    assert!(
        result_lower[[0]] <= 14.0,
        "lower {} must be <= 14.0",
        result_lower[[0]]
    );
    assert!(
        result_upper[[0]] >= 14.0,
        "upper {} must be >= 14.0",
        result_upper[[0]]
    );
    assert!(
        result_lower[[1]] <= 32.0,
        "lower {} must be <= 32.0",
        result_lower[[1]]
    );
    assert!(
        result_upper[[1]] >= 32.0,
        "upper {} must be >= 32.0",
        result_upper[[1]]
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_negative_input() {
    // Test case that triggered bug #229: negative input with interval coefficients
    // Coefficient: [0, 0.1], Input: [-1.0, -0.5]
    // Products: 0*-1=0, 0*-0.5=0, 0.1*-1=-0.1, 0.1*-0.5=-0.05
    // Result: [min(-0.1, 0), max(0, -0.05)] = [-0.1, 0]
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0_f32]).unwrap();
    let a_upper = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.1_f32]).unwrap();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0_f32]).unwrap();
    let x_upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5_f32]).unwrap();

    let (result_lower, result_upper) =
        batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper)
            .expect("valid negative-input interval matvec should succeed");

    // Result should be [-0.1, 0] (sound interval)
    assert!((result_lower[[0]] - (-0.1)).abs() < 1e-6);
    assert!((result_upper[[0]] - 0.0).abs() < 1e-6);
    // Critical: lower <= upper (the bug that #229 caught)
    assert!(result_lower[[0]] <= result_upper[[0]]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_mixed_signs() {
    // Coefficient: [-0.5, 0.5], Input: [-1.0, 1.0]
    // Products: -0.5*-1=0.5, -0.5*1=-0.5, 0.5*-1=-0.5, 0.5*1=0.5
    // Result: [min(-0.5, 0.5), max(-0.5, 0.5)] = [-0.5, 0.5]
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-0.5_f32]).unwrap();
    let a_upper = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.5_f32]).unwrap();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0_f32]).unwrap();
    let x_upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0_f32]).unwrap();

    let (result_lower, result_upper) =
        batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper)
            .expect("valid mixed-sign interval matvec should succeed");

    assert!((result_lower[[0]] - (-0.5)).abs() < 1e-6);
    assert!((result_upper[[0]] - 0.5).abs() < 1e-6);
    assert!(result_lower[[0]] <= result_upper[[0]]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_with_batches() {
    // 2 batches, 2x2 matrices, 2-element vectors
    let a_lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0_f32],
    )
    .unwrap();
    let a_upper = a_lower.clone(); // Deterministic
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0_f32]).unwrap();
    let x_upper = x_lower.clone(); // Point input

    let (result_lower, result_upper) =
        batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper)
            .expect("valid batched interval matvec should succeed");

    assert_eq!(result_lower.shape(), &[2, 2]);
    assert_eq!(result_upper.shape(), &[2, 2]);
    // With deterministic coefficients and point inputs, lower == upper
    // Batch 0: identity @ [1, 2] = [1, 2]
    assert!((result_lower[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((result_lower[[0, 1]] - 2.0).abs() < 1e-6);
    assert!((result_upper[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((result_upper[[0, 1]] - 2.0).abs() < 1e-6);
    // Batch 1: 2*identity @ [3, 4] = [6, 8]
    assert!((result_lower[[1, 0]] - 6.0).abs() < 1e-6);
    assert!((result_lower[[1, 1]] - 8.0).abs() < 1e-6);
    assert!((result_upper[[1, 0]] - 6.0).abs() < 1e-6);
    assert!((result_upper[[1, 1]] - 8.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_batch_mismatch_error() {
    // a has batch size 2, x has batch size 3 - must return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_batch_dim_ordering_mismatch_error() {
    // Same batch size product but different batch dimension ordering - must return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 2]), vec![1.0; 8]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0; 4]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_zero_batch_shape() {
    // Zero-sized batch should return empty outputs with correct shape
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[0, 2, 2]), Vec::<f32>::new()).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[0, 2]), Vec::<f32>::new()).unwrap();
    let x_upper = x_lower.clone();

    let (result_lower, result_upper) =
        batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper)
            .expect("zero-sized batch is valid and should return empty output");

    assert_eq!(result_lower.shape(), &[0, 2]);
    assert_eq!(result_upper.shape(), &[0, 2]);
}

// =========================================================================
// batched_interval_matvec_checked tests
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_checked_valid_inputs() {
    // Valid inputs should return Ok with same result as unchecked
    let a_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec_checked(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_ok());

    let (checked_lower, checked_upper) = result.unwrap();
    let (unchecked_lower, unchecked_upper) =
        batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper)
            .expect("unchecked alias should match checked behavior on valid inputs");

    assert_eq!(checked_lower, unchecked_lower);
    assert_eq!(checked_upper, unchecked_upper);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_checked_batch_mismatch_error() {
    // a has batch size 2, x has batch size 3 - should return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec_checked(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_checked_inner_dim_mismatch_error() {
    // Inner dimensions don't match - should return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap(); // 4 != 3
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec_checked(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_checked_insufficient_dims_error() {
    // Coefficient array has only 1 dimension - should return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec_checked(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::InvalidSpec(_)));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_checked_input_shape_mismatch_error() {
    // x_lower and x_upper have different shapes - should return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap();

    let result = batched_interval_matvec_checked(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_checked_coeff_shape_mismatch_error() {
    // a_lower and a_upper have different shapes - should return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec_checked(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

// =========================================================================
// Error propagation tests for batched_interval_matvec
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_degenerate_input_returns_error() {
    // Degenerate input: coefficient array with < 2 dimensions must return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::InvalidSpec(_)));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_empty_input_returns_error() {
    // Degenerate input: empty x array must return Err
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_coeff_mismatch_returns_error() {
    // a_lower and a_upper have different shapes
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_input_mismatch_returns_error() {
    // x_lower and x_upper have different shapes
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_inner_dim_mismatch_returns_error() {
    // Inner dimensions don't match: a is [2, 3], x is [4] (should be 3)
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_batch_mismatch_returns_error() {
    // Batch dimensions don't match: a has batch [2], x has batch [3]
    let a_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3, 4]), vec![1.0; 12]).unwrap();
    let x_upper = x_lower.clone();

    let result = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_valid_input_unchanged() {
    // Valid input should produce sound finite bounds
    let a_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let a_upper = a_lower.clone();
    let x_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x_upper = x_lower.clone();

    let (lower, upper) = batched_interval_matvec(&a_lower, &a_upper, &x_lower, &x_upper)
        .expect("valid input should succeed");

    // Result should be [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
    // Directed rounding (#2391) widens by at most 1 ULP per side.
    assert_eq!(lower.shape(), &[2]);
    assert!(lower[[0]] <= 14.0, "lower {} must be <= 14.0", lower[[0]]);
    assert!(upper[[0]] >= 14.0, "upper {} must be >= 14.0", upper[[0]]);
    assert!(lower[[1]] <= 32.0, "lower {} must be <= 32.0", lower[[1]]);
    assert!(upper[[1]] >= 32.0, "upper {} must be >= 32.0", upper[[1]]);
    // Bounds should be close to true values (within a few ULPs)
    assert!((lower[[0]] - 14.0).abs() < 1e-5);
    assert!((upper[[1]] - 32.0).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_directed_rounding_soundness() {
    // Verify directed rounding: lower bound <= true value <= upper bound.
    // Use values whose f64 dot product falls between two adjacent f32 values,
    // so round-to-nearest could go either way.
    // 1.0000001 * 1.0000001 + 1.0000001 * 1.0000001 accumulated in f64 then
    // cast to f32 — the directed rounding must ensure the interval is sound.
    let val = 1.0000001_f32;
    let a = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![val; 4]).unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[4]), vec![val; 4]).unwrap();

    // Point-valued inputs: a_lower == a_upper, x_lower == x_upper
    let (result_lower, result_upper) = batched_interval_matvec(&a, &a, &x, &x)
        .expect("point-valued interval matvec should succeed");

    // Compute true dot product in f64 for reference
    let true_val = (val as f64) * (val as f64) * 4.0;
    let true_f32 = true_val as f32;

    // Soundness: lower <= true <= upper (directed rounding guarantee)
    assert!(
        result_lower[[0]] <= true_f32,
        "lower bound {} must be <= true value {true_f32}",
        result_lower[[0]]
    );
    assert!(
        result_upper[[0]] >= true_f32,
        "upper bound {} must be >= true value {true_f32}",
        result_upper[[0]]
    );
    // With directed rounding on point-valued intervals, next_down_f32(x) < next_up_f32(x)
    // for any finite non-zero x. Without directed rounding, lower == upper (plain `as f32`).
    // This assertion would FAIL without the next_down/next_up fix. (#2391)
    assert!(
        result_lower[[0]] < result_upper[[0]],
        "directed rounding must widen point-valued interval: lower {} must be strictly < upper {} (#2391)",
        result_lower[[0]], result_upper[[0]]
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_interval_matvec_directed_rounding_large_sum() {
    // Large sum that stresses f64→f32 rounding: many terms of 1e6 + 1e-1
    // The f64 sum is exact enough that the f32 cast direction matters.
    let n = 100;
    let val = 1e6_f32 + 0.1;
    let a = ArrayD::from_shape_vec(IxDyn(&[1, n]), vec![val; n]).unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), vec![1.0_f32; n]).unwrap();

    let (result_lower, result_upper) =
        batched_interval_matvec(&a, &a, &x, &x).expect("large-sum interval matvec should succeed");

    // True sum in f64
    let true_sum: f64 = (0..n).map(|_| val as f64).sum();
    let true_f32 = true_sum as f32;

    // Soundness check
    assert!(
        result_lower[[0]] <= true_f32,
        "lower bound {} must be <= true value {true_f32}",
        result_lower[[0]]
    );
    assert!(
        result_upper[[0]] >= true_f32,
        "upper bound {} must be >= true value {true_f32}",
        result_upper[[0]]
    );
    // With directed rounding, lower and upper MUST differ for finite non-zero sums.
    // Without the fix, both would be `true_f32` (plain `as f32`). (#2391)
    assert!(
        result_lower[[0]] < result_upper[[0]],
        "directed rounding must widen point-valued interval: lower {} must be strictly < upper {} (#2391)",
        result_lower[[0]], result_upper[[0]]
    );
}
