// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for BilinearCrownLayer overflow and NaN guardrails.
//!
//! These regression tests verify that BilinearCrownLayer correctly rejects
//! inputs that would cause overflow or produce NaN in McCormick relaxation:
//! - Bounds exceeding MCCORMICK_MAX_MAGNITUDE (sqrt(f32::MAX) ~ 1.84e19)
//! - NaN or Inf in input bounds (BilinearCrown's own guards)
//! - Negative scale factors (would flip bounds incorrectly)

use super::*;
use crate::layers::BilinearCrownLayer;
use ndarray::{Array2, ArrayD, IxDyn};

fn identity_downstream(m: usize, n: usize) -> BatchedLinearBounds {
    let z_size = m * n;
    BatchedLinearBounds::from_parts_unchecked(
        Array2::eye(z_size).into_dyn(),
        ArrayD::zeros(IxDyn(&[z_size])),
        Array2::eye(z_size).into_dyn(),
        ArrayD::zeros(IxDyn(&[z_size])),
        vec![m, n],
        vec![m, n],
    )
}

/// Create identity downstream bounds for BilinearCrown testing.
/// For a 2x2 matmul output (m=2, n=2), we need 4x4 identity coefficients.
fn identity_downstream_for_2x2_output() -> BatchedLinearBounds {
    identity_downstream(2, 2)
}

/// Assert concretized bounds are NaN-free and ordered (lower <= upper).
/// concretize_sound's new_repaired(Widen) replaces NaN pairs with [-inf, +inf] (#3423);
/// this helper verifies the contract is upheld.
fn assert_concretized_sound(bounds: &BoundedTensor, label: &str) {
    for (j, (&lo, &hi)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        assert!(
            !lo.is_nan() && !hi.is_nan(),
            "{}: concretized bounds[{}] contain NaN (lo={}, hi={})",
            label,
            j,
            lo,
            hi
        );
        assert!(
            lo <= hi,
            "{}: concretized lower[{}]={} > upper[{}]={}",
            label,
            j,
            lo,
            j,
            hi
        );
    }
}

/// Test that bounds exceeding MCCORMICK_MAX_MAGNITUDE are rejected.
/// McCormick computes products lx*ly, ux*uy which overflow if |bound| > sqrt(f32::MAX).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_overflow_threshold_rejects() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));

    // Create downstream bounds (identity for 2x2 output)
    let downstream = identity_downstream_for_2x2_output();

    // Create input bounds with values exceeding MCCORMICK_MAX_MAGNITUDE (1.84e19)
    let huge_value = 2.0e19_f32;
    let input_a = BoundedTensor::new(
        arr2(&[[-huge_value, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[huge_value, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[-1.0, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let result = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);
    assert!(
        result.is_err(),
        "BilinearCrown should reject bounds exceeding overflow threshold"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("overflow") || err_msg.contains("infinite"),
        "Error message should mention overflow/infinite: {}",
        err_msg
    );
}

// Note: BoundedTensor::new now rejects NaN/Inf at runtime in all builds.
// BilinearCrown still keeps its own guards because callers may use
// new_unchecked for fuzzing/proptest or prevalidated fast paths.

/// Test that negative scale is rejected (would flip bounds incorrectly).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_negative_scale_rejects() {
    let bilinear = BilinearCrownLayer::new(true, Some(-0.125)); // negative scale
    let downstream = identity_downstream_for_2x2_output();

    let input_a = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let result = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);
    assert!(
        result.is_err(),
        "BilinearCrown should reject negative scale"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("negative scale"),
        "Error message should mention negative scale: {}",
        err_msg
    );
}

/// Test that BilinearCrown succeeds with valid bounded inputs.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_valid_inputs_succeed() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Valid bounded inputs
    let input_a = BoundedTensor::new(
        arr2(&[[-1.0, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[-1.0, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .unwrap_or_else(|err| {
            panic!(
                "BilinearCrown should succeed with valid bounded inputs: {:?}",
                err
            )
        });

    // Verify no NaN or Inf in results
    let has_bad_a = lb_a
        .lower_a
        .iter()
        .chain(lb_a.upper_a.iter())
        .chain(lb_a.lower_b.iter())
        .chain(lb_a.upper_b.iter())
        .any(|&v| v.is_nan() || v.is_infinite());
    let has_bad_b = lb_b
        .lower_a
        .iter()
        .chain(lb_b.upper_a.iter())
        .chain(lb_b.lower_b.iter())
        .chain(lb_b.upper_b.iter())
        .any(|&v| v.is_nan() || v.is_infinite());

    assert!(!has_bad_a, "Result lb_a should not contain NaN or Inf");
    assert!(!has_bad_b, "Result lb_b should not contain NaN or Inf");
}

/// Test that bounds just above the threshold are rejected.
/// The check is `v.abs() > MCCORMICK_MAX_MAGNITUDE` (strictly greater than 1.84e19).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_above_threshold_rejects() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Value slightly above threshold (1.841e19 > 1.84e19) should be rejected
    let above_threshold = 1.841e19_f32;
    let input_a = BoundedTensor::new(
        arr2(&[[-above_threshold, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[above_threshold, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let result = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);
    assert!(
        result.is_err(),
        "BilinearCrown should reject bounds above overflow threshold"
    );
}

/// Test that BilinearCrown handles bad bounds in input_b (not just input_a).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_bad_bounds_in_input_b_rejects() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Valid input_a
    let input_a = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    // Bad input_b with overflow value
    let huge_value = 2.0e19_f32;
    let input_b = BoundedTensor::new(
        arr2(&[[-huge_value, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[huge_value, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let result = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);
    assert!(
        result.is_err(),
        "BilinearCrown should reject bad bounds in input_b"
    );
}

/// Test that BilinearCrown's own guards catch NaN when bypassing BoundedTensor checks.
/// Uses new_unchecked to bypass BoundedTensor's debug_assert, testing BilinearCrown directly.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_nan_caught_by_bilinear_guard() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Use new_unchecked to bypass BoundedTensor's panic on NaN
    let input_a = BoundedTensor::new_unchecked(
        arr2(&[[f32::NAN, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let result = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);
    assert!(result.is_err(), "BilinearCrown should reject NaN in bounds");
}

/// Test that BilinearCrown's own guards catch Inf when bypassing BoundedTensor checks.
/// Uses new_unchecked to bypass BoundedTensor's debug_assert, testing BilinearCrown directly.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_inf_caught_by_bilinear_guard() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Use new_unchecked to bypass BoundedTensor's panic on Inf
    let input_a = BoundedTensor::new_unchecked(
        arr2(&[[f32::NEG_INFINITY, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[f32::INFINITY, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let result = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);
    assert!(result.is_err(), "BilinearCrown should reject Inf in bounds");
}

/// Test that bounds at exactly the threshold are accepted.
/// The check is `v.abs() > MCCORMICK_MAX_MAGNITUDE` (strictly greater), so exactly at threshold passes.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_at_threshold_accepts() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Value exactly at threshold (1.84e19) should be accepted (check is >)
    let at_threshold = 1.84e19_f32;
    let input_a = BoundedTensor::new(
        arr2(&[[-at_threshold, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[at_threshold, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .expect("BilinearCrown should accept bounds exactly at threshold");
    // Threshold-boundary inputs must not produce NaN or Inf in McCormick backward.
    assert!(lb_a.lower_a.iter().all(|v| v.is_finite()), "Q: NaN/Inf");
    assert!(lb_b.lower_a.iter().all(|v| v.is_finite()), "K: NaN/Inf");
}

/// Test that BilinearCrown works with scale=None (no scaling).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_no_scale_succeeds() {
    let bilinear = BilinearCrownLayer::new(true, None); // No scale factor
    let downstream = identity_downstream_for_2x2_output();

    let input_a = BoundedTensor::new(
        arr2(&[[-1.0, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[-1.0, -1.0], [-1.0, -1.0]]).into_dyn(),
        arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
    )
    .unwrap();

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .expect("BilinearCrown should succeed with scale=None");
    assert!(lb_a.lower_a.iter().all(|v| v.is_finite()), "Q: NaN/Inf");
    assert!(lb_b.lower_a.iter().all(|v| v.is_finite()), "K: NaN/Inf");
    // Symmetric [-1,1]×[-1,1] → non-trivial McCormick coefficients.
    assert!(
        lb_a.lower_a.iter().any(|&v| v != 0.0),
        "Q coefs should be non-trivial"
    );
}

/// Test soundness: concretized output bounds must be ordered (lower <= upper).
///
/// After sign-split broadcast McCormick backward (#286 Approach A), CROWN coefficient
/// matrices `lower_a` and `upper_a` are NOT guaranteed to be element-wise ordered
/// because they are computed independently for each bound direction (matching
/// auto_LiRPA `propagate_A_xy`). The correctness property is that concretized bounds
/// are ordered: lower(y) <= upper(y) after dotting with input intervals.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_soundness_concretized_ordered() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Test with various bound configurations
    let test_cases = [
        // (a_lower, a_upper, b_lower, b_upper)
        ([-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]), // symmetric
        ([-100.0, 0.0], [0.01, 1.0], [-1.0, -1.0], [1.0, 1.0]), // asymmetric a
        ([-1.0, -1.0], [1.0, 1.0], [-100.0, 0.0], [0.01, 1.0]), // asymmetric b
        ([0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]),     // non-negative
        ([-1.0, -1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0]), // non-positive
    ];

    for (i, (a_l, a_u, b_l, b_u)) in test_cases.iter().enumerate() {
        let input_a = BoundedTensor::new(
            arr2(&[[a_l[0], a_l[1]], [a_l[0], a_l[1]]]).into_dyn(),
            arr2(&[[a_u[0], a_u[1]], [a_u[0], a_u[1]]]).into_dyn(),
        )
        .unwrap();

        let input_b = BoundedTensor::new(
            arr2(&[[b_l[0], b_l[1]], [b_l[0], b_l[1]]]).into_dyn(),
            arr2(&[[b_u[0], b_u[1]], [b_u[0], b_u[1]]]).into_dyn(),
        )
        .unwrap();

        let (lb_a, lb_b) = bilinear
            .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
            .unwrap_or_else(|err| panic!("Case {}: BilinearCrown failed: {:?}", i, err));

        // Verify concretized bounds are ordered: lower <= upper
        // This is the actual soundness property, not element-wise coefficient ordering.
        let concretized_a = lb_a
            .concretize_sound(&input_a)
            .unwrap_or_else(|err| panic!("Case {}: concretize Q failed: {:?}", i, err));
        assert_concretized_sound(&concretized_a, &format!("Case {}: Q", i));

        let concretized_b = lb_b
            .concretize_sound(&input_b)
            .unwrap_or_else(|err| panic!("Case {}: concretize K failed: {:?}", i, err));
        assert_concretized_sound(&concretized_b, &format!("Case {}: K", i));
    }
}

/// Test with point bounds (lower == upper, zero-width interval).
/// McCormick relaxation should still produce valid bounds.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_point_bounds() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Point bounds: lower == upper
    let input_a = BoundedTensor::new(
        arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn(),
        arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn(),
        arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn(),
    )
    .unwrap();

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .unwrap_or_else(|err| panic!("BilinearCrown should handle point bounds: {:?}", err));

    // Verify no NaN or Inf in output
    let has_nan_or_inf = lb_a.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite());

    assert!(
        !has_nan_or_inf,
        "Point bounds should not produce NaN or Inf"
    );
}

/// Test with zero-centered bounds spanning negative to positive.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_zero_crossing_bounds() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // Bounds crossing zero
    let input_a = BoundedTensor::new(
        arr2(&[[-0.5, -0.5], [-0.5, -0.5]]).into_dyn(),
        arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[-0.5, -0.5], [-0.5, -0.5]]).into_dyn(),
        arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn(),
    )
    .unwrap();

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .unwrap_or_else(|err| {
            panic!(
                "BilinearCrown should handle zero-crossing bounds: {:?}",
                err
            )
        });

    // Verify no NaN or Inf in output
    let has_nan_or_inf = lb_a.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite());
    assert!(
        !has_nan_or_inf,
        "Zero-crossing bounds should not produce NaN or Inf"
    );

    // Verify concretized bound ordering for zero-crossing case.
    // Note: raw coefficient ordering (lower_a <= upper_a) is NOT guaranteed
    // for sign-split McCormick — coefficients are computed independently per
    // bound direction. Only concretized bounds must satisfy lower <= upper.
    let concretized_a = lb_a
        .concretize_sound(&input_a)
        .expect("Zero-crossing: concretize Q failed");
    assert_concretized_sound(&concretized_a, "Zero-crossing: Q");

    let concretized_b = lb_b
        .concretize_sound(&input_b)
        .expect("Zero-crossing: concretize K failed");
    assert_concretized_sound(&concretized_b, "Zero-crossing: K");
}

/// Test with zero bounds (0.0, 0.0) - edge case for McCormick midpoint computation.
/// Catches potential division-by-zero or degenerate cases.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_zero_bounds() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream_for_2x2_output();

    // All zero bounds: both lower and upper are 0
    let input_a = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
        arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn(),
    )
    .unwrap();

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .unwrap_or_else(|err| panic!("BilinearCrown should handle zero bounds: {:?}", err));

    // Verify no NaN or Inf in output
    let has_nan_or_inf = lb_a.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite());
    assert!(!has_nan_or_inf, "Zero bounds should not produce NaN or Inf");
}

/// Test batch-reduced global intervals with 3D batched inputs.
/// Verifies that the fix for #292 correctly computes min lower / max upper
/// across batch positions for sound McCormick plane selection.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_batched_inputs_global_intervals() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));

    // Create 3D batched inputs: [batch=2, m=2, k=2]
    // Batch 0 has bounds [-1, 1], Batch 1 has bounds [-2, 2]
    // Global min lower should be -2, global max upper should be 2
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                // Batch 0: tighter bounds
                -1.0, -1.0, -1.0, -1.0, // Batch 1: wider bounds
                -2.0, -2.0, -2.0, -2.0,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                // Batch 0
                1.0, 1.0, 1.0, 1.0, // Batch 1
                2.0, 2.0, 2.0, 2.0,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    // Input B: same batch structure [batch=2, n=2, k=2] for transpose_b=true
    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                // Batch 0: tighter bounds
                -0.5, -0.5, -0.5, -0.5, // Batch 1: wider bounds
                -1.5, -1.5, -1.5, -1.5,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                // Batch 0
                0.5, 0.5, 0.5, 0.5, // Batch 1
                1.5, 1.5, 1.5, 1.5,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    // Downstream bounds for output [m=2, n=2] flattened to [4]
    // Note: BilinearCrown currently uses unbatched downstream (per-batch handled via global intervals)
    let z_size = 4;
    let downstream = BatchedLinearBounds::from_parts_unchecked(
        Array2::eye(z_size).into_dyn(),
        ArrayD::zeros(IxDyn(&[z_size])),
        Array2::eye(z_size).into_dyn(),
        ArrayD::zeros(IxDyn(&[z_size])),
        vec![2, 2],
        vec![2, 2],
    );

    let (lb_a, lb_b) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a, &input_b)
        .expect("BilinearCrown should support batched inputs via global interval reduction");

    // Check no NaN or Inf
    let has_nan_or_inf = lb_a.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_a.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.lower_b.iter().any(|&v| v.is_nan() || v.is_infinite())
        || lb_b.upper_b.iter().any(|&v| v.is_nan() || v.is_infinite());
    assert!(
        !has_nan_or_inf,
        "Batched inputs should not produce NaN or Inf"
    );

    // Bounds ordering: lower_b <= upper_b
    for (lo, hi) in lb_a.lower_b.iter().zip(lb_a.upper_b.iter()) {
        assert!(lo <= hi, "lb_a: lower_b {} > upper_b {}", lo, hi);
    }
    for (lo, hi) in lb_b.lower_b.iter().zip(lb_b.upper_b.iter()) {
        assert!(lo <= hi, "lb_b: lower_b {} > upper_b {}", lo, hi);
    }
}

/// Test multi-dim batch shapes [2,3,...] produce the same values as flat batch [6,...].
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_global_intervals_multi_dim_batch_matches_flat_batch() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));
    let downstream = identity_downstream(1, 1);
    let a_lo = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -100.0];
    let a_hi = vec![1.0, 2.0, 3.0, 4.0, 5.0, 200.0];
    let b_lo = vec![-0.1, -0.2, -0.3, -0.4, -0.5, -300.0];
    let b_hi = vec![0.1, 0.2, 0.3, 0.4, 0.5, 6.0];
    let mk = |s: &[usize], lo: Vec<f32>, hi: Vec<f32>| {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(s), lo).unwrap(),
            ArrayD::from_shape_vec(IxDyn(s), hi).unwrap(),
        )
        .unwrap()
    };
    let input_a_multi = mk(&[2, 3, 1, 1], a_lo.clone(), a_hi.clone());
    let input_b_multi = mk(&[2, 3, 1, 1], b_lo.clone(), b_hi.clone());
    let input_a_flat = mk(&[6, 1, 1], a_lo, a_hi);
    let input_b_flat = mk(&[6, 1, 1], b_lo, b_hi);

    let (q_multi, k_multi) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a_multi, &input_b_multi)
        .expect("multi-dim batch should succeed");
    let (q_flat, k_flat) = bilinear
        .propagate_linear_batched_binary(&downstream, &input_a_flat, &input_b_flat)
        .expect("flat batch should succeed");

    // Flatten to compare values across different batch shapes ([2,3,1,1] vs [6,1,1]).
    for (label, m, f) in [("q", &q_multi, &q_flat), ("k", &k_multi, &k_flat)] {
        for (name, a, b) in [
            ("la", m.lower_a(), f.lower_a()),
            ("ua", m.upper_a(), f.upper_a()),
            ("lb", m.lower_b(), f.lower_b()),
            ("ub", m.upper_b(), f.upper_b()),
        ] {
            assert_eq!(a.len(), b.len(), "{label}.{name} length mismatch");
            for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
                assert!((va - vb).abs() <= 1e-6, "{label}.{name}[{i}]: {va} vs {vb}");
            }
        }
    }
}

// =========================================================================
// Alpha-CROWN BilinearCrown tests (#295)
// =========================================================================

/// Test BilinearCrown with alpha parameters produces valid bounds.
///
/// Alpha parameters interpolate between McCormick plane pairs:
/// - r_l = 0: L2 plane (upper corner), r_l = 1: L1 plane (lower corner)
/// - r_u = 0: U2 plane (upper corner), r_u = 1: U1 plane (lower corner)
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_with_alpha_valid_bounds() {
    use ndarray::Array4;

    let bilinear = BilinearCrownLayer::new(true, Some(0.125));

    // Input A: 2x2 matrix with bounds in reasonable range
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -1.0, -1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Input B: 2x2 matrix (transposed, so K is 2x2)
    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -1.0, -1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
    )
    .unwrap();

    let downstream = identity_downstream_for_2x2_output();

    // Test with alpha = 1.0 (default, uses L1/U1 planes). Shape [4, m, n, k] (#3287).
    let alphas = Array4::ones((4, 2, 2, 2));
    let result = bilinear.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &input_a,
        &input_b,
        Some(&alphas),
    );

    assert!(
        result.is_ok(),
        "Alpha-BilinearCrown should succeed with valid inputs"
    );
    let (lb_a, lb_b) = result.unwrap();

    // Verify no NaN or Inf
    assert!(
        !lb_a.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite()),
        "lb_a.lower_a should not contain NaN or Inf"
    );
    assert!(
        !lb_a.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite()),
        "lb_a.upper_a should not contain NaN or Inf"
    );
    assert!(
        !lb_b.lower_a.iter().any(|&v| v.is_nan() || v.is_infinite()),
        "lb_b.lower_a should not contain NaN or Inf"
    );
    assert!(
        !lb_b.upper_a.iter().any(|&v| v.is_nan() || v.is_infinite()),
        "lb_b.upper_a should not contain NaN or Inf"
    );

    // Verify bounds ordering
    for (lo, hi) in lb_a.lower_b.iter().zip(lb_a.upper_b.iter()) {
        assert!(lo <= hi, "Alpha bounds: lower_b {} > upper_b {}", lo, hi);
    }
    for (lo, hi) in lb_b.lower_b.iter().zip(lb_b.upper_b.iter()) {
        assert!(lo <= hi, "Alpha bounds: lower_b {} > upper_b {}", lo, hi);
    }
}

/// Test alpha shape computation.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_alpha_shape() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));

    // For Q: [m=3, k=4], K: [n=5, k=4] (transposed), output is [m=3, n=5]
    let (m, n, k) = bilinear.alpha_shape(&[3, 4], &[5, 4]).unwrap();
    assert_eq!(m, 3, "m should be 3");
    assert_eq!(n, 5, "n should be 5");
    assert_eq!(k, 4, "k should be 4");

    // For non-transposed: Q: [m=3, k=4], K: [k=4, n=5]
    let bilinear_no_transpose = BilinearCrownLayer::new(false, None);
    let (m2, n2, k2) = bilinear_no_transpose.alpha_shape(&[3, 4], &[4, 5]).unwrap();
    assert_eq!(m2, 3, "m should be 3");
    assert_eq!(n2, 5, "n should be 5");
    assert_eq!(k2, 4, "k should be 4");
}

/// Test that alpha init produces correct shape.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_init_alpha() {
    let bilinear = BilinearCrownLayer::new(true, None);

    // Q: [m=2, k=3], K: [n=4, k=3] (transposed)
    let alphas = bilinear.init_alpha(&[2, 3], &[4, 3]).unwrap();

    assert_eq!(
        alphas.shape(),
        &[4, 2, 4, 3],
        "Alpha shape should be [4, m, n, k] — 4 channels for direction-dependent r_l/r_u (#3287)"
    );

    // All initialized to 1.0
    assert!(
        alphas.iter().all(|&v| (v - 1.0).abs() < 1e-6),
        "All alpha values should be initialized to 1.0"
    );
}

/// Test that None alphas falls back to fixed selection (same as original method).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_with_alpha_none_fallback() {
    let bilinear = BilinearCrownLayer::new(true, Some(0.125));

    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, -0.5, -0.5, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, -0.5, -0.5, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
    )
    .unwrap();

    let downstream = identity_downstream_for_2x2_output();

    // Call with None - should produce same result as original method
    let result_with_none =
        bilinear.propagate_linear_batched_binary_with_alpha(&downstream, &input_a, &input_b, None);
    let result_original = bilinear.propagate_linear_batched_binary(&downstream, &input_a, &input_b);

    assert!(
        result_with_none.is_ok(),
        "Alpha method with None should succeed"
    );
    assert!(result_original.is_ok(), "Original method should succeed");

    let (lb_a_none, lb_b_none) = result_with_none.unwrap();
    let (lb_a_orig, lb_b_orig) = result_original.unwrap();

    // Results should be identical
    assert_eq!(lb_a_none.lower_a, lb_a_orig.lower_a, "lower_a should match");
    assert_eq!(lb_a_none.upper_a, lb_a_orig.upper_a, "upper_a should match");
    assert_eq!(lb_a_none.lower_b, lb_a_orig.lower_b, "lower_b should match");
    assert_eq!(lb_a_none.upper_b, lb_a_orig.upper_b, "upper_b should match");
    assert_eq!(
        lb_b_none.lower_a, lb_b_orig.lower_a,
        "lb_b lower_a should match"
    );
    assert_eq!(
        lb_b_none.upper_a, lb_b_orig.upper_a,
        "lb_b upper_a should match"
    );
    assert_eq!(
        lb_b_none.lower_b, lb_b_orig.lower_b,
        "lb_b lower_b should match"
    );
    assert_eq!(
        lb_b_none.upper_b, lb_b_orig.upper_b,
        "lb_b upper_b should match"
    );
}

/// Test alpha = 0 vs alpha = 1 gives different bounds (for non-symmetric inputs).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_alpha_affects_bounds() {
    use ndarray::Array4;

    let bilinear = BilinearCrownLayer::new(true, Some(1.0));

    // Asymmetric bounds to ensure different McCormick planes give different results
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 2.0, 2.0, 2.0]).unwrap(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 2.0, 2.0, 2.0]).unwrap(),
    )
    .unwrap();

    let downstream = identity_downstream_for_2x2_output();

    // Alpha = 0 (uses L2/U2 planes). Shape [4, m, n, k] (#3287).
    let alphas_zero = Array4::zeros((4, 2, 2, 2));
    let result_zero = bilinear.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &input_a,
        &input_b,
        Some(&alphas_zero),
    );

    // Alpha = 1 (uses L1/U1 planes). Shape [4, m, n, k] (#3287).
    let alphas_one = Array4::ones((4, 2, 2, 2));
    let result_one = bilinear.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &input_a,
        &input_b,
        Some(&alphas_one),
    );

    assert!(result_zero.is_ok(), "Alpha=0 should succeed");
    assert!(result_one.is_ok(), "Alpha=1 should succeed");

    let (lb_a_zero, lb_b_zero) = result_zero.unwrap();
    let (lb_a_one, lb_b_one) = result_one.unwrap();

    // For non-negative inputs [0, 2], the McCormick planes differ:
    // L1: z ≥ 0*y + 0*x - 0 = 0 (tight at origin)
    // L2: z ≥ 2*y + 2*x - 4 (tight at (2,2))
    // The bounds should differ unless inputs are perfectly symmetric
    // At minimum, coefficients should be different
    let differs = lb_a_zero.lower_a != lb_a_one.lower_a
        || lb_a_zero.upper_a != lb_a_one.upper_a
        || lb_a_zero.lower_b != lb_a_one.lower_b
        || lb_a_zero.upper_b != lb_a_one.upper_b
        || lb_b_zero.lower_a != lb_b_one.lower_a
        || lb_b_zero.upper_a != lb_b_one.upper_a
        || lb_b_zero.lower_b != lb_b_one.lower_b
        || lb_b_zero.upper_b != lb_b_one.upper_b;

    assert!(
        differs,
        "Alpha=0 and alpha=1 should produce different bounds for asymmetric inputs"
    );
}

/// Test alpha-bilinear with point bounds (lower == upper).
///
/// Point bounds collapse the McCormick relaxation to exact values regardless of alpha.
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_with_alpha_point_bounds() {
    use ndarray::Array4;

    let bilinear = BilinearCrownLayer::new(true, Some(1.0));

    // Point bounds: lower == upper (zero-width interval)
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.0, 1.5, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.0, 1.5, 2.0]).unwrap(),
    )
    .unwrap();

    let downstream = identity_downstream_for_2x2_output();

    // With point bounds, alpha value shouldn't matter. Shape [4, m, n, k] (#3287).
    let alphas_zero = Array4::zeros((4, 2, 2, 2));
    let alphas_one = Array4::ones((4, 2, 2, 2));

    let result_zero = bilinear.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &input_a,
        &input_b,
        Some(&alphas_zero),
    );
    let result_one = bilinear.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &input_a,
        &input_b,
        Some(&alphas_one),
    );

    assert!(
        result_zero.is_ok(),
        "Alpha point bounds with r=0 should succeed"
    );
    assert!(
        result_one.is_ok(),
        "Alpha point bounds with r=1 should succeed"
    );

    let (lb_a_zero, _) = result_zero.unwrap();
    let (lb_a_one, _) = result_one.unwrap();

    // For point bounds, McCormick is exact - all alpha values should give same result
    // (within floating point tolerance)
    for (v0, v1) in lb_a_zero.lower_a.iter().zip(lb_a_one.lower_a.iter()) {
        assert!(
            (v0 - v1).abs() < 1e-5,
            "Point bounds should give same lower_a regardless of alpha: {} vs {}",
            v0,
            v1
        );
    }

    // Verify no NaN or Inf
    assert!(
        !lb_a_zero
            .lower_a
            .iter()
            .any(|&v| v.is_nan() || v.is_infinite()),
        "Point bounds should not produce NaN or Inf"
    );
}
