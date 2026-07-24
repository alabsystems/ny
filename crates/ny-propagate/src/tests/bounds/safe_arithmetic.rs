// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

// --- Tests for safe_add_for_bounds (line 696) ---

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_returns_nonzero() {
    // Kills mutant: replace safe_add_for_bounds -> f32 with 0.0
    let result = safe_add_for_bounds(1.0, 2.0);
    assert_eq!(result, 3.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_returns_negative() {
    // Kills mutant: replace safe_add_for_bounds -> f32 with 1.0 or 0.0
    let result = safe_add_for_bounds(-5.0, 2.0);
    assert_eq!(result, -3.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_returns_correct_value() {
    // Kills mutant: replace safe_add_for_bounds -> f32 with -1.0
    let result = safe_add_for_bounds(0.5, 0.3);
    assert!((result - 0.8).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_inf_handling() {
    // Test that inf + finite = inf
    let result = safe_add_for_bounds(f32::INFINITY, 1.0);
    assert!(result.is_infinite() && result > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_propagates_nan() {
    let result = safe_add_for_bounds(f32::NAN, f32::INFINITY);
    assert!(result.is_nan());
}

// --- Tests for safe_array_add (lines 715) ---

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_nan_to_conservative_lower() {
    // When inf + (-inf) = NaN, should become -inf for lower bounds
    // Kills mutant: replace && with ||
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::NEG_INFINITY);
    let result = safe_array_add(&a, &b, true).unwrap(); // is_lower = true
    assert!(result[[0]].is_infinite() && result[[0]] < 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_nan_to_conservative_upper() {
    // When inf + (-inf) = NaN, should become +inf for upper bounds
    // Kills mutant: replace || with &&
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::NEG_INFINITY);
    let result = safe_array_add(&a, &b, false).unwrap(); // is_lower = false
    assert!(result[[0]].is_infinite() && result[[0]] > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_preserves_normal_values() {
    // Normal addition should work correctly
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), 1.0f32);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), 2.0f32);
    let result = safe_array_add(&a, &b, false).unwrap();
    assert!((result[[0]] - 3.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_inf_plus_finite() {
    // inf + finite should remain inf (no NaN)
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[2]), 1.0f32);
    let result = safe_array_add(&a, &b, false).unwrap();
    assert!(result[[0]].is_infinite() && result[[0]] > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_supports_broadcasting() {
    use ndarray::ArrayD;
    let a = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0f32, 2.0f32]).unwrap();
    let b = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.5f32, -0.25f32]).unwrap();

    let result = safe_array_add(&a, &b, false).unwrap();
    assert_eq!(result.shape(), &[1, 2]);
    assert!((result[[0, 0]] - 1.5).abs() < 1e-6);
    assert!((result[[0, 1]] - 1.75).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_broadcasting_with_is_lower() {
    use ndarray::ArrayD;
    // Test that is_lower=true also works with broadcasting
    let a = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3]),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.1f32, 0.2, 0.3]).unwrap();

    let result_lower = safe_array_add(&a, &b, true).unwrap();
    let result_upper = safe_array_add(&a, &b, false).unwrap();

    // Both should produce the same result for finite values
    assert_eq!(result_lower.shape(), &[2, 3]);
    assert_eq!(result_upper.shape(), &[2, 3]);
    assert!((result_lower[[0, 0]] - 1.1).abs() < 1e-6);
    assert!((result_lower[[1, 2]] - 6.3).abs() < 1e-6);
    assert_eq!(result_lower, result_upper); // Same for finite values
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_broadcasting_with_inf_nan() {
    use ndarray::ArrayD;
    // Test inf+(-inf)=NaN case with broadcasting - should become conservative bound
    let a = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![f32::INFINITY, 1.0f32]).unwrap();
    let b = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![f32::NEG_INFINITY, 2.0f32]).unwrap();

    let result_lower = safe_array_add(&a, &b, true).unwrap();
    let result_upper = safe_array_add(&a, &b, false).unwrap();

    // inf + (-inf) = NaN -> conservative bound
    assert!(result_lower[[0, 0]].is_infinite() && result_lower[[0, 0]] < 0.0); // -inf for lower
    assert!(result_upper[[0, 0]].is_infinite() && result_upper[[0, 0]] > 0.0); // +inf for upper

    // Normal addition should work
    assert!((result_lower[[0, 1]] - 3.0).abs() < 1e-6);
    assert!((result_upper[[0, 1]] - 3.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_mul_for_bounds_zero_times_inf_is_zero() {
    // Kills mutant: replace || with && in safe_mul_for_bounds
    let v = safe_mul_for_bounds(0.0, f32::INFINITY);
    assert_eq!(v, 0.0);
    assert!(!v.is_nan());
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_with_polarity_inf_plus_nan_propagates_nan() {
    // Kills mutant: replace || with && in safe_add_for_bounds_with_polarity
    let v = safe_add_for_bounds_with_polarity(f32::INFINITY, f32::NAN, false);
    assert!(v.is_nan(), "NaN input must propagate (not be sanitized)");
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_for_bounds_with_polarity_does_not_overconservatively_flip_sign() {
    // Kills mutant: replace && with || in safe_add_for_bounds_with_polarity
    let v = safe_add_for_bounds_with_polarity(f32::INFINITY, 1.0, true);
    assert!(v.is_infinite() && v > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_nan_from_nan_plus_inf_propagates_nan() {
    // Kills mutant: replace || with && in safe_array_add
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NAN);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY);
    let r = safe_array_add(&a, &b, false).unwrap();
    assert!(
        r[[0]].is_nan(),
        "NaN input must propagate (not be sanitized)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_inf_plus_finite_lower_does_not_flip_to_neg_inf() {
    // Kills mutant: replace && with || in safe_array_add
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0f32);
    let r = safe_array_add(&a, &b, true).unwrap();
    assert!(r[[0]].is_infinite() && r[[0]] > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_array_add_shape_mismatch_returns_error() {
    // Kills mutant: replace ok_or_else with unwrap in safe_array_add_checked
    use ndarray::ArrayD;
    let a = ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), 1.0f32);
    let b = ArrayD::from_elem(ndarray::IxDyn(&[4]), 2.0f32);

    let err = safe_array_add(&a, &b, false).unwrap_err();
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(got, vec![4]);
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }

    let err_checked = safe_array_add_checked(&a, &b, false).unwrap_err();
    match err_checked {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(got, vec![4]);
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_lower_for_bounds_inf_minus_inf_is_neg_inf() {
    // Kills mutant: replace is_nan with is_infinite in safe_add_lower_for_bounds
    let v = safe_add_lower_for_bounds(f32::INFINITY, f32::NEG_INFINITY);
    assert!(v.is_infinite() && v.is_sign_negative());
}

#[ntest::timeout(10000)]
#[test]
fn test_safe_add_upper_for_bounds_inf_minus_inf_is_pos_inf() {
    // Kills mutant: replace is_nan with is_infinite in safe_add_upper_for_bounds
    let v = safe_add_upper_for_bounds(f32::INFINITY, f32::NEG_INFINITY);
    assert!(v.is_infinite() && v.is_sign_positive());
}

#[ntest::timeout(10000)]
#[test]
fn test_interval_mul_for_bounds_nan_returns_unbounded() {
    // Kills mutant: remove NaN short-circuit in interval_mul_for_bounds
    let (lower, upper) = interval_mul_for_bounds(f32::NAN, 1.0, -1.0, 2.0);
    assert!(lower.is_infinite() && lower.is_sign_negative());
    assert!(upper.is_infinite() && upper.is_sign_positive());
}

#[ntest::timeout(10000)]
#[test]
fn test_interval_mul_for_bounds_all_infinite_products_returns_unbounded() {
    // Kills mutant: remove all-infinite-products guard in interval_mul_for_bounds
    let (lower, upper) = interval_mul_for_bounds(
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
    );
    assert!(lower.is_infinite() && lower.is_sign_negative());
    assert!(upper.is_infinite() && upper.is_sign_positive());
}
