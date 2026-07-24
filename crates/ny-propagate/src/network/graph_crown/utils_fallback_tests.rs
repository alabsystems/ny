// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Direct tests for `GraphNetwork::partial_crown_fallback` (#3265).
//!
//! This function is the last line of defense in graph CROWN backward
//! propagation. When a layer's CROWN backward fails or produces non-finite
//! coefficients, it concretizes accumulated linear bounds using IBP bounds.
//! All three code paths are tested:
//! 1. Normal path: both inputs finite → concretize_sound()
//! 2. Sanitized path: either input non-finite → sanitize_bounds_for_fallback()
//! 3. Reshape path: output shape differs → reshape result

use super::GraphNetwork;
use crate::bounds::BatchedLinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Helper: create a 1D `BoundedTensor` from slices.
fn bounded_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower shape valid"),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper shape valid"),
    )
    .expect("bounds valid")
}

/// Helper: create a 1D `BoundedTensor` allowing infinite values.
fn bounded_1d_allow_inf(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new_allow_infinite(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower shape valid"),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper shape valid"),
    )
    .expect("bounds valid")
}

/// #3265 acceptance criterion 1: Normal path — identity bounds + finite IBP.
///
/// With identity linear bounds (A=I, b=0), concretize_sound should return
/// bounds equal to the IBP bounds (with 1-ULP directed rounding widening).
/// The result must soundly contain the true output for any input in the
/// IBP range.
#[test]
fn test_partial_crown_fallback_normal_path_identity_3265() {
    let node_lb = BatchedLinearBounds::identity(&[3]).expect("identity should succeed");
    let ibp_bounds = bounded_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);

    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[3])
        .expect("fallback should succeed");

    // Identity concretization: result ≈ IBP bounds (with ULP widening).
    assert_eq!(result.shape(), &[3]);
    for i in 0..3 {
        let lb = result.lower()[[i]];
        let ub = result.upper()[[i]];
        let ibp_lb = ibp_bounds.lower()[[i]];
        let ibp_ub = ibp_bounds.upper()[[i]];

        // Sound containment: result bounds must contain IBP bounds
        // (concretize_sound widens by 1 ULP, so result_lb <= ibp_lb).
        assert!(
            lb <= ibp_lb,
            "dim {i}: lower bound {lb} must be <= IBP lower {ibp_lb} (sound widening)"
        );
        assert!(
            ub >= ibp_ub,
            "dim {i}: upper bound {ub} must be >= IBP upper {ibp_ub} (sound widening)"
        );

        // Bounds should be close to IBP (within a few ULPs, not degraded to Inf).
        assert!(
            lb.is_finite(),
            "dim {i}: lower bound should be finite, got {lb}"
        );
        assert!(
            ub.is_finite(),
            "dim {i}: upper bound should be finite, got {ub}"
        );
    }
}

/// #3265 acceptance criterion 1 extended: non-identity linear bounds produce
/// tighter-than-IBP bounds when the linear relationship is well-conditioned.
///
/// Linear bounds: y_0 = 2*x_0 + 1, y_1 = 3*x_1 - 1
/// Input x in [1, 4] x [2, 5]
/// Expected: y_0 in [3, 9], y_1 in [5, 14]
#[test]
fn test_partial_crown_fallback_normal_path_scaling_3265() {
    // A = [[2, 0], [0, 3]], b_lower = [1, -1], b_upper = [1, -1]
    let lower_a =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 0.0, 0.0, 3.0]).expect("shape valid");
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).expect("shape valid");
    let upper_a = lower_a.clone();
    let upper_b = lower_b.clone();

    let node_lb = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        vec![2],
        vec![2],
    );

    let ibp_bounds = bounded_1d(&[1.0, 2.0], &[4.0, 5.0]);

    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[2])
        .expect("fallback should succeed");

    assert_eq!(result.shape(), &[2]);

    // y_0 = 2*x_0 + 1: range [2*1+1, 2*4+1] = [3, 9]
    let lb0 = result.lower()[[0]];
    let ub0 = result.upper()[[0]];
    assert!(lb0 <= 3.0, "y_0 lower should be <= 3.0 (sound), got {lb0}");
    assert!(ub0 >= 9.0, "y_0 upper should be >= 9.0 (sound), got {ub0}");
    assert!(lb0.is_finite(), "y_0 lower should be finite, got {lb0}");
    assert!(ub0.is_finite(), "y_0 upper should be finite, got {ub0}");

    // y_1 = 3*x_1 - 1: range [3*2-1, 3*5-1] = [5, 14]
    let lb1 = result.lower()[[1]];
    let ub1 = result.upper()[[1]];
    assert!(lb1 <= 5.0, "y_1 lower should be <= 5.0 (sound), got {lb1}");
    assert!(
        ub1 >= 14.0,
        "y_1 upper should be >= 14.0 (sound), got {ub1}"
    );
    assert!(lb1.is_finite(), "y_1 lower should be finite, got {lb1}");
    assert!(ub1.is_finite(), "y_1 upper should be finite, got {ub1}");
}

/// #3265 acceptance criterion 2: Sanitized path — IBP bounds with Inf.
///
/// When IBP bounds contain Inf, partial_crown_fallback must sanitize
/// (not concretize), and the result must be valid (no NaN, no inversions).
#[test]
fn test_partial_crown_fallback_sanitized_ibp_inf_3265() {
    let node_lb = BatchedLinearBounds::identity(&[3]).expect("identity should succeed");
    let ibp_bounds =
        bounded_1d_allow_inf(&[1.0, f32::NEG_INFINITY, 3.0], &[4.0, f32::INFINITY, 6.0]);

    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[3])
        .expect("fallback should succeed");

    assert_eq!(result.shape(), &[3]);

    // No NaN in result.
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "sanitized result lower must not contain NaN"
    );
    assert!(
        !result.upper().iter().any(|v| v.is_nan()),
        "sanitized result upper must not contain NaN"
    );

    // No inversions: lower <= upper everywhere.
    for i in 0..3 {
        let lb = result.lower()[[i]];
        let ub = result.upper()[[i]];
        assert!(
            lb <= ub,
            "dim {i}: lower {lb} must be <= upper {ub} (no inversion)"
        );
    }

    // Finite elements should be preserved (or widened).
    // Element 0: IBP=[1,4] → should be [1,4] or wider (sanitize keeps finite values).
    let lb0 = result.lower()[[0]];
    let ub0 = result.upper()[[0]];
    assert!(
        lb0 <= 1.0,
        "dim 0: sanitized lower should be <= 1.0, got {lb0}"
    );
    assert!(
        ub0 >= 4.0,
        "dim 0: sanitized upper should be >= 4.0, got {ub0}"
    );

    // Element 1: IBP=[-Inf, +Inf] → should remain [-Inf, +Inf].
    assert_eq!(
        result.lower()[[1]],
        f32::NEG_INFINITY,
        "dim 1: Inf lower should stay -Inf"
    );
    assert_eq!(
        result.upper()[[1]],
        f32::INFINITY,
        "dim 1: Inf upper should stay +Inf"
    );
}

/// #3265 acceptance criterion 2 extended: node_lb with Inf coefficients,
/// output shape matches IBP shape.
#[test]
fn test_partial_crown_fallback_lb_inf_triggers_sanitized_path_3265() {
    // BatchedLinearBounds with one Inf coefficient in lower_a.
    let lower_a = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![1.0, f32::INFINITY, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0],
    )
    .expect("shape valid");
    let lower_b = ArrayD::zeros(IxDyn(&[3]));
    let upper_a = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![1.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0],
    )
    .expect("shape valid");
    let upper_b = ArrayD::zeros(IxDyn(&[3]));

    let node_lb = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        vec![3],
        vec![3],
    );

    let ibp_bounds = bounded_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);

    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[3])
        .expect("fallback should succeed even with Inf coefficients");

    assert_eq!(result.shape(), &[3]);

    // No NaN.
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "result must not contain NaN"
    );

    // No inversions.
    for i in 0..3 {
        let lb = result.lower()[[i]];
        let ub = result.upper()[[i]];
        assert!(lb <= ub, "dim {i}: lower {lb} must be <= upper {ub}");
    }

    // Sanitized path returns IBP bounds (possibly with Inf for non-finite).
    // Since IBP was all finite, the sanitized result equals the IBP bounds.
    for i in 0..3 {
        let lb = result.lower()[[i]];
        let ub = result.upper()[[i]];
        assert!(
            lb <= ibp_bounds.lower()[[i]],
            "dim {i}: sanitized lower {lb} should be <= IBP lower {} (containment)",
            ibp_bounds.lower()[[i]]
        );
        assert!(
            ub >= ibp_bounds.upper()[[i]],
            "dim {i}: sanitized upper {ub} should be >= IBP upper {} (containment)",
            ibp_bounds.upper()[[i]]
        );
    }
}

/// #3265: Inf in bias of node_lb also triggers sanitized path.
#[test]
fn test_partial_crown_fallback_lb_inf_bias_triggers_sanitized_3265() {
    let lower_a =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("shape valid");
    let lower_b =
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, f32::NEG_INFINITY]).expect("shape valid");
    let upper_a = lower_a.clone();
    let upper_b =
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, f32::INFINITY]).expect("shape valid");

    let node_lb = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        vec![2],
        vec![2],
    );

    let ibp_bounds = bounded_1d(&[1.0, 2.0], &[3.0, 4.0]);

    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[2])
        .expect("fallback should succeed with Inf bias");

    assert_eq!(result.shape(), &[2]);
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "no NaN in result"
    );
    assert!(
        !result.upper().iter().any(|v| v.is_nan()),
        "no NaN in result"
    );
    for i in 0..2 {
        assert!(
            result.lower()[[i]] <= result.upper()[[i]],
            "dim {i}: no inversion"
        );
    }
}

/// #3265 acceptance criterion 3: Reshape path — output shape differs from
/// concretized shape.
///
/// When the concretized result has a different shape than output_shape,
/// partial_crown_fallback reshapes it. This tests the reshape path with
/// compatible element counts.
#[test]
fn test_partial_crown_fallback_reshape_path_3265() {
    // Create identity bounds for shape [6] (flat).
    let node_lb = BatchedLinearBounds::identity(&[6]).expect("identity should succeed");
    let ibp_bounds = bounded_1d(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );

    // Request output shape [2, 3] — same element count (6), different shape.
    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[2, 3])
        .expect("fallback with reshape should succeed");

    // Shape must match requested output_shape.
    assert_eq!(
        result.shape(),
        &[2, 3],
        "result shape should be [2, 3] after reshape"
    );

    // All elements should be finite (identity + finite IBP).
    assert!(
        result.lower().iter().all(|v| v.is_finite()),
        "all lower bounds should be finite"
    );
    assert!(
        result.upper().iter().all(|v| v.is_finite()),
        "all upper bounds should be finite"
    );

    // Soundness: reshaped bounds should still contain the original IBP
    // values (just in a different shape).
    let flat_lower: Vec<f32> = result.lower().iter().copied().collect();
    let flat_upper: Vec<f32> = result.upper().iter().copied().collect();
    let ibp_lower: Vec<f32> = ibp_bounds.lower().iter().copied().collect();
    let ibp_upper: Vec<f32> = ibp_bounds.upper().iter().copied().collect();
    for i in 0..6 {
        assert!(
            flat_lower[i] <= ibp_lower[i],
            "element {i}: reshaped lower {} must be <= IBP lower {} (sound)",
            flat_lower[i],
            ibp_lower[i]
        );
        assert!(
            flat_upper[i] >= ibp_upper[i],
            "element {i}: reshaped upper {} must be >= IBP upper {} (sound)",
            flat_upper[i],
            ibp_upper[i]
        );
    }
}

/// Helper: create finite 2x2 identity-like BatchedLinearBounds for non-finite tests.
fn finite_2x2_bounds() -> BatchedLinearBounds {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).expect("shape");
    BatchedLinearBounds::from_parts_unchecked(a.clone(), b.clone(), a, b, vec![2], vec![2])
}

/// #3265: all-finite coefficients are not flagged.
#[test]
fn test_has_non_finite_coefficients_clean_3265() {
    assert!(
        !GraphNetwork::has_non_finite_coefficients(&finite_2x2_bounds()),
        "all-finite bounds should return false"
    );
}

/// #3265: Inf in lower_a is detected.
#[test]
fn test_has_non_finite_coefficients_inf_lower_a_3265() {
    let a =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::INFINITY, 0.0, 0.0, 1.0]).expect("shape");
    let b = ArrayD::zeros(IxDyn(&[2]));
    let fa = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
    let lb = BatchedLinearBounds::from_parts_unchecked(a, b.clone(), fa, b, vec![2], vec![2]);
    assert!(
        GraphNetwork::has_non_finite_coefficients(&lb),
        "Inf in lower_a"
    );
}

/// #3265: Inf in upper_a is detected.
#[test]
fn test_has_non_finite_coefficients_inf_upper_a_3265() {
    let fa = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
    let b = ArrayD::zeros(IxDyn(&[2]));
    let ua = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, f32::NEG_INFINITY])
        .expect("shape");
    let lb = BatchedLinearBounds::from_parts_unchecked(fa, b.clone(), ua, b, vec![2], vec![2]);
    assert!(
        GraphNetwork::has_non_finite_coefficients(&lb),
        "Inf in upper_a"
    );
}

/// #3265: Inf in lower_b is detected.
#[test]
fn test_has_non_finite_coefficients_inf_lower_b_3265() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
    let inf_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, f32::INFINITY]).expect("shape");
    let b = ArrayD::zeros(IxDyn(&[2]));
    let lb = BatchedLinearBounds::from_parts_unchecked(a.clone(), inf_b, a, b, vec![2], vec![2]);
    assert!(
        GraphNetwork::has_non_finite_coefficients(&lb),
        "Inf in lower_b"
    );
}

/// #3265: Inf in upper_b is detected.
#[test]
fn test_has_non_finite_coefficients_inf_upper_b_3265() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
    let b = ArrayD::zeros(IxDyn(&[2]));
    let inf_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).expect("shape");
    let lb = BatchedLinearBounds::from_parts_unchecked(a.clone(), b, a, inf_b, vec![2], vec![2]);
    assert!(
        GraphNetwork::has_non_finite_coefficients(&lb),
        "Inf in upper_b"
    );
}

/// #3265 acceptance criterion 4 (optional): partial CROWN with non-trivial
/// linear bounds produces tighter bounds than pure IBP for the supported
/// portion.
///
/// When linear bounds represent y = 0.5*x + 1 (slope 0.5, not identity),
/// the concretized CROWN bounds should be tighter than just using IBP
/// bounds on the output (which would be the raw pre-computed IBP).
#[test]
fn test_partial_crown_tighter_than_identity_3265() {
    // Linear bounds: y = 0.5*x + 1 (contracting map).
    // Input x in [0, 10] → y in [1, 6].
    // IBP at output (from the network's IBP pass) might be wider if the
    // network has nonlinearities that IBP over-approximates.
    let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.5]).expect("shape valid");
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).expect("shape valid");
    let upper_a = lower_a.clone();
    let upper_b = lower_b.clone();

    let node_lb = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        vec![1],
        vec![1],
    );

    let ibp_bounds = bounded_1d(&[0.0], &[10.0]);

    let result = GraphNetwork::partial_crown_fallback(&node_lb, &ibp_bounds, &[1])
        .expect("fallback should succeed");

    // Expected: y = 0.5*x + 1, x in [0, 10] → y in [1, 6].
    let lb = result.lower()[[0]];
    let ub = result.upper()[[0]];

    assert!(lb <= 1.0, "lower should be <= 1.0 (sound), got {lb}");
    assert!(ub >= 6.0, "upper should be >= 6.0 (sound), got {ub}");

    // Tightness: bounds should be close to [1, 6], not [-inf, +inf].
    assert!(lb > -1.0, "lower should be tighter than -1.0, got {lb}");
    assert!(ub < 8.0, "upper should be tighter than 8.0, got {ub}");
}
