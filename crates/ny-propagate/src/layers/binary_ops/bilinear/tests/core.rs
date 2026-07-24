// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::super::*;

/// Assert no NaN in bounds and bias ordering is valid.
pub(super) fn assert_bounds_no_nan_ordered(label: &str, bounds: &crate::BatchedLinearBounds) {
    assert!(
        !bounds.lower_a().iter().any(|v| v.is_nan()),
        "{label} lower_a has NaN"
    );
    assert!(
        !bounds.upper_a().iter().any(|v| v.is_nan()),
        "{label} upper_a has NaN"
    );
    assert!(
        !bounds.lower_b().iter().any(|v| v.is_nan()),
        "{label} lower_b has NaN"
    );
    assert!(
        !bounds.upper_b().iter().any(|v| v.is_nan()),
        "{label} upper_b has NaN"
    );
    for (l, u) in bounds.lower_b().iter().zip(bounds.upper_b().iter()) {
        if l.is_finite() && u.is_finite() {
            assert!(l <= u, "{label} bias lower {l} > upper {u}");
        }
    }
}

/// Assert sign-split broadcast bounds are at least as tight as interval-mul.
///
/// Tighter means: lower_b >= (higher lower bound) and upper_b <= (lower upper bound).
/// For coefficient matrices (lower_a, upper_a), sign-split values may differ
/// because the composition method is different, but the concretized bounds are tighter.
pub(super) fn assert_bounds_at_least_as_tight(
    label: &str,
    broadcast: &crate::BatchedLinearBounds,
    interval_mul: &crate::BatchedLinearBounds,
    tol: f32,
) {
    // Bias bounds: broadcast lower >= interval-mul lower (tighter lower)
    for (idx, (bc, iv)) in broadcast
        .lower_b()
        .iter()
        .zip(interval_mul.lower_b().iter())
        .enumerate()
    {
        assert!(
            *bc >= *iv - tol,
            "{label} lower_b[{idx}]: broadcast {bc} < interval-mul {iv} (should be >=)"
        );
    }
    // Bias bounds: broadcast upper <= interval-mul upper (tighter upper)
    for (idx, (bc, iv)) in broadcast
        .upper_b()
        .iter()
        .zip(interval_mul.upper_b().iter())
        .enumerate()
    {
        assert!(
            *bc <= *iv + tol,
            "{label} upper_b[{idx}]: broadcast {bc} > interval-mul {iv} (should be <=)"
        );
    }
}

/// Concretize interval bounds at a point: sum of interval_mul(A, x) + b.
pub(super) fn concretize_interval_at(
    bounds: &crate::BatchedLinearBounds,
    x: &ArrayD<f32>,
    out_idx: usize,
) -> (f64, f64) {
    let size = x.len();
    let mut lower_sum = bounds.lower_b()[out_idx] as f64;
    let mut upper_sum = bounds.upper_b()[out_idx] as f64;
    for idx in 0..size {
        let val = x[idx] as f64;
        let al = bounds.lower_a()[[out_idx, idx]] as f64;
        let au = bounds.upper_a()[[out_idx, idx]] as f64;
        lower_sum += if val >= 0.0 { al * val } else { au * val };
        upper_sum += if val >= 0.0 { au * val } else { al * val };
    }
    (lower_sum, upper_sum)
}

#[test]
fn constructor_rejects_non_finite_scale_4307() {
    let err = BilinearCrownLayer::try_new(false, Some(f32::NEG_INFINITY))
        .expect_err("non-finite Bilinear scale should be rejected");
    assert!(matches!(err, ny_core::NyError::InvalidSpec(_)));
}

/// Test interpolated_mccormick formula correctness at boundary cases.
///
/// Verifies that r=0 produces L2/U2 and r=1 produces L1/U1 planes.
#[ntest::timeout(10000)]
#[test]
fn test_interpolated_mccormick_r0_gives_l2_u2() {
    // x ∈ [1, 3], y ∈ [2, 4]
    let x_l = 1.0_f32;
    let x_u = 3.0_f32;
    let y_l = 2.0_f32;
    let y_u = 4.0_f32;

    // r = 0 should give L2/U2
    let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
        interpolated_mccormick(x_l, x_u, y_l, y_u, 0.0, 0.0);

    // L2: z ≥ y_u*x + x_u*y - x_u*y_u = 4*x + 3*y - 12
    assert!(
        (alpha_l - y_u).abs() < 1e-6,
        "L2 alpha should be y_u=4, got {}",
        alpha_l
    );
    assert!(
        (beta_l - x_u).abs() < 1e-6,
        "L2 beta should be x_u=3, got {}",
        beta_l
    );
    assert!(
        (ny_l - (-x_u * y_u)).abs() < 1e-6,
        "L2 ny should be -12, got {}",
        ny_l
    );

    // U2: z ≤ y_l*x + x_u*y - x_u*y_l = 2*x + 3*y - 6
    assert!(
        (alpha_u - y_l).abs() < 1e-6,
        "U2 alpha should be y_l=2, got {}",
        alpha_u
    );
    assert!(
        (beta_u - x_u).abs() < 1e-6,
        "U2 beta should be x_u=3, got {}",
        beta_u
    );
    assert!(
        (ny_u - (-x_u * y_l)).abs() < 1e-6,
        "U2 ny should be -6, got {}",
        ny_u
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_interpolated_mccormick_r1_gives_l1_u1() {
    // x ∈ [1, 3], y ∈ [2, 4]
    let x_l = 1.0_f32;
    let x_u = 3.0_f32;
    let y_l = 2.0_f32;
    let y_u = 4.0_f32;

    // r = 1 should give L1/U1
    let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
        interpolated_mccormick(x_l, x_u, y_l, y_u, 1.0, 1.0);

    // L1: z ≥ y_l*x + x_l*y - x_l*y_l = 2*x + 1*y - 2
    assert!(
        (alpha_l - y_l).abs() < 1e-6,
        "L1 alpha should be y_l=2, got {}",
        alpha_l
    );
    assert!(
        (beta_l - x_l).abs() < 1e-6,
        "L1 beta should be x_l=1, got {}",
        beta_l
    );
    assert!(
        (ny_l - (-x_l * y_l)).abs() < 1e-6,
        "L1 ny should be -2, got {}",
        ny_l
    );

    // U1: z ≤ y_u*x + x_l*y - x_l*y_u = 4*x + 1*y - 4
    assert!(
        (alpha_u - y_u).abs() < 1e-6,
        "U1 alpha should be y_u=4, got {}",
        alpha_u
    );
    assert!(
        (beta_u - x_l).abs() < 1e-6,
        "U1 beta should be x_l=1, got {}",
        beta_u
    );
    assert!(
        (ny_u - (-x_l * y_u)).abs() < 1e-6,
        "U1 ny should be -4, got {}",
        ny_u
    );
}

/// Test that interpolation at r=0.5 produces the midpoint coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_interpolated_mccormick_midpoint() {
    let x_l = 0.0_f32;
    let x_u = 2.0_f32;
    let y_l = 0.0_f32;
    let y_u = 2.0_f32;

    let (alpha_l, beta_l, ny_l, _, _, _) = interpolated_mccormick(x_l, x_u, y_l, y_u, 0.5, 0.5);

    // L1: coeffs (y_l=0, x_l=0, -0) = (0, 0, 0)
    // L2: coeffs (y_u=2, x_u=2, -4) = (2, 2, -4)
    // Midpoint: ((0+2)/2, (0+2)/2, (0-4)/2) = (1, 1, -2)
    assert!(
        (alpha_l - 1.0).abs() < 1e-6,
        "midpoint alpha_l should be 1, got {}",
        alpha_l
    );
    assert!(
        (beta_l - 1.0).abs() < 1e-6,
        "midpoint beta_l should be 1, got {}",
        beta_l
    );
    assert!(
        (ny_l - (-2.0)).abs() < 1e-6,
        "midpoint ny_l should be -2, got {}",
        ny_l
    );
}

/// Test soundness: for any point in the input box, the true z = x*y must satisfy
/// computed_lower ≤ z ≤ computed_upper.
#[ntest::timeout(10000)]
#[test]
fn test_interpolated_mccormick_soundness() {
    let x_l = -1.0_f32;
    let x_u = 2.0_f32;
    let y_l = -1.0_f32;
    let y_u = 3.0_f32;

    // Test with various r values
    for &r in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
            interpolated_mccormick(x_l, x_u, y_l, y_u, r, r);

        // Check soundness at corners and midpoints
        let test_points = [
            (x_l, y_l),
            (x_l, y_u),
            (x_u, y_l),
            (x_u, y_u),
            (f32::midpoint(x_l, x_u), f32::midpoint(y_l, y_u)),
        ];

        for &(x, y) in &test_points {
            let z_true = x * y;
            let z_lower = alpha_l * x + beta_l * y + ny_l;
            let z_upper = alpha_u * x + beta_u * y + ny_u;

            assert!(
                z_lower <= z_true + 1e-5,
                "Soundness: lower={} should be <= true z={} at ({}, {}), r={}",
                z_lower,
                z_true,
                x,
                y,
                r
            );
            assert!(
                z_true <= z_upper + 1e-5,
                "Soundness: true z={} should be <= upper={} at ({}, {}), r={}",
                z_true,
                z_upper,
                x,
                y,
                r
            );
        }
    }
}

/// Regression test: zero batch dimension in `propagate_linear_batched_binary`
/// must return an error, not panic with `% 0`. (#2819)
#[ntest::timeout(10000)]
#[test]
fn test_batched_binary_zero_batch_dim_returns_error() {
    let layer = BilinearCrownLayer::new(false, None);
    // Shape [0, 2, 3]: batch=0, m=2, k=3
    let a_lower = ArrayD::zeros(IxDyn(&[0, 2, 3]));
    let a_upper = ArrayD::zeros(IxDyn(&[0, 2, 3]));
    let a_bounds = BoundedTensor::new(a_lower, a_upper).unwrap();
    // Shape [0, 3, 4]: batch=0, k=3, n=4
    let b_lower = ArrayD::zeros(IxDyn(&[0, 3, 4]));
    let b_upper = ArrayD::zeros(IxDyn(&[0, 3, 4]));
    let b_bounds = BoundedTensor::new(b_lower, b_upper).unwrap();

    // Output shape would be [m, n] = [2, 4], z_size = 8
    let downstream = crate::BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = layer.propagate_linear_batched_binary(&downstream, &a_bounds, &b_bounds);
    assert!(
        result.is_err(),
        "Expected error for zero batch dimension, got Ok"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("zero-valued batch dimension"),
        "Expected zero-batch error, got: {err_msg}"
    );
}

/// Regression test: zero batch dimension in alpha-parameterized variant
/// must return an error, not panic with `% 0`. (#2819)
#[ntest::timeout(10000)]
#[test]
fn test_batched_binary_with_alpha_zero_batch_dim_returns_error() {
    let layer = BilinearCrownLayer::new(false, None);
    // Shape [0, 2, 3]: batch=0, m=2, k=3
    let a_lower = ArrayD::zeros(IxDyn(&[0, 2, 3]));
    let a_upper = ArrayD::zeros(IxDyn(&[0, 2, 3]));
    let a_bounds = BoundedTensor::new(a_lower, a_upper).unwrap();
    // Shape [0, 3, 4]: batch=0, k=3, n=4
    let b_lower = ArrayD::zeros(IxDyn(&[0, 3, 4]));
    let b_upper = ArrayD::zeros(IxDyn(&[0, 3, 4]));
    let b_bounds = BoundedTensor::new(b_lower, b_upper).unwrap();

    let downstream = crate::BatchedLinearBounds::identity(&[2, 4]).unwrap();
    // Alpha shape: [4, m, n, k] = [4, 2, 4, 3] — direction-dependent r_l/r_u (#3287)
    let alphas = ndarray::Array4::ones((4, 2, 4, 3));

    let result = layer.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &a_bounds,
        &b_bounds,
        Some(&alphas),
    );
    assert!(
        result.is_err(),
        "Expected error for zero batch dimension, got Ok"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("zero-valued batch dimension"),
        "Expected zero-batch error, got: {err_msg}"
    );
}

/// Regression test: NaN in input bounds must propagate through batch-reduced
/// global intervals, NOT be absorbed by f32::min/max. (#3120)
///
/// Before the fix, `q_l_global.min(NaN)` returned `q_l_global` (NaN absorbed),
/// producing clean-looking McCormick coefficients from corrupted intervals.
/// After the fix, `nan_propagating_min` ensures NaN propagates through the
/// entire CROWN backward computation.
#[ntest::timeout(10000)]
#[test]
fn test_batched_binary_nan_input_bounds_propagate_3120() {
    let layer = BilinearCrownLayer::new(false, None);
    // Shape [2, 2, 3]: batch=2, m=2, k=3
    // Batch 0: valid bounds, Batch 1: NaN in lower bound
    let mut a_lower = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 1.0_f32);
    let a_upper = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 3.0_f32);
    // Inject NaN into batch 1, position [0, 0]
    a_lower[[1, 0, 0]] = f32::NAN;

    let a_bounds = BoundedTensor::new_unchecked(a_lower, a_upper).unwrap();

    // Shape [2, 3, 4]: batch=2, k=3, n=4
    let b_lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 2.0_f32);
    let b_upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 4.0_f32);
    let b_bounds = BoundedTensor::new(b_lower, b_upper).unwrap();

    // Output shape: [m, n] = [2, 4], z_size = 8
    let downstream = crate::BatchedLinearBounds::identity(&[2, 4]).unwrap();

    let result = layer.propagate_linear_batched_binary(&downstream, &a_bounds, &b_bounds);

    // The result should either be an error (NaN detected early) or
    // contain NaN in the output bounds (NaN propagated through McCormick).
    // It must NOT produce clean, finite bounds from NaN input.
    match result {
        Err(_) => {
            // Early NaN detection is acceptable
        }
        Ok((bounds_a, _bounds_b)) => {
            // If it returns Ok, the McCormick coefficients and bias must
            // contain NaN for the affected positions (i=0, l=0).
            let has_nan = bounds_a.lower_a.iter().any(|v| v.is_nan())
                || bounds_a.upper_a.iter().any(|v| v.is_nan())
                || bounds_a.lower_b.iter().any(|v| v.is_nan())
                || bounds_a.upper_b.iter().any(|v| v.is_nan());
            assert!(
                has_nan,
                "NaN in input A bounds must propagate to output bounds, not be silently absorbed"
            );
        }
    }
}

/// Same as above but for the alpha-parameterized variant. (#3120)
#[ntest::timeout(10000)]
#[test]
fn test_batched_binary_with_alpha_nan_input_bounds_propagate_3120() {
    let layer = BilinearCrownLayer::new(false, None);
    // Shape [2, 2, 3]: batch=2, m=2, k=3
    let mut a_lower = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 1.0_f32);
    let a_upper = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 3.0_f32);
    // Inject NaN into batch 1, position [0, 0]
    a_lower[[1, 0, 0]] = f32::NAN;
    let a_bounds = BoundedTensor::new_unchecked(a_lower, a_upper).unwrap();

    // Shape [2, 3, 4]: batch=2, k=3, n=4
    let b_lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 2.0_f32);
    let b_upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 4.0_f32);
    let b_bounds = BoundedTensor::new(b_lower, b_upper).unwrap();

    let downstream = crate::BatchedLinearBounds::identity(&[2, 4]).unwrap();
    // Alpha shape: [4, m, n, k] = [4, 2, 4, 3] — direction-dependent r_l/r_u (#3287)
    let alphas = ndarray::Array4::ones((4, 2, 4, 3));

    let result = layer.propagate_linear_batched_binary_with_alpha(
        &downstream,
        &a_bounds,
        &b_bounds,
        Some(&alphas),
    );

    match result {
        Err(_) => {
            // Early NaN detection is acceptable
        }
        Ok((bounds_a, _bounds_b)) => {
            let has_nan = bounds_a.lower_a.iter().any(|v| v.is_nan())
                || bounds_a.upper_a.iter().any(|v| v.is_nan())
                || bounds_a.lower_b.iter().any(|v| v.is_nan())
                || bounds_a.upper_b.iter().any(|v| v.is_nan());
            assert!(
                has_nan,
                "NaN in input A bounds must propagate to output bounds (alpha path), not be silently absorbed"
            );
        }
    }
}

/// Test that batched downstream (from identity_for_attention) composes correctly
/// with the flat McCormick matrices via tile_to_batch.
///
/// For attention with shape [batch=1, heads=2, seq=2, seq=2]:
/// - identity_for_attention produces [1, 2, 4, 4] (flat=seq*seq=4)
/// - McCormick produces flat [z_size, q_size] = [4, q_size]
/// - tile_to_batch lifts McCormick to [1, 2, 4, q_size]
/// - broadcast composition succeeds with matching batch dims
#[ntest::timeout(10000)]
#[test]
fn test_batched_attention_identity_full_composition() {
    let layer = BilinearCrownLayer::new(true, None);

    // Attention inputs: Q=[1, 2, 2, 3], K=[1, 2, 2, 3] (batch=1, heads=2, seq=2, head_dim=3)
    let a_lower = ArrayD::from_elem(IxDyn(&[1, 2, 2, 3]), -1.0_f32);
    let a_upper = ArrayD::from_elem(IxDyn(&[1, 2, 2, 3]), 1.0_f32);
    let a_bounds = BoundedTensor::new(a_lower, a_upper).unwrap();

    let b_lower = ArrayD::from_elem(IxDyn(&[1, 2, 2, 3]), -1.0_f32);
    let b_upper = ArrayD::from_elem(IxDyn(&[1, 2, 2, 3]), 1.0_f32);
    let b_bounds = BoundedTensor::new(b_lower, b_upper).unwrap();

    // Downstream from identity_for_attention: [1, 2, 4, 4] where 4 = seq*seq
    let downstream = crate::BatchedLinearBounds::identity_for_attention(&[1, 2, 2, 2]).unwrap();
    assert_eq!(
        downstream.lower_a().shape(),
        &[1, 2, 4, 4],
        "identity should have shape [batch, heads, flat, flat]"
    );

    // This should succeed now: tile_to_batch lifts McCormick to [1, 2, z_size, q_size]
    let result = layer.propagate_linear_batched_binary(&downstream, &a_bounds, &b_bounds);
    assert!(
        result.is_ok(),
        "Batched attention composition should succeed, got: {:?}",
        result.err()
    );

    let (bounds_a, bounds_b) = result.unwrap();

    // bounds_a should map from Q-space through attention to downstream spec
    // With identity downstream, the composed bounds should have matching batch dims [1, 2]
    let a_ndim = bounds_a.lower_a().ndim();
    assert!(
        a_ndim >= 2,
        "Result bounds_a should be at least 2D, got {}D",
        a_ndim
    );

    // The composed output dimension should match the downstream output dim (flat=4)
    // and the input dimension should match Q's flattened space (q_size = m*k = 2*3 = 6)
    let a_shape = bounds_a.lower_a().shape();
    let out_dim = a_shape[a_ndim - 2];
    let in_dim = a_shape[a_ndim - 1];
    assert_eq!(out_dim, 4, "Output dim should be flat=seq*seq=4");
    assert_eq!(in_dim, 6, "Input dim should be q_size=m*k=2*3=6");

    // bounds_b should map from K-space with transpose_b=true: k_size = n*k = 2*3 = 6
    let b_shape = bounds_b.lower_a().shape();
    let b_ndim = bounds_b.lower_a().ndim();
    let b_in_dim = b_shape[b_ndim - 1];
    assert_eq!(b_in_dim, 6, "K input dim should be k_size=n*k=2*3=6");

    // Verify no NaN in output (inputs are clean)
    assert!(
        !bounds_a.lower_a().iter().any(|v| v.is_nan()),
        "bounds_a lower_a should not contain NaN"
    );
    assert!(
        !bounds_b.lower_a().iter().any(|v| v.is_nan()),
        "bounds_b lower_a should not contain NaN"
    );
}

/// Test that tile_to_batch with empty batch shape is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_tile_to_batch_empty_is_noop() {
    let bounds = crate::BatchedLinearBounds::identity(&[4]).unwrap();
    let tiled = bounds.tile_to_batch(&[]).unwrap();
    assert_eq!(tiled.lower_a().shape(), bounds.lower_a().shape());
}

/// Test that tile_to_batch correctly replicates 2D matrices along batch dims.
#[ntest::timeout(10000)]
#[test]
fn test_tile_to_batch_replication() {
    let bounds = crate::BatchedLinearBounds::identity(&[3]).unwrap();
    let tiled = bounds.tile_to_batch(&[2, 4]).unwrap();

    // Should be [2, 4, 3, 3]
    assert_eq!(tiled.lower_a().shape(), &[2, 4, 3, 3]);
    assert_eq!(tiled.upper_a().shape(), &[2, 4, 3, 3]);
    assert_eq!(tiled.lower_b().shape(), &[2, 4, 3]);
    assert_eq!(tiled.upper_b().shape(), &[2, 4, 3]);

    // Each batch position should be the identity matrix
    for b0 in 0..2 {
        for b1 in 0..4 {
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_eq!(
                        tiled.lower_a()[[b0, b1, i, j]],
                        expected,
                        "lower_a[{b0},{b1},{i},{j}] should be {expected}"
                    );
                }
            }
        }
    }
}

/// Test McCormick broadcast backward at production scale (seq=65).
/// Verifies shapes are correct, no NaN, valid bias ordering.
#[ntest::timeout(60000)]
#[test]
fn test_mccormick_production_threshold() {
    let (m, n, k) = (65, 65, 4); // z_size=4225, production-scale attention
    let z_size = m * n;
    let out_dim = 3;

    let q_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[m, k]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[m, k]), 1.0_f32),
    )
    .unwrap();
    let k_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[n, k]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[n, k]), 0.5_f32),
    )
    .unwrap();

    let downstream = crate::BatchedLinearBounds::new(
        ArrayD::from_elem(IxDyn(&[out_dim, z_size]), 0.1_f32),
        ArrayD::from_elem(IxDyn(&[out_dim]), -0.1_f32),
        ArrayD::from_elem(IxDyn(&[out_dim, z_size]), 0.3_f32),
        ArrayD::from_elem(IxDyn(&[out_dim]), 0.1_f32),
        vec![z_size],
        vec![out_dim],
    )
    .unwrap();

    // Routes through broadcast McCormick composition
    let layer = BilinearCrownLayer::new(true, Some(0.125));
    let (bq, bk) = layer
        .propagate_linear_batched_binary(&downstream, &q_bounds, &k_bounds)
        .expect("McCormick at production threshold should succeed");

    // Verify shapes: Q=[out_dim, m*k], K=[out_dim, n*k]
    assert_eq!(bq.lower_a().shape(), &[out_dim, m * k]);
    assert_eq!(bk.lower_a().shape(), &[out_dim, n * k]);

    // Verify no NaN in coefficients
    for (label, bounds) in [("Q", &bq), ("K", &bk)] {
        assert!(
            !bounds.lower_a().iter().any(|v| v.is_nan()),
            "{label} lower_a NaN"
        );
        assert!(
            !bounds.upper_a().iter().any(|v| v.is_nan()),
            "{label} upper_a NaN"
        );
        for (l, u) in bounds.lower_b().iter().zip(bounds.upper_b().iter()) {
            assert!(l <= u || l.is_nan() || u.is_nan(), "{label} bias {l} > {u}");
        }
    }
}

/// Soundness: BilinearRelaxation bounds contain true matmul output.
#[test]
fn test_bilinear_relaxation_soundness_contains_true_output() {
    let (m, n, k, scale) = (3, 4, 2, 0.5_f32);

    let q_lower =
        ArrayD::from_shape_vec(IxDyn(&[m, k]), vec![-1.0, 0.0, -0.5, 0.5, 0.2, -0.3]).unwrap();
    let q_upper =
        ArrayD::from_shape_vec(IxDyn(&[m, k]), vec![0.5, 1.0, 0.5, 1.5, 0.8, 0.7]).unwrap();
    let q_bounds = BoundedTensor::new(q_lower.clone(), q_upper.clone()).unwrap();

    let k_lower = ArrayD::from_shape_vec(
        IxDyn(&[n, k]),
        vec![-0.5, -1.0, 0.0, 0.5, -0.3, 0.2, 0.1, -0.4],
    )
    .unwrap();
    let k_upper =
        ArrayD::from_shape_vec(IxDyn(&[n, k]), vec![0.5, 0.0, 1.0, 1.5, 0.7, 1.2, 1.1, 0.6])
            .unwrap();
    let k_bounds = BoundedTensor::new(k_lower.clone(), k_upper.clone()).unwrap();

    let z_size = m * n;
    let eye_vec = ndarray::Array2::<f32>::eye(z_size)
        .into_raw_vec_and_offset()
        .0;
    let relaxation =
        BilinearRelaxation::from_bounds(&q_bounds, &k_bounds, true, Some(scale)).unwrap();
    let downstream = crate::BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[z_size, z_size]), eye_vec.clone()).unwrap(),
        ArrayD::zeros(IxDyn(&[z_size])),
        ArrayD::from_shape_vec(IxDyn(&[z_size, z_size]), eye_vec).unwrap(),
        ArrayD::zeros(IxDyn(&[z_size])),
        vec![z_size],
        vec![z_size],
    )
    .unwrap();

    let (bq, bk) = relaxation.compose_backward(&downstream).unwrap();

    // Verify bounds contain true Q@K^T at interval midpoint.
    let q_mid = (&q_lower + &q_upper).mapv(|v| v * 0.5);
    let k_mid = (&k_lower + &k_upper).mapv(|v| v * 0.5);
    let q_2d = q_mid.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let k_2d = k_mid.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let true_output = q_2d.dot(&k_2d.t()) * scale;
    let q_flat = q_mid.into_shape_with_order(IxDyn(&[m * k])).unwrap();
    let k_flat = k_mid.into_shape_with_order(IxDyn(&[n * k])).unwrap();

    for o in 0..(m * n) {
        let (q_lo, q_up) = concretize_interval_at(&bq, &q_flat, o);
        let (k_lo, k_up) = concretize_interval_at(&bk, &k_flat, o);
        let true_val = true_output[[o / n, o % n]] as f64;
        assert!(q_lo + k_lo <= true_val + 1e-4, "lower > true at {o}");
        assert!(q_up + k_up >= true_val - 1e-4, "upper < true at {o}");
    }
}
