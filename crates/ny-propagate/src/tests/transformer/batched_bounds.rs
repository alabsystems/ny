// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BatchedLinearBounds construction and composition tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity() {
    // Test identity bounds for 1D shape
    let bounds = BatchedLinearBounds::identity(&[4]).unwrap();
    assert_eq!(bounds.lower_a.shape(), &[4, 4]);
    assert_eq!(bounds.lower_b.shape(), &[4]);

    // Check it's identity
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((bounds.lower_a[[i, j]] - expected).abs() < 1e-6);
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_2d() {
    // Test identity bounds for 2D shape (batch, hidden)
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();
    assert_eq!(bounds.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(bounds.lower_b.shape(), &[2, 4]);

    // Check each batch position has identity
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (bounds.lower_a[[b, i, j]] - expected).abs() < 1e-6,
                    "lower_a[{}, {}, {}] = {}, expected {}",
                    b,
                    i,
                    j,
                    bounds.lower_a[[b, i, j]],
                    expected
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_3d() {
    // Test identity bounds for 3D shape (batch, seq, hidden)
    let bounds = BatchedLinearBounds::identity(&[1, 4, 8]).unwrap();
    assert_eq!(bounds.lower_a.shape(), &[1, 4, 8, 8]);
    assert_eq!(bounds.lower_b.shape(), &[1, 4, 8]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_concretize_identity() {
    // Identity bounds should return input unchanged
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap(),
    )
    .unwrap();

    let output = bounds.concretize(&input).unwrap();

    // Output should equal input for identity bounds
    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (output.lower()[[i, j]] - input.lower()[[i, j]]).abs() < 1e-5,
                "lower[{}, {}] mismatch",
                i,
                j
            );
            assert!(
                (output.upper()[[i, j]] - input.upper()[[i, j]]).abs() < 1e-5,
                "upper[{}, {}] mismatch",
                i,
                j
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_concretize_allows_vector_like_reshape() {
    // Allow vector-like reshape when element counts match.
    let bounds = BatchedLinearBounds::identity(&[8]).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 8]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 8]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap(),
    )
    .unwrap();

    let output = bounds.concretize(&input).unwrap();
    assert_eq!(output.lower().shape(), &[8]);
    assert_eq!(output.upper().shape(), &[8]);

    for (idx, (low, high)) in output.lower().iter().zip(output.upper().iter()).enumerate() {
        assert!(low <= high, "output bounds inverted at {}", idx);
    }

    for (idx, (out, inp)) in output.lower().iter().zip(input.lower().iter()).enumerate() {
        assert!((out - inp).abs() < 1e-5, "lower[{}] mismatch", idx);
    }

    for (idx, (out, inp)) in output.upper().iter().zip(input.upper().iter()).enumerate() {
        assert!((out - inp).abs() < 1e-5, "upper[{}] mismatch", idx);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_concretize_shape_mismatch() {
    let bounds = BatchedLinearBounds::identity(&[2, 4]).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();

    let err = bounds.concretize(&input).unwrap_err();
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_concretize_bias_shape_mismatch() {
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::zeros(IxDyn(&[2, 4, 4])),
        ArrayD::zeros(IxDyn(&[3])),
        ArrayD::zeros(IxDyn(&[2, 4, 4])),
        ArrayD::zeros(IxDyn(&[3])),
        vec![2, 4],
        vec![2, 4],
    );
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();

    let err = bounds.concretize(&input).unwrap_err();
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention() {
    // Test identity_for_attention for attention-shaped output
    // Attention output shape: [batch=1, heads=2, seq=4, seq=4]
    let shape = [1_usize, 2, 4, 4];
    let bounds = BatchedLinearBounds::identity_for_attention(&shape);

    // Should return Some for small attention shapes
    assert!(
        bounds.is_some(),
        "identity_for_attention should succeed for small seq"
    );
    let bounds = bounds.unwrap();

    // A shape should be [batch=1, heads=2, flat_size=16, flat_size=16]
    let flat_size = 4 * 4;
    assert_eq!(
        bounds.lower_a.shape(),
        &[1, 2, flat_size, flat_size],
        "lower_a shape mismatch"
    );
    assert_eq!(
        bounds.lower_b.shape(),
        &[1, 2, flat_size],
        "lower_b shape mismatch"
    );

    // Check identity structure per head
    for b in 0..1 {
        for h in 0..2 {
            for i in 0..flat_size {
                for j in 0..flat_size {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (bounds.lower_a[[b, h, i, j]] - expected).abs() < 1e-6,
                        "lower_a[{}, {}, {}, {}] = {}, expected {}",
                        b,
                        h,
                        i,
                        j,
                        bounds.lower_a[[b, h, i, j]],
                        expected
                    );
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_identity_for_attention_rejects_large_seq() {
    // For seq > 64 (flat_size > 4096), should return None to avoid memory issues
    // seq=65 gives flat_size=4225 > 4096
    let shape = [1_usize, 1, 65, 65];
    let bounds = BatchedLinearBounds::identity_for_attention(&shape);
    assert!(
        bounds.is_none(),
        "identity_for_attention should reject seq > 64"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_identity_for_attention_rejects_non_square() {
    // Non-square attention output should return None
    let shape = [1_usize, 2, 4, 8];
    let bounds = BatchedLinearBounds::identity_for_attention(&shape);
    assert!(
        bounds.is_none(),
        "identity_for_attention should reject non-square attention"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_identity_for_attention_rejects_wrong_dims() {
    // Non-4D shapes should return None
    let shape_3d = [1_usize, 4, 4];
    assert!(
        BatchedLinearBounds::identity_for_attention(&shape_3d).is_none(),
        "Should reject 3D shape"
    );

    let shape_5d = [1_usize, 1, 2, 4, 4];
    assert!(
        BatchedLinearBounds::identity_for_attention(&shape_5d).is_none(),
        "Should reject 5D shape"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_identity() {
    // Composing two identity bounds should give identity bounds
    let shape = [2_usize, 4]; // batch=2, dim=4
    let id1 = BatchedLinearBounds::identity(&shape).unwrap();
    let id2 = BatchedLinearBounds::identity(&shape).unwrap();

    let composed = id1
        .compose(&id2)
        .expect("Compose should succeed for compatible identities");

    // Check that composed coefficient matrices are identity-like
    // A_composed = I @ I = I
    let expected_a_shape = [2, 4, 4]; // [batch, out_dim, in_dim]
    assert_eq!(composed.lower_a.shape(), expected_a_shape);
    assert_eq!(composed.upper_a.shape(), expected_a_shape);

    // Each batch should have identity matrix
    for b in 0..2 {
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got_lower = composed.lower_a[[b, i, j]];
                let got_upper = composed.upper_a[[b, i, j]];
                assert!(
                    (got_lower - expected).abs() < 1e-6,
                    "Expected lower[{},{},{}] = {}, got {}",
                    b,
                    i,
                    j,
                    expected,
                    got_lower
                );
                assert!(
                    (got_upper - expected).abs() < 1e-6,
                    "Expected upper[{},{},{}] = {}, got {}",
                    b,
                    i,
                    j,
                    expected,
                    got_upper
                );
            }
        }
    }

    // Bias should be zero
    for val in composed.lower_b.iter() {
        assert!(val.abs() < 1e-6, "Expected zero bias, got {}", val);
    }
    for val in composed.upper_b.iter() {
        assert!(val.abs() < 1e-6, "Expected zero bias, got {}", val);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_scale() {
    // Composing a 2x scale with a 3x scale should give 6x scale
    let batch = 1;
    let dim = 2;

    // Create 2x scale bounds: y = 2*x
    let mut eye2: Array2<f32> = Array2::eye(dim);
    eye2.mapv_inplace(|v| v * 2.0);
    let scale_2 = BatchedLinearBounds::from_parts_unchecked(
        eye2.clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        eye2.into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        vec![batch, dim],
        vec![batch, dim],
    );

    // Create 3x scale bounds: z = 3*y
    let mut eye3: Array2<f32> = Array2::eye(dim);
    eye3.mapv_inplace(|v| v * 3.0);
    let scale_3 = BatchedLinearBounds::from_parts_unchecked(
        eye3.clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        eye3.into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        vec![batch, dim],
        vec![batch, dim],
    );

    // Compose: z = 3 * (2 * x) = 6 * x
    let composed = scale_2.compose(&scale_3).expect("Compose should succeed");

    // Check that result is 6x identity
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 6.0 } else { 0.0 };
            let got = composed.lower_a[[0, i, j]];
            assert!(
                (got - expected).abs() < 1e-5,
                "Expected composed[{},{}] = {}, got {}",
                i,
                j,
                expected,
                got
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_with_bias() {
    // Test that bias composition works: z = A2(A1*x + b1) + b2 = A2*A1*x + A2*b1 + b2
    let batch = 1;
    let dim = 2;

    // y = 2*x + [1, 2]
    let mut a1: Array2<f32> = Array2::eye(dim);
    a1.mapv_inplace(|v| v * 2.0);
    let bounds1 = BatchedLinearBounds::from_parts_unchecked(
        a1.clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![1.0, 2.0]).unwrap(),
        a1.into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![1.0, 2.0]).unwrap(),
        vec![batch, dim],
        vec![batch, dim],
    );

    // z = y + [3, 4] (identity transform with bias)
    let eye: Array2<f32> = Array2::eye(dim);
    let bounds2 = BatchedLinearBounds::from_parts_unchecked(
        eye.clone()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![3.0, 4.0]).unwrap(),
        eye.into_dyn()
            .into_shape_with_order(IxDyn(&[batch, dim, dim]))
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim]), vec![3.0, 4.0]).unwrap(),
        vec![batch, dim],
        vec![batch, dim],
    );

    // z = (2*x + [1,2]) + [3,4] = 2*x + [4,6]
    let composed = bounds1.compose(&bounds2).expect("Compose should succeed");

    // Check coefficient matrix is 2*I
    assert!((composed.lower_a[[0, 0, 0]] - 2.0).abs() < 1e-5);
    assert!((composed.lower_a[[0, 1, 1]] - 2.0).abs() < 1e-5);
    assert!(composed.lower_a[[0, 0, 1]].abs() < 1e-5);
    assert!(composed.lower_a[[0, 1, 0]].abs() < 1e-5);

    // Check bias is [4, 6] (b1=[1,2] passed through identity A2, plus b2=[3,4])
    assert!(
        (composed.lower_b[[0, 0]] - 4.0).abs() < 1e-5,
        "Expected bias[0] = 4, got {}",
        composed.lower_b[[0, 0]]
    );
    assert!(
        (composed.lower_b[[0, 1]] - 6.0).abs() < 1e-5,
        "Expected bias[1] = 6, got {}",
        composed.lower_b[[0, 1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_avoids_nan_from_0_times_inf() {
    // Regression test: saturated coefficients (±inf) can appear in bounds, and composing
    // bounds must not introduce NaNs via 0 * inf.
    let batch = 1;
    let dim = 2;

    // y = A1 @ x where A1 contains +inf on the diagonal (synthetic saturation)
    let a1 = vec![
        f32::INFINITY,
        0.0, //
        0.0,
        f32::INFINITY, //
    ];
    let bounds1 = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1.clone()).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        vec![batch, dim],
        vec![batch, dim],
    );

    // z = A2 @ y with A2 = 0 (should produce z = 0 regardless of A1)
    let a2 = vec![
        0.0, 0.0, //
        0.0, 0.0, //
    ];
    let bounds2 = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2.clone()).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        vec![batch, dim],
        vec![batch, dim],
    );

    let composed = bounds1.compose(&bounds2).expect("Compose should succeed");

    for v in composed
        .lower_a
        .iter()
        .chain(composed.upper_a.iter())
        .chain(composed.lower_b.iter())
        .chain(composed.upper_b.iter())
    {
        assert!(!v.is_nan(), "compose produced NaN");
    }

    // Coefficients should be near zero (0 * inf treated as 0).
    // compose() applies directed rounding: next_down_f32(0.0) = -1e-45 for lower,
    // next_up_f32(0.0) = 1e-45 for upper. This is the sound 1-ULP widening.
    for v in composed.lower_a.iter() {
        assert!(v.abs() < 1e-30, "lower_a should be ~0, got {}", v);
        assert!(*v <= 0.0, "lower_a should be <= 0 (sound), got {}", v);
    }
    for v in composed.upper_a.iter() {
        assert!(v.abs() < 1e-30, "upper_a should be ~0, got {}", v);
        assert!(*v >= 0.0, "upper_a should be >= 0 (sound), got {}", v);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_avoids_nan_from_inf_minus_inf_sum() {
    // Regression test: interval sums like (+inf) + (-inf) must widen, not become NaN.
    let batch = 1;
    let dim = 2;

    // y = A1 @ x where two rows contribute +inf and -inf to the same output when composed.
    // A1:
    //   [ +inf, 0 ]
    //   [ -inf, 0 ]
    let a1_lower = vec![
        f32::INFINITY,
        0.0, //
        f32::NEG_INFINITY,
        0.0, //
    ];
    let a1_upper = vec![
        f32::INFINITY,
        0.0, //
        f32::NEG_INFINITY,
        0.0, //
    ];
    let bounds1 = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1_lower).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a1_upper).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        vec![batch, dim],
        vec![batch, dim],
    );

    // z = A2 @ y with A2[0,:] = [1, 1], so z0 = y0 + y1 includes +inf + (-inf).
    let a2 = vec![
        1.0, 1.0, //
        0.0, 0.0, //
    ];
    let bounds2 = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2.clone()).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        ArrayD::from_shape_vec(IxDyn(&[batch, dim, dim]), a2).unwrap(),
        ArrayD::zeros(IxDyn(&[batch, dim])),
        vec![batch, dim],
        vec![batch, dim],
    );

    let composed = bounds1.compose(&bounds2).expect("Compose should succeed");

    for v in composed
        .lower_a
        .iter()
        .chain(composed.upper_a.iter())
        .chain(composed.lower_b.iter())
        .chain(composed.upper_b.iter())
    {
        assert!(!v.is_nan(), "compose produced NaN");
    }

    assert!(
        composed.lower_a[[0, 0, 0]].is_infinite() && composed.lower_a[[0, 0, 0]].is_sign_negative(),
        "Expected widened lower=-inf, got {}",
        composed.lower_a[[0, 0, 0]]
    );
    assert!(
        composed.upper_a[[0, 0, 0]].is_infinite() && composed.upper_a[[0, 0, 0]].is_sign_positive(),
        "Expected widened upper=+inf, got {}",
        composed.upper_a[[0, 0, 0]]
    );
}
