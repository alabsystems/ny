// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for LinearBounds.

use super::{checked_bounds, unchecked_bounds};
use crate::bounds::LinearBounds;
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::{next_down_f32, next_up_f32};
use std::mem::size_of;
use std::time::{Duration, Instant};

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_identity() {
    let bounds = LinearBounds::identity(3);

    // Check identity matrix
    assert_eq!(bounds.lower_a.nrows(), 3);
    assert_eq!(bounds.lower_a.ncols(), 3);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(bounds.lower_a[[i, j]], expected);
            assert_eq!(bounds.upper_a[[i, j]], expected);
        }
    }

    // Check zero bias
    assert_eq!(bounds.lower_b, Array1::<f32>::zeros(3));
    assert_eq!(bounds.upper_b, Array1::<f32>::zeros(3));
}

#[ntest::timeout(5000)]
#[test]
fn deadline_aware_identity_matches_legacy_and_refuses_atomically() {
    let expected = LinearBounds::identity(3);
    let live = crate::tests::with_crown_dense_budget_mb("2048", || {
        LinearBounds::try_identity_with_deadline(
            3,
            Some(Instant::now() + Duration::from_secs(30)),
            137,
        )
        .expect("live finite identity seed")
    });
    assert_linear_bounds_exact(&live, &expected);

    let expired = LinearBounds::try_identity_with_deadline(
        3,
        Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("one second fits before now"),
        ),
        0,
    )
    .expect_err("expired identity seed must be terminal before allocation");
    assert!(expired.is_deadline_exceeded());

    let memory = crate::tests::with_crown_dense_budget_mb("0", || {
        LinearBounds::try_identity_with_deadline(
            3,
            Some(Instant::now() + Duration::from_secs(30)),
            0,
        )
        .expect_err("finite identity seed must obey the dense budget")
    });
    assert!(matches!(memory, NyError::CpuMemoryExceeded { .. }));

    let legacy = crate::tests::with_crown_dense_budget_mb("0", || {
        LinearBounds::try_identity_with_deadline(3, None, usize::MAX)
            .expect("None must delegate exactly to legacy identity")
    });
    assert_linear_bounds_exact(&legacy, &expected);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_num_outputs_inputs() {
    let bounds = LinearBounds {
        lower_a: Array2::zeros((5, 3)),
        lower_b: Array1::zeros(5),
        upper_a: Array2::zeros((5, 3)),
        upper_b: Array1::zeros(5),
        lower_a_err: None,
        upper_a_err: None,
    };
    assert_eq!(bounds.num_outputs(), 5);
    assert_eq!(bounds.num_inputs(), 3);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_memory_bytes_includes_coefficient_errors() {
    let mut bounds = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: Array1::zeros(2),
        upper_a: Array2::zeros((2, 3)),
        upper_b: Array1::zeros(2),
        lower_a_err: None,
        upper_a_err: None,
    };

    // 12 coefficients plus 4 biases, all f32.
    assert_eq!(bounds.memory_bytes(), 16 * size_of::<f32>());

    bounds.lower_a_err = Some(Array2::zeros((2, 3)));
    assert_eq!(
        bounds.memory_bytes(),
        22 * size_of::<f32>(),
        "the lower coefficient-error allocation must be counted"
    );

    bounds.lower_a_err = None;
    bounds.upper_a_err = Some(Array2::zeros((2, 3)));
    assert_eq!(
        bounds.memory_bytes(),
        22 * size_of::<f32>(),
        "the upper coefficient-error allocation must be counted independently"
    );

    bounds.lower_a_err = Some(Array2::zeros((2, 3)));
    assert_eq!(
        bounds.memory_bytes(),
        28 * size_of::<f32>(),
        "both coefficient-error allocations must be counted"
    );
}

fn beta_split_deadline_fixture() -> LinearBounds {
    LinearBounds {
        lower_a: array![[0.25, -1.0], [3.5, 0.125], [-2.0, 4.0]],
        lower_b: array![0.5, -0.25, 1.0],
        upper_a: array![[0.75, 2.0], [-1.5, 0.5], [3.0, -4.0]],
        upper_b: array![1.5, 0.25, -1.0],
        lower_a_err: Some(Array2::from_elem((3, 2), 1.0e-6)),
        upper_a_err: Some(Array2::from_elem((3, 2), 2.0e-6)),
    }
}

fn assert_linear_bounds_exact(actual: &LinearBounds, expected: &LinearBounds) {
    assert_eq!(actual.lower_a, expected.lower_a);
    assert_eq!(actual.lower_b, expected.lower_b);
    assert_eq!(actual.upper_a, expected.upper_a);
    assert_eq!(actual.upper_b, expected.upper_b);
    assert_eq!(actual.lower_a_err, expected.lower_a_err);
    assert_eq!(actual.upper_a_err, expected.upper_a_err);
}

#[ntest::timeout(5000)]
#[test]
fn beta_split_live_deadline_matches_legacy_exactly() {
    let mut expected = beta_split_deadline_fixture();
    let mut actual = expected.clone();
    expected.apply_beta_split_to_column(1, 0.375);

    actual
        .apply_beta_split_to_column_with_deadline(
            1,
            0.375,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect("live deadline beta mutation");

    assert_linear_bounds_exact(&actual, &expected);
}

#[ntest::timeout(5000)]
#[test]
fn beta_split_expired_deadline_is_atomic() {
    let mut actual = beta_split_deadline_fixture();
    let expected = actual.clone();
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("one-second deadline subtraction");

    let error = actual
        .apply_beta_split_to_column_with_deadline(1, 0.375, Some(expired))
        .expect_err("expired beta mutation must be terminal");

    assert!(error.is_deadline_exceeded());
    assert_linear_bounds_exact(&actual, &expected);
}

#[ntest::timeout(5000)]
#[test]
fn beta_split_none_keeps_the_in_place_legacy_path() {
    let mut actual = beta_split_deadline_fixture();
    let mut expected = actual.clone();
    let lower_ptr = actual.lower_a.as_ptr();
    let upper_ptr = actual.upper_a.as_ptr();
    expected.apply_beta_split_to_column(0, -0.25);

    actual
        .apply_beta_split_to_column_with_deadline(0, -0.25, None)
        .expect("no-deadline beta mutation");

    assert_eq!(actual.lower_a.as_ptr(), lower_ptr);
    assert_eq!(actual.upper_a.as_ptr(), upper_ptr);
    assert_linear_bounds_exact(&actual, &expected);
}

#[ntest::timeout(5000)]
#[test]
fn deadline_aware_linear_clone_is_exact_atomic_and_memory_bounded() {
    let mut bounds = LinearBounds {
        lower_a: array![[1.0_f32, -2.0], [3.0, -4.0]],
        lower_b: array![0.25_f32, -0.5],
        upper_a: array![[5.0_f32, -6.0], [7.0, -8.0]],
        upper_b: array![0.75_f32, -1.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    bounds.lower_a_err = Some(array![[0.0_f32, 0.125], [0.25, 0.5]]);
    bounds.upper_a_err = Some(array![[0.75_f32, 1.0], [1.25, 1.5]]);

    let cloned = crate::tests::with_crown_dense_budget_mb("2048", || {
        bounds
            .try_clone_with_deadline(Some(Instant::now() + Duration::from_secs(1)), 137)
            .unwrap()
    });
    assert_eq!(cloned.lower_a, bounds.lower_a);
    assert_eq!(cloned.lower_b, bounds.lower_b);
    assert_eq!(cloned.upper_a, bounds.upper_a);
    assert_eq!(cloned.upper_b, bounds.upper_b);
    assert_eq!(cloned.lower_a_err, bounds.lower_a_err);
    assert_eq!(cloned.upper_a_err, bounds.upper_a_err);

    let lower_only = crate::tests::with_crown_dense_budget_mb("2048", || {
        bounds.try_clone_lower_a_with_deadline(None, 137).unwrap()
    });
    assert_eq!(lower_only, bounds.lower_a);

    let before = format!("{bounds:?}");
    let expired = bounds
        .try_clone_with_deadline(
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("one millisecond fits before now"),
            ),
            0,
        )
        .unwrap_err();
    assert!(matches!(expired, NyError::DeadlineExceeded(_)));
    assert_eq!(format!("{bounds:?}"), before);

    let memory = crate::tests::with_crown_dense_budget_mb("0", || {
        bounds.try_clone_with_deadline(None, 0).unwrap_err()
    });
    assert!(matches!(memory, NyError::CpuMemoryExceeded { .. }));
    assert_eq!(format!("{bounds:?}"), before);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_new_rejects_malformed_shapes() {
    let err = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        array![0.0],
        Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap(),
        array![0.0, 0.0],
    )
    .unwrap_err();

    let msg = err.to_string();
    assert!(
        msg.contains("invariant violated"),
        "expected invariant violation error, got: {msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_identity_preserves_input() {
    let identity = LinearBounds::identity(4);

    // Input bounds: [1, 2, 3, 4] to [5, 6, 7, 8]
    let input = checked_bounds(
        array![1.0_f32, 2.0, 3.0, 4.0].into_dyn(),
        array![5.0_f32, 6.0, 7.0, 8.0].into_dyn(),
    );

    let output = identity.concretize(&input);

    // Identity should preserve bounds exactly
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[5.0, 6.0, 7.0, 8.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_handles_non_standard_layout_input() {
    let identity = LinearBounds::identity(6);
    let lower = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .permuted_axes([1, 0])
        .into_dyn();
    let upper = array![[1.5_f32, 2.5, 3.5], [4.5, 5.5, 6.5]]
        .permuted_axes([1, 0])
        .into_dyn();
    let input = checked_bounds(lower, upper);

    let output = identity.concretize(&input);

    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[1.5, 4.5, 2.5, 5.5, 3.5, 6.5]
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_with_positive_coeffs() {
    // y = 2*x, bounds [1, 3]
    let bounds = LinearBounds {
        lower_a: Array2::from_elem((1, 1), 2.0),
        lower_b: Array1::zeros(1),
        upper_a: Array2::from_elem((1, 1), 2.0),
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = checked_bounds(array![1.0_f32].into_dyn(), array![3.0_f32].into_dyn());

    let output = bounds.concretize(&input);

    // y = 2*x with x in [1,3] gives y in [2, 6]
    assert!((output.lower()[[0]] - 2.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 6.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_with_negative_coeffs() {
    // y = -2*x, bounds x in [1, 3]
    let bounds = LinearBounds {
        lower_a: Array2::from_elem((1, 1), -2.0),
        lower_b: Array1::zeros(1),
        upper_a: Array2::from_elem((1, 1), -2.0),
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = checked_bounds(array![1.0_f32].into_dyn(), array![3.0_f32].into_dyn());

    let output = bounds.concretize(&input);

    // y = -2*x with x in [1,3] gives y in [-6, -2]
    assert!((output.lower()[[0]] - (-6.0)).abs() < 1e-6);
    assert!((output.upper()[[0]] - (-2.0)).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_with_bias() {
    // y = x + 10
    let bounds = LinearBounds {
        lower_a: Array2::from_elem((1, 1), 1.0),
        lower_b: array![10.0_f32],
        upper_a: Array2::from_elem((1, 1), 1.0),
        upper_b: array![10.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = checked_bounds(array![1.0_f32].into_dyn(), array![3.0_f32].into_dyn());

    let output = bounds.concretize(&input);

    // y = x + 10 with x in [1,3] gives y in [11, 13]
    assert!((output.lower()[[0]] - 11.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 13.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_mixed_coeffs() {
    // y = x1 - x2, with x1 in [1,5], x2 in [2,4]
    // lower(y) = min(x1) - max(x2) = 1 - 4 = -3
    // upper(y) = max(x1) - min(x2) = 5 - 2 = 3
    let bounds = LinearBounds {
        lower_a: array![[1.0_f32, -1.0]],
        lower_b: array![0.0_f32],
        upper_a: array![[1.0_f32, -1.0]],
        upper_b: array![0.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = checked_bounds(
        array![1.0_f32, 2.0].into_dyn(),
        array![5.0_f32, 4.0].into_dyn(),
    );

    let output = bounds.concretize(&input);

    assert!((output.lower()[[0]] - (-3.0)).abs() < 1e-6);
    assert!((output.upper()[[0]] - 3.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_with_inf_and_zero_coeff() {
    // Coefficient is 0, input is infinite - should produce 0, not NaN
    let bounds = LinearBounds {
        lower_a: array![[0.0_f32]],
        lower_b: array![5.0_f32],
        upper_a: array![[0.0_f32]],
        upper_b: array![5.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = unchecked_bounds(
        array![f32::NEG_INFINITY].into_dyn(),
        array![f32::INFINITY].into_dyn(),
    );

    let output = bounds.concretize(&input);

    // 0 * inf = 0, so output = bias = 5
    assert_eq!(output.lower()[[0]], 5.0);
    assert_eq!(output.upper()[[0]], 5.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_malformed_shapes_fail_closed() {
    // Malformed internal bounds (upper has 2 outputs while lower has 1).
    // concretize should fail closed to conservative [-inf, +inf] bounds.
    let bounds = LinearBounds {
        lower_a: array![[1.0_f32]],
        lower_b: array![0.0_f32],
        upper_a: array![[1.0_f32], [2.0_f32]],
        upper_b: array![0.0_f32, 0.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = checked_bounds(array![0.0_f32].into_dyn(), array![1.0_f32].into_dyn());
    let output = bounds.concretize(&input);

    assert_eq!(output.shape(), &[1]);
    assert_eq!(output.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(output.upper()[[0]], f32::INFINITY);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_checked_malformed_shapes_returns_err() {
    let bounds = LinearBounds {
        lower_a: array![[1.0_f32]],
        lower_b: array![0.0_f32],
        upper_a: array![[1.0_f32], [2.0_f32]],
        upper_b: array![0.0_f32, 0.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(array![0.0_f32].into_dyn(), array![1.0_f32].into_dyn());

    let err = bounds.concretize_checked(&input).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("invariant violated"),
        "expected invariant violation error, got: {msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_basic() {
    // Identity transformation with L2 ball
    let bounds = LinearBounds::identity(2);
    let x_hat = array![1.0_f32, 2.0];
    let rho = 0.5;

    let result = bounds.concretize_l2_ball(&x_hat, rho).unwrap();

    // For identity A=I, ||a||_2 = 1 for each row
    // lower = x_hat - rho * 1 = [0.5, 1.5]
    // upper = x_hat + rho * 1 = [1.5, 2.5]
    let lower = result.lower().as_slice().unwrap();
    let upper = result.upper().as_slice().unwrap();

    assert!((lower[0] - 0.5).abs() < 1e-5);
    assert!((lower[1] - 1.5).abs() < 1e-5);
    assert!((upper[0] - 1.5).abs() < 1e-5);
    assert!((upper[1] - 2.5).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_zero_radius() {
    let bounds = LinearBounds::identity(2);
    let x_hat = array![3.0_f32, 4.0];

    let result = bounds.concretize_l2_ball(&x_hat, 0.0).unwrap();

    // Zero radius: bounds should equal x_hat exactly
    let lower = result.lower().as_slice().unwrap();
    let upper = result.upper().as_slice().unwrap();

    assert!((lower[0] - 3.0).abs() < 1e-5);
    assert!((lower[1] - 4.0).abs() < 1e-5);
    assert!((upper[0] - 3.0).abs() < 1e-5);
    assert!((upper[1] - 4.0).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_negative_rho() {
    let bounds = LinearBounds::identity(2);
    let x_hat = array![1.0_f32, 2.0];

    let result = bounds.concretize_l2_ball(&x_hat, -1.0);
    assert!(result.is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_shape_mismatch() {
    let bounds = LinearBounds::identity(3);
    let x_hat = array![1.0_f32, 2.0]; // Wrong size

    let result = bounds.concretize_l2_ball(&x_hat, 1.0);
    assert!(result.is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_with_transform() {
    // y = 2*x, x_hat = [1], rho = 0.5
    // ||a||_2 = 2
    // lower = 2*1 - 0.5*2 = 1
    // upper = 2*1 + 0.5*2 = 3
    let bounds = LinearBounds {
        lower_a: array![[2.0_f32]],
        lower_b: array![0.0_f32],
        upper_a: array![[2.0_f32]],
        upper_b: array![0.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };

    let x_hat = array![1.0_f32];
    let result = bounds.concretize_l2_ball(&x_hat, 0.5).unwrap();

    let lower = result.lower().as_slice().unwrap();
    let upper = result.upper().as_slice().unwrap();

    assert!((lower[0] - 1.0).abs() < 1e-5);
    assert!((upper[0] - 3.0).abs() < 1e-5);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_large_matrix() {
    // Test with a larger matrix to ensure no indexing issues
    let n = 100;
    let bounds = LinearBounds::identity(n);

    let input = checked_bounds(
        ArrayD::from_elem(IxDyn(&[n]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[n]), 1.0_f32),
    );

    let output = bounds.concretize(&input);

    assert_eq!(output.lower().shape(), &[n]);
    assert_eq!(output.upper().shape(), &[n]);
}

#[ntest::timeout(5000)]
#[test]
fn test_linear_bounds_single_element() {
    let bounds = LinearBounds::identity(1);

    let input = checked_bounds(array![5.0_f32].into_dyn(), array![10.0_f32].into_dyn());

    let output = bounds.concretize(&input);

    assert_eq!(output.lower()[[0]], 5.0);
    assert_eq!(output.upper()[[0]], 10.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_widens_bounds() {
    // concretize_sound() should return bounds at least as wide as concretize()
    let lower_a = Array2::from_shape_vec((2, 3), vec![1.0, -0.5, 0.3, -0.2, 0.8, -0.1]).unwrap();
    let upper_a = Array2::from_shape_vec((2, 3), vec![1.2, -0.3, 0.5, 0.0, 1.0, 0.1]).unwrap();
    let lower_b = array![0.1, -0.2];
    let upper_b = array![0.3, 0.1];
    let bounds = LinearBounds {
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        array![-1.0_f32, 0.5, -0.5].into_dyn(),
        array![1.0_f32, 1.5, 0.5].into_dyn(),
    );

    let normal = bounds.concretize(&input);
    let sound = bounds.concretize_sound(&input);

    // Sound lower bounds must be <= normal lower bounds (widened toward -inf)
    for i in 0..2 {
        assert!(
            sound.lower()[[i]] <= normal.lower()[[i]],
            "sound lower[{i}]={} should be <= normal lower[{i}]={}",
            sound.lower()[[i]],
            normal.lower()[[i]]
        );
    }
    // Sound upper bounds must be >= normal upper bounds (widened toward +inf)
    for i in 0..2 {
        assert!(
            sound.upper()[[i]] >= normal.upper()[[i]],
            "sound upper[{i}]={} should be >= normal upper[{i}]={}",
            sound.upper()[[i]],
            normal.upper()[[i]]
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_concretize_l2_ball_directed_rounding() {
    let bounds = LinearBounds::identity(2);
    let x_hat = array![0.5_f32, -0.3];
    let rho = 0.1_f32;

    let result = bounds.concretize_l2_ball(&x_hat, rho).unwrap();

    // For identity bounds the mathematical extrema are x_hat[i] ± rho. Check
    // against those exact binary64 sums, not against a fixed ULP count: the
    // certified dot/norm reductions may already have moved their binary64
    // enclosure outward before the final directed binary32 conversion.
    for i in 0..2 {
        let exact_lower = f64::from(x_hat[i]) - f64::from(rho);
        let exact_upper = f64::from(x_hat[i]) + f64::from(rho);
        assert!(
            f64::from(result.lower()[[i]]) <= exact_lower,
            "l2 lower[{i}]={} should enclose exact {exact_lower:e}",
            result.lower()[[i]],
        );
        assert!(
            f64::from(result.upper()[[i]]) >= exact_upper,
            "l2 upper[{i}]={} should enclose exact {exact_upper:e}",
            result.upper()[[i]],
        );
    }
}

/// concretize_l2_ball should repair inverted bounds (lower > upper) to [-inf, +inf]
/// instead of propagating as Err. This matches the per-element inversion repair in
/// concretize_sound, ensuring SDP-CROWN doesn't hard-fail on numerical instability.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_l2_ball_repairs_inverted_bounds() {
    // Craft LinearBounds where lower_a coefficients give a larger result than upper_a,
    // simulating CROWN backward producing inconsistent coefficient matrices.
    // With x_hat=[0], rho=0.1:
    //   lower = lower_a*x_hat + lower_b - rho*||lower_a|| = 0 + 10 - 0.1*1 = 9.9
    //   upper = upper_a*x_hat + upper_b + rho*||upper_a|| = 0 + 1 + 0.1*1 = 1.1
    // So lower(9.9) > upper(1.1) — an inversion.
    let bounds = LinearBounds {
        lower_a: array![[1.0_f32]],
        lower_b: array![10.0_f32],
        upper_a: array![[1.0_f32]],
        upper_b: array![1.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };
    let x_hat = array![0.0_f32];
    let rho = 0.1_f32;

    // Before the fix, this would return Err from BoundedTensor::new_allow_infinite.
    // After the fix, it should return Ok with the inverted element widened to [-inf, +inf].
    let result = bounds.concretize_l2_ball(&x_hat, rho).unwrap();
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// A malformed linear proof object must fail closed before L2 concretization.
/// Returning a partially finite result for a NaN coefficient would make it too
/// easy for a caller to treat corrupted proof state as an ordinary enclosure.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_l2_ball_nan_coefficients_fail_closed() {
    let bounds = LinearBounds {
        lower_a: array![[f32::NAN]],
        lower_b: array![0.0_f32],
        upper_a: array![[1.0_f32]],
        upper_b: array![0.0_f32],
        lower_a_err: None,
        upper_a_err: None,
    };
    let x_hat = array![1.0_f32];
    let rho = 0.5_f32;

    assert!(bounds.concretize_l2_ball(&x_hat, rho).is_err());
}

/// concretize_checked should return Ok with sound results when dimensions match.
/// #2239: concretize_checked now delegates to concretize_sound (directed rounding),
/// so lower bounds may be 1 ULP below and upper bounds 1 ULP above exact values.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_checked_success() {
    let bounds = LinearBounds::identity(3);
    let input = checked_bounds(
        array![1.0_f32, 2.0, 3.0].into_dyn(),
        array![4.0_f32, 5.0, 6.0].into_dyn(),
    );

    let result = bounds.concretize_checked(&input).unwrap();
    // Sound: lower bounds must be ≤ true values, upper bounds must be ≥ true values.
    for (i, &expected_lo) in [1.0_f32, 2.0, 3.0].iter().enumerate() {
        assert!(
            result.lower()[[i]] <= expected_lo,
            "lower[{i}] should be <= {expected_lo}"
        );
    }
    for (i, &expected_hi) in [4.0_f32, 5.0, 6.0].iter().enumerate() {
        assert!(
            result.upper()[[i]] >= expected_hi,
            "upper[{i}] should be >= {expected_hi}"
        );
    }
}

/// concretize_checked should return Err(ShapeMismatch) when input dim != num_inputs.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_checked_dimension_mismatch() {
    let bounds = LinearBounds::identity(3); // expects 3 inputs
    let input = checked_bounds(
        array![1.0_f32, 2.0].into_dyn(), // only 2 elements
        array![3.0_f32, 4.0].into_dyn(),
    );

    let result = bounds.concretize_checked(&input);
    assert!(result.is_err(), "should error on dimension mismatch");
}

/// concretize_checked should agree with concretize_sound when dimensions match.
/// #2239: concretize_checked now delegates to concretize_sound, not concretize.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_checked_matches_concretize_sound() {
    let lower_a = Array2::from_shape_vec((2, 3), vec![1.0, -0.5, 0.3, -0.2, 0.8, -0.1]).unwrap();
    let upper_a = Array2::from_shape_vec((2, 3), vec![1.2, -0.3, 0.5, 0.0, 1.0, 0.1]).unwrap();
    let bounds = LinearBounds {
        lower_a,
        lower_b: array![0.1, -0.2],
        upper_a,
        upper_b: array![0.3, 0.1],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        array![-1.0_f32, 0.5, -0.5].into_dyn(),
        array![1.0_f32, 1.5, 0.5].into_dyn(),
    );

    let checked = bounds.concretize_checked(&input).unwrap();
    let sound = bounds.concretize_sound(&input);

    assert_eq!(
        checked.lower().as_slice().unwrap(),
        sound.lower().as_slice().unwrap(),
    );
    assert_eq!(
        checked.upper().as_slice().unwrap(),
        sound.upper().as_slice().unwrap(),
    );
}

/// concretize_sound should widen bounds even on identity (1 ULP).
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_identity_widens_by_ulp() {
    let bounds = LinearBounds::identity(2);
    let input = checked_bounds(
        array![1.0_f32, -3.0].into_dyn(),
        array![2.0_f32, -1.0].into_dyn(),
    );

    let normal = bounds.concretize(&input);
    let sound = bounds.concretize_sound(&input);

    // Identity concretize is exact; sound version must be 1 ULP wider.
    assert_eq!(sound.lower()[[0]], next_down_f32(normal.lower()[[0]]));
    assert_eq!(sound.lower()[[1]], next_down_f32(normal.lower()[[1]]));
    assert_eq!(sound.upper()[[0]], next_up_f32(normal.upper()[[0]]));
    assert_eq!(sound.upper()[[1]], next_up_f32(normal.upper()[[1]]));
}

/// concretize_sound with infinite input bounds should not produce NaN.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_with_infinite_inputs() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    // First dimension is [-inf, inf], second is [1, 2].
    // Since A[0,0]=0, the infinite dimension should contribute 0 (safe 0*inf=0).
    let input = unchecked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 2.0]).unwrap(),
    );

    let sound = bounds.concretize_sound(&input);
    assert!(
        sound.lower()[[0]].is_finite(),
        "lower should be finite when coefficient is 0: got {}",
        sound.lower()[[0]],
    );
    assert!(
        sound.upper()[[0]].is_finite(),
        "upper should be finite when coefficient is 0: got {}",
        sound.upper()[[0]],
    );
    // 0*inf + 1*1 = 1 for lower, 0*inf + 1*2 = 2 for upper
    assert!(sound.lower()[[0]] <= 1.0);
    assert!(sound.upper()[[0]] >= 2.0);
}

/// concretize with NaN in input bounds should produce conservative bounds, not NaN.
///
/// Matches BatchedLinearBounds::concretize behavior (via interval_mul_for_bounds)
/// which returns (-inf, +inf) for NaN inputs. See #2210.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_nan_input_bounds_returns_conservative() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    // NaN in lower bound of first dimension.
    // Lower accumulator: la.max(0.0)*in_l[0] = 1.0*NaN → NaN → -inf in concretize_f64_inner.
    // f64_to_bounded_tensor repair (#2287): when lower is non-finite (-inf),
    // the entire element is widened to [-inf, +inf] (sound conservative fallback).
    let input = unchecked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    assert!(
        !result.lower()[[0]].is_nan(),
        "concretize should not produce NaN lower bound; got NaN"
    );
    assert!(
        !result.upper()[[0]].is_nan(),
        "concretize should not produce NaN upper bound; got NaN"
    );
    // Lower accumulator poisoned by NaN → -inf, then repair widens both to [-inf, +inf].
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// concretize_sound with NaN in input bounds should produce conservative bounds.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_nan_input_bounds_returns_conservative() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    // NaN in upper bound of second dimension.
    // Upper accumulator: ua.max(0.0)*in_u[1] = 1.0*NaN → NaN → +inf
    // Lower accumulator: la.max(0.0)*in_l = clean, la.min(0.0)=0 so safe_mul(0,NaN)=0 → clean
    let input = unchecked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, f32::NAN]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);
    assert!(
        !result.lower()[[0]].is_nan(),
        "concretize_sound should not produce NaN lower bound; got NaN"
    );
    assert!(
        !result.upper()[[0]].is_nan(),
        "concretize_sound should not produce NaN upper bound; got NaN"
    );
    // Upper accumulator poisoned by NaN → non-finite after inner computation.
    // new_repaired(Widen) (#2287, #3423) widens the entire element to [-inf, +inf]
    // when either bound is non-finite (sound conservative fallback).
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// concretize with NaN in bias should produce conservative bounds for that direction.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_nan_bias_returns_conservative() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        lower_b: array![f32::NAN],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    assert!(
        !result.lower()[[0]].is_nan(),
        "concretize should not produce NaN lower bound from NaN bias; got NaN"
    );
    // Lower bias is NaN → lower accumulator poisoned → -inf in concretize_f64_inner.
    // f64_to_bounded_tensor repair (#2287): non-finite lower triggers symmetric
    // widening of entire element to [-inf, +inf].
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert!(!result.upper()[[0]].is_nan());
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// concretize with NaN in both lower and upper input bounds → both bounds conservative.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_nan_both_input_bounds_returns_conservative() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    // NaN in both lower and upper bounds.
    let input = unchecked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    assert!(!result.lower()[[0]].is_nan());
    assert!(!result.upper()[[0]].is_nan());
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// concretize with mismatched input dimension returns conservative bounds instead of panicking.
/// Regression test for #2222.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_input_dimension_mismatch_returns_conservative() {
    // LinearBounds with 3 inputs, but input_bounds has 2 elements
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    );

    // Should return conservative [-inf, +inf] instead of panicking
    let result = bounds.concretize(&input);
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// concretize_sound with mismatched input dimension returns conservative bounds instead of panicking.
/// Regression test for #2222.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_input_dimension_mismatch_returns_conservative() {
    // LinearBounds with 3 inputs, but input_bounds has 2 elements
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    );

    // Should return conservative [-inf, +inf] instead of panicking
    let result = bounds.concretize_sound(&input);
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// concretize with too many inputs also returns conservative bounds.
/// Regression test for #2222.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_too_many_inputs_returns_conservative() {
    // LinearBounds with 2 inputs, but input_bounds has 4 elements
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// `concretize_sound` with malformed internal shapes (lower_a/upper_a dimension
/// mismatch) returns conservative `[-inf, +inf]` bounds instead of panicking.
/// Regression test for #2222 — ensures `validate_internal_shapes` guard covers
/// the sound concretization path, not just `concretize`.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_malformed_internal_shapes_returns_conservative() {
    // lower_a is (1, 2) but upper_a is (1, 3) — shape mismatch.
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// `concretize_sound` with too many inputs returns conservative bounds.
/// Complements the existing `concretize` too-many-inputs test.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_too_many_inputs_returns_conservative() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
}

/// `concretize_sound` repairs inverted bounds (lower > upper) per-element.
///
/// Constructs LinearBounds where the lower bound coefficients intentionally
/// produce a value higher than the upper bound coefficients for one output,
/// simulating the numerical instability that causes CROWN backward to produce
/// inversions. Verifies that concretize_sound repairs the inverted element
/// to [-inf, +inf] while leaving valid elements intact (#2287).
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_repairs_inversions() {
    // Two outputs, one input.
    // Output 0: lower_a=10, lower_b=0, upper_a=1, upper_b=0
    //   For input [1, 2]: lower = 10*1 = 10, upper = 1*2 = 2 → inverted (10 > 2)
    // Output 1: lower_a=1, lower_b=0, upper_a=2, upper_b=0
    //   For input [1, 2]: lower = 1*1 = 1, upper = 2*2 = 4 → valid (1 <= 4)
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((2, 1), vec![10.0, 1.0]).unwrap(),
        lower_b: array![0.0, 0.0],
        upper_a: Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap(),
        upper_b: array![0.0, 0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);

    // Output 0 should be repaired to [-inf, +inf]
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "inverted output 0 lower should be -inf"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "inverted output 0 upper should be +inf"
    );

    // Output 1 should be valid (with directed rounding applied)
    let lb1 = result.lower()[[1]];
    let ub1 = result.upper()[[1]];
    assert!(lb1.is_finite(), "valid output 1 lower should be finite");
    assert!(ub1.is_finite(), "valid output 1 upper should be finite");
    assert!(lb1 <= ub1, "valid output 1 should not be inverted");
    // Lower bound of output 1: lower_a * input_lower = 1*1 = 1, with next_down
    assert!(lb1 <= 1.0, "lower bound should be at most 1.0");
    // Upper bound of output 1: upper_a * input_upper = 2*2 = 4, with next_up
    assert!(ub1 >= 4.0, "upper bound should be at least 4.0");
}

#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_preserves_certified_finite_inversion_provenance() {
    let bounds = LinearBounds {
        lower_a: Array2::zeros((1, 1)),
        lower_b: array![1.0],
        upper_a: Array2::zeros((1, 1)),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(array![0.0].into_dyn(), array![0.0].into_dyn());

    let ordinary = bounds.concretize_sound(&input);
    let result = bounds.concretize_sound_with_infeasibility(&input);

    assert!(result.certified_finite_inversion);
    let gap = result.row_finite_gaps[0].expect("finite singleton gap");
    assert!((gap - 1.0).abs() <= 2.0 * f64::EPSILON);
    assert_eq!(result.max_finite_gap, Some(gap));
    assert_eq!(result.bounds.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.bounds.upper()[[0]], f32::INFINITY);
    assert_eq!(ordinary.lower(), result.bounds.lower());
    assert_eq!(ordinary.upper(), result.bounds.upper());
}

#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_preserves_ordered_mixed_row_gap_provenance() {
    let bounds = LinearBounds {
        lower_a: Array2::zeros((3, 1)),
        lower_b: array![1.0, -2.0, f32::NEG_INFINITY],
        upper_a: Array2::zeros((3, 1)),
        upper_b: array![0.0, 3.0, f32::INFINITY],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(array![0.0].into_dyn(), array![0.0].into_dyn());

    let ordinary = bounds.concretize_sound(&input);
    let result = bounds.concretize_sound_with_infeasibility(&input);

    assert_eq!(result.row_finite_gaps.len(), 3);
    let positive_gap = result.row_finite_gaps[0].expect("finite inverted-row gap");
    let negative_gap = result.row_finite_gaps[1].expect("finite ordered-row gap");
    assert!((positive_gap - 1.0).abs() <= 2.0 * f64::EPSILON);
    assert!((negative_gap + 5.0).abs() <= 8.0 * f64::EPSILON * 5.0);
    assert_eq!(result.row_finite_gaps[2], None);
    assert_eq!(result.max_finite_gap, Some(positive_gap));
    assert!(result.certified_finite_inversion);
    assert_eq!(ordinary.lower(), result.bounds.lower());
    assert_eq!(ordinary.upper(), result.bounds.upper());
}

#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_fallbacks_never_claim_infeasibility() {
    let input = checked_bounds(array![0.0].into_dyn(), array![1.0].into_dyn());

    let conservative = LinearBounds {
        lower_a: Array2::zeros((1, 1)),
        lower_b: array![f32::NEG_INFINITY],
        upper_a: Array2::zeros((1, 1)),
        upper_b: array![f32::INFINITY],
        lower_a_err: None,
        upper_a_err: None,
    }
    .concretize_sound_with_infeasibility(&input);
    assert!(!conservative.certified_finite_inversion);
    assert_eq!(conservative.row_finite_gaps, vec![None]);

    let nan = LinearBounds {
        lower_a: Array2::zeros((1, 1)),
        lower_b: array![f32::NAN],
        upper_a: Array2::zeros((1, 1)),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    }
    .concretize_sound_with_infeasibility(&input);
    assert!(!nan.certified_finite_inversion);
    assert_eq!(nan.row_finite_gaps, vec![None]);

    let overflow = LinearBounds {
        lower_a: Array2::from_elem((1, 1), f32::MAX),
        lower_b: array![0.0],
        upper_a: Array2::from_elem((1, 1), -f32::MAX),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    }
    .concretize_sound_with_infeasibility(&input);
    assert!(!overflow.certified_finite_inversion);
    assert_eq!(overflow.row_finite_gaps, vec![None]);
}

/// NaN in A-matrix coefficients must produce conservative bounds, not silently zero.
///
/// Regression test for #2415: Rust's `f32::max(NaN, 0.0) = 0.0` (IEEE 754-2008)
/// silently absorbed NaN coefficients into zero, dropping the neuron's contribution.
/// The fix uses NaN-propagating max/min so NaN poisons the accumulator and triggers
/// the NaN guard → conservative [-inf, +inf] bounds.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_nan_a_matrix_coefficient_returns_conservative() {
    // NaN in lower_a coefficient: the neuron's contribution should NOT silently vanish.
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![f32::NAN, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    assert!(
        !result.lower()[[0]].is_nan(),
        "NaN A-matrix coefficient must not produce NaN output"
    );
    // The NaN should poison the lower accumulator → -inf → repair to [-inf, +inf].
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "NaN lower_a coefficient must produce conservative lower bound"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "NaN lower_a coefficient must produce conservative upper bound"
    );
}

/// NaN in upper_a coefficient must also produce conservative bounds.
///
/// Regression test for #2415 (upper path).
#[ntest::timeout(5000)]
#[test]
fn test_concretize_nan_upper_a_coefficient_returns_conservative() {
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![1.0, f32::NAN]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);
    assert!(
        !result.upper()[[0]].is_nan(),
        "NaN upper_a coefficient must not produce NaN output"
    );
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "NaN upper_a must produce conservative lower bound"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "NaN upper_a must produce conservative upper bound"
    );
}

/// A NaN anywhere in a directly-constructed proof carrier invalidates the whole object.
///
/// Regression test for the proof-boundary firewall: rows from a malformed
/// `LinearBounds` cannot be trusted independently, even when some look finite.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_nan_coefficient_invalidates_entire_proof_object() {
    // Two outputs: row 0 has NaN in lower_a, row 1 is clean.
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((2, 2), vec![f32::NAN, 1.0, 1.0, 1.0]).unwrap(),
        lower_b: array![0.0, 0.0],
        upper_a: Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        upper_b: array![0.0, 0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);
    // Whole-object validation fails closed for both rows.
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
    assert_eq!(result.lower()[[1]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[1]], f32::INFINITY);
}
/// #1932: concretize defense-in-depth — coefficients exceeding CROWN_COEFF_MAX
/// but below f32::MAX must produce conservative bounds for the affected row.
///
/// This tests the *secondary* defense path in `concretize_f64_inner`:
/// coefficients that slipped past the backward pass magnitude check (e.g., from
/// convolution backward which uses `is_finite()` instead of `is_crown_coeff_safe()`)
/// are caught at concretization time and degraded to [-inf, +inf].
///
/// Note: even when only lower_a exceeds CROWN_COEFF_MAX and upper_a is normal,
/// `f64_to_bounded_tensor` repair widens BOTH to [-inf, +inf] for that element
/// because a non-finite lower triggers the `!l.is_finite()` repair. This is sound.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_large_coeff_exceeding_crown_coeff_max_1932() {
    use ny_core::CROWN_COEFF_MAX;

    // Two outputs, two inputs. Row 0 has lower_a coefficient = 1e15 (> CROWN_COEFF_MAX = 1e10).
    // Row 1 has normal coefficients.
    let large_coeff = CROWN_COEFF_MAX * 1e5; // 1e15, well above threshold but below f32::MAX
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((2, 2), vec![large_coeff, 1.0, 1.0, 1.0]).unwrap(),
        lower_b: array![0.0, 0.0],
        upper_a: Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        upper_b: array![0.0, 0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    // Row 0: lower_a has large coefficient → lower becomes -inf in concretize_f64_inner,
    // then f64_to_bounded_tensor repair widens entire element to [-inf, +inf].
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "lower_a > CROWN_COEFF_MAX must degrade lower to -inf"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "repair widens entire element to +inf when lower is -inf"
    );
    // Row 1: all normal → finite bounds
    assert!(
        result.lower()[[1]].is_finite(),
        "clean row 1 lower should be finite"
    );
    assert!(
        result.upper()[[1]].is_finite(),
        "clean row 1 upper should be finite"
    );
}

/// #1932/#3202: When only upper_a exceeds CROWN_COEFF_MAX, the guard should
/// degrade upper to +inf. KNOWN FAILURE until #3202 fix: dot-product write-back
/// overwrites the infinity. This is defense-in-depth consistency (P2), not
/// direct soundness — f64 intermediates prevent overflow for sub-f32::MAX coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_large_upper_coeff_degrade_1932() {
    use ny_core::CROWN_COEFF_MAX;

    let large_coeff = CROWN_COEFF_MAX * 100.0; // 1e12
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![large_coeff, 1.0]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    // Correct behavior: upper_a > CROWN_COEFF_MAX → upper degrades to +inf,
    // then f64_to_bounded_tensor repair widens to [-inf, +inf].
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "upper_a > CROWN_COEFF_MAX must degrade upper to +inf (blocked by #3202)"
    );
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "repair must widen lower to -inf when upper is +inf (blocked by #3202)"
    );
}

/// #1932: concretize defense-in-depth — negative large coefficients also trigger guard.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_large_negative_coeff_triggers_guard_1932() {
    use ny_core::CROWN_COEFF_MAX;

    let large_neg = -(CROWN_COEFF_MAX * 10.0); // -1e11
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 1), vec![large_neg]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 1), vec![large_neg]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    );

    let result = bounds.concretize(&input);
    // Both lower and upper have |coeff| > CROWN_COEFF_MAX → both degraded
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "negative large lower_a must degrade to -inf"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "negative large upper_a must degrade to +inf"
    );
}

/// #1932: concretize_sound also applies the CROWN_COEFF_MAX defense-in-depth.
/// Both lower_a and upper_a exceed threshold → both degrade.
#[ntest::timeout(5000)]
#[test]
fn test_concretize_sound_large_coeff_defense_1932() {
    use ny_core::CROWN_COEFF_MAX;

    let large_coeff = CROWN_COEFF_MAX * 1e5;
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 1), vec![large_coeff]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 1), vec![large_coeff]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    );

    let result = bounds.concretize_sound(&input);
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "concretize_sound: lower_a > CROWN_COEFF_MAX must degrade to -inf"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "concretize_sound: upper_a > CROWN_COEFF_MAX must degrade to +inf"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_exact_crown_transport_sentinel_is_never_a_coefficient() {
    use ny_core::CROWN_COEFF_MAX;

    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 1), vec![CROWN_COEFF_MAX]).unwrap(),
        lower_b: array![0.0],
        upper_a: Array2::from_shape_vec((1, 1), vec![-CROWN_COEFF_MAX]).unwrap(),
        upper_b: array![0.0],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = checked_bounds(array![0.0_f32].into_dyn(), array![0.0_f32].into_dyn());

    for result in [bounds.concretize(&input), bounds.concretize_sound(&input)] {
        assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
        assert_eq!(result.upper()[[0]], f32::INFINITY);
    }
}

// #2907 concretize panic-cliff tests moved to panic_cliff.rs
// #2977 NaN validation tests moved to linear_validation.rs

// --- new_or_conservative tests (moved from linear.rs inline module) ---

#[ntest::timeout(5000)]
#[test]
fn test_new_or_conservative_finite_passthrough() {
    let la = array![[1.0, 2.0], [3.0, 4.0]];
    let lb = array![0.5, -0.5];
    let ua = array![[1.5, 2.5], [3.5, 4.5]];
    let ub = array![0.6, -0.4];
    let result =
        LinearBounds::new_or_conservative(la.clone(), lb.clone(), ua.clone(), ub.clone()).unwrap();
    assert_eq!(result.lower_a(), &la);
    assert_eq!(result.lower_b(), &lb);
    assert_eq!(result.upper_a(), &ua);
    assert_eq!(result.upper_b(), &ub);
}

#[ntest::timeout(5000)]
#[test]
fn test_new_or_conservative_nan_coefficient_falls_back() {
    let la = Array2::from_shape_vec((2, 2), vec![f32::NAN, 2.0, 3.0, 4.0]).unwrap();
    let lb = array![0.5, -0.5];
    let ua = array![[1.5, 2.5], [3.5, 4.5]];
    let ub = array![0.6, -0.4];
    let result = LinearBounds::new_or_conservative(la, lb, ua, ub).unwrap();
    assert!(result.lower_a().iter().all(|v| *v == 0.0));
    assert!(result.upper_a().iter().all(|v| *v == 0.0));
    assert!(result.lower_b().iter().all(|v| *v == f32::NEG_INFINITY));
    assert!(result.upper_b().iter().all(|v| *v == f32::INFINITY));
}

#[ntest::timeout(5000)]
#[test]
fn test_new_or_conservative_inf_coefficient_falls_back() {
    let la = Array2::from_shape_vec((1, 2), vec![f32::INFINITY, 2.0]).unwrap();
    let lb = array![0.5];
    let ua = array![[1.5, 2.5]];
    let ub = array![0.6];
    let result = LinearBounds::new_or_conservative(la, lb, ua, ub).unwrap();
    assert!(result.lower_a().iter().all(|v| *v == 0.0));
    assert_eq!(result.num_outputs(), 1);
    assert_eq!(result.num_inputs(), 2);
}

#[ntest::timeout(5000)]
#[test]
fn test_new_or_conservative_nan_bias_falls_back() {
    let la = array![[1.0, 2.0]];
    let lb = Array1::from_vec(vec![f32::NAN]);
    let ua = array![[1.5, 2.5]];
    let ub = array![0.6];
    let result = LinearBounds::new_or_conservative(la, lb, ua, ub).unwrap();
    assert!(result.lower_a().iter().all(|v| *v == 0.0));
}

#[ntest::timeout(5000)]
#[test]
fn test_new_or_conservative_inf_bias_passthrough() {
    let la = array![[1.0, 2.0]];
    let lb = Array1::from_vec(vec![f32::NEG_INFINITY]);
    let ua = array![[1.5, 2.5]];
    let ub = Array1::from_vec(vec![f32::INFINITY]);
    let result = LinearBounds::new_or_conservative(la.clone(), lb, ua.clone(), ub).unwrap();
    assert_eq!(result.lower_a(), &la);
    assert_eq!(result.upper_a(), &ua);
}

#[ntest::timeout(5000)]
#[test]
fn test_new_rejects_nan_with_error() {
    let la = Array2::from_shape_vec((1, 2), vec![f32::NAN, 2.0]).unwrap();
    let lb = array![0.5];
    let ua = array![[1.5, 2.5]];
    let ub = array![0.6];
    let result = LinearBounds::new(la, lb, ua, ub);
    assert!(result.is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_conservative_shape() {
    let c = LinearBounds::conservative(3, 5);
    assert_eq!(c.num_outputs(), 3);
    assert_eq!(c.num_inputs(), 5);
    assert!(c.lower_a().iter().all(|v| *v == 0.0));
    assert!(c.upper_a().iter().all(|v| *v == 0.0));
    assert!(c.lower_b().iter().all(|v| *v == f32::NEG_INFINITY));
    assert!(c.upper_b().iter().all(|v| *v == f32::INFINITY));
}

// --- new_repaired() tests (#3423 Step 2) ---

/// Conservative strategy: NaN coefficient → 0.0, finite preserved.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_conservative_nan_coefficient() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((2, 2), vec![f32::NAN, 1.0, 2.0, 3.0]).unwrap();
    let lb = array![0.5, -0.5];
    let ua = Array2::from_shape_vec((2, 2), vec![4.0, 5.0, 6.0, 7.0]).unwrap();
    let ub = array![0.1, 0.2];
    let result = LinearBounds::new_repaired(
        la,
        lb.clone(),
        ua.clone(),
        ub.clone(),
        RepairStrategy::Conservative,
    )
    .unwrap();
    // NaN in lower_a[0,0] → 0.0, rest preserved
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.lower_a()[[0, 1]], 1.0);
    assert_eq!(result.lower_a()[[1, 0]], 2.0);
    assert_eq!(result.upper_a(), &ua);
    assert_eq!(result.lower_b(), &lb);
    assert_eq!(result.upper_b(), &ub);
}

/// Conservative strategy: Inf coefficient → 0.0.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_conservative_inf_coefficient() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 2), vec![f32::INFINITY, 1.0]).unwrap();
    let lb = array![0.0];
    let ua = Array2::from_shape_vec((1, 2), vec![1.0, f32::NEG_INFINITY]).unwrap();
    let ub = array![0.0];
    let result = LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Conservative).unwrap();
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.lower_a()[[0, 1]], 1.0);
    assert_eq!(result.upper_a()[[0, 0]], 1.0);
    assert_eq!(result.upper_a()[[0, 1]], 0.0);
}

/// Conservative strategy: NaN bias → ±Inf.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_conservative_nan_bias() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    let lb = array![f32::NAN, 0.5];
    let ua = Array2::from_shape_vec((2, 1), vec![3.0, 4.0]).unwrap();
    let ub = array![0.1, f32::NAN];
    let result =
        LinearBounds::new_repaired(la.clone(), lb, ua.clone(), ub, RepairStrategy::Conservative)
            .unwrap();
    // NaN in lower_b → -inf, NaN in upper_b → +inf
    assert_eq!(result.lower_b()[0], f32::NEG_INFINITY);
    assert_eq!(result.lower_b()[1], 0.5);
    assert_eq!(result.upper_b()[0], 0.1);
    assert_eq!(result.upper_b()[1], f32::INFINITY);
    // Coefficients unchanged
    assert_eq!(result.lower_a(), &la);
    assert_eq!(result.upper_a(), &ua);
}

/// Conservative preserves ±Inf in biases (already valid).
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_conservative_preserves_inf_bias() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let lb = array![f32::NEG_INFINITY];
    let ua = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
    let ub = array![f32::INFINITY];
    let result = LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Conservative).unwrap();
    assert_eq!(result.lower_b()[0], f32::NEG_INFINITY);
    assert_eq!(result.upper_b()[0], f32::INFINITY);
}

/// Widen strategy: NaN coefficient → 0.0, Inf coefficient left as-is.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_widen_nan_vs_inf_coefficient() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 3), vec![f32::NAN, f32::INFINITY, 1.0]).unwrap();
    let lb = array![0.0];
    let ua = Array2::from_shape_vec((1, 3), vec![2.0, f32::NAN, f32::NEG_INFINITY]).unwrap();
    let ub = array![0.0];
    let result = LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Widen).unwrap();
    // NaN → 0.0, Inf → kept
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert_eq!(result.lower_a()[[0, 1]], f32::INFINITY);
    assert_eq!(result.lower_a()[[0, 2]], 1.0);
    assert_eq!(result.upper_a()[[0, 0]], 2.0);
    assert_eq!(result.upper_a()[[0, 1]], 0.0);
    assert_eq!(result.upper_a()[[0, 2]], f32::NEG_INFINITY);
}

/// Widen strategy: NaN bias → ±Inf.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_widen_nan_bias() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let lb = array![f32::NAN];
    let ua = Array2::from_shape_vec((1, 1), vec![2.0]).unwrap();
    let ub = array![f32::NAN];
    let result = LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Widen).unwrap();
    assert_eq!(result.lower_b()[0], f32::NEG_INFINITY);
    assert_eq!(result.upper_b()[0], f32::INFINITY);
}

/// Strict strategy rejects NaN (same as new()).
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_strict_rejects_nan() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 1), vec![f32::NAN]).unwrap();
    let lb = array![0.0];
    let ua = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let ub = array![0.0];
    assert!(LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Strict).is_err());
}

/// Strict strategy rejects Inf in coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_strict_rejects_inf_coefficient() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 1), vec![f32::INFINITY]).unwrap();
    let lb = array![0.0];
    let ua = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let ub = array![0.0];
    assert!(LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Strict).is_err());
}

/// Shape mismatch is rejected regardless of strategy.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_shape_mismatch() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let lb = array![0.0];
    let ua = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let ub = array![0.0, 0.0];
    assert!(LinearBounds::new_repaired(la, lb, ua, ub, RepairStrategy::Conservative).is_err());
}

/// Finite inputs pass through unchanged for all strategies.
#[ntest::timeout(5000)]
#[test]
fn test_new_repaired_finite_passthrough() {
    use ny_tensor::RepairStrategy;

    let la = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let lb = array![0.5, -0.5];
    let ua = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let ub = array![0.1, 0.2];

    for strategy in [
        RepairStrategy::Conservative,
        RepairStrategy::Widen,
        RepairStrategy::Strict,
    ] {
        let result =
            LinearBounds::new_repaired(la.clone(), lb.clone(), ua.clone(), ub.clone(), strategy)
                .unwrap();
        assert_eq!(result.lower_a(), &la, "strategy={strategy:?}");
        assert_eq!(result.upper_a(), &ua, "strategy={strategy:?}");
        assert_eq!(result.lower_b(), &lb, "strategy={strategy:?}");
        assert_eq!(result.upper_b(), &ub, "strategy={strategy:?}");
    }
}

/// Regression #2220 / #concretize-soundness-hardening: non-batched concretize_sound
/// with n=4096 must contain the **f64-exact-product** reference under heavy
/// cancellation. As of the soundness-hardening sweep the non-batched scalar path
/// forms each per-term product in f64 (operands cast f32→f64, so f32×f32 products
/// are EXACT), matching the production batched path (`batched/concretize.rs`
/// concretize_blas_posneg, which "casts f32→f64 so products are exact"). Forming the
/// product in f32 (the prior behavior) rounds each term to nearest — up to 0.5 f32-ULP
/// inward — and with alternating-sign coefficients (large cancellation) that inward
/// bias accumulates beyond the final 1-ULP directed cast, making `concretize_sound`
/// UNSOUND w.r.t. the true real linear form. The f64-product result encloses the true
/// value; here we check it brackets the f64-exact reference (a sound, tighter bound).
///
/// Mirrors test_batched_concretize_sound_n4096_contains_f64_reference from
/// batched_linear.rs but targets the non-batched LinearBounds path.
#[ntest::timeout(60000)]
#[test]
fn test_concretize_sound_n4096_contains_f32_product_reference() {
    let n = 4096;
    let scale = 1.0 / (n as f32).sqrt();
    let a_data: Vec<f32> = (0..n)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0f32 } else { -1.0 };
            sign * (1.0 + j as f32 * 1e-5) * scale
        })
        .collect();
    let x_data: Vec<f32> = (0..n).map(|j| 0.5 + j as f32 * 1e-4).collect();

    // f64-exact-product reference: promote each operand to f64 BEFORE the product
    // (f32×f32 fits in f64's 53-bit significand, so the product is exact), then
    // accumulate in f64. This matches the scalar path semantics after the
    // soundness-hardening sweep (safe_mul_for_bounds_f64 on f64-promoted operands).
    let ref_sum: f64 = a_data
        .iter()
        .zip(&x_data)
        .map(|(&a, &x)| (a.max(0.0) as f64) * (x as f64) + (a.min(0.0) as f64) * (x as f64))
        .sum();
    let ref_lo = next_down_f32(ref_sum as f32);
    let ref_hi = next_up_f32(ref_sum as f32);

    let lower_a = Array2::from_shape_vec((1, n), a_data).unwrap();
    let bounds = LinearBounds {
        lower_a: lower_a.clone(),
        lower_b: Array1::zeros(1),
        upper_a: lower_a,
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };
    // Point input: lower == upper, so concretize_sound computes A @ x + b.
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[n]), x_data.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), x_data).unwrap(),
    );

    let sound = bounds.concretize_sound(&input);
    let sl = sound.lower().as_slice().unwrap()[0];
    let su = sound.upper().as_slice().unwrap()[0];

    // Sound bounds must bracket the f64-exact-product reference.
    assert!(
        sl <= ref_lo,
        "concretize_sound lower {sl} > f64-product ref {ref_lo} \
         (gap: {} ULPs). Non-batched scalar path must form per-term products \
         in f64 (exact) to match the production batched path; see #2220 / \
         #concretize-soundness-hardening.",
        (sl.to_bits() as i64 - ref_lo.to_bits() as i64).unsigned_abs()
    );
    assert!(
        su >= ref_hi,
        "concretize_sound upper {su} < f64-product ref {ref_hi} \
         (gap: {} ULPs). Non-batched scalar path must form per-term products \
         in f64 (exact) to match the production batched path; see #2220 / \
         #concretize-soundness-hardening.",
        (ref_hi.to_bits() as i64 - su.to_bits() as i64).unsigned_abs()
    );
}
