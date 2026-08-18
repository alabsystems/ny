// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use proptest::prelude::*;

#[test]
fn concretize_sound_survives_three_term_binary64_cancellation() {
    let large = 2.0_f32.powi(30);
    let coefficients = arr2(&[[large, 1.0, -large]]);
    let bounds = LinearBounds::new(
        coefficients.clone(),
        arr1(&[0.0]),
        coefficients,
        arr1(&[0.0]),
    )
    .unwrap();
    let point = arr1(&[large, 1.0, large]).into_dyn();
    let input = BoundedTensor::new(point.clone(), point).unwrap();

    // Exact value: 2^60 + 1 - 2^60 = 1. Nearest binary64 accumulation
    // loses the middle one, which one final binary32 ULP at zero cannot cover.
    let result = bounds.concretize_sound(&input);
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];
    assert!(lower <= 1.0 && upper >= 1.0, "[{lower:e}, {upper:e}]");
    assert!(lower > 0.99 && upper < 1.01, "[{lower:e}, {upper:e}]");
}

#[test]
fn concretize_l2_zero_radius_survives_three_term_binary64_cancellation() {
    let large = 2.0_f32.powi(30);
    let coefficients = arr2(&[[large, 1.0, -large]]);
    let bounds = LinearBounds::new(
        coefficients.clone(),
        arr1(&[0.0]),
        coefficients,
        arr1(&[0.0]),
    )
    .unwrap();
    let result = bounds
        .concretize_l2_ball(&arr1(&[large, 1.0, large]), 0.0)
        .unwrap();
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];
    assert!(lower <= 1.0 && upper >= 1.0, "[{lower:e}, {upper:e}]");
    assert!(lower > 0.99 && upper < 1.01, "[{lower:e}, {upper:e}]");
}

#[test]
fn concretize_l2_ball_discharges_coefficient_error_over_the_whole_ball() {
    let coefficients = arr2(&[[0.0_f32]]);
    let mut bounds = LinearBounds::new(
        coefficients.clone(),
        arr1(&[0.0]),
        coefficients,
        arr1(&[0.0]),
    )
    .unwrap();
    bounds.set_coeff_err(arr2(&[[1.0]]), arr2(&[[1.0]]));

    // For x in the radius-3 ball around 2, |x| may reach 5.  A true
    // coefficient anywhere in [-1,1] therefore requires at least [-5,5].
    let result = bounds.concretize_l2_ball(&arr1(&[2.0]), 3.0).unwrap();
    assert!(result.lower()[[0]] <= -5.0, "lower={}", result.lower()[[0]]);
    assert!(result.upper()[[0]] >= 5.0, "upper={}", result.upper()[[0]]);
}

#[test]
fn coeff_error_penalty_discharge_stays_outward_after_bias_cancellation() {
    let large = 2.0_f32.powi(50);
    let bias = 2.0_f32.powi(100);
    let coefficients = arr2(&[[0.0, 0.0]]);
    let mut bounds = LinearBounds::new(
        coefficients.clone(),
        arr1(&[bias]),
        coefficients,
        arr1(&[-bias]),
    )
    .unwrap();
    let error = arr2(&[[large, 1.0]]);
    bounds.set_coeff_err(error.clone(), error);
    bounds.fold_coeff_err_into_bias(&[large, 1.0], &[large, 1.0]);

    // Exact penalty is 2^100 + 1. A nearest-f64 sum loses the one; subtracting
    // from the nearly equal bias cancels to zero, where final f32 widening is
    // far too small. Lower and upper folds must enclose the exact ±1 residue.
    assert!(bounds.lower_b()[0] <= -1.0, "lower={}", bounds.lower_b()[0]);
    assert!(bounds.upper_b()[0] >= 1.0, "upper={}", bounds.upper_b()[0]);
}

#[test]
fn malformed_coefficient_error_fails_closed() {
    let coefficients = arr2(&[[0.0, 0.0]]);
    let mut bounds = LinearBounds::new(
        coefficients.clone(),
        arr1(&[0.0]),
        coefficients,
        arr1(&[0.0]),
    )
    .unwrap();
    bounds.set_coeff_err(arr2(&[[-1.0, 0.0]]), arr2(&[[f32::NAN, 0.0]]));
    let point = arr1(&[1.0, 1.0]).into_dyn();
    let result = bounds.concretize_sound(&BoundedTensor::new(point.clone(), point).unwrap());
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);

    let mut shape_mismatch = LinearBounds::identity(2);
    shape_mismatch.set_coeff_err(arr2(&[[0.0]]), arr2(&[[0.0]]));
    assert_eq!(shape_mismatch.lower_b()[0], f32::NEG_INFINITY);
    assert_eq!(shape_mismatch.upper_b()[0], f32::INFINITY);

    // Defense in depth for crate-internal construction sites: a malformed
    // carrier must degrade before a fold can index or under-count it.
    let mut directly_malformed = LinearBounds::identity(2);
    directly_malformed.lower_a_err = Some(arr2(&[[-1.0, 0.0], [0.0, 0.0]]));
    directly_malformed.upper_a_err = Some(arr2(&[[0.0, 0.0], [0.0, 0.0]]));
    directly_malformed.fold_coeff_err_into_bias(&[1.0, 1.0], &[1.0, 1.0]);
    assert!(directly_malformed
        .lower_b()
        .iter()
        .all(|&value| value == f32::NEG_INFINITY));
    assert!(directly_malformed
        .upper_b()
        .iter()
        .all(|&value| value == f32::INFINITY));

    let constructor_mismatch = LinearBounds::new_or_conservative_with_err(
        arr2(&[[1.0, 0.0]]),
        arr1(&[0.0]),
        arr2(&[[1.0, 0.0]]),
        arr1(&[0.0]),
        arr2(&[[0.0]]),
        arr2(&[[0.0]]),
    )
    .unwrap();
    assert_eq!(constructor_mismatch.lower_b()[0], f32::NEG_INFINITY);
    assert_eq!(constructor_mismatch.upper_b()[0], f32::INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_concretize_handles_inf_by_avoiding_zero_times_inf() {
    // Kills mutants in LinearBounds::concretize:
    // - has_inf detection (|| -> &&)
    // - 0 * inf guard (!= -> ==)
    // - arithmetic operator mutations inside the safe path loop
    let bounds = LinearBounds::new(
        arr2(&[[0.0, 1.0, -2.0], [0.0, -1.0, 0.5]]),
        arr1(&[0.1, -0.2]),
        arr2(&[[0.0, 1.0, -2.0], [0.0, -1.0, 0.5]]),
        arr1(&[0.3, 0.4]),
    )
    .unwrap();

    // Note: BoundedTensor::new() debug-asserts against Inf/NaN; use new_unchecked for this test.
    let input = unchecked_bounds(
        arr1(&[0.0, -1.0, 2.0]).into_dyn(),
        arr1(&[f32::INFINITY, 3.0, 4.0]).into_dyn(),
    );

    let out = bounds.concretize(&input);
    for &v in out.lower().iter().chain(out.upper().iter()) {
        assert!(!v.is_nan(), "unexpected NaN in concretize output");
    }

    let lower = out.lower().as_slice().unwrap();
    let upper = out.upper().as_slice().unwrap();
    assert!((lower[0] - (-8.9)).abs() < 1e-6, "lower[0]={}", lower[0]);
    assert!((upper[0] - (-0.7)).abs() < 1e-6, "upper[0]={}", upper[0]);
    assert!((lower[1] - (-2.2)).abs() < 1e-6, "lower[1]={}", lower[1]);
    assert!((upper[1] - 3.4).abs() < 1e-6, "upper[1]={}", upper[1]);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_rho_validation() {
    // Kills mutants: replace < with == / <= on rho validation
    let bounds = LinearBounds::identity(2);
    let x_hat = arr1(&[0.0, 0.0]);
    assert!(bounds.concretize_l2_ball(&x_hat, -1.0).is_err());

    let out = bounds.concretize_l2_ball(&x_hat, 0.0).unwrap();
    // With directed rounding, lower bounds are widened toward -inf (1 ULP)
    // and upper bounds toward +inf (1 ULP). The exact value is 0.0.
    // next_down_f32(0.0) = -1.4e-45 (smallest negative subnormal)
    // next_up_f32(0.0) = 1.4e-45 (smallest positive subnormal)
    use ny_tensor::{next_down_f32, next_up_f32};
    for &v in out.lower().as_slice().unwrap() {
        assert!(v <= 0.0, "lower bound should be <= 0.0, got {v}");
        assert_eq!(
            v,
            next_down_f32(0.0),
            "lower should be exactly next_down(0.0)"
        );
    }
    for &v in out.upper().as_slice().unwrap() {
        assert!(v >= 0.0, "upper bound should be >= 0.0, got {v}");
        assert_eq!(v, next_up_f32(0.0), "upper should be exactly next_up(0.0)");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_concretize_l2_ball_matches_closed_form() {
    // Kills arithmetic operator mutants inside concretize_l2_ball
    let bounds = LinearBounds::new(
        arr2(&[[3.0, 4.0]]),
        arr1(&[0.5]),
        arr2(&[[3.0, 4.0]]),
        arr1(&[-1.0]),
    )
    .unwrap();
    let x_hat = arr1(&[1.0, 2.0]);
    let rho = 2.0;
    let out = bounds.concretize_l2_ball(&x_hat, rho).unwrap();

    // dot = a^T x + b, ||a|| = 5
    // lower = (0.5 + 11) - 2*5 = 1.5
    // upper = (-1 + 11) + 2*5 = 20
    let lower = out.lower().as_slice().unwrap()[0];
    let upper = out.upper().as_slice().unwrap()[0];
    // With directed rounding (1 ULP widening), lower <= exact <= upper.
    // The exact values are 1.5 and 20.0, but next_down/next_up shift by 1 ULP.
    // ULP at 1.5 ≈ 1.19e-7, ULP at 20.0 ≈ 1.91e-6.
    // Allow 2 ULP margin (1 from round-to-nearest + 1 from next_down/next_up).
    assert!(lower <= 1.5, "lower={lower} should be <= 1.5");
    assert!(
        (lower - 1.5).abs() < 3e-7,
        "lower={lower} too far from 1.5 (> 2 ULPs)"
    );
    assert!(upper >= 20.0, "upper={upper} should be >= 20.0");
    assert!(
        (upper - 20.0).abs() < 4e-6,
        "upper={upper} too far from 20.0 (> 2 ULPs)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_empty_shape() {
    // Kills mutant: replace || with && in BatchedLinearBounds::identity
    let b = BatchedLinearBounds::identity(&[]).unwrap();
    assert_eq!(b.input_shape, vec![1]);
    assert_eq!(b.output_shape, vec![1]);
    assert_eq!(b.lower_a.shape(), &[1, 1]);
    assert_eq!(b.lower_b.shape(), &[1]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_small_identity_matrix() {
    // Kills mutants in out_dim/in_dim accessors.
    let b = BatchedLinearBounds::identity_for_attention(&[1, 1, 2, 2]).unwrap();
    assert_eq!(b.in_dim(), 4);
    assert_eq!(b.out_dim(), 4);
    assert_eq!(b.input_shape, vec![1, 1, 4]);
    assert_eq!(b.output_shape, vec![1, 1, 4]);

    let a = b
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    assert_eq!(a[[0, 0, 0, 0]], 1.0);
    assert_eq!(a[[0, 0, 0, 1]], 0.0);
    assert_eq!(a[[0, 0, 3, 3]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_over_limit_returns_none() {
    // Kills mutants in total_elements/max_elements arithmetic and comparisons.
    // Baseline returns None before allocating large matrices.
    assert!(BatchedLinearBounds::identity_for_attention(&[3, 1, 64, 64]).is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_mid_size_is_some() {
    // Kills mutants in max_elements arithmetic that would incorrectly reject moderate shapes.
    let b = BatchedLinearBounds::identity_for_attention(&[2, 1, 20, 20]).unwrap();
    assert_eq!(b.in_dim(), 400);
    assert_eq!(b.out_dim(), 400);
    assert_eq!(b.lower_a.shape(), &[2, 1, 400, 400]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_2d_exact() {
    // Kills mutants in BatchedLinearBounds::compose (shape checks and arithmetic).
    let a1 = arr2(&[[1.0, 2.0], [0.0, -1.0]]).into_dyn();
    let b1 = arr1(&[1.0, -1.0]).into_dyn();
    let self_bounds =
        BatchedLinearBounds::from_parts_unchecked(a1.clone(), b1.clone(), a1, b1, vec![2], vec![2]);

    let a2 = arr2(&[[2.0, 0.0], [1.0, 1.0]]).into_dyn();
    let b2 = arr1(&[0.5, 2.0]).into_dyn();
    let other_bounds =
        BatchedLinearBounds::from_parts_unchecked(a2.clone(), b2.clone(), a2, b2, vec![2], vec![2]);

    let composed = self_bounds.compose(&other_bounds).unwrap();
    assert_eq!(composed.input_shape, vec![2]);
    assert_eq!(composed.output_shape, vec![2]);

    let a = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let b = composed
        .lower_b
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    // compose() uses f64 accumulators with directed rounding, so lower bounds
    // may be up to 1 ULP below exact and upper bounds up to 1 ULP above.
    // For point bounds (lower == upper), both get widened outward.
    assert!((a[[0, 0]] - 2.0).abs() < 1e-5, "a[[0,0]]={}", a[[0, 0]]);
    assert!((a[[0, 1]] - 4.0).abs() < 1e-5, "a[[0,1]]={}", a[[0, 1]]);
    assert!((a[[1, 0]] - 1.0).abs() < 1e-5, "a[[1,0]]={}", a[[1, 0]]);
    assert!((a[[1, 1]] - 1.0).abs() < 1e-5, "a[[1,1]]={}", a[[1, 1]]);
    assert!((b[0] - 2.5).abs() < 1e-5, "b[0]={}", b[0]);
    assert!((b[1] - 2.0).abs() < 1e-5, "b[1]={}", b[1]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_rectangular_exact() {
    // Kills mutants in dimension indexing inside compose (e.g., len-2 arithmetic).
    let a1 = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).into_dyn(); // 3x2
    let b1 = arr1(&[0.0, 1.0, 2.0]).into_dyn(); // 3
    let self_bounds =
        BatchedLinearBounds::from_parts_unchecked(a1.clone(), b1.clone(), a1, b1, vec![2], vec![3]);

    let a2 = arr2(&[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]).into_dyn(); // 2x3
    let b2 = arr1(&[1.0, -1.0]).into_dyn(); // 2
    let other_bounds =
        BatchedLinearBounds::from_parts_unchecked(a2.clone(), b2.clone(), a2, b2, vec![3], vec![2]);

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    let b = composed
        .lower_b
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    // compose() uses f64 accumulators with directed rounding (1 ULP widening).
    assert!((a[[0, 0]] - 4.0).abs() < 1e-5, "a[[0,0]]={}", a[[0, 0]]);
    assert!((a[[0, 1]] - 5.0).abs() < 1e-5, "a[[0,1]]={}", a[[0, 1]]);
    assert!((a[[1, 0]] - 0.0).abs() < 1e-5, "a[[1,0]]={}", a[[1, 0]]);
    assert!((a[[1, 1]] - 1.0).abs() < 1e-5, "a[[1,1]]={}", a[[1, 1]]);
    assert!((b[0] - 9.0).abs() < 1e-5, "b[0]={}", b[0]);
    assert!((b[1] - 0.0).abs() < 1e-5, "b[1]={}", b[1]);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_invalid_ndim_is_error() {
    // Kills mutant: replace || with && in BatchedLinearBounds::compose ndim validation.
    use ndarray::ArrayD;
    let bad_self = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::zeros(ndarray::IxDyn(&[1])),
        ArrayD::zeros(ndarray::IxDyn(&[1])),
        ArrayD::zeros(ndarray::IxDyn(&[1])),
        ArrayD::zeros(ndarray::IxDyn(&[1])),
        vec![1],
        vec![1],
    );
    let other = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::zeros(ndarray::IxDyn(&[1, 1])),
        ArrayD::zeros(ndarray::IxDyn(&[1])),
        ArrayD::zeros(ndarray::IxDyn(&[1, 1])),
        ArrayD::zeros(ndarray::IxDyn(&[1])),
        vec![1],
        vec![1],
    );
    assert!(bad_self.compose(&other).is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_nan_in_coeffs_widens_to_infinity() {
    // Kills mutants: replace || with && in compose::interval_mul_for_bounds NaN check.
    // Note: from_parts_unchecked now debug_asserts against NaN (#2979), so we use
    // direct struct construction to inject NaN for testing compose() NaN handling.
    let self_bounds = BatchedLinearBounds::from_parts_unchecked(
        arr2(&[[2.0]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr2(&[[2.0]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        vec![1],
        vec![1],
    );
    let other_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[1.0]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_a: arr2(&[[f32::NAN]]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
        lower_a_err: None,
        upper_a_err: None,
    };

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a_l = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    let a_u = composed
        .upper_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    assert!(a_l.is_infinite() && a_l < 0.0, "a_l={}", a_l);
    assert!(a_u.is_infinite() && a_u > 0.0, "a_u={}", a_u);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_nan_in_self_upper_widens_to_infinity() {
    // Kills missed mutant: OR->AND for the b_u.is_nan() clause in interval_mul_for_bounds.
    // Note: from_parts_unchecked now debug_asserts against NaN (#2979), so we use
    // direct struct construction for the NaN-containing bounds.
    let self_bounds = BatchedLinearBounds {
        lower_a: arr2(&[[2.0]]).into_dyn(),
        lower_b: arr1(&[0.0]).into_dyn(),
        upper_a: arr2(&[[f32::NAN]]).into_dyn(),
        upper_b: arr1(&[0.0]).into_dyn(),
        input_shape: vec![1],
        output_shape: vec![1],
        lower_a_err: None,
        upper_a_err: None,
    };
    let other_bounds = BatchedLinearBounds::from_parts_unchecked(
        arr2(&[[1.0]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr2(&[[1.0]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        vec![1],
        vec![1],
    );

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a_l = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    let a_u = composed
        .upper_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    assert!(a_l.is_infinite() && a_l < 0.0, "a_l={}", a_l);
    assert!(a_u.is_infinite() && a_u > 0.0, "a_u={}", a_u);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_compose_all_inf_products_widens_to_infinity() {
    // Kills missed mutant: OR->AND in interval_mul_for_bounds sentinel check.
    let self_bounds = BatchedLinearBounds::from_parts_unchecked(
        arr2(&[[f32::INFINITY]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr2(&[[f32::INFINITY]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        vec![1],
        vec![1],
    );
    let other_bounds = BatchedLinearBounds::from_parts_unchecked(
        arr2(&[[1.0]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr2(&[[1.0]]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        vec![1],
        vec![1],
    );

    let composed = self_bounds.compose(&other_bounds).unwrap();
    let a_l = composed
        .lower_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    let a_u = composed
        .upper_a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()[[0, 0]];
    assert!(a_l.is_infinite() && a_l < 0.0, "a_l={}", a_l);
    assert!(a_u.is_infinite() && a_u > 0.0, "a_u={}", a_u);
}

// ========== Phase 4 proptest: NaN rejection ==========
// Reference: designs/2026-02-25-validated-linear-bounds.md §Verification

use ndarray::{Array1, Array2};

/// Generate an N×M matrix of finite f32 values.
fn finite_matrix(n: usize, m: usize) -> impl Strategy<Value = Array2<f32>> {
    proptest::collection::vec(
        proptest::num::f32::NORMAL | proptest::num::f32::SUBNORMAL | proptest::num::f32::ZERO,
        n * m,
    )
    .prop_map(move |v| Array2::from_shape_vec((n, m), v).unwrap())
}

/// Generate an N-element vector of finite f32 values.
fn finite_vector(n: usize) -> impl Strategy<Value = Array1<f32>> {
    proptest::collection::vec(
        proptest::num::f32::NORMAL | proptest::num::f32::SUBNORMAL | proptest::num::f32::ZERO,
        n,
    )
    .prop_map(Array1::from_vec)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// LinearBounds::new() must accept any finite-valued inputs.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_bounds_new_accepts_finite(
        la in finite_matrix(3, 4),
        lb in finite_vector(3),
        ua in finite_matrix(3, 4),
        ub in finite_vector(3),
    ) {
        let result = LinearBounds::new(la, lb, ua, ub);
        prop_assert!(result.is_ok(), "new() rejected finite values: {:?}", result.err());
    }

    /// LinearBounds::new() must reject NaN in lower_a coefficients.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_bounds_new_rejects_nan_in_lower_a(
        la in finite_matrix(3, 4),
        lb in finite_vector(3),
        ua in finite_matrix(3, 4),
        ub in finite_vector(3),
        nan_row in 0..3usize,
        nan_col in 0..4usize,
    ) {
        let mut la = la;
        la[[nan_row, nan_col]] = f32::NAN;
        let result = LinearBounds::new(la, lb, ua, ub);
        prop_assert!(result.is_err(), "new() accepted NaN in lower_a");
    }

    /// LinearBounds::new() must reject NaN in upper_a coefficients.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_bounds_new_rejects_nan_in_upper_a(
        la in finite_matrix(3, 4),
        lb in finite_vector(3),
        ua in finite_matrix(3, 4),
        ub in finite_vector(3),
        nan_row in 0..3usize,
        nan_col in 0..4usize,
    ) {
        let mut ua = ua;
        ua[[nan_row, nan_col]] = f32::NAN;
        let result = LinearBounds::new(la, lb, ua, ub);
        prop_assert!(result.is_err(), "new() accepted NaN in upper_a");
    }

    /// LinearBounds::new() must reject Inf in coefficients (Inf = unbounded
    /// proportion to input, not a valid linear relaxation).
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_bounds_new_rejects_inf_in_coefficients(
        la in finite_matrix(3, 4),
        lb in finite_vector(3),
        ua in finite_matrix(3, 4),
        ub in finite_vector(3),
        nan_row in 0..3usize,
        nan_col in 0..4usize,
        is_lower in proptest::bool::ANY,
        is_neg in proptest::bool::ANY,
    ) {
        let mut la = la;
        let mut ua = ua;
        let inf_val = if is_neg { f32::NEG_INFINITY } else { f32::INFINITY };
        if is_lower {
            la[[nan_row, nan_col]] = inf_val;
        } else {
            ua[[nan_row, nan_col]] = inf_val;
        }
        let result = LinearBounds::new(la, lb, ua, ub);
        prop_assert!(result.is_err(), "new() accepted Inf in coefficients");
    }

    /// LinearBounds::new() must reject NaN in biases.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_bounds_new_rejects_nan_in_bias(
        la in finite_matrix(3, 4),
        lb in finite_vector(3),
        ua in finite_matrix(3, 4),
        ub in finite_vector(3),
        nan_idx in 0..3usize,
        is_lower in proptest::bool::ANY,
    ) {
        let mut lb = lb;
        let mut ub = ub;
        if is_lower {
            lb[nan_idx] = f32::NAN;
        } else {
            ub[nan_idx] = f32::NAN;
        }
        let result = LinearBounds::new(la, lb, ua, ub);
        prop_assert!(result.is_err(), "new() accepted NaN in bias");
    }

    /// LinearBounds::new() must allow ±Inf in biases (conservative bounds).
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_linear_bounds_new_accepts_inf_in_bias(
        la in finite_matrix(3, 4),
        ua in finite_matrix(3, 4),
        inf_idx in 0..3usize,
    ) {
        let mut lb = Array1::zeros(3);
        let mut ub = Array1::zeros(3);
        lb[inf_idx] = f32::NEG_INFINITY;
        ub[inf_idx] = f32::INFINITY;
        let result = LinearBounds::new(la, lb, ua, ub);
        prop_assert!(result.is_ok(), "new() rejected ±Inf bias: {:?}", result.err());
    }
}
