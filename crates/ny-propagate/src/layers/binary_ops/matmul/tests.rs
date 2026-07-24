// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{array, Array2, ArrayD, IxDyn};

use super::helpers::view_batch_2d;
use super::*;
use crate::LinearBounds;

fn bounded_from_2d(lower: Array2<f32>, upper: Array2<f32>) -> BoundedTensor {
    BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).expect("valid bounds")
}

#[ntest::timeout(10000)]
#[test]
fn economic_ibp_requires_mode_and_perturbation() {
    let lower_a = array![[0.0, 1.0], [2.0, 3.0]];
    let upper_a = array![[1.0, 2.0], [3.0, 4.0]];
    let lower_b = array![[0.0, -1.0], [1.0, 2.0]];
    let upper_b = array![[0.0, 1.0], [2.0, 3.0]];

    let a = bounded_from_2d(lower_a, upper_a);
    let b = bounded_from_2d(lower_b, upper_b);

    let standard = MatMulLayer::new(false, None);
    assert!(!standard.should_use_economic_ibp(&a, &b));

    let economic = MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic);
    assert!(economic.should_use_economic_ibp(&a, &b));

    let b_concrete = BoundedTensor::concrete(b.lower().clone()).expect("concrete tensor");
    assert!(!economic.should_use_economic_ibp(&a, &b_concrete));
}

#[test]
fn constructor_rejects_non_finite_scale_4307() {
    let err = MatMulLayer::try_new(false, Some(f32::NAN))
        .expect_err("non-finite MatMul scale should be rejected");
    assert!(matches!(err, NyError::InvalidSpec(_)));

    let err =
        MatMulLayer::try_new_with_ibp_mode(false, Some(f32::INFINITY), MatMulIbpMode::Economic)
            .expect_err("non-finite MatMul scale should be rejected regardless of IBP mode");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn economic_ibp_rejects_non_finite_bounds() {
    let lower_a = array![[0.0, 1.0], [2.0, 3.0]];
    let upper_a = array![[1.0, 2.0], [3.0, 4.0]];
    let lower_b = array![[0.0, f32::NEG_INFINITY], [1.0, 2.0]];
    let upper_b = array![[0.0, f32::INFINITY], [2.0, 3.0]];

    let a = bounded_from_2d(lower_a, upper_a);
    let b = BoundedTensor::new_unchecked(lower_b.into_dyn(), upper_b.into_dyn())
        .expect("unchecked bounds");

    let economic = MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic);
    assert!(!economic.should_use_economic_ibp(&a, &b));
}

#[ntest::timeout(10000)]
#[test]
fn view_batch_2d_extracts_expected_slice() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), (0..8).map(|v| v as f32).collect())
        .expect("shape");
    let upper = lower.mapv(|v| v + 1.0);
    let bounds = BoundedTensor::new(lower, upper).expect("bounds");

    let (lower_view, upper_view) = view_batch_2d(&bounds, &[1], "test view").expect("view");
    assert_eq!(lower_view.shape(), &[2, 2]);
    assert_eq!(upper_view.shape(), &[2, 2]);
    assert_eq!(lower_view[[0, 0]], 4.0);
    assert_eq!(upper_view[[1, 1]], 8.0);
}

#[ntest::timeout(10000)]
#[test]
fn ibp_rejects_mismatched_batch_dims() {
    let matmul = MatMulLayer::new(false, None);
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 1, 1]), 1.0_f32),
    )
    .expect("input_a");
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[3, 1, 1]), 1.0_f32),
    )
    .expect("input_b");

    let err = matmul
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("batch mismatch must return an error");
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn batched_crown_rejects_mismatched_batch_dims() {
    let matmul = MatMulLayer::new(false, None);
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 1, 1]), 1.0_f32),
    )
    .expect("input_a");
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[3, 1, 1]), 1.0_f32),
    )
    .expect("input_b");
    let bounds = BatchedLinearBounds::identity(&[1]).expect("identity bounds");

    let err = matmul
        .propagate_linear_batched_binary(&bounds, &input_a, &input_b)
        .expect_err("batch mismatch must return an error");
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn batched_crown_includes_nonzero_batches_in_coefficients() {
    let matmul = MatMulLayer::new(false, None);

    // Two batches for 1x1 @ 1x1. Batch 0 is all-zero (no contribution), batch 1 is all-one.
    // If batch offsets are incorrectly applied to c_flat/a_flat/b_flat, batch 1 gets skipped.
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.0_f32, 1.0_f32]).expect("shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.0_f32, 1.0_f32]).expect("shape"),
    )
    .expect("input_a");
    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.0_f32, 1.0_f32]).expect("shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.0_f32, 1.0_f32]).expect("shape"),
    )
    .expect("input_b");

    let bounds = BatchedLinearBounds::identity(&[1]).expect("identity bounds");
    let (bounds_a, bounds_b) = matmul
        .propagate_linear_batched_binary(&bounds, &input_a, &input_b)
        .expect("batched crown");

    // Batch 1 must contribute non-zero McCormick slopes to both operands.
    assert!(
        bounds_a.lower_a[[0, 0]] != 0.0 || bounds_a.upper_a[[0, 0]] != 0.0,
        "A coefficients unexpectedly zero; non-zero batch was skipped"
    );
    assert!(
        bounds_b.lower_a[[0, 0]] != 0.0 || bounds_b.upper_a[[0, 0]] != 0.0,
        "B coefficients unexpectedly zero; non-zero batch was skipped"
    );
}

// ============================================================================
// IBP happy-path tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn ibp_2x2_matmul_bounds_contain_corners() {
    // A in [[1,2],[3,4]] to [[2,3],[4,5]], B in [[-1,0],[1,2]] to [[0,1],[2,3]]
    // C = A @ B. Test that IBP bounds contain all 16 corner evaluations.
    let matmul = MatMulLayer::new(false, None);
    let a = bounded_from_2d(
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[2.0, 3.0], [4.0, 5.0]],
    );
    let b = bounded_from_2d(
        array![[-1.0, 0.0], [1.0, 2.0]],
        array![[0.0, 1.0], [2.0, 3.0]],
    );

    let result = matmul
        .propagate_ibp_binary(&a, &b)
        .expect("IBP should succeed");

    // Enumerate corners: A_lower/A_upper x B_lower/B_upper
    let a_corners: [Array2<f32>; 2] = [
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[2.0, 3.0], [4.0, 5.0]],
    ];
    let b_corners: [Array2<f32>; 2] = [
        array![[-1.0, 0.0], [1.0, 2.0]],
        array![[0.0, 1.0], [2.0, 3.0]],
    ];

    let tol = 1e-5;
    for a_corner in &a_corners {
        for b_corner in &b_corners {
            let c = a_corner.dot(b_corner);
            for i in 0..2_usize {
                for j in 0..2_usize {
                    let lo = result.lower()[[i, j]];
                    let hi = result.upper()[[i, j]];
                    let true_val = c[[i, j]];
                    assert!(
                        lo - tol <= true_val,
                        "IBP lower bound unsound at [{i},{j}]: lower={lo} > true={true_val}",
                    );
                    assert!(
                        true_val <= hi + tol,
                        "IBP upper bound unsound at [{i},{j}]: true={true_val} > upper={hi}",
                    );
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn ibp_concrete_inputs_produce_tight_bounds() {
    // When A and B are concrete (point inputs), IBP bounds should be tight.
    let matmul = MatMulLayer::new(false, None);
    let a_val = array![[1.0, 2.0], [3.0, 4.0]];
    let b_val = array![[5.0, 6.0], [7.0, 8.0]];

    let a = bounded_from_2d(a_val.clone(), a_val.clone());
    let b = bounded_from_2d(b_val.clone(), b_val.clone());

    let result = matmul
        .propagate_ibp_binary(&a, &b)
        .expect("IBP should succeed");
    let expected = a_val.dot(&b_val);

    let tol = 1e-5;
    for i in 0..2_usize {
        for j in 0..2_usize {
            let lo = result.lower()[[i, j]];
            let hi = result.upper()[[i, j]];
            let exp = expected[[i, j]];
            assert!(
                (lo - exp).abs() < tol,
                "Concrete IBP lower should be tight at [{i},{j}]: got {lo} expected {exp}",
            );
            assert!(
                (hi - exp).abs() < tol,
                "Concrete IBP upper should be tight at [{i},{j}]: got {hi} expected {exp}",
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn ibp_with_transpose_b() {
    let matmul = MatMulLayer::new(true, None);
    let a = bounded_from_2d(array![[1.0, 2.0]], array![[3.0, 4.0]]);
    // B is [2,1] but transpose_b means we compute A @ B^T, so B^T is [1,2]
    let b = bounded_from_2d(array![[1.0, 0.5]], array![[2.0, 1.5]]);

    let result = matmul
        .propagate_ibp_binary(&a, &b)
        .expect("IBP with transpose_b");

    // Corner: A=[1,2], B=[1,0.5] => A @ B^T = 1*1 + 2*0.5 = 2.0
    // Corner: A=[3,4], B=[2,1.5] => A @ B^T = 3*2 + 4*1.5 = 12.0
    let tol = 1e-5;
    assert!(
        result.lower()[[0, 0]] - tol <= 2.0,
        "lower should be <= 2.0"
    );
    assert!(
        result.upper()[[0, 0]] + tol >= 12.0,
        "upper should be >= 12.0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn ibp_with_scale() {
    let matmul = MatMulLayer::new(false, Some(0.5));
    let a = bounded_from_2d(array![[2.0]], array![[4.0]]);
    let b = bounded_from_2d(array![[3.0]], array![[5.0]]);

    let result = matmul.propagate_ibp_binary(&a, &b).expect("IBP with scale");

    // Corners: 2*3=6, 2*5=10, 4*3=12, 4*5=20 => [6, 20] * 0.5 = [3.0, 10.0]
    let tol = 1e-5;
    assert!(
        (result.lower()[[0, 0]] - 3.0).abs() < tol,
        "scaled lower: {} expected 3.0",
        result.lower()[[0, 0]]
    );
    assert!(
        (result.upper()[[0, 0]] - 10.0).abs() < tol,
        "scaled upper: {} expected 10.0",
        result.upper()[[0, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn ibp_preserves_large_finite_products_2549() {
    let matmul = MatMulLayer::new(false, None);
    let a = bounded_from_2d(array![[1.0e5_f32]], array![[1.0e5_f32]]);
    let b = bounded_from_2d(array![[2.0e5_f32]], array![[2.0e5_f32]]);

    let result = matmul
        .propagate_ibp_binary(&a, &b)
        .expect("IBP should succeed");
    let expected = 2.0e10_f32;

    assert!(result.lower()[[0, 0]].is_finite());
    assert!(result.upper()[[0, 0]].is_finite());
    assert!(
        (result.lower()[[0, 0]] - expected).abs() / expected < 1e-6,
        "lower unexpectedly changed: {} (expected {expected})",
        result.lower()[[0, 0]]
    );
    assert!(
        (result.upper()[[0, 0]] - expected).abs() / expected < 1e-6,
        "upper unexpectedly changed: {} (expected {expected})",
        result.upper()[[0, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn ibp_with_negative_scale_swaps_bounds() {
    let matmul = MatMulLayer::new(false, Some(-1.0));
    let a = bounded_from_2d(array![[2.0]], array![[4.0]]);
    let b = bounded_from_2d(array![[3.0]], array![[5.0]]);

    let result = matmul
        .propagate_ibp_binary(&a, &b)
        .expect("IBP with negative scale");

    // Unscaled: [6, 20] => scaled by -1: [-20, -6]
    let tol = 1e-5;
    assert!(
        (result.lower()[[0, 0]] - (-20.0)).abs() < tol,
        "neg-scaled lower: {} expected -20.0",
        result.lower()[[0, 0]]
    );
    assert!(
        (result.upper()[[0, 0]] - (-6.0)).abs() < tol,
        "neg-scaled upper: {} expected -6.0",
        result.upper()[[0, 0]]
    );
}

// ============================================================================
// Economic IBP soundness
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn economic_ibp_bounds_contain_standard_ibp_bounds() {
    // Economic IBP should produce bounds that are at least as wide as standard IBP
    // (and ideally close to the same).
    let a = bounded_from_2d(
        array![[0.5, -1.0], [2.0, 0.0]],
        array![[1.5, 1.0], [3.0, 1.0]],
    );
    let b = bounded_from_2d(
        array![[-0.5, 0.0], [1.0, -1.0]],
        array![[0.5, 1.0], [2.0, 0.0]],
    );

    let standard = MatMulLayer::new(false, None);
    let economic = MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic);

    let std_result = standard.propagate_ibp_binary(&a, &b).expect("standard IBP");
    let econ_result = economic.propagate_ibp_binary(&a, &b).expect("economic IBP");

    // Economic bounds must contain standard bounds (economic may be looser)
    let tol = 1e-4;
    for idx in std_result.lower().indexed_iter() {
        let (ix, &std_val) = idx;
        assert!(
            econ_result.lower()[&ix] - tol <= std_val,
            "Economic lower bound tighter than standard at {:?}: econ={} > std={}",
            ix,
            econ_result.lower()[&ix],
            std_val
        );
    }
    for idx in std_result.upper().indexed_iter() {
        let (ix, &std_val) = idx;
        assert!(
            econ_result.upper()[&ix] + tol >= std_val,
            "Economic upper bound tighter than standard at {:?}: econ={} < std={}",
            ix,
            econ_result.upper()[&ix],
            std_val
        );
    }
}

// ============================================================================
// eval() and jacobian tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn eval_basic_matmul() {
    let matmul = MatMulLayer::new(false, None);
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];

    let result = matmul.eval(&a, &b).expect("eval");
    let expected = a.dot(&b); // [[19, 22], [43, 50]]

    assert_eq!(result, expected);
}

#[ntest::timeout(10000)]
#[test]
fn eval_with_transpose_b() {
    let matmul = MatMulLayer::new(true, None);
    let a = array![[1.0, 2.0]];
    let b = array![[3.0, 4.0]];

    // A @ B^T = [1,2] @ [3,4]^T = [[1*3 + 2*4]] = [[11]]
    let result = matmul.eval(&a, &b).expect("eval transpose_b");
    assert_eq!(result, array![[11.0]]);
}

#[ntest::timeout(10000)]
#[test]
fn eval_with_scale() {
    let matmul = MatMulLayer::new(false, Some(0.5));
    let a = array![[2.0, 0.0], [0.0, 2.0]];
    let b = array![[1.0, 0.0], [0.0, 1.0]];

    let result = matmul.eval(&a, &b).expect("eval with scale");
    // 2*I @ I * 0.5 = I
    let expected = array![[1.0, 0.0], [0.0, 1.0]];
    assert_eq!(result, expected);
}

#[ntest::timeout(10000)]
#[test]
fn jacobian_wrt_a_is_scaled_b_transpose() {
    let matmul = MatMulLayer::new(false, None);
    let b = array![[1.0, 2.0], [3.0, 4.0]];

    let j_a = matmul.jacobian_wrt_a(&b);
    // jacobian_wrt_a returns B^T (possibly with scale)
    assert_eq!(j_a, b.t().to_owned());
}

#[ntest::timeout(10000)]
#[test]
fn jacobian_wrt_a_with_scale() {
    let matmul = MatMulLayer::new(false, Some(2.0));
    let b = array![[1.0, 3.0], [2.0, 4.0]];

    let j_a = matmul.jacobian_wrt_a(&b);
    let expected = b.t().mapv(|v| v * 2.0);
    assert_eq!(j_a, expected);
}

#[ntest::timeout(10000)]
#[test]
fn jacobian_wrt_b_is_scaled_a_transpose() {
    let matmul = MatMulLayer::new(false, None);
    let a = array![[1.0, 2.0], [3.0, 4.0]];

    let j_b = matmul.jacobian_wrt_b(&a);
    assert_eq!(j_b, a.t().to_owned());
}

#[ntest::timeout(10000)]
#[test]
fn jacobian_wrt_b_with_scale() {
    let matmul = MatMulLayer::new(false, Some(0.5));
    let a = array![[2.0, 4.0], [6.0, 8.0]];

    let j_b = matmul.jacobian_wrt_b(&a);
    let expected = a.t().mapv(|v| v * 0.5);
    assert_eq!(j_b, expected);
}

// ============================================================================
// CROWN backward (unbatched) tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn crown_backward_identity_bounds_contain_corners() {
    // C = A @ B for A in [1,3], B in [2,4] (both 1x1 "matrices")
    // C is a scalar = a * b, a in [1,3], b in [2,4]
    // True range: [2, 12]
    let matmul = MatMulLayer::new(false, None);
    let a_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 3.0_f32),
    )
    .expect("a_bounds");
    let b_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), 2.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 4.0_f32),
    )
    .expect("b_bounds");

    // Identity bounds on the 1-element output
    let identity = LinearBounds::identity(1);

    let (bounds_a, bounds_b) = matmul
        .propagate_linear_binary(&identity, &a_bounds, &b_bounds)
        .expect("CROWN backward");

    // Concretize: for each corner (a, b), compute the linear approximation
    // and verify it's a valid relaxation.
    let tol = 1e-4;
    for &a_val in &[1.0_f32, 3.0] {
        for &b_val in &[2.0_f32, 4.0] {
            let true_output = a_val * b_val;

            // Lower bound: bounds_a.lower_a * a + bounds_b.lower_a * b + (sum of biases)
            let crown_lower = bounds_a.lower_a[[0, 0]] * a_val
                + bounds_b.lower_a[[0, 0]] * b_val
                + bounds_a.lower_b[0]
                + bounds_b.lower_b[0];
            let crown_upper = bounds_a.upper_a[[0, 0]] * a_val
                + bounds_b.upper_a[[0, 0]] * b_val
                + bounds_a.upper_b[0]
                + bounds_b.upper_b[0];

            assert!(
                crown_lower - tol <= true_output,
                "CROWN lower unsound at (a={a_val},b={b_val}): {crown_lower} > {true_output}"
            );
            assert!(
                crown_upper + tol >= true_output,
                "CROWN upper unsound at (a={a_val},b={b_val}): {crown_upper} < {true_output}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn crown_backward_rejects_infinite_input_bounds() {
    let matmul = MatMulLayer::new(false, None);
    let a_bounds = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1, 1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 1]), f32::INFINITY),
    )
    .expect("a_bounds");
    let b_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 2.0_f32),
    )
    .expect("b_bounds");

    let identity = LinearBounds::identity(1);

    let err = matmul
        .propagate_linear_binary(&identity, &a_bounds, &b_bounds)
        .expect_err("should reject infinite bounds");
    match err {
        NyError::UnsupportedOp(_) => {}
        other => panic!("expected UnsupportedOp, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn crown_backward_2x1_matmul_soundness() {
    // A is [2,1], B is [1,1] => C = A @ B is [2,1], flattened to 2 elements.
    // A in [[1],[2]] to [[3],[4]], B in [[0.5]] to [[1.5]]
    let matmul = MatMulLayer::new(false, None);
    let a_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0_f32, 2.0]).expect("shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![3.0_f32, 4.0]).expect("shape"),
    )
    .expect("a_bounds");
    let b_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), 0.5_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.5_f32),
    )
    .expect("b_bounds");

    // Identity bounds on C (2 elements)
    let identity = LinearBounds::identity(2);

    let (bounds_a, bounds_b) = matmul
        .propagate_linear_binary(&identity, &a_bounds, &b_bounds)
        .expect("CROWN backward 2x1");

    // Test corners: a0 in [1,3], a1 in [2,4], b in [0.5, 1.5]
    // C[0] = a0 * b, C[1] = a1 * b
    let tol = 1e-4;
    for &a0 in &[1.0_f32, 3.0] {
        for &a1 in &[2.0_f32, 4.0] {
            for &b in &[0.5_f32, 1.5] {
                let c0 = a0 * b;
                let c1 = a1 * b;

                // Reconstruct: bounds_a has shape (2, 2) [num_outputs=2, total_a_size=2]
                // bounds_b has shape (2, 1) [num_outputs=2, total_b_size=1]
                for out_idx in 0..2 {
                    let true_c = if out_idx == 0 { c0 } else { c1 };
                    let crown_lower = bounds_a.lower_a[[out_idx, 0]] * a0
                        + bounds_a.lower_a[[out_idx, 1]] * a1
                        + bounds_b.lower_a[[out_idx, 0]] * b
                        + bounds_a.lower_b[out_idx]
                        + bounds_b.lower_b[out_idx];
                    let crown_upper = bounds_a.upper_a[[out_idx, 0]] * a0
                        + bounds_a.upper_a[[out_idx, 1]] * a1
                        + bounds_b.upper_a[[out_idx, 0]] * b
                        + bounds_a.upper_b[out_idx]
                        + bounds_b.upper_b[out_idx];

                    assert!(
                        crown_lower - tol <= true_c,
                        "CROWN lower unsound at output {out_idx}, (a0={a0},a1={a1},b={b}): \
                         {crown_lower} > {true_c}"
                    );
                    assert!(
                        crown_upper + tol >= true_c,
                        "CROWN upper unsound at output {out_idx}, (a0={a0},a1={a1},b={b}): \
                         {crown_upper} < {true_c}"
                    );
                }
            }
        }
    }
}

// ============================================================================
// decode_batch_index zero-dimension guard (#2806)
// ============================================================================

/// Regression test for #2806: `decode_batch_index` must return an error
/// when batch_dims contains a zero, not panic from integer division-by-zero.
#[ntest::timeout(10000)]
#[test]
fn test_decode_batch_index_zero_dimension_returns_error_2806() {
    use super::shape::decode_batch_index;

    let err = decode_batch_index(0, &[2, 0, 3]).unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued batch dimension"),
        "Expected zero-valued batch dimension error, got: {err}"
    );
}

/// Verify `decode_batch_index` still works correctly for valid batch dims.
#[ntest::timeout(10000)]
#[test]
fn test_decode_batch_index_valid_dims() {
    use super::shape::decode_batch_index;

    // batch_dims [2, 3]: flat index 4 -> [1, 1]
    let indices = decode_batch_index(4, &[2, 3]).expect("valid batch dims");
    assert_eq!(indices, vec![1, 1]);

    // batch_dims [2, 3]: flat index 0 -> [0, 0]
    let indices = decode_batch_index(0, &[2, 3]).expect("valid batch dims");
    assert_eq!(indices, vec![0, 0]);

    // batch_dims [2, 3]: flat index 5 -> [1, 2]
    let indices = decode_batch_index(5, &[2, 3]).expect("valid batch dims");
    assert_eq!(indices, vec![1, 2]);
}

/// Regression test for #2237: the scratch-based decoder should overwrite and
/// truncate an existing buffer instead of relying on fresh Vec allocation.
#[ntest::timeout(10000)]
#[test]
fn test_decode_batch_index_into_reuses_existing_scratch_2237() {
    use super::shape::decode_batch_index_into;

    let mut scratch = vec![99, 98, 97, 96];
    decode_batch_index_into(4, &[2, 3], &mut scratch).expect("valid batch dims");
    assert_eq!(scratch, vec![1, 1]);

    decode_batch_index_into(5, &[2, 3], &mut scratch).expect("valid batch dims");
    assert_eq!(scratch, vec![1, 2]);
}
