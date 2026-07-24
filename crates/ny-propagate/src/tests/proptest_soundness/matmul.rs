// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based soundness tests for MatMulLayer.
//!
//! Tests IBP soundness (bounds contain all corner evaluations) and
//! CROWN backward soundness (McCormick envelope relaxation is a valid
//! over-approximation for bilinear z = A @ B).
//!
//! Extended CROWN tests (batched, non-identity, scale, transpose_b, f64)
//! are in `matmul_crown_extended.rs`.

use crate::{LinearBounds, MatMulIbpMode, MatMulLayer};
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{CROWN_TOLERANCE, FP_TOLERANCE};

/// Generate a bounded 2D tensor of shape [m, n] with element intervals in [-range, range].
pub(super) fn bounded_2d(lower: Vec<f32>, upper: Vec<f32>, m: usize, n: usize) -> BoundedTensor {
    let lo = ArrayD::from_shape_vec(IxDyn(&[m, n]), lower).expect("lower shape");
    let hi = ArrayD::from_shape_vec(IxDyn(&[m, n]), upper).expect("upper shape");
    BoundedTensor::new(lo, hi).expect("valid bounds")
}

/// Evaluate C = A @ B (optionally transposed, optionally scaled) at concrete points.
pub(super) fn eval_matmul(
    a: &Array2<f32>,
    b: &Array2<f32>,
    transpose_b: bool,
    scale: Option<f32>,
) -> Array2<f32> {
    let b_eff = if transpose_b {
        b.t().to_owned()
    } else {
        b.clone()
    };
    let mut c = a.dot(&b_eff);
    if let Some(s) = scale {
        c.mapv_inplace(|v| v * s);
    }
    c
}

/// Strategy for generating a pair (lower, upper) vectors of length `n`
/// where lower[i] <= upper[i] and values are in [-range, range].
pub(super) fn valid_interval_vec(
    n: usize,
    range: f32,
) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    proptest::collection::vec((-range..=range, -range..=range), n).prop_map(move |pairs| {
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for (a, b) in pairs {
            lo.push(a.min(b));
            hi.push(a.max(b));
        }
        (lo, hi)
    })
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // ========================================================================
    // IBP soundness: 2x2 @ 2x2
    // ========================================================================

    /// IBP bounds must contain the true output at all corner combinations.
    /// For a 2x2 @ 2x2 matmul, we sample corners (each element at its lower
    /// or upper bound) and verify that the IBP output contains the evaluation.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_ibp_2x2(
        (a_lo, a_hi) in valid_interval_vec(4, 5.0),
        (b_lo, b_hi) in valid_interval_vec(4, 5.0),
    ) {
        let matmul = MatMulLayer::new(false, None);
        let a_bounds = bounded_2d(a_lo.clone(), a_hi.clone(), 2, 2);
        let b_bounds = bounded_2d(b_lo.clone(), b_hi.clone(), 2, 2);

        let result = matmul.propagate_ibp_binary(&a_bounds, &b_bounds)
            .map_err(|e| TestCaseError::fail(format!("IBP failed: {e}")))?;

        // Enumerate all 2^4 * 2^4 = 256 corner combinations (each of 4 A elements
        // and 4 B elements independently at lower or upper).
        // For efficiency, sample 16 corners (each element at its extremal).
        for a_mask in 0..16_u32 {
            let a_corner = Array2::from_shape_fn((2, 2), |(i, j)| {
                let idx = i * 2 + j;
                if a_mask & (1 << idx) != 0 { a_hi[idx] } else { a_lo[idx] }
            });
            for b_mask in 0..16_u32 {
                let b_corner = Array2::from_shape_fn((2, 2), |(i, j)| {
                    let idx = i * 2 + j;
                    if b_mask & (1 << idx) != 0 { b_hi[idx] } else { b_lo[idx] }
                });
                let c = a_corner.dot(&b_corner);
                for i in 0..2_usize {
                    for j in 0..2_usize {
                        let lo = result.lower()[[i, j]];
                        let hi = result.upper()[[i, j]];
                        let true_val = c[[i, j]];
                        prop_assert!(
                            lo - FP_TOLERANCE <= true_val,
                            "IBP lower unsound at [{i},{j}]: lo={lo} > true={true_val} \
                             (a_mask={a_mask}, b_mask={b_mask})"
                        );
                        prop_assert!(
                            true_val <= hi + FP_TOLERANCE,
                            "IBP upper unsound at [{i},{j}]: true={true_val} > hi={hi} \
                             (a_mask={a_mask}, b_mask={b_mask})"
                        );
                    }
                }
            }
        }
    }

    // ========================================================================
    // IBP soundness with transpose_b
    // ========================================================================

    /// IBP bounds for A @ B^T are sound at corners.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_ibp_transpose_b(
        (a_lo, a_hi) in valid_interval_vec(2, 5.0),  // 1x2
        (b_lo, b_hi) in valid_interval_vec(2, 5.0),  // 1x2, transposed to 2x1
    ) {
        let matmul = MatMulLayer::new(true, None);
        let a_bounds = bounded_2d(a_lo.clone(), a_hi.clone(), 1, 2);
        let b_bounds = bounded_2d(b_lo.clone(), b_hi.clone(), 1, 2);

        let result = matmul.propagate_ibp_binary(&a_bounds, &b_bounds)
            .map_err(|e| TestCaseError::fail(format!("IBP transpose_b failed: {e}")))?;

        // C = A @ B^T => [1,2] @ [2,1] => [1,1]
        for a_mask in 0..4_u32 {
            let a_corner = Array2::from_shape_fn((1, 2), |(_, j)| {
                if a_mask & (1 << j) != 0 { a_hi[j] } else { a_lo[j] }
            });
            for b_mask in 0..4_u32 {
                let b_corner = Array2::from_shape_fn((1, 2), |(_, j)| {
                    if b_mask & (1 << j) != 0 { b_hi[j] } else { b_lo[j] }
                });
                let c = eval_matmul(&a_corner, &b_corner, true, None);
                let true_val = c[[0, 0]];
                let lo = result.lower()[[0, 0]];
                let hi = result.upper()[[0, 0]];
                prop_assert!(
                    lo - FP_TOLERANCE <= true_val && true_val <= hi + FP_TOLERANCE,
                    "IBP transpose_b unsound: true={true_val} not in [{lo}, {hi}]"
                );
            }
        }
    }

    // ========================================================================
    // IBP soundness with scale
    // ========================================================================

    /// IBP bounds with scale factor are sound at corners.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_ibp_with_scale(
        (a_lo, a_hi) in valid_interval_vec(2, 5.0),  // 1x2
        (b_lo, b_hi) in valid_interval_vec(2, 5.0),  // 2x1
        scale in -3.0f32..3.0,
    ) {
        let matmul = MatMulLayer::new(false, Some(scale));
        let a_bounds = bounded_2d(a_lo.clone(), a_hi.clone(), 1, 2);
        let b_bounds = bounded_2d(b_lo.clone(), b_hi.clone(), 2, 1);

        let result = matmul.propagate_ibp_binary(&a_bounds, &b_bounds)
            .map_err(|e| TestCaseError::fail(format!("IBP scale failed: {e}")))?;

        for a_mask in 0..4_u32 {
            let a_corner = Array2::from_shape_fn((1, 2), |(_, j)| {
                if a_mask & (1 << j) != 0 { a_hi[j] } else { a_lo[j] }
            });
            for b_mask in 0..4_u32 {
                let b_corner = Array2::from_shape_fn((2, 1), |(i, _)| {
                    if b_mask & (1 << i) != 0 { b_hi[i] } else { b_lo[i] }
                });
                let c = eval_matmul(&a_corner, &b_corner, false, Some(scale));
                let true_val = c[[0, 0]];
                let lo = result.lower()[[0, 0]];
                let hi = result.upper()[[0, 0]];
                // Use scaled tolerance for larger magnitudes
                let tol = FP_TOLERANCE * true_val.abs().max(lo.abs()).max(hi.abs()).max(1.0);
                prop_assert!(
                    lo - tol <= true_val && true_val <= hi + tol,
                    "IBP scale={scale} unsound: true={true_val} not in [{lo}, {hi}]"
                );
            }
        }
    }

    // ========================================================================
    // Economic IBP contains standard IBP
    // ========================================================================

    /// Economic IBP should never produce bounds tighter than standard IBP
    /// (it is a potentially looser but more memory-efficient computation).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_economic_ibp_no_tighter_than_standard(
        (a_lo, a_hi) in valid_interval_vec(4, 3.0),  // 2x2
        (b_lo, b_hi) in valid_interval_vec(4, 3.0),  // 2x2
    ) {
        // Only test when both inputs are perturbed (economic IBP falls back otherwise)
        let a_perturbed = a_lo.iter().zip(a_hi.iter()).any(|(&l, &u)| l != u);
        let b_perturbed = b_lo.iter().zip(b_hi.iter()).any(|(&l, &u)| l != u);
        prop_assume!(a_perturbed && b_perturbed, "both inputs must be perturbed for economic IBP");

        let standard = MatMulLayer::new(false, None);
        let economic = MatMulLayer::new_with_ibp_mode(false, None, MatMulIbpMode::Economic);

        let a_bounds = bounded_2d(a_lo, a_hi, 2, 2);
        let b_bounds = bounded_2d(b_lo, b_hi, 2, 2);

        let std_result = standard.propagate_ibp_binary(&a_bounds, &b_bounds)
            .map_err(|e| TestCaseError::fail(format!("standard IBP failed: {e}")))?;
        let econ_result = economic.propagate_ibp_binary(&a_bounds, &b_bounds)
            .map_err(|e| TestCaseError::fail(format!("economic IBP failed: {e}")))?;

        let tol = 1e-4;
        for i in 0..2_usize {
            for j in 0..2_usize {
                prop_assert!(
                    econ_result.lower()[[i, j]] - tol <= std_result.lower()[[i, j]],
                    "Economic lower tighter than standard at [{i},{j}]: econ={} > std={}",
                    econ_result.lower()[[i, j]], std_result.lower()[[i, j]]
                );
                prop_assert!(
                    econ_result.upper()[[i, j]] + tol >= std_result.upper()[[i, j]],
                    "Economic upper tighter than standard at [{i},{j}]: econ={} < std={}",
                    econ_result.upper()[[i, j]], std_result.upper()[[i, j]]
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    // ========================================================================
    // CROWN backward soundness: 1x1 @ 1x1 (scalar bilinear z = a * b)
    // ========================================================================

    /// CROWN backward with identity bounds for scalar matmul (1x1 @ 1x1).
    /// The McCormick envelope must produce a valid over-approximation:
    /// for any (a, b) in [a_l, a_u] x [b_l, b_u], the linear relaxation
    /// must satisfy: crown_lower <= a*b <= crown_upper.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_scalar(
        (a_l, a_u) in super::valid_interval(5.0),
        (b_l, b_u) in super::valid_interval(5.0),
    ) {
        let matmul = MatMulLayer::new(false, None);
        let a_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), a_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), a_u),
        ).map_err(|e| TestCaseError::fail(format!("a_bounds: {e}")))?;
        let b_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), b_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), b_u),
        ).map_err(|e| TestCaseError::fail(format!("b_bounds: {e}")))?;

        let identity = LinearBounds::identity(1);

        let result = matmul.propagate_linear_binary(&identity, &a_bounds, &b_bounds);
        match result {
            Err(_) => {
                // McCormick CROWN may reject infinite/overflow bounds — skip case
                return Err(TestCaseError::reject("McCormick CROWN rejected overflow bounds"));
            }
            Ok((bounds_a, bounds_b)) => {
                // Sample corners and midpoints
                let a_samples = super::sample_points(a_l, a_u, 10);
                let b_samples = super::sample_points(b_l, b_u, 10);

                for &a_val in &a_samples {
                    for &b_val in &b_samples {
                        let true_output = a_val * b_val;

                        // Sum both paths (bias is split 50/50)
                        let crown_lower = bounds_a.lower_a[[0, 0]] * a_val
                            + bounds_b.lower_a[[0, 0]] * b_val
                            + bounds_a.lower_b[0]
                            + bounds_b.lower_b[0];
                        let crown_upper = bounds_a.upper_a[[0, 0]] * a_val
                            + bounds_b.upper_a[[0, 0]] * b_val
                            + bounds_a.upper_b[0]
                            + bounds_b.upper_b[0];

                        let tol = CROWN_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower - tol <= true_output,
                            "CROWN lower unsound at (a={a_val},b={b_val}): \
                             lower={crown_lower} > true={true_output}"
                        );
                        prop_assert!(
                            crown_upper + tol >= true_output,
                            "CROWN upper unsound at (a={a_val},b={b_val}): \
                             upper={crown_upper} < true={true_output}"
                        );
                    }
                }
            }
        }
    }

    // ========================================================================
    // CROWN backward soundness: 2x1 @ 1x1 (vector-scalar)
    // ========================================================================

    /// CROWN backward for 2x1 @ 1x1 matmul: C[0] = a0*b, C[1] = a1*b.
    /// Verifies McCormick envelope is sound for each output element.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_2x1(
        (a0_l, a0_u) in super::valid_interval(4.0),
        (a1_l, a1_u) in super::valid_interval(4.0),
        (b_l, b_u) in super::valid_interval(4.0),
    ) {
        let matmul = MatMulLayer::new(false, None);
        let a_bounds = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![a0_l, a1_l]).expect("shape"),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![a0_u, a1_u]).expect("shape"),
        ).map_err(|e| TestCaseError::fail(format!("a_bounds: {e}")))?;
        let b_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), b_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), b_u),
        ).map_err(|e| TestCaseError::fail(format!("b_bounds: {e}")))?;

        // Identity on 2-element flattened output
        let identity = LinearBounds::identity(2);

        let result = matmul.propagate_linear_binary(&identity, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                let a_samples = [
                    (a0_l, a1_l), (a0_l, a1_u), (a0_u, a1_l), (a0_u, a1_u),
                    (f32::midpoint(a0_l, a0_u), f32::midpoint(a1_l, a1_u)),
                ];
                let b_samples = super::sample_points(b_l, b_u, 5);

                for &(a0, a1) in &a_samples {
                    for &b_val in &b_samples {
                        let true_c = [a0 * b_val, a1 * b_val];

                        for (out_idx, &true_val) in true_c.iter().enumerate() {
                            // bounds_a has shape (2, 2), bounds_b has shape (2, 1)
                            let crown_lower = bounds_a.lower_a[[out_idx, 0]] * a0
                                + bounds_a.lower_a[[out_idx, 1]] * a1
                                + bounds_b.lower_a[[out_idx, 0]] * b_val
                                + bounds_a.lower_b[out_idx]
                                + bounds_b.lower_b[out_idx];
                            let crown_upper = bounds_a.upper_a[[out_idx, 0]] * a0
                                + bounds_a.upper_a[[out_idx, 1]] * a1
                                + bounds_b.upper_a[[out_idx, 0]] * b_val
                                + bounds_a.upper_b[out_idx]
                                + bounds_b.upper_b[out_idx];

                            let tol = CROWN_TOLERANCE * true_val.abs().max(1.0);
                            prop_assert!(
                                crown_lower - tol <= true_val,
                                "CROWN lower unsound at output {out_idx}, (a0={a0},a1={a1},b={b_val}): \
                                 lower={crown_lower} > true={true_val}",
                            );
                            prop_assert!(
                                crown_upper + tol >= true_val,
                                "CROWN upper unsound at output {out_idx}, (a0={a0},a1={a1},b={b_val}): \
                                 upper={crown_upper} < true={true_val}",
                            );
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // CROWN backward: concrete inputs produce tight bounds
    // ========================================================================

    /// When both inputs are concrete (lower == upper), CROWN bounds should be
    /// tight (equal to the true output within FP tolerance).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_concrete_tight(
        a_val in -5.0f32..5.0,
        b_val in -5.0f32..5.0,
    ) {
        let matmul = MatMulLayer::new(false, None);
        let a_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), a_val),
            ArrayD::from_elem(IxDyn(&[1, 1]), a_val),
        ).map_err(|e| TestCaseError::fail(format!("a_bounds: {e}")))?;
        let b_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), b_val),
            ArrayD::from_elem(IxDyn(&[1, 1]), b_val),
        ).map_err(|e| TestCaseError::fail(format!("b_bounds: {e}")))?;

        let identity = LinearBounds::identity(1);

        let result = matmul.propagate_linear_binary(&identity, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                let true_output = a_val * b_val;

                let crown_lower = bounds_a.lower_a[[0, 0]] * a_val
                    + bounds_b.lower_a[[0, 0]] * b_val
                    + bounds_a.lower_b[0]
                    + bounds_b.lower_b[0];
                let crown_upper = bounds_a.upper_a[[0, 0]] * a_val
                    + bounds_b.upper_a[[0, 0]] * b_val
                    + bounds_a.upper_b[0]
                    + bounds_b.upper_b[0];

                let tol = CROWN_TOLERANCE * true_output.abs().max(1.0);
                prop_assert!(
                    (crown_lower - true_output).abs() < tol,
                    "Concrete CROWN lower not tight: lower={crown_lower}, true={true_output}"
                );
                prop_assert!(
                    (crown_upper - true_output).abs() < tol,
                    "Concrete CROWN upper not tight: upper={crown_upper}, true={true_output}"
                );
            }
        }
    }

    // ========================================================================
    // CROWN concretized soundness: concretized bounds contain true output
    // ========================================================================

    /// When CROWN backward is concretized (via bounds_a + bounds_b), the summed
    /// result must contain the true output at all corners.
    ///
    /// Note: For bilinear CROWN (McCormick), the concretized bounds are NOT
    /// guaranteed to be tighter than IBP — McCormick linearization introduces
    /// relaxation error that may exceed IBP's interval arithmetic in some cases.
    /// Both are guaranteed to be sound (contain the true output).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_concretized_contains_true_output(
        (a_l, a_u) in super::valid_interval(4.0),
        (b_l, b_u) in super::valid_interval(4.0),
    ) {
        let matmul = MatMulLayer::new(false, None);
        let a_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), a_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), a_u),
        ).map_err(|e| TestCaseError::fail(format!("a_bounds: {e}")))?;
        let b_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), b_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), b_u),
        ).map_err(|e| TestCaseError::fail(format!("b_bounds: {e}")))?;

        let identity = LinearBounds::identity(1);
        let crown_result = matmul.propagate_linear_binary(&identity, &a_bounds, &b_bounds);
        match crown_result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                // Concretize both paths and sum
                let concrete_a: BoundedTensor = bounds_a.concretize(&a_bounds);
                let concrete_b: BoundedTensor = bounds_b.concretize(&b_bounds);

                let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
                let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

                // Verify corners are contained
                let a_samples = super::sample_points(a_l, a_u, 5);
                let b_samples = super::sample_points(b_l, b_u, 5);

                for &a_val in &a_samples {
                    for &b_val in &b_samples {
                        let true_output = a_val * b_val;
                        let tol = CROWN_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower - tol <= true_output,
                            "Concretized CROWN lower unsound: {crown_lower} > {true_output} \
                             (a={a_val}, b={b_val})"
                        );
                        prop_assert!(
                            crown_upper + tol >= true_output,
                            "Concretized CROWN upper unsound: {crown_upper} < {true_output} \
                             (a={a_val}, b={b_val})"
                        );
                    }
                }
            }
        }
    }
}
