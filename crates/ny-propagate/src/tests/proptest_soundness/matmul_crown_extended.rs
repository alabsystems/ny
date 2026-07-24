// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extended MatMul CROWN proptest soundness coverage.
//!
//! Tests batched CROWN, non-identity incoming coefficients, asymmetric
//! incoming bounds, transpose_b, scale, and f64 directed rounding.
//! Split from matmul.rs to stay under the 1000-line file limit.
//! Part of #2170.

use crate::{BatchedLinearBounds, LinearBounds, MatMulLayer};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use proptest::prelude::*;

use super::matmul::{bounded_2d, eval_matmul, valid_interval_vec};
use super::CROWN_TOLERANCE;

// ============================================================================
// Batched CROWN backward soundness (propagate_linear_batched_binary)
// Part of #2170.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Batched CROWN backward with identity bounds for scalar matmul (1x1 @ 1x1).
    /// Exercises `propagate_linear_batched_binary` — the N-D batched version
    /// of McCormick CROWN used in transformer attention.
    /// Part of #2170.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_batched_crown_scalar(
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

        let identity = BatchedLinearBounds::identity(&[1])
            .map_err(|e| TestCaseError::fail(format!("identity: {e}")))?;

        let result = matmul.propagate_linear_batched_binary(&identity, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                // Concretize each path against its input interval and sum
                let concrete_a = bounds_a.concretize(&a_bounds)
                    .map_err(|e| TestCaseError::fail(format!("concretize_a: {e}")))?;
                let concrete_b = bounds_b.concretize(&b_bounds)
                    .map_err(|e| TestCaseError::fail(format!("concretize_b: {e}")))?;

                let crown_lower = concrete_a.lower().iter().next().unwrap()
                    + concrete_b.lower().iter().next().unwrap();
                let crown_upper = concrete_a.upper().iter().next().unwrap()
                    + concrete_b.upper().iter().next().unwrap();

                // Sample concrete points and verify containment
                let a_samples = super::sample_points(a_l, a_u, 10);
                let b_samples = super::sample_points(b_l, b_u, 10);

                for &a_val in &a_samples {
                    for &b_val in &b_samples {
                        let true_output = a_val * b_val;
                        let tol = CROWN_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower - tol <= true_output,
                            "Batched CROWN lower unsound at (a={a_val},b={b_val}): \
                             {crown_lower} > {true_output}"
                        );
                        prop_assert!(
                            crown_upper + tol >= true_output,
                            "Batched CROWN upper unsound at (a={a_val},b={b_val}): \
                             {crown_upper} < {true_output}"
                        );
                    }
                }
            }
        }
    }

    /// Batched CROWN backward for 2x1 @ 1x1 matmul: C[0] = a0*b, C[1] = a1*b.
    /// Tests batched path with multi-element output.
    /// Part of #2170.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_batched_crown_2x1(
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
        let identity = BatchedLinearBounds::identity(&[2])
            .map_err(|e| TestCaseError::fail(format!("identity: {e}")))?;

        let result = matmul.propagate_linear_batched_binary(&identity, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                let concrete_a = bounds_a.concretize(&a_bounds)
                    .map_err(|e| TestCaseError::fail(format!("concretize_a: {e}")))?;
                let concrete_b = bounds_b.concretize(&b_bounds)
                    .map_err(|e| TestCaseError::fail(format!("concretize_b: {e}")))?;

                let a_samples = [
                    (a0_l, a1_l), (a0_l, a1_u), (a0_u, a1_l), (a0_u, a1_u),
                    (f32::midpoint(a0_l, a0_u), f32::midpoint(a1_l, a1_u)),
                ];
                let b_samples = super::sample_points(b_l, b_u, 5);

                for &(a0, a1) in &a_samples {
                    for &b_val in &b_samples {
                        let true_c = [a0 * b_val, a1 * b_val];

                        for (out_idx, &true_val) in true_c.iter().enumerate() {
                            let crown_lower = concrete_a.lower()[[out_idx]]
                                + concrete_b.lower()[[out_idx]];
                            let crown_upper = concrete_a.upper()[[out_idx]]
                                + concrete_b.upper()[[out_idx]];

                            let tol = CROWN_TOLERANCE * true_val.abs().max(1.0);
                            prop_assert!(
                                crown_lower - tol <= true_val,
                                "Batched CROWN lower unsound at output {out_idx}, \
                                 (a0={a0},a1={a1},b={b_val}): {crown_lower} > {true_val}"
                            );
                            prop_assert!(
                                crown_upper + tol >= true_val,
                                "Batched CROWN upper unsound at output {out_idx}, \
                                 (a0={a0},a1={a1},b={b_val}): {crown_upper} < {true_val}"
                            );
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Non-identity incoming CROWN bounds
// Tests McCormick plane selection with positive and negative incoming
// coefficients (sign-switching logic).
// Part of #2170.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// CROWN backward with random (positive or negative) incoming coefficient.
    /// When c < 0, the McCormick plane selection must swap bound directions.
    /// Part of #2170.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_nonidentity_coeff(
        (a_l, a_u) in super::valid_interval(4.0),
        (b_l, b_u) in super::valid_interval(4.0),
        c in -3.0f32..3.0,
        d in -2.0f32..2.0,
    ) {
        // Skip c=0 (degenerate: output is constant bias)
        prop_assume!(c.abs() >= 1e-6, "c must be non-zero");

        let matmul = MatMulLayer::new(false, None);
        let a_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), a_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), a_u),
        ).map_err(|e| TestCaseError::fail(format!("a_bounds: {e}")))?;
        let b_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), b_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), b_u),
        ).map_err(|e| TestCaseError::fail(format!("b_bounds: {e}")))?;

        // Non-identity incoming: lower and upper share the same coefficient c
        // (symmetric case — asymmetric is tested separately below).
        let incoming = LinearBounds::new(
            Array2::from_elem((1, 1), c),
            Array1::from_vec(vec![d]),
            Array2::from_elem((1, 1), c),
            Array1::from_vec(vec![d]),
        ).map_err(|e| TestCaseError::fail(format!("incoming: {e}")))?;

        let result = matmul.propagate_linear_binary(&incoming, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                let a_samples = super::sample_points(a_l, a_u, 10);
                let b_samples = super::sample_points(b_l, b_u, 10);

                for &a_val in &a_samples {
                    for &b_val in &b_samples {
                        // True incoming function value: c * (a*b) + d
                        let true_output = c * (a_val * b_val) + d;

                        // CROWN propagated bounds (summing both paths)
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
                            "Non-identity CROWN lower unsound at (a={a_val},b={b_val},c={c},d={d}): \
                             {crown_lower} > {true_output}"
                        );
                        prop_assert!(
                            crown_upper + tol >= true_output,
                            "Non-identity CROWN upper unsound at (a={a_val},b={b_val},c={c},d={d}): \
                             {crown_upper} < {true_output}"
                        );
                    }
                }
            }
        }
    }

    /// CROWN backward with asymmetric incoming bounds (lower_a != upper_a).
    /// This occurs after ReLU relaxation where the lower and upper relaxation
    /// slopes differ. Tests the full asymmetric McCormick case.
    /// Part of #2170.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_asymmetric_incoming(
        (a_l, a_u) in super::valid_interval(4.0),
        (b_l, b_u) in super::valid_interval(4.0),
        c_lower in -3.0f32..3.0,
        c_upper in -3.0f32..3.0,
        d_lower in -2.0f32..2.0,
        d_upper in -2.0f32..2.0,
    ) {
        // Skip degenerate cases
        prop_assume!(c_lower.abs() >= 1e-6 || c_upper.abs() >= 1e-6, "at least one coefficient must be non-zero");

        let matmul = MatMulLayer::new(false, None);
        let a_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), a_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), a_u),
        ).map_err(|e| TestCaseError::fail(format!("a_bounds: {e}")))?;
        let b_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), b_l),
            ArrayD::from_elem(IxDyn(&[1, 1]), b_u),
        ).map_err(|e| TestCaseError::fail(format!("b_bounds: {e}")))?;

        // Asymmetric incoming: lower and upper paths have different coefficients.
        let incoming = LinearBounds::new(
            Array2::from_elem((1, 1), c_lower),
            Array1::from_vec(vec![d_lower]),
            Array2::from_elem((1, 1), c_upper),
            Array1::from_vec(vec![d_upper]),
        ).map_err(|e| TestCaseError::fail(format!("incoming: {e}")))?;

        let result = matmul.propagate_linear_binary(&incoming, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                let a_samples = super::sample_points(a_l, a_u, 10);
                let b_samples = super::sample_points(b_l, b_u, 10);

                for &a_val in &a_samples {
                    for &b_val in &b_samples {
                        let z = a_val * b_val;
                        // True incoming function values
                        let true_lower = c_lower * z + d_lower;
                        let true_upper = c_upper * z + d_upper;

                        // CROWN propagated bounds (summing both paths)
                        let crown_lower = bounds_a.lower_a[[0, 0]] * a_val
                            + bounds_b.lower_a[[0, 0]] * b_val
                            + bounds_a.lower_b[0]
                            + bounds_b.lower_b[0];
                        let crown_upper = bounds_a.upper_a[[0, 0]] * a_val
                            + bounds_b.upper_a[[0, 0]] * b_val
                            + bounds_a.upper_b[0]
                            + bounds_b.upper_b[0];

                        let tol_lower = CROWN_TOLERANCE * true_lower.abs().max(1.0);
                        let tol_upper = CROWN_TOLERANCE * true_upper.abs().max(1.0);
                        prop_assert!(
                            crown_lower - tol_lower <= true_lower,
                            "Asymmetric CROWN lower unsound at (a={a_val},b={b_val}): \
                             crown={crown_lower} > true={true_lower} \
                             (c_l={c_lower},c_u={c_upper},d_l={d_lower},d_u={d_upper})"
                        );
                        prop_assert!(
                            crown_upper + tol_upper >= true_upper,
                            "Asymmetric CROWN upper unsound at (a={a_val},b={b_val}): \
                             crown={crown_upper} < true={true_upper} \
                             (c_l={c_lower},c_u={c_upper},d_l={d_lower},d_u={d_upper})"
                        );
                    }
                }
            }
        }
    }
}

// ============================================================================
// CROWN backward with transpose_b and scale parameters
// Part of #2170: Prover finding — zero CROWN proptest coverage for these params.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// CROWN backward with transpose_b: A [1,2] @ B^T [1,2] → [1,1].
    /// Verifies McCormick envelope is sound when B is transposed.
    /// Part of #2170.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_transpose_b(
        (a_lo, a_hi) in valid_interval_vec(2, 5.0),  // 1x2
        (b_lo, b_hi) in valid_interval_vec(2, 5.0),  // 1x2 (transposed to 2x1)
    ) {
        let matmul = MatMulLayer::new(true, None);
        let a_bounds = bounded_2d(a_lo.clone(), a_hi.clone(), 1, 2);
        let b_bounds = bounded_2d(b_lo.clone(), b_hi.clone(), 1, 2);

        let identity = LinearBounds::identity(1);
        let result = matmul.propagate_linear_binary(&identity, &a_bounds, &b_bounds);
        match result {
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                // Concretize and verify soundness at corners
                let concrete_a: BoundedTensor = bounds_a.concretize(&a_bounds);
                let concrete_b: BoundedTensor = bounds_b.concretize(&b_bounds);

                let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
                let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

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
                        let tol = CROWN_TOLERANCE * true_val.abs().max(1.0);
                        prop_assert!(
                            crown_lower - tol <= true_val,
                            "CROWN transpose_b lower unsound: {crown_lower} > {true_val}"
                        );
                        prop_assert!(
                            crown_upper + tol >= true_val,
                            "CROWN transpose_b upper unsound: {crown_upper} < {true_val}"
                        );
                    }
                }
            }
        }
    }

    /// CROWN backward with scale: scalar matmul C = scale * (a * b).
    /// Verifies McCormick envelope is sound when scale is applied.
    /// Part of #2170.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_matmul_crown_with_scale(
        (a_l, a_u) in super::valid_interval(5.0),
        (b_l, b_u) in super::valid_interval(5.0),
        scale in -3.0f32..3.0,
    ) {
        let matmul = MatMulLayer::new(false, Some(scale));
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
            Err(_) => return Err(TestCaseError::reject("McCormick CROWN rejected inputs")),
            Ok((bounds_a, bounds_b)) => {
                let a_samples = super::sample_points(a_l, a_u, 10);
                let b_samples = super::sample_points(b_l, b_u, 10);

                for &a_val in &a_samples {
                    for &b_val in &b_samples {
                        let true_output = scale * a_val * b_val;

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
                            "CROWN scale={scale} lower unsound: {crown_lower} > {true_output} \
                             (a={a_val}, b={b_val})"
                        );
                        prop_assert!(
                            crown_upper + tol >= true_output,
                            "CROWN scale={scale} upper unsound: {crown_upper} < {true_output} \
                             (a={a_val}, b={b_val})"
                        );
                    }
                }
            }
        }
    }
}

// ============================================================================
// f64 precision regression tests for directed rounding (#2164, #2183)
// ============================================================================

/// Setup for f64 directed rounding tests: creates point-interval inputs
/// that trigger f32 vs f64 accumulation divergence.
/// Returns (a_bounds, b_bounds, true_total_f64, expected_lower_half, expected_upper_half).
fn directed_rounding_setup(k: usize) -> (BoundedTensor, BoundedTensor, f64, f32, f32) {
    let a_bounds = bounded_2d(vec![1.0; k], vec![1.0; k], 1, k);
    let b_bounds = bounded_2d(vec![-0.1_f32; k], vec![-0.1_f32; k], k, 1);

    let true_total_f64: f64 = (0..k).map(|_| 0.1_f32 as f64).sum();
    let true_half_f64 = true_total_f64 * 0.5;
    let cast_half = true_half_f64 as f32;

    // Guardrail: f32 accumulation must diverge from f64 for this test to be meaningful.
    let mut f32_total = 0.0_f32;
    for _ in 0..k {
        f32_total += 0.1_f32;
    }
    assert_ne!(
        f32_total.to_bits(),
        (true_total_f64 as f32).to_bits(),
        "test setup must exercise f64 vs f32 accumulation divergence",
    );

    (
        a_bounds,
        b_bounds,
        true_total_f64,
        next_down_f32(cast_half),
        next_up_f32(cast_half),
    )
}

/// Assert that a bias value uses directed rounding: lower uses next_down, upper uses next_up.
fn assert_directed_rounding_bias(
    lower_b: f32,
    upper_b: f32,
    expected_lower: f32,
    expected_upper: f32,
    true_half: f64,
) {
    assert_eq!(
        lower_b.to_bits(),
        expected_lower.to_bits(),
        "lower_b must use next_down_f32"
    );
    assert_eq!(
        upper_b.to_bits(),
        expected_upper.to_bits(),
        "upper_b must use next_up_f32"
    );
    assert!(
        (lower_b as f64) <= true_half,
        "lower_b must stay <= true f64 bias"
    );
    assert!(
        (upper_b as f64) >= true_half,
        "upper_b must stay >= true f64 bias"
    );
}

/// Regression for #2183: MatMul CROWN bias path must accumulate in f64 and
/// apply directed rounding on the final f32 cast.
#[ntest::timeout(10000)]
#[test]
fn soundness_matmul_crown_directed_rounding_bias_2183() {
    let (a_bounds, b_bounds, true_total_f64, exp_lower, exp_upper) = directed_rounding_setup(100);
    let true_half_f64 = true_total_f64 * 0.5;

    let incoming = LinearBounds::new(
        Array2::from_elem((1, 1), 1.0_f32),
        Array1::zeros(1),
        Array2::from_elem((1, 1), 1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();

    let (bounds_a, bounds_b) = MatMulLayer::new(false, None)
        .propagate_linear_binary(&incoming, &a_bounds, &b_bounds)
        .expect("matmul CROWN failed");

    for branch in [&bounds_a, &bounds_b] {
        assert_directed_rounding_bias(
            branch.lower_b[0],
            branch.upper_b[0],
            exp_lower,
            exp_upper,
            true_half_f64,
        );
    }
}

/// Regression for #2164/#2183: Batched CROWN bias path must also accumulate
/// in f64 and apply directed rounding, same as the unbatched path.
/// Part of #2170.
#[ntest::timeout(10000)]
#[test]
fn soundness_matmul_batched_crown_directed_rounding_bias() {
    let (a_bounds, b_bounds, true_total_f64, exp_lower, exp_upper) = directed_rounding_setup(100);
    let true_half_f64 = true_total_f64 * 0.5;

    let identity = BatchedLinearBounds::identity(&[1]).expect("identity");
    let (bounds_a, bounds_b) = MatMulLayer::new(false, None)
        .propagate_linear_batched_binary(&identity, &a_bounds, &b_bounds)
        .expect("batched matmul CROWN failed");

    // Concretize and check that bounds bracket the true output (-10.0).
    let concrete_a = bounds_a.concretize(&a_bounds).expect("concretize_a");
    let concrete_b = bounds_b.concretize(&b_bounds).expect("concretize_b");
    let total_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
    let total_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];
    let true_output = -10.0_f32;
    assert!(
        total_lower <= true_output + 1e-4,
        "lower: {total_lower} > {true_output}"
    );
    assert!(
        total_upper >= true_output - 1e-4,
        "upper: {total_upper} < {true_output}"
    );

    // Check each half individually matches directed rounding pattern
    for branch in [&bounds_a, &bounds_b] {
        let lb = *branch.lower_b.iter().next().unwrap();
        let ub = *branch.upper_b.iter().next().unwrap();
        assert_directed_rounding_bias(lb, ub, exp_lower, exp_upper, true_half_f64);
    }
}
