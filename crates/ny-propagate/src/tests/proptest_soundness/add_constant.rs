// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::arithmetic::AddConstantLayer;
use crate::layers::common::BoundPropagation;
use crate::*;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// f32 tolerance for comparing batched (sum_axis * c) vs non-batched (dot) paths.
/// These are algebraically identical but can differ by a few ULPs from reduction order.
const EQUIV_TOLERANCE: f32 = 2e-5;

/// Tight tolerance for affine CROWN soundness.
/// AddConstant is an exact affine transform — only FP rounding introduces error.
const AFFINE_CROWN_TOLERANCE: f32 = 1e-5;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Batched AddConstant CROWN must match non-batched affine substitution.
    #[ntest::timeout(10000)]
    #[test]
    fn add_constant_batched_matches_non_batched(
        c in -10.0f32..10.0,
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let linear_bounds = LinearBounds::new(
            Array2::from_shape_vec((2, 3), lower_a_vals).unwrap(),
            Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((2, 3), upper_a_vals).unwrap(),
            Array1::from_vec(upper_b_vals),
        ).unwrap();

        let layer = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), c));
        let expected = layer.propagate_linear(&linear_bounds).unwrap().into_owned();

        let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            linear_bounds.lower_a.clone().into_dyn(),
            linear_bounds.lower_b.clone().into_dyn(),
            linear_bounds.upper_a.clone().into_dyn(),
            linear_bounds.upper_b.clone().into_dyn(),
            vec![linear_bounds.num_inputs()],
            vec![linear_bounds.num_outputs()],
        );

        let actual = layer.propagate_linear_batched(&batched_bounds).unwrap();

        prop_assert_eq!(actual.lower_a.shape(), expected.lower_a.shape());
        prop_assert_eq!(actual.upper_a.shape(), expected.upper_a.shape());
        prop_assert_eq!(actual.lower_b.shape(), expected.lower_b.shape());
        prop_assert_eq!(actual.upper_b.shape(), expected.upper_b.shape());
        prop_assert_eq!(actual.input_shape, batched_bounds.input_shape);
        prop_assert_eq!(actual.output_shape, batched_bounds.output_shape);

        for (idx, (&a, &e)) in actual.lower_a.iter().zip(expected.lower_a.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "lower_a mismatch at {idx}: actual={a}, expected={e}"
            );
        }

        for (idx, (&a, &e)) in actual.upper_a.iter().zip(expected.upper_a.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "upper_a mismatch at {idx}: actual={a}, expected={e}"
            );
        }

        for (idx, (&a, &e)) in actual.lower_b.iter().zip(expected.lower_b.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "lower_b mismatch at {idx}: actual={a}, expected={e}"
            );
        }

        for (idx, (&a, &e)) in actual.upper_b.iter().zip(expected.upper_b.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "upper_b mismatch at {idx}: actual={a}, expected={e}"
            );
        }
    }
}

// =============================================================================
// ADDCONSTANT CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// AddConstant (y = x + c) CROWN backward soundness with identity incoming.
    ///
    /// For y = x + c, CROWN backward shifts bias: b_new = b + A@c.
    /// Tests CROWN-IBP equivalence and sampling soundness.
    /// Part of #3157.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_addconstant_crown_identity(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        c in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), c));

        // IBP reference
        let ibp_output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // CROWN backward
        let identity = LinearBounds::identity(3);
        let crown_result = layer.propagate_linear(&identity)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);

        // CROWN-IBP equivalence (AddConstant is linear, so should match closely)
        for i in 0..3 {
            prop_assert!(
                (concrete.lower()[[i]] - ibp_output.lower()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "AddConstant CROWN-IBP lower mismatch at {i}: crown={}, ibp={}",
                concrete.lower()[[i]], ibp_output.lower()[[i]]
            );
            prop_assert!(
                (concrete.upper()[[i]] - ibp_output.upper()[[i]]).abs() <= AFFINE_CROWN_TOLERANCE,
                "AddConstant CROWN-IBP upper mismatch at {i}: crown={}, ibp={}",
                concrete.upper()[[i]], ibp_output.upper()[[i]]
            );
        }

        // Soundness via sampling
        let spts = sample_points(0.0, 1.0, 7);
        for &t0 in &spts {
            for &t1 in &spts {
                for &t2 in &spts {
                    let x0 = l0 + t0 * (u0 - l0);
                    let x1 = l1 + t1 * (u1 - l1);
                    let x2 = l2 + t2 * (u2 - l2);
                    let y = arr1(&[x0 + c, x1 + c, x2 + c]);

                    for i in 0..3 {
                        prop_assert!(
                            y[i] >= concrete.lower()[[i]] - AFFINE_CROWN_TOLERANCE,
                            "AddConstant lower violation at {i}: y={} < lb={}",
                            y[i], concrete.lower()[[i]]
                        );
                        prop_assert!(
                            y[i] <= concrete.upper()[[i]] + AFFINE_CROWN_TOLERANCE,
                            "AddConstant upper violation at {i}: y={} > ub={}",
                            y[i], concrete.upper()[[i]]
                        );
                    }
                }
            }
        }
    }

    /// AddConstant CROWN with non-identity incoming and negative coefficients.
    ///
    /// Tests that CROWN composition correctly handles the bias shift A@c
    /// when incoming coefficient matrix has mixed-sign entries.
    /// Part of #3157.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_addconstant_crown_negative_coeffs(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        c in -3.0f32..3.0,
        k0 in -2.0f32..2.0,
        k1 in -2.0f32..2.0,
        k2 in -2.0f32..2.0,
    ) {
        prop_assume!(k0.abs() > 0.01 || k1.abs() > 0.01 || k2.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), c));

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![k0, k1, k2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![k0, k1, k2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let crown_result = layer.propagate_linear(&incoming)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_linear failed: {e}")
            ))?;
        let concrete = crown_result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        // Verify: k . (x + c) must be in [crown_lower, crown_upper]
        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let combined = k0 * (x0 + c) + k1 * (x1 + c) + k2 * (x2 + c);

                    prop_assert!(
                        combined >= crown_lower - AFFINE_CROWN_TOLERANCE,
                        "AddConstant negative_coeffs lower violation: \
                         k.(x+c)={combined} < lb={crown_lower}, c={c}"
                    );
                    prop_assert!(
                        combined <= crown_upper + AFFINE_CROWN_TOLERANCE,
                        "AddConstant negative_coeffs upper violation: \
                         k.(x+c)={combined} > ub={crown_upper}, c={c}"
                    );
                }
            }
        }
    }
}
