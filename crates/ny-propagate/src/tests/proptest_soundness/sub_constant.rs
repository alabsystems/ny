// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched-vs-non-batched equivalence proptest for SubConstant CROWN backward.
//!
//! Mirrors the AddConstant batched proptest (`add_constant_batched_matches_non_batched`)
//! to validate that the batched f64 accumulation + directed rounding path (#2423)
//! produces results consistent with the non-batched path.
//!
//! Part of #2423.

use crate::layers::arithmetic::SubConstantLayer;
use crate::layers::common::BoundPropagation;
use crate::*;
use ndarray::{Array2, ArrayD, IxDyn};
use proptest::prelude::*;

/// f32 tolerance for comparing batched (sum_axis * c) vs non-batched (dot) paths.
/// These are algebraically identical but can differ by a few ULPs from reduction order.
const EQUIV_TOLERANCE: f32 = 2e-5;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// SubConstant (y = x - c) batched CROWN must match non-batched affine substitution.
    ///
    /// The batched path uses f64 sum_axis + f64 multiply + directed rounding (#2423).
    /// The non-batched path uses scalar_row_sum_f64 + directed rounding.
    /// Both should produce equivalent results within FP tolerance.
    #[ntest::timeout(10000)]
    #[test]
    fn sub_constant_batched_matches_non_batched(
        c in -10.0f32..10.0,
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let linear_bounds = LinearBounds::new(
            Array2::from_shape_vec((2, 3), lower_a_vals).unwrap(),
            ndarray::Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((2, 3), upper_a_vals).unwrap(),
            ndarray::Array1::from_vec(upper_b_vals),
        ).unwrap();

        let layer = SubConstantLayer::scalar(c);
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

    /// SubConstant reverse (y = c - x) batched CROWN must match non-batched.
    ///
    /// The reverse path negates coefficient matrices and adds bias, which
    /// the batched path must handle identically to the non-batched path.
    #[ntest::timeout(10000)]
    #[test]
    fn sub_constant_reverse_batched_matches_non_batched(
        c in -10.0f32..10.0,
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
    ) {
        let linear_bounds = LinearBounds::new(
            Array2::from_shape_vec((2, 3), lower_a_vals).unwrap(),
            ndarray::Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((2, 3), upper_a_vals).unwrap(),
            ndarray::Array1::from_vec(upper_b_vals),
        ).unwrap();

        let layer = SubConstantLayer::new_reverse(ArrayD::from_elem(IxDyn(&[]), c));
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
                "reverse lower_a mismatch at {idx}: actual={a}, expected={e}"
            );
        }

        for (idx, (&a, &e)) in actual.upper_a.iter().zip(expected.upper_a.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "reverse upper_a mismatch at {idx}: actual={a}, expected={e}"
            );
        }

        for (idx, (&a, &e)) in actual.lower_b.iter().zip(expected.lower_b.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "reverse lower_b mismatch at {idx}: actual={a}, expected={e}"
            );
        }

        for (idx, (&a, &e)) in actual.upper_b.iter().zip(expected.upper_b.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= EQUIV_TOLERANCE,
                "reverse upper_b mismatch at {idx}: actual={a}, expected={e}"
            );
        }
    }
}
