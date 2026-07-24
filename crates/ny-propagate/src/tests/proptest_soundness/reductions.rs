// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::common::BoundPropagation;
use crate::layers::{Layer, ReduceMeanLayer, ReduceSumLayer};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

// =============================================================================
// REDUCTION OPERATION SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ReduceMean IBP soundness: for any x in [l, u], mean(x) is in computed bounds.
    /// Uses keepdims=true to preserve output shape for easier indexing.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_mean_ibp_1d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn()
        ).unwrap();

        // Use keepdims=true so output has shape [1] instead of []
        let reduce_mean = ReduceMeanLayer::new(vec![0], true);
        let output = reduce_mean.propagate_ibp(&input).unwrap();

        // Test corner points
        let corners = vec![
            arr1(&[l0, l1, l2, l3]),
            arr1(&[u0, u1, u2, u3]),
            arr1(&[u0, l1, l2, l3]),
            arr1(&[l0, u1, l2, l3]),
            arr1(&[l0, l1, u2, l3]),
            arr1(&[l0, l1, l2, u3]),
            arr1(&[f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2), f32::midpoint(l3, u3)]),
        ];

        for x in corners {
            let mean_x = x.mean().unwrap();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= mean_x && mean_x <= output.upper()[[0]] + FP_TOLERANCE,
                "ReduceMean soundness violation: mean({:?})={} not in [{}, {}]",
                x, mean_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// ReduceSum IBP soundness: for any x in [l, u], sum(x) is in computed bounds.
    /// Uses keepdims=true to preserve output shape for easier indexing.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_sum_ibp_1d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        // Use keepdims=true so output has shape [1] instead of []
        let reduce_sum = ReduceSumLayer::new(vec![0], true);
        let output = reduce_sum.propagate_ibp(&input).unwrap();

        // Test corner points
        let corners = vec![
            arr1(&[l0, l1, l2]),
            arr1(&[u0, u1, u2]),
            arr1(&[u0, l1, l2]),
            arr1(&[l0, u1, l2]),
            arr1(&[l0, l1, u2]),
            arr1(&[f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)]),
        ];

        for x in corners {
            let sum_x = x.sum();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= sum_x && sum_x <= output.upper()[[0]] + FP_TOLERANCE,
                "ReduceSum soundness violation: sum({:?})={} not in [{}, {}]",
                x, sum_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}

// =============================================================================
// CROWN BACKWARD SOUNDNESS TESTS FOR REDUCTIONS
// =============================================================================
// These tests verify that CROWN backward propagation through ReduceMean/ReduceSum
// is sound. Since reductions are linear operations, CROWN with identity incoming
// bounds should produce results that exactly match IBP (within FP tolerance).
//
// Critical coverage gap: prior to these tests, only IBP proptests existed for
// reductions. The identity-fallback bug (#1733) in propagate_linear was only
// caught by manual coefficient inspection tests, not by randomized testing.
//
// These tests exercise the `propagate_crown_backward` trait method via the Layer
// enum, which is the actual dispatch path used during network-level CROWN
// propagation. Unit tests call `propagate_linear_with_bounds` directly, which
// would still pass even if the trait dispatch were broken.

/// Helper: concretize CROWN linear bounds against a pre-activation tensor.
/// Returns (lower, upper) vectors of the concretized bounds.
fn concretize_crown_1d(
    result: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ReduceMean CROWN backward soundness via trait dispatch.
    ///
    /// Uses `Layer::ReduceMean` enum variant to call `propagate_crown_backward`
    /// (the trait method), not the inherent `propagate_linear_with_bounds`.
    /// This tests the actual call path used during network CROWN propagation.
    ///
    /// For a 2x3 input reduced over axis -1 with keepdims=true:
    /// - Output is 2x1 (flattened to 2 for LinearBounds)
    /// - CROWN backward should produce a (2, 6) coefficient matrix with 1/3 entries
    /// - Concretized bounds should match IBP exactly (linear operation = no relaxation gap)
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_mean_crown_backward_trait_dispatch(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceMeanLayer::new(vec![-1], true);
        let layer = Layer::ReduceMean(reduce.clone());

        // Use identity incoming bounds (2 outputs for 2-element reduced output)
        let identity = LinearBounds::identity(2);

        // Call via trait dispatch (the actual path used during CROWN propagation)
        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        // Verify coefficient matrix shape: (2 outputs, 6 inputs)
        prop_assert_eq!(result.lower_a.nrows(), 2);
        prop_assert_eq!(result.lower_a.ncols(), 6);

        // Concretize and compare against IBP
        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

        // For linear ops, CROWN should match IBP exactly (within FP tolerance)
        for i in 0..2 {
            let ibp_l = ibp_result.lower().iter().nth(i).copied().unwrap();
            let ibp_u = ibp_result.upper().iter().nth(i).copied().unwrap();

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "ReduceMean CROWN-IBP mismatch at row {i}: crown_lower={} ibp_lower={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "ReduceMean CROWN-IBP mismatch at row {i}: crown_upper={} ibp_upper={ibp_u}",
                crown_upper[i]
            );
        }

        // Also verify pointwise soundness: sampled mean(x) must be within bounds
        for (row_idx, row_lower, row_upper) in &[
            (0usize, &[l0, l1, l2][..], &[u0, u1, u2][..]),
            (1, &[l3, l4, l5][..], &[u3, u4, u5][..]),
        ] {
            let samples: Vec<Vec<f32>> = (0..3)
                .map(|j| sample_points(row_lower[j], row_upper[j], 5))
                .collect();

            // Check a few sampled combinations
            for &s0 in &samples[0] {
                for &s1 in &samples[1] {
                    for &s2 in &samples[2] {
                        let mean_x = (s0 + s1 + s2) / 3.0;
                        prop_assert!(
                            crown_lower[*row_idx] - FP_TOLERANCE <= mean_x,
                            "ReduceMean lower bound violation at row {row_idx}: \
                             lb={} > mean({s0},{s1},{s2})={mean_x}",
                            crown_lower[*row_idx]
                        );
                        prop_assert!(
                            mean_x <= crown_upper[*row_idx] + FP_TOLERANCE,
                            "ReduceMean upper bound violation at row {row_idx}: \
                             mean({s0},{s1},{s2})={mean_x} > ub={}",
                            crown_upper[*row_idx]
                        );
                    }
                }
            }
        }
    }

    /// ReduceSum CROWN backward soundness via trait dispatch.
    ///
    /// Same approach as ReduceMean but with sum (no 1/n scaling).
    /// Exercises `Layer::ReduceSum` enum → `propagate_crown_backward` trait dispatch.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_sum_crown_backward_trait_dispatch(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceSumLayer::new(vec![-1], true);
        let layer = Layer::ReduceSum(reduce.clone());

        let identity = LinearBounds::identity(2);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        prop_assert_eq!(result.lower_a.nrows(), 2);
        prop_assert_eq!(result.lower_a.ncols(), 6);

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

        for i in 0..2 {
            let ibp_l = ibp_result.lower().iter().nth(i).copied().unwrap();
            let ibp_u = ibp_result.upper().iter().nth(i).copied().unwrap();

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "ReduceSum CROWN-IBP mismatch at row {i}: crown_lower={} ibp_lower={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "ReduceSum CROWN-IBP mismatch at row {i}: crown_upper={} ibp_upper={ibp_u}",
                crown_upper[i]
            );
        }

        // Pointwise soundness
        for (row_idx, row_lower, row_upper) in &[
            (0usize, &[l0, l1, l2][..], &[u0, u1, u2][..]),
            (1, &[l3, l4, l5][..], &[u3, u4, u5][..]),
        ] {
            let samples: Vec<Vec<f32>> = (0..3)
                .map(|j| sample_points(row_lower[j], row_upper[j], 5))
                .collect();

            for &s0 in &samples[0] {
                for &s1 in &samples[1] {
                    for &s2 in &samples[2] {
                        let sum_x = s0 + s1 + s2;
                        prop_assert!(
                            crown_lower[*row_idx] - FP_TOLERANCE <= sum_x,
                            "ReduceSum lower bound violation at row {row_idx}: \
                             lb={} > sum({s0},{s1},{s2})={sum_x}",
                            crown_lower[*row_idx]
                        );
                        prop_assert!(
                            sum_x <= crown_upper[*row_idx] + FP_TOLERANCE,
                            "ReduceSum upper bound violation at row {row_idx}: \
                             sum({s0},{s1},{s2})={sum_x} > ub={}",
                            crown_upper[*row_idx]
                        );
                    }
                }
            }
        }
    }

    /// ReduceSum CROWN backward with non-identity incoming bounds (including negative coefficients).
    ///
    /// Same approach as ReduceMean non-identity test but with sum (coefficient 1.0 vs 1/n).
    /// Exercises negative coefficients which flip the lower/upper bound relationship.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reduce_sum_crown_nonidentity_incoming(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceSumLayer::new(vec![-1], true);
        let layer = Layer::ReduceSum(reduce);

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // f(x) = c0 * sum(x[0:3]) + c1 * sum(x[3:6])
        let samples: Vec<Vec<f32>> = (0..6)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2),
                    3 => (l3, u3), 4 => (l4, u4), _ => (l5, u5),
                };
                sample_points(lj, uj, 3)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        for &s4 in &samples[4] {
                            for &s5 in &samples[5] {
                                let sum_row0 = s0 + s1 + s2;
                                let sum_row1 = s3 + s4 + s5;
                                let true_output = c0 * sum_row0 + c1 * sum_row1;

                                let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                                prop_assert!(
                                    crown_lower[0] - scale_tol <= true_output,
                                    "ReduceSum CROWN lower violated: lb={} > true={true_output} \
                                     (c0={c0}, c1={c1})",
                                    crown_lower[0]
                                );
                                prop_assert!(
                                    true_output <= crown_upper[0] + scale_tol,
                                    "ReduceSum CROWN upper violated: true={true_output} > ub={} \
                                     (c0={c0}, c1={c1})",
                                    crown_upper[0]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// ReduceSum CROWN backward with keepdims=false.
    ///
    /// Tests the filter-based coordinate computation path (used when keepdims=false)
    /// instead of the map-based path (keepdims=true).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_sum_crown_no_keepdims(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceSumLayer::new(vec![-1], false);
        let layer = Layer::ReduceSum(reduce.clone());

        let identity = LinearBounds::identity(2);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward (no keepdims) failed: {e}")
            ))?;

        prop_assert_eq!(result.lower_a.nrows(), 2);
        prop_assert_eq!(result.lower_a.ncols(), 6);

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

        for i in 0..2 {
            let ibp_l = ibp_result.lower().iter().nth(i).copied().unwrap();
            let ibp_u = ibp_result.upper().iter().nth(i).copied().unwrap();

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "ReduceSum no-keepdims CROWN-IBP mismatch at row {i}: \
                 crown_lower={} ibp_lower={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "ReduceSum no-keepdims CROWN-IBP mismatch at row {i}: \
                 crown_upper={} ibp_upper={ibp_u}",
                crown_upper[i]
            );
        }
    }

    /// ReduceSum CROWN backward with asymmetric lower_a/upper_a.
    ///
    /// Tests where lower and upper coefficient matrices differ, which occurs
    /// after passing through nonlinear layers in multi-layer CROWN backward.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reduce_sum_crown_asymmetric_bounds(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
    ) {
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lc0, lc1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![uc0, uc1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceSumLayer::new(vec![-1], true);
        let layer = Layer::ReduceSum(reduce);

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward (asymmetric) failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let samples: Vec<Vec<f32>> = (0..6)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2),
                    3 => (l3, u3), 4 => (l4, u4), _ => (l5, u5),
                };
                sample_points(lj, uj, 3)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        for &s4 in &samples[4] {
                            for &s5 in &samples[5] {
                                let sum_row0 = s0 + s1 + s2;
                                let sum_row1 = s3 + s4 + s5;

                                let lower_output = lc0 * sum_row0 + lc1 * sum_row1;
                                let upper_output = uc0 * sum_row0 + uc1 * sum_row1;

                                let lower_tol = FP_TOLERANCE * lower_output.abs().max(1.0);
                                let upper_tol = FP_TOLERANCE * upper_output.abs().max(1.0);

                                prop_assert!(
                                    crown_lower[0] - lower_tol <= lower_output,
                                    "Asymmetric lower violated: crown_lb={} > lower_f={lower_output} \
                                     (lc0={lc0}, lc1={lc1})",
                                    crown_lower[0]
                                );
                                prop_assert!(
                                    upper_output <= crown_upper[0] + upper_tol,
                                    "Asymmetric upper violated: upper_f={upper_output} > crown_ub={} \
                                     (uc0={uc0}, uc1={uc1})",
                                    crown_upper[0]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// ReduceMean CROWN backward with non-identity incoming bounds.
    ///
    /// Tests with arbitrary incoming linear bounds (not just identity), which exercises
    /// the full coefficient multiplication path. Uses the Layer enum for trait dispatch.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reduce_mean_crown_nonidentity_incoming(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        // Non-identity incoming bounds: one output row with coefficients [c0, c1]
        // applied to the 2-element reduction output.
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceMeanLayer::new(vec![-1], true);
        let layer = Layer::ReduceMean(reduce);

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Verify against sampled true outputs:
        // f(x) = c0 * mean(x[0:3]) + c1 * mean(x[3:6])
        let samples: Vec<Vec<f32>> = (0..6)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2),
                    3 => (l3, u3), 4 => (l4, u4), _ => (l5, u5),
                };
                sample_points(lj, uj, 3)
            })
            .collect();

        // Check all combinations (3^6 = 729 total)
        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        for &s4 in &samples[4] {
                            for &s5 in &samples[5] {
                                let mean_row0 = (s0 + s1 + s2) / 3.0;
                                let mean_row1 = (s3 + s4 + s5) / 3.0;
                                let true_output = c0 * mean_row0 + c1 * mean_row1;

                                let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                                prop_assert!(
                                    crown_lower[0] - scale_tol <= true_output,
                                    "ReduceMean CROWN lower violated: lb={} > true={true_output} \
                                     (c0={c0}, c1={c1})",
                                    crown_lower[0]
                                );
                                prop_assert!(
                                    true_output <= crown_upper[0] + scale_tol,
                                    "ReduceMean CROWN upper violated: true={true_output} > ub={} \
                                     (c0={c0}, c1={c1})",
                                    crown_upper[0]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// ReduceMean CROWN backward with keepdims=false.
    ///
    /// The keepdims=false path uses filter-based coordinate computation (reduction.rs:234-239)
    /// instead of map-based (reduction.rs:228-232). This exercises a different code path
    /// that was identified as untested in the reflection iteration.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_mean_crown_no_keepdims(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        // keepdims=false: output shape is [2] (axis -1 removed), not [2, 1]
        let reduce = ReduceMeanLayer::new(vec![-1], false);
        let layer = Layer::ReduceMean(reduce.clone());

        let identity = LinearBounds::identity(2);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward (no keepdims) failed: {e}")
            ))?;

        // Shape: (2 outputs, 6 inputs) — same as keepdims=true
        prop_assert_eq!(result.lower_a.nrows(), 2);
        prop_assert_eq!(result.lower_a.ncols(), 6);

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

        // CROWN should match IBP for linear ops
        for i in 0..2 {
            let ibp_l = ibp_result.lower().iter().nth(i).copied().unwrap();
            let ibp_u = ibp_result.upper().iter().nth(i).copied().unwrap();

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "ReduceMean no-keepdims CROWN-IBP mismatch at row {i}: \
                 crown_lower={} ibp_lower={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "ReduceMean no-keepdims CROWN-IBP mismatch at row {i}: \
                 crown_upper={} ibp_upper={ibp_u}",
                crown_upper[i]
            );
        }
    }

    /// ReduceSum batched CROWN backward with identity bounds (#287).
    ///
    /// Tests propagate_linear_batched for ReduceSum (last axis, keepdims=true).
    /// For input [2, 3] reduced to [2, 1], identity batched CROWN backward
    /// should produce bounds matching IBP (linear operation = exact).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_sum_batched_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceSumLayer::new(vec![-1], true);

        // Output shape: [2, 1]
        let bounds = BatchedLinearBounds::identity(&[2, 1]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = reduce.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("ReduceSum batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        let ibp = reduce.propagate_ibp(&pre_activation).unwrap();
        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        // The batched concretization now accumulates the dot product in f64
        // (BLAS DGEMV on f32-cast operands) and rounds only the single f64->f32
        // cast — the conservative `gamma_{2n+2}*S` envelope of the earlier
        // f32-BLAS path is gone. CROWN therefore matches IBP up to ordinary FP
        // rounding again: a result-relative `FP_TOLERANCE` (scaled by the result
        // magnitude, which can reach ~30 for three summed inputs of |x|<=10).
        for i in 0..2 {
            let tol = FP_TOLERANCE * ibp_lower[i].abs().max(ibp_upper[i].abs()).max(1.0);
            prop_assert!(
                (conc_lower[i] - ibp_lower[i]).abs() < tol,
                "ReduceSum batched CROWN-IBP lower mismatch at {i}: batched={} ibp={} tol={tol}",
                conc_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (conc_upper[i] - ibp_upper[i]).abs() < tol,
                "ReduceSum batched CROWN-IBP upper mismatch at {i}: batched={} ibp={} tol={tol}",
                conc_upper[i], ibp_upper[i]
            );
        }
    }

    /// ReduceMean batched CROWN backward with identity bounds (#287).
    ///
    /// Tests propagate_linear_batched for ReduceMean (last axis, keepdims=true).
    /// For input [2, 3] reduced to [2, 1], identity batched CROWN backward
    /// should produce bounds matching IBP (linear operation with scale 1/3).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_mean_batched_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceMeanLayer::new(vec![-1], true);

        // Output shape: [2, 1]
        let bounds = BatchedLinearBounds::identity(&[2, 1]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = reduce.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("ReduceMean batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        let ibp = reduce.propagate_ibp(&pre_activation).unwrap();
        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        // As in the ReduceSum case: the f64-accumulate batched path drops the
        // `gamma_{2n+2}*S` envelope, so CROWN matches IBP up to ordinary FP
        // rounding. ReduceMean coeffs are 1/3, so the result magnitude is ~mean
        // of |x|<=10; use a result-relative `FP_TOLERANCE`.
        for i in 0..2 {
            let tol = FP_TOLERANCE * ibp_lower[i].abs().max(ibp_upper[i].abs()).max(1.0);
            prop_assert!(
                (conc_lower[i] - ibp_lower[i]).abs() < tol,
                "ReduceMean batched CROWN-IBP lower mismatch at {i}: batched={} ibp={} tol={tol}",
                conc_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (conc_upper[i] - ibp_upper[i]).abs() < tol,
                "ReduceMean batched CROWN-IBP upper mismatch at {i}: batched={} ibp={} tol={tol}",
                conc_upper[i], ibp_upper[i]
            );
        }
    }

    /// ReduceSum batched CROWN backward with pointwise soundness (#287).
    ///
    /// For ReduceSum(axis=-1, keepdims=true) on [3, 4], verify that for every
    /// sampled input point, the reduced output lies within concretized bounds.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reduce_sum_batched_crown_pointwise(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceSumLayer::new(vec![-1], true);

        let bounds = BatchedLinearBounds::identity(&[2, 1]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = reduce.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("ReduceSum batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        // Verify pointwise: sampled sums within bounds
        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3), (l4, u4), (l5, u5)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    let sum0 = s0 + s1 + s2;
                    prop_assert!(
                        concrete.lower().iter().next().copied().unwrap() - FP_TOLERANCE <= sum0
                        && sum0 <= concrete.upper().iter().next().copied().unwrap() + FP_TOLERANCE,
                        "ReduceSum batched row 0: sum({s0},{s1},{s2})={sum0} not in bounds"
                    );
                }
            }
        }
        for &s3 in &samples[3] {
            for &s4 in &samples[4] {
                for &s5 in &samples[5] {
                    let sum1 = s3 + s4 + s5;
                    prop_assert!(
                        concrete.lower().iter().nth(1).copied().unwrap() - FP_TOLERANCE <= sum1
                        && sum1 <= concrete.upper().iter().nth(1).copied().unwrap() + FP_TOLERANCE,
                        "ReduceSum batched row 1: sum({s3},{s4},{s5})={sum1} not in bounds"
                    );
                }
            }
        }
    }

    /// ReduceMean CROWN backward with asymmetric lower_a/upper_a.
    ///
    /// Self-audit fix: the non-identity test used lower_a == upper_a, which
    /// would miss bugs where only one coefficient matrix is correctly transformed.
    /// This test uses different coefficients for lower_a and upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reduce_mean_crown_asymmetric_bounds(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
    ) {
        // Asymmetric incoming bounds: lower_a != upper_a
        // This occurs after passing through nonlinear layers in CROWN backward.
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lc0, lc1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![uc0, uc1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let reduce = ReduceMeanLayer::new(vec![-1], true);
        let layer = Layer::ReduceMean(reduce);

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward (asymmetric) failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Sample and verify: concretized bounds contain all possible f(x).
        let samples: Vec<Vec<f32>> = (0..6)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2),
                    3 => (l3, u3), 4 => (l4, u4), _ => (l5, u5),
                };
                sample_points(lj, uj, 3)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        for &s4 in &samples[4] {
                            for &s5 in &samples[5] {
                                let mean_row0 = (s0 + s1 + s2) / 3.0;
                                let mean_row1 = (s3 + s4 + s5) / 3.0;

                                // Lower bound uses lower_a coefficients
                                let lower_output = lc0 * mean_row0 + lc1 * mean_row1;
                                // Upper bound uses upper_a coefficients
                                let upper_output = uc0 * mean_row0 + uc1 * mean_row1;

                                let lower_tol = FP_TOLERANCE * lower_output.abs().max(1.0);
                                let upper_tol = FP_TOLERANCE * upper_output.abs().max(1.0);

                                prop_assert!(
                                    crown_lower[0] - lower_tol <= lower_output,
                                    "Asymmetric lower violated: crown_lb={} > lower_f={lower_output} \
                                     (lc0={lc0}, lc1={lc1})",
                                    crown_lower[0]
                                );
                                prop_assert!(
                                    upper_output <= crown_upper[0] + upper_tol,
                                    "Asymmetric upper violated: upper_f={upper_output} > crown_ub={} \
                                     (uc0={uc0}, uc1={uc1})",
                                    crown_upper[0]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
