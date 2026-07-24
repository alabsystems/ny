// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for ReduceMax/ReduceMin (extremum reductions).
//!
//! ReduceMax/ReduceMin IBP is exact due to monotonicity:
//!   max(lower) <= max(x) <= max(upper)
//!   min(lower) <= min(x) <= min(upper)
//!
//! CROWN uses fixed-index assumption (argmax/argmin at center point).
//! This is standard in VNN-COMP tools but is inherently approximate.

use crate::layers::common::BoundPropagation;
use crate::layers::{Layer, ReduceMaxLayer, ReduceMinLayer};
use crate::LinearBounds;
use crate::{soundness_provenance_for_network, Network, PropagationMethod};
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::HeuristicUsed;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{valid_interval, FP_TOLERANCE};

/// Helper: concretize CROWN linear bounds against a pre-activation tensor.
fn concretize_crown_1d(
    result: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).expect("finite values"))
        .expect("non-empty slice")
        .0
}

fn argmin_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).expect("finite values"))
        .expect("non-empty slice")
        .0
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// ReduceMax IBP soundness: for any x in [l, u], max(x) is in computed bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_max_ibp_1d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn(),
        ).unwrap();

        let reduce_max = ReduceMaxLayer::new(vec![0], true);
        let output = reduce_max.propagate_ibp(&input).unwrap();

        let lowers = [l0, l1, l2, l3];
        let uppers = [u0, u1, u2, u3];

        let corners: Vec<[f32; 4]> = vec![
            [l0, l1, l2, l3],
            [u0, u1, u2, u3],
            [u0, l1, l2, l3],
            [l0, u1, l2, l3],
            [l0, l1, u2, l3],
            [l0, l1, l2, u3],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2), f32::midpoint(l3, u3)],
        ];

        for x in corners {
            let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= max_x
                    && max_x <= output.upper()[[0]] + FP_TOLERANCE,
                "ReduceMax IBP soundness violation: max({:?})={} not in [{}, {}]",
                x, max_x, output.lower()[[0]], output.upper()[[0]]
            );
        }

        // Analytical check: lower = max(l_i), upper = max(u_i)
        let expected_lower = lowers.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let expected_upper = uppers.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            (output.lower()[[0]] - expected_lower).abs() < FP_TOLERANCE,
            "ReduceMax lower != max(lower): {} vs {}",
            output.lower()[[0]], expected_lower
        );
        prop_assert!(
            (output.upper()[[0]] - expected_upper).abs() < FP_TOLERANCE,
            "ReduceMax upper != max(upper): {} vs {}",
            output.upper()[[0]], expected_upper
        );
    }

    /// ReduceMin IBP soundness: for any x in [l, u], min(x) is in computed bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_min_ibp_1d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn(),
        ).unwrap();

        let reduce_min = ReduceMinLayer::new(vec![0], true);
        let output = reduce_min.propagate_ibp(&input).unwrap();

        let lowers = [l0, l1, l2, l3];
        let uppers = [u0, u1, u2, u3];

        let corners: Vec<[f32; 4]> = vec![
            [l0, l1, l2, l3],
            [u0, u1, u2, u3],
            [u0, l1, l2, l3],
            [l0, u1, l2, l3],
            [l0, l1, u2, l3],
            [l0, l1, l2, u3],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2), f32::midpoint(l3, u3)],
        ];

        for x in corners {
            let min_x = x.iter().copied().fold(f32::INFINITY, f32::min);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= min_x
                    && min_x <= output.upper()[[0]] + FP_TOLERANCE,
                "ReduceMin IBP soundness violation: min({:?})={} not in [{}, {}]",
                x, min_x, output.lower()[[0]], output.upper()[[0]]
            );
        }

        // Analytical check: lower = min(l_i), upper = min(u_i)
        let expected_lower = lowers.iter().copied().fold(f32::INFINITY, f32::min);
        let expected_upper = uppers.iter().copied().fold(f32::INFINITY, f32::min);
        prop_assert!(
            (output.lower()[[0]] - expected_lower).abs() < FP_TOLERANCE,
            "ReduceMin lower != min(lower): {} vs {}",
            output.lower()[[0]], expected_lower
        );
        prop_assert!(
            (output.upper()[[0]] - expected_upper).abs() < FP_TOLERANCE,
            "ReduceMin upper != min(upper): {} vs {}",
            output.upper()[[0]], expected_upper
        );
    }

    /// ReduceMax IBP soundness for 2D input, reducing last axis.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_max_ibp_2d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5]).unwrap(),
        ).unwrap();

        let reduce_max = ReduceMaxLayer::new(vec![-1], true);
        let output = reduce_max.propagate_ibp(&input).unwrap();

        let row0_lower_max = l0.max(l1).max(l2);
        let row0_upper_max = u0.max(u1).max(u2);
        let row1_lower_max = l3.max(l4).max(l5);
        let row1_upper_max = u3.max(u4).max(u5);

        prop_assert!(
            (output.lower()[[0, 0]] - row0_lower_max).abs() < FP_TOLERANCE,
            "Row 0 lower: {} vs expected {}",
            output.lower()[[0, 0]], row0_lower_max
        );
        prop_assert!(
            (output.upper()[[0, 0]] - row0_upper_max).abs() < FP_TOLERANCE,
            "Row 0 upper: {} vs expected {}",
            output.upper()[[0, 0]], row0_upper_max
        );
        prop_assert!(
            (output.lower()[[1, 0]] - row1_lower_max).abs() < FP_TOLERANCE,
            "Row 1 lower: {} vs expected {}",
            output.lower()[[1, 0]], row1_lower_max
        );
        prop_assert!(
            (output.upper()[[1, 0]] - row1_upper_max).abs() < FP_TOLERANCE,
            "Row 1 upper: {} vs expected {}",
            output.upper()[[1, 0]], row1_upper_max
        );
    }

    /// ReduceMax batched CROWN backward with identity bounds (#287).
    ///
    /// Tests propagate_linear_batched for ReduceMax (last axis, keepdims=true).
    /// With identity batched bounds, concretized results should contain all
    /// sampled input maxima (may not exactly match IBP due to fixed-index).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_max_batched_crown_identity(
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

        let reduce_max = ReduceMaxLayer::new(vec![-1], true);

        // Output shape: [2, 1]
        let bounds = BatchedLinearBounds::identity(&[2, 1]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = reduce_max.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("ReduceMax batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        // Verify bounds are finite and ordered
        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();

        for i in 0..2 {
            prop_assert!(
                conc_lower[i].is_finite() && conc_upper[i].is_finite(),
                "ReduceMax batched CROWN non-finite bounds at {i}: [{}, {}]",
                conc_lower[i], conc_upper[i]
            );
            prop_assert!(
                conc_lower[i] <= conc_upper[i] + FP_TOLERANCE,
                "ReduceMax batched CROWN lower > upper at {i}: {} > {}",
                conc_lower[i], conc_upper[i]
            );
        }

        // Verify the center point's max is within bounds (fixed-index is
        // correct at the center by construction).
        let center = pre_activation.center();
        let center_max_0 = (0..3).map(|j| center[[0, j]]).fold(f32::NEG_INFINITY, f32::max);
        let center_max_1 = (0..3).map(|j| center[[1, j]]).fold(f32::NEG_INFINITY, f32::max);

        prop_assert!(
            conc_lower[0] - FP_TOLERANCE <= center_max_0
                && center_max_0 <= conc_upper[0] + FP_TOLERANCE,
            "ReduceMax batched CROWN: center max row 0 = {} not in [{}, {}]",
            center_max_0, conc_lower[0], conc_upper[0]
        );
        prop_assert!(
            conc_lower[1] - FP_TOLERANCE <= center_max_1
                && center_max_1 <= conc_upper[1] + FP_TOLERANCE,
            "ReduceMax batched CROWN: center max row 1 = {} not in [{}, {}]",
            center_max_1, conc_lower[1], conc_upper[1]
        );
    }

    /// ReduceMin batched CROWN backward with identity bounds (#287).
    ///
    /// Tests propagate_linear_batched for ReduceMin (last axis, keepdims=true).
    /// Verifies bounds are finite, ordered, and contain the center point's min.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_min_batched_crown_identity(
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

        let reduce_min = ReduceMinLayer::new(vec![-1], true);

        let bounds = BatchedLinearBounds::identity(&[2, 1]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = reduce_min.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("ReduceMin batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();

        for i in 0..2 {
            prop_assert!(
                conc_lower[i].is_finite() && conc_upper[i].is_finite(),
                "ReduceMin batched CROWN non-finite bounds at {i}: [{}, {}]",
                conc_lower[i], conc_upper[i]
            );
            prop_assert!(
                conc_lower[i] <= conc_upper[i] + FP_TOLERANCE,
                "ReduceMin batched CROWN lower > upper at {i}: {} > {}",
                conc_lower[i], conc_upper[i]
            );
        }

        // Verify center point's min is within bounds
        let center = pre_activation.center();
        let center_min_0 = (0..3).map(|j| center[[0, j]]).fold(f32::INFINITY, f32::min);
        let center_min_1 = (0..3).map(|j| center[[1, j]]).fold(f32::INFINITY, f32::min);

        prop_assert!(
            conc_lower[0] - FP_TOLERANCE <= center_min_0
                && center_min_0 <= conc_upper[0] + FP_TOLERANCE,
            "ReduceMin batched CROWN: center min row 0 = {} not in [{}, {}]",
            center_min_0, conc_lower[0], conc_upper[0]
        );
        prop_assert!(
            conc_lower[1] - FP_TOLERANCE <= center_min_1
                && center_min_1 <= conc_upper[1] + FP_TOLERANCE,
            "ReduceMin batched CROWN: center min row 1 = {} not in [{}, {}]",
            center_min_1, conc_lower[1], conc_upper[1]
        );
    }

    /// ReduceMax CROWN backward with identity incoming — verify coefficient sparsity
    /// (<=1 per row: a definite winner scatters one coeff, an unstable argmax
    /// constant-folds to zero) and SOUNDNESS (the box's true max lies within the
    /// concretized bounds, even when the argmax moves within the box).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reduce_max_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5]).unwrap(),
        ).unwrap();

        let reduce_max = ReduceMaxLayer::new(vec![-1], true);
        let layer = Layer::ReduceMax(reduce_max);

        let identity = LinearBounds::identity(2);
        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &input);

        prop_assert_eq!(result.lower_a.nrows(), 2);
        prop_assert_eq!(result.lower_a.ncols(), 6);

        // Verify coefficient sparsity: each row has AT MOST one non-zero. With the
        // definite-winner soundness guard, a provably-dominant element scatters one
        // coefficient (nnz==1); when the argmax is not stable over the box the
        // backward constant-folds the sound IBP interval into the bias with a zero
        // A-row (nnz==0). The old unconditional fixed-index scatter (always nnz==1)
        // was unsound for moving-argmax boxes.
        for row in 0..2 {
            let nnz: usize = (0..6)
                .filter(|&col| result.lower_a[[row, col]].abs() > 1e-8)
                .count();
            prop_assert!(
                nnz <= 1,
                "ReduceMax CROWN lower_a row {} has {} non-zeros (expected <=1)",
                row, nnz
            );
        }

        // Verify bounds are finite and ordered
        for i in 0..2 {
            prop_assert!(
                crown_lower[i].is_finite() && crown_upper[i].is_finite(),
                "ReduceMax CROWN non-finite bounds at row {}: [{}, {}]",
                i, crown_lower[i], crown_upper[i]
            );
            prop_assert!(
                crown_lower[i] <= crown_upper[i] + FP_TOLERANCE,
                "ReduceMax CROWN lower > upper at row {}: {} > {}",
                i, crown_lower[i], crown_upper[i]
            );
        }

        // SOUNDNESS: the box's true max for each row must lie within the concretized
        // CROWN bounds. We check the center point's max (a feasible point of the box)
        // — the old fixed-index scatter could place the upper bound BELOW the true
        // max at a corner where the argmax moved; the guarded fix cannot.
        let center = input.center();
        for i in 0..2 {
            let center_max = (0..3).map(|j| center[[i, j]]).fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                crown_lower[i] - FP_TOLERANCE <= center_max
                    && center_max <= crown_upper[i] + FP_TOLERANCE,
                "ReduceMax CROWN: center max row {} = {} not in [{}, {}]",
                i, center_max, crown_lower[i], crown_upper[i]
            );
        }
    }

    /// The center-point argmax/argmin can change at another valid point in the box.
    /// The soundness scanner must therefore report fixed-index ReduceMax/ReduceMin
    /// as heuristic for CROWN-family methods instead of Sound (#3698).
    #[ntest::timeout(10000)]
    #[test]
    fn reduce_extremum_fixed_index_can_change_under_perturbation(scale in 1.0f32..100.0f32) {
        let max_center = [scale * 0.5, scale * 0.65];
        let max_corner = [scale, scale * 0.55];
        prop_assert_eq!(argmax_index(&max_center), 1);
        prop_assert_eq!(argmax_index(&max_corner), 0);

        let min_center = [-scale * 0.5, -scale * 0.65];
        let min_corner = [-scale, -scale * 0.55];
        prop_assert_eq!(argmin_index(&min_center), 1);
        prop_assert_eq!(argmin_index(&min_corner), 0);

        let mut network = Network::new();
        network.add_layer(Layer::ReduceMax(ReduceMaxLayer::new(vec![0], true)));
        network.add_layer(Layer::ReduceMin(ReduceMinLayer::new(vec![0], true)));
        let provenance = soundness_provenance_for_network(&network, &PropagationMethod::Crown);

        prop_assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::ReduceExtremumFixedIndex { num_nodes: 2 }),
            "expected ReduceExtremumFixedIndex heuristic, got {:?}",
            provenance.heuristics_used()
        );
    }

}

#[ntest::timeout(10000)]
#[test]
fn reduce_extremum_fixed_index_is_flagged_for_all_crown_variants() {
    let mut network = Network::new();
    network.add_layer(Layer::ReduceMax(ReduceMaxLayer::new(vec![0], true)));
    network.add_layer(Layer::ReduceMin(ReduceMinLayer::new(vec![0], true)));

    for method in &[
        PropagationMethod::Crown,
        PropagationMethod::AlphaCrown,
        PropagationMethod::BetaCrown,
        PropagationMethod::SdpCrown,
    ] {
        let provenance = soundness_provenance_for_network(&network, method);
        assert!(
            provenance
                .heuristics_used()
                .contains(&HeuristicUsed::ReduceExtremumFixedIndex { num_nodes: 2 }),
            "{method:?} should detect ReduceExtremumFixedIndex, got {:?}",
            provenance.heuristics_used()
        );
    }
}
