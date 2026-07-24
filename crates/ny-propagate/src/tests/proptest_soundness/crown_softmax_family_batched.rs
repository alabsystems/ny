// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::Layer;
use crate::layers::{CausalSoftmaxLayer, LogSoftmaxLayer, LogSumExpLayer};
use crate::BatchedLinearBounds;
use ndarray::{ArrayD, Dimension, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{causal_softmax, logsoftmax};

const BATCHED_LOGSOFTMAX_TOLERANCE: f32 = 1e-4;
const BATCHED_CAUSAL_TOLERANCE: f32 = 1e-4;
const BATCHED_LOGSUMEXP_TOLERANCE: f32 = 1e-4;

fn sample_tensor(lower: &ArrayD<f32>, upper: &ArrayD<f32>, sample_idx: usize) -> ArrayD<f32> {
    ArrayD::from_shape_fn(lower.raw_dim(), |idx| {
        let flat_idx = idx.slice().iter().fold(0usize, |acc, &value| {
            acc.wrapping_mul(17).wrapping_add(value)
        });
        let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (flat_idx as u32)) as f32
            / u32::MAX as f32;
        let l = lower[idx.clone()];
        let u = upper[idx];
        l + (u - l) * t
    })
}

fn logsumexp_row(row: &[f32]) -> f32 {
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = row.iter().map(|&value| (value - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(100) })]

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsoftmax_batched_dispatch(
        intervals in prop::collection::vec(super::valid_interval(2.0), 6),
    ) {
        let lower_vals: Vec<f32> = intervals.iter().map(|(l, _)| *l).collect();
        let upper_vals: Vec<f32> = intervals.iter().map(|(_, u)| *u).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_vals).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_vals).unwrap(),
        ).unwrap();
        let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();
        let layer = Layer::LogSoftmax(LogSoftmaxLayer::new(-1).with_sound_mode(true));

        let result = layer
            .propagate_crown_backward_batched(&bounds, Some(&input), None)
            .map_err(|e| TestCaseError::fail(
                format!("batched LogSoftmax dispatch failed: {e}")
            ))?;
        let concretized = result.concretize(&input).unwrap();

        for sample_idx in 0..12 {
            let sample = sample_tensor(input.lower(), input.upper(), sample_idx);
            for row_idx in 0..2 {
                let row = sample
                    .slice(ndarray::s![row_idx, ..])
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();
                let actual = logsoftmax(&row);
                for col_idx in 0..3 {
                    let lb = concretized.lower()[[row_idx, col_idx]];
                    let ub = concretized.upper()[[row_idx, col_idx]];
                    prop_assert!(
                        actual[col_idx] >= lb - BATCHED_LOGSOFTMAX_TOLERANCE,
                        "batched LogSoftmax lower violation at [{row_idx},{col_idx}]: actual={} lb={}",
                        actual[col_idx],
                        lb
                    );
                    prop_assert!(
                        actual[col_idx] <= ub + BATCHED_LOGSOFTMAX_TOLERANCE,
                        "batched LogSoftmax upper violation at [{row_idx},{col_idx}]: actual={} ub={}",
                        actual[col_idx],
                        ub
                    );
                }
            }
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_causal_softmax_batched_dispatch(
        intervals in prop::collection::vec(super::valid_interval(2.0), 12),
    ) {
        let lower_vals: Vec<f32> = intervals.iter().map(|(l, _)| *l).collect();
        let upper_vals: Vec<f32> = intervals.iter().map(|(_, u)| *u).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), lower_vals).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), upper_vals).unwrap(),
        ).unwrap();
        let bounds = BatchedLinearBounds::identity(&[2, 2, 3]).unwrap();
        let layer = Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1));

        let result = layer
            .propagate_crown_backward_batched(&bounds, Some(&input), None)
            .map_err(|e| TestCaseError::fail(
                format!("batched CausalSoftmax dispatch failed: {e}")
            ))?;
        let concretized = result.concretize(&input).unwrap();

        for sample_idx in 0..10 {
            let sample = sample_tensor(input.lower(), input.upper(), sample_idx);
            for batch_idx in 0..2 {
                let matrix = sample
                    .slice(ndarray::s![batch_idx, .., ..])
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let actual = causal_softmax(&matrix);
                for row_idx in 0..2 {
                    for col_idx in 0..3 {
                        let lb = concretized.lower()[[batch_idx, row_idx, col_idx]];
                        let ub = concretized.upper()[[batch_idx, row_idx, col_idx]];
                        prop_assert!(
                            actual[[row_idx, col_idx]] >= lb - BATCHED_CAUSAL_TOLERANCE,
                            "batched CausalSoftmax lower violation at [{batch_idx},{row_idx},{col_idx}]: actual={} lb={}",
                            actual[[row_idx, col_idx]],
                            lb
                        );
                        prop_assert!(
                            actual[[row_idx, col_idx]] <= ub + BATCHED_CAUSAL_TOLERANCE,
                            "batched CausalSoftmax upper violation at [{batch_idx},{row_idx},{col_idx}]: actual={} ub={}",
                            actual[[row_idx, col_idx]],
                            ub
                        );
                    }
                }
            }
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsumexp_batched_dispatch_last_axis_keepdims(
        intervals in prop::collection::vec(super::valid_interval(2.0), 12),
    ) {
        let lower_vals: Vec<f32> = intervals.iter().map(|(l, _)| *l).collect();
        let upper_vals: Vec<f32> = intervals.iter().map(|(_, u)| *u).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), lower_vals).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), upper_vals).unwrap(),
        ).unwrap();
        let bounds = BatchedLinearBounds::identity(&[2, 2, 1]).unwrap();
        let layer = Layer::LogSumExp(LogSumExpLayer::new(vec![-1], true));

        let result = layer
            .propagate_crown_backward_batched(&bounds, Some(&input), None)
            .map_err(|e| TestCaseError::fail(
                format!("batched LogSumExp dispatch failed: {e}")
            ))?;
        let concretized = result.concretize(&input).unwrap();

        for sample_idx in 0..10 {
            let sample = sample_tensor(input.lower(), input.upper(), sample_idx);
            for batch_idx in 0..2 {
                for row_idx in 0..2 {
                    let row = sample
                        .slice(ndarray::s![batch_idx, row_idx, ..])
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .unwrap();
                    let actual = logsumexp_row(row.as_slice().unwrap());
                    let lb = concretized.lower()[[batch_idx, row_idx, 0]];
                    let ub = concretized.upper()[[batch_idx, row_idx, 0]];
                    prop_assert!(
                        actual >= lb - BATCHED_LOGSUMEXP_TOLERANCE,
                        "batched LogSumExp lower violation at [{batch_idx},{row_idx},0]: actual={} lb={}",
                        actual,
                        lb
                    );
                    prop_assert!(
                        actual <= ub + BATCHED_LOGSUMEXP_TOLERANCE,
                        "batched LogSumExp upper violation at [{batch_idx},{row_idx},0]: actual={} ub={}",
                        actual,
                        ub
                    );
                }
            }
        }
    }
}
