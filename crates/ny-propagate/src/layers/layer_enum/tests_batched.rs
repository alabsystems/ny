// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::BatchedLinearBounds;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

#[test]
fn test_layer_conv1d_batched_crown_dispatch_uses_engine_3622() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![2.0_f32]).expect("kernel");
    let layer = Layer::Conv1d(Conv1dLayer::new(kernel, Some(arr1(&[0.5_f32])), 1, 0)?);
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let pre_activation = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[2, 1, 3])),
        ArrayD::zeros(IxDyn(&[2, 1, 3])),
    )?;
    let expected = layer.propagate_crown_backward_batched(&bounds, Some(&pre_activation), None)?;
    let engine = CountingGemmEngine::new();
    let actual =
        layer.propagate_crown_backward_batched(&bounds, Some(&pre_activation), Some(&engine))?;

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3622 regression: Layer::Conv1d batched dispatcher should forward GemmEngine, got {calls} calls"
    );

    for (idx, (&actual_value, &expected_value)) in actual
        .lower_a()
        .iter()
        .zip(expected.lower_a().iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "lower_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_a()
        .iter()
        .zip(expected.upper_a().iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "upper_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    Ok(())
}

#[test]
fn test_layer_softmax_batched_crown_flat_grouped_dispatch_matches_direct() -> Result<()> {
    let layer = SoftmaxLayer::new(-1);
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, 0.5, 1.5, 2.5])
            .expect("shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
            .expect("shape should be valid"),
    )?;
    let flat_bounds = BatchedLinearBounds::identity(&[2, 3])?.flatten_to_block_diagonal()?;

    let expected = layer.propagate_linear_batched_with_bounds(
        &flat_bounds,
        &pre_activation,
        layer.soundness_mode(),
    )?;
    let actual = Layer::Softmax(layer).propagate_crown_backward_batched(
        &flat_bounds,
        Some(&pre_activation),
        None,
    )?;

    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_b(), expected.upper_b());
    Ok(())
}

#[test]
fn test_layer_logsoftmax_batched_crown_dispatch_matches_direct() -> Result<()> {
    let layer = LogSoftmaxLayer::new(-1).with_sound_mode(true);
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, 0.5, 1.5, 2.5])
            .expect("shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
            .expect("shape should be valid"),
    )?;
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;

    let expected = layer.propagate_linear_batched_with_bounds(
        &bounds,
        &pre_activation,
        layer.soundness_mode(),
    )?;
    let actual = Layer::LogSoftmax(layer).propagate_crown_backward_batched(
        &bounds,
        Some(&pre_activation),
        None,
    )?;

    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_b(), expected.upper_b());
    Ok(())
}

#[test]
fn test_layer_causal_softmax_batched_crown_dispatch_matches_direct() -> Result<()> {
    let layer = CausalSoftmaxLayer::new(-1);
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -1.0, 0.0, 1.0, 0.5, 1.0, 1.5, -0.5, 0.25, 0.75, 1.0, 1.5, 2.0,
            ],
        )
        .expect("shape should be valid"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -0.25, 0.75, 1.75, 1.0, 1.5, 2.0, 0.0, 0.75, 1.25, 1.5, 2.0, 2.5,
            ],
        )
        .expect("shape should be valid"),
    )?;
    let bounds = BatchedLinearBounds::identity(&[2, 2, 3])?;

    let expected = layer.propagate_linear_batched_with_bounds(
        &bounds,
        &pre_activation,
        layer.soundness_mode(),
    )?;
    let actual = Layer::CausalSoftmax(layer).propagate_crown_backward_batched(
        &bounds,
        Some(&pre_activation),
        None,
    )?;

    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_b(), expected.upper_b());
    Ok(())
}

#[test]
fn test_layer_causal_softmax_batched_crown_flat_grouped_returns_unsupported() -> Result<()> {
    let layer = Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1));
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -1.0, 0.0, 1.0, 0.5, 1.0, 1.5, -0.5, 0.25, 0.75, 1.0, 1.5, 2.0,
            ],
        )
        .expect("shape should be valid"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -0.25, 0.75, 1.75, 1.0, 1.5, 2.0, 0.0, 0.75, 1.25, 1.5, 2.0, 2.5,
            ],
        )
        .expect("shape should be valid"),
    )?;
    let flat_bounds = BatchedLinearBounds::identity(&[2, 2, 3])?.flatten_to_block_diagonal()?;

    match layer.propagate_crown_backward_batched(&flat_bounds, Some(&pre_activation), None) {
        Err(NyError::UnsupportedOp(reason)) => {
            assert!(
                reason.contains("flat block-diagonal grouped bounds"),
                "expected flat-grouped UnsupportedOp, got: {reason}"
            );
        }
        Err(other) => panic!("expected UnsupportedOp for flat-grouped CausalSoftmax, got {other}"),
        Ok(_) => panic!("expected flat-grouped CausalSoftmax to reject block-diagonal bounds"),
    }

    Ok(())
}

#[test]
fn test_layer_logsoftmax_batched_crown_flat_grouped_returns_unsupported() -> Result<()> {
    let layer = Layer::LogSoftmax(LogSoftmaxLayer::new(-1).with_sound_mode(true));
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, 0.5, 1.5, 2.5])
            .expect("shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
            .expect("shape should be valid"),
    )?;
    let flat_bounds = BatchedLinearBounds::identity(&[2, 3])?.flatten_to_block_diagonal()?;

    match layer.propagate_crown_backward_batched(&flat_bounds, Some(&pre_activation), None) {
        Err(NyError::UnsupportedOp(reason)) => {
            assert!(
                reason.contains("flat block-diagonal grouped bounds"),
                "expected flat-grouped UnsupportedOp, got: {reason}"
            );
        }
        Err(other) => panic!("expected UnsupportedOp for flat-grouped LogSoftmax, got {other}"),
        Ok(_) => panic!("expected flat-grouped LogSoftmax to reject block-diagonal bounds"),
    }

    Ok(())
}

#[test]
fn test_layer_logsumexp_batched_crown_flat_grouped_returns_unsupported() -> Result<()> {
    let layer = Layer::LogSumExp(LogSumExpLayer::new(vec![-1], true));
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -1.0, 0.0, 1.0, 0.5, 1.0, 1.5, -0.5, 0.25, 0.75, 1.0, 1.5, 2.0,
            ],
        )
        .expect("shape should be valid"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -0.25, 0.75, 1.75, 1.0, 1.5, 2.0, 0.0, 0.75, 1.25, 1.5, 2.0, 2.5,
            ],
        )
        .expect("shape should be valid"),
    )?;
    let flat_bounds = BatchedLinearBounds::identity(&[2, 2, 3])?.flatten_to_block_diagonal()?;

    match layer.propagate_crown_backward_batched(&flat_bounds, Some(&pre_activation), None) {
        Err(NyError::UnsupportedOp(reason)) => {
            assert!(
                reason.contains("flat block-diagonal grouped bounds"),
                "expected flat-grouped UnsupportedOp, got: {reason}"
            );
        }
        Err(other) => panic!("expected UnsupportedOp for flat-grouped LogSumExp, got {other}"),
        Ok(_) => panic!("expected flat-grouped LogSumExp to reject block-diagonal bounds"),
    }

    Ok(())
}

#[test]
fn test_layer_cumsum_batched_crown_dispatch_matches_direct() -> Result<()> {
    let layer = CumsumLayer::new(-1, false, false);
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, 0.5, 1.5, 2.5])
            .expect("shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
            .expect("shape should be valid"),
    )?;
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let expected = layer.propagate_linear_batched(&bounds, &pre_activation)?;
    let actual = Layer::CumSum(layer).propagate_crown_backward_batched(
        &bounds,
        Some(&pre_activation),
        None,
    )?;

    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_b(), expected.upper_b());
    Ok(())
}

#[test]
fn test_layer_logsumexp_batched_crown_dispatch_matches_direct() -> Result<()> {
    let layer = LogSumExpLayer::new(vec![-1], true);
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -1.0, 0.0, 1.0, 0.5, 1.0, 1.5, -0.5, 0.25, 0.75, 1.0, 1.5, 2.0,
            ],
        )
        .expect("shape should be valid"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                -0.25, 0.75, 1.75, 1.0, 1.5, 2.0, 0.0, 0.75, 1.25, 1.5, 2.0, 2.5,
            ],
        )
        .expect("shape should be valid"),
    )?;
    let bounds = BatchedLinearBounds::identity(&[2, 2, 1])?;

    let expected = layer.propagate_linear_batched_with_bounds(&bounds, &pre_activation)?;
    let actual = Layer::LogSumExp(layer).propagate_crown_backward_batched(
        &bounds,
        Some(&pre_activation),
        None,
    )?;

    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_b(), expected.upper_b());
    Ok(())
}
