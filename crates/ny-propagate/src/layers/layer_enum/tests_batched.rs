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
fn test_layer_erf_batched_crown_dispatch_matches_direct() -> Result<()> {
    let layer = ErfLayer::new();
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-3.0, -1.0, -0.25, 0.0, 0.5, 2.0])
            .expect("valid lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-2.0, 0.5, 0.25, 1.0, 1.5, 3.0])
            .expect("valid upper shape"),
    )?;
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;

    let expected = layer.propagate_linear_batched_with_bounds(&bounds, &pre_activation)?;
    let actual =
        Layer::Erf(layer).propagate_crown_backward_batched(&bounds, Some(&pre_activation), None)?;

    assert_eq!(actual.lower_a(), expected.lower_a());
    assert_eq!(actual.upper_a(), expected.upper_a());
    assert_eq!(actual.lower_b(), expected.lower_b());
    assert_eq!(actual.upper_b(), expected.upper_b());
    Ok(())
}

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

/// #cgan-shape-err-compose: the four pure 1:1 shape remaps must PROPAGATE an
/// incoming certified coefficient error, not take the dispatcher's output-box
/// discharge.
///
/// Their batched backward is `Ok(bounds.clone())` — a literal IDENTITY on the
/// coefficient matrix (a reshape is metadata; the flat coefficient layout is
/// unchanged) — so an incoming certificate `|A_stored - A_true| <= e` holds
/// VERBATIM on the result. Discharging instead folds a u-scale RELATIVE error
/// into an ABSOLUTE width penalty at intermediate-box magnitude, the mechanism
/// that made BatchNorm the dominant cGAN CROWN looseness before eb651c59.
///
/// The observable is the err ITSELF surviving on the result together with an
/// UNWIDENED bias: a discharge is visible as `lower_a_err == None` plus a bias
/// pushed outward by `Σ_j max(|y_j|)·e_j`. Asserting only "no panic" would pass
/// on the defective routing, so both halves are load-bearing.
#[test]
fn batched_shape_remaps_propagate_incoming_coeff_err_and_do_not_widen_bias() -> Result<()> {
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
            .expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-1.0, 0.0, 0.5, 1.5, 2.0, 3.0]).expect("upper"),
    )?;
    let rows = 2usize;
    let n = 6usize;
    let a = ArrayD::from_shape_vec(
        IxDyn(&[1, rows, n]),
        (0..rows * n)
            .map(|i| ((i % 5) as f32) - 2.0)
            .collect::<Vec<f32>>(),
    )
    .expect("a");
    let b = ArrayD::from_shape_vec(IxDyn(&[1, rows]), vec![0.0f32; rows]).expect("b");
    let err = ArrayD::from_shape_vec(IxDyn(&[1, rows, n]), vec![0.125f32; rows * n]).expect("err");

    let cases: Vec<(&str, Layer)> = vec![
        ("Reshape", Layer::Reshape(ReshapeLayer::new(vec![1, 6]))),
        ("Flatten", Layer::Flatten(FlattenLayer::new(1))),
        ("Squeeze", Layer::Squeeze(SqueezeLayer::new(0))),
        ("Unsqueeze", Layer::Unsqueeze(UnsqueezeLayer::new(0))),
    ];

    for (tag, layer) in cases {
        let mut bounds = BatchedLinearBounds::new_or_conservative(
            a.clone(),
            b.clone(),
            a.clone(),
            b.clone(),
            vec![1, 6],
            vec![1, rows],
        )?;
        bounds.lower_a_err = Some(err.clone());
        bounds.upper_a_err = Some(err.clone());
        assert!(
            bounds.has_coeff_err(),
            "{tag}: fixture must carry coeff err"
        );

        let out = layer.propagate_crown_backward_batched(&bounds, Some(&pre), None)?;

        assert!(
            out.lower_a_err.is_some() && out.upper_a_err.is_some(),
            "{tag}: incoming coeff err was DISCHARGED by the batched dispatcher; \
             a 1:1 shape remap must PROPAGATE it (#cgan-shape-err-compose)"
        );
        assert_eq!(
            out.lower_b(),
            b,
            "{tag}: lower bias widened => the output-box discharge ran"
        );
        assert_eq!(
            out.upper_b(),
            b,
            "{tag}: upper bias widened => the output-box discharge ran"
        );
    }
    Ok(())
}
