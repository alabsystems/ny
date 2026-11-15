// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Authored reshape controls for complete property rows and local batch rows.

use super::{Layer, ReshapeLayer};
use crate::layers::BoundPropagation;
use crate::BatchedLinearBounds;
use ndarray::{ArrayD, Dimension, IxDyn};
use ny_core::{reshape_copy_axis_sentinel, NyError, Result};
use ny_tensor::BoundedTensor;

fn pre(shape: &[usize]) -> Result<BoundedTensor> {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), -0.5),
        ArrayD::from_elem(IxDyn(shape), 0.75),
    )
}

fn carrier(a_shape: &[usize], input: &[usize], output: &[usize]) -> Result<BatchedLinearBounds> {
    let b_shape = &a_shape[..a_shape.len() - 1];
    let mut bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_fn(IxDyn(a_shape), |index| {
            let coordinate = index[index.ndim() - 1];
            if coordinate % 11 == 0 {
                -0.0
            } else {
                (coordinate % 7) as f32 * 0.125 - 0.375
            }
        }),
        ArrayD::from_elem(IxDyn(b_shape), -1024.0),
        ArrayD::from_shape_fn(IxDyn(a_shape), |index| {
            (index[index.ndim() - 1] % 5) as f32 * 0.0625 - 0.125
        }),
        ArrayD::from_elem(IxDyn(b_shape), 1024.0),
        input.to_vec(),
        output.to_vec(),
    )?;
    bounds.lower_a_err = Some(ArrayD::from_shape_fn(IxDyn(a_shape), |index| {
        (index[index.ndim() - 1] % 3) as f32 * 0.015625
    }));
    bounds.upper_a_err = Some(ArrayD::from_shape_fn(IxDyn(a_shape), |index| {
        (index[index.ndim() - 1] % 7) as f32 * 0.0078125
    }));
    Ok(bounds)
}

fn assert_bits(left: &ArrayD<f32>, right: &ArrayD<f32>) {
    assert_eq!(left.shape(), right.shape());
    assert!(left
        .iter()
        .zip(right.iter())
        .all(|(&a, &b)| a.to_bits() == b.to_bits()));
}

fn assert_numeric_parts_unchanged(before: &BatchedLinearBounds, after: &BatchedLinearBounds) {
    assert_bits(before.lower_a(), after.lower_a());
    assert_bits(before.upper_a(), after.upper_a());
    assert_bits(before.lower_b(), after.lower_b());
    assert_bits(before.upper_b(), after.upper_b());
    for (before, after) in [
        (&before.lower_a_err, &after.lower_a_err),
        (&before.upper_a_err, &after.upper_a_err),
    ] {
        match (before, after) {
            (Some(before), Some(after)) => assert_bits(before, after),
            (None, None) => {}
            _ => panic!("reshape changed independent coefficient-error ownership"),
        }
    }
    assert_eq!(before.output_shape(), after.output_shape());
}

#[test]
fn batched_reshape_original_vit_full_columns_preserve_property_rows_and_errors() -> Result<()> {
    // Original authored Q/V geometry. Ten logit rows remain ten property rows,
    // rather than becoming 17 sequence positions or three attention heads.
    let pre = pre(&[17, 48])?;
    let reshape = ReshapeLayer::new(vec![17, 3, 16]);
    let incoming = carrier(&[10, 816], &[17, 3, 16], &[10])?;
    let authored_output = reshape.propagate_ibp(&pre)?;
    let before = incoming.concretize_sound(&authored_output)?;
    let result =
        Layer::Reshape(reshape).propagate_crown_backward_batched(&incoming, Some(&pre), None)?;
    assert_eq!(result.input_shape(), &[17, 48]);
    assert_eq!(result.lower_a().shape(), &[10, 816]);
    assert_numeric_parts_unchanged(&incoming, &result);
    // Exercise the actual next consumer on the original input layout, including
    // both signed coefficient functions and their independent E*abs(x) charges.
    let after = result.concretize_sound(&pre)?;
    assert_eq!(after.len(), 10);
    assert_bits(before.lower(), after.lower());
    assert_bits(before.upper(), after.upper());
    Ok(())
}

#[test]
fn batched_reshape_global_copy_infer_and_singleton_carriers_keep_lazy_errors() -> Result<()> {
    let pre = pre(&[2, 6])?;
    let copy_first = reshape_copy_axis_sentinel(0).expect("bounded copy axis");
    for target in [vec![0, 3, -1], vec![copy_first, 3, -1]] {
        for upper_only in [false, true] {
            let mut incoming = carrier(&[1, 3, 12], &[2, 3, 2], &[1, 3])?;
            incoming.lower_a_err = None;
            if !upper_only {
                incoming.upper_a_err = None;
            }
            let result = Layer::Reshape(ReshapeLayer::new(target.clone()))
                .propagate_crown_backward_batched(&incoming, Some(&pre), None)?;
            assert_eq!(result.input_shape(), &[2, 6]);
            assert_eq!(result.lower_a().shape(), &[1, 3, 12]);
            assert_numeric_parts_unchanged(&incoming, &result);
        }
    }
    Ok(())
}

#[test]
fn batched_reshape_rejects_stale_global_metadata_with_the_same_volume() -> Result<()> {
    let pre = pre(&[17, 48])?;
    let layer = Layer::Reshape(ReshapeLayer::new(vec![17, 3, 16]));
    for stale in [&[3, 17, 16][..], &[17, 48][..], &[816][..]] {
        let incoming = carrier(&[10, 816], stale, &[10])?;
        assert!(matches!(
            layer.propagate_crown_backward_batched(&incoming, Some(&pre), None),
            Err(NyError::InvalidSpec(_))
        ));
    }
    Ok(())
}

#[test]
fn batched_reshape_rejects_grouped_head_rows_without_an_exact_block_remap() -> Result<()> {
    let pre = pre(&[17, 48])?;
    let incoming = carrier(&[17, 3, 2, 16], &[17, 3, 16], &[17, 3, 2])?;
    assert!(matches!(
        Layer::Reshape(ReshapeLayer::new(vec![17, 3, 16])).propagate_crown_backward_batched(
            &incoming,
            Some(&pre),
            None
        ),
        Err(NyError::InvalidSpec(_))
    ));
    Ok(())
}

#[test]
fn batched_reshape_local_feature_flatten_preserves_batch_ownership() -> Result<()> {
    let pre = pre(&[2, 2, 3])?;
    let incoming = carrier(&[2, 3, 6], &[2, 6], &[2, 3])?;
    let result = Layer::Reshape(ReshapeLayer::new(vec![0, -1])).propagate_crown_backward_batched(
        &incoming,
        Some(&pre),
        None,
    )?;
    assert_eq!(result.input_shape(), &[2, 6]);
    assert_numeric_parts_unchanged(&incoming, &result);
    let foreign_pre = self::pre(&[3, 4])?;
    assert!(matches!(
        Layer::Reshape(ReshapeLayer::new(vec![2, 6])).propagate_crown_backward_batched(
            &incoming,
            Some(&foreign_pre),
            None
        ),
        Err(NyError::InvalidSpec(_))
    ));
    Ok(())
}

#[test]
fn batched_reshape_rejects_incoherent_property_and_error_geometry() -> Result<()> {
    let pre = pre(&[2, 6])?;
    let layer = Layer::Reshape(ReshapeLayer::new(vec![2, 3, 2]));
    let wrong_rows = carrier(&[3, 12], &[2, 3, 2], &[4])?;
    assert!(matches!(
        layer.propagate_crown_backward_batched(&wrong_rows, Some(&pre), None),
        Err(NyError::InvalidSpec(_))
    ));
    let mut wrong_error = carrier(&[3, 12], &[2, 3, 2], &[3])?;
    wrong_error.lower_a_err = Some(ArrayD::zeros(IxDyn(&[3, 11])));
    assert!(matches!(
        layer.propagate_crown_backward_batched(&wrong_error, Some(&pre), None),
        Err(NyError::ShapeMismatch { .. })
    ));
    Ok(())
}

#[test]
fn batched_reshape_rejects_overflowing_authored_target_before_remapping() -> Result<()> {
    let pre = pre(&[1])?;
    let incoming = carrier(&[1, 1], &[1], &[1])?;
    assert!(matches!(
        Layer::Reshape(ReshapeLayer::new(vec![i64::MAX, i64::MAX, -1]))
            .propagate_crown_backward_batched(&incoming, Some(&pre), None),
        Err(NyError::InvalidSpec(_))
    ));
    Ok(())
}
