// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{FlattenLayer, ReshapeLayer};
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

fn make_bounded(shape: &[usize], lower_start: f32) -> BoundedTensor {
    let len = shape.iter().product();
    let lower = ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..len).map(|i| lower_start + i as f32).collect(),
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..len).map(|i| lower_start + 100.0 + i as f32).collect(),
    )
    .unwrap();
    BoundedTensor::new(lower, upper).unwrap()
}

#[ntest::timeout(5000)]
#[test]
fn reshape_preserve_leading_axis_prefixes_batched_output_4093() {
    let input = make_bounded(&[5, 2, 3], 1.0);
    let layer = ReshapeLayer::new(vec![3, 2]);

    let output = layer
        .propagate_ibp_preserve_leading_axis(&input)
        .expect("reshape preserve-leading-axis should keep the restart extent");

    assert_eq!(output.shape(), &[5, 3, 2]);
    assert_eq!(
        output.lower().iter().copied().collect::<Vec<_>>(),
        input.lower().iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        output.upper().iter().copied().collect::<Vec<_>>(),
        input.upper().iter().copied().collect::<Vec<_>>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn reshape_preserve_leading_axis_rejects_mismatched_products_4093() {
    let input = make_bounded(&[4, 2, 3], 0.0);
    let layer = ReshapeLayer::new(vec![5]);

    let err = layer
        .propagate_ibp_preserve_leading_axis(&input)
        .expect_err("reshape preserve-leading-axis should reject incompatible sample products");

    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected shape mismatch, got {err:?}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn flatten_preserve_leading_axis_restores_positive_axis_4093() {
    let input = make_bounded(&[7, 1, 2, 3], 10.0);
    let layer = FlattenLayer::new(0);

    let output = layer
        .propagate_ibp_preserve_leading_axis(&input)
        .expect("flatten preserve-leading-axis should restore the dropped batch axis");

    assert_eq!(output.shape(), &[7, 6]);
    assert_eq!(
        output.lower().iter().copied().collect::<Vec<_>>(),
        input.lower().iter().copied().collect::<Vec<_>>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn flatten_preserve_leading_axis_keeps_negative_axis_4093() {
    let input = make_bounded(&[4, 2, 3, 5], -5.0);
    let layer = FlattenLayer::new(-2);

    let output = layer
        .propagate_ibp_preserve_leading_axis(&input)
        .expect("flatten preserve-leading-axis should leave negative axes end-relative");

    assert_eq!(output.shape(), &[8, 15]);
    assert_eq!(
        output.upper().iter().copied().collect::<Vec<_>>(),
        input.upper().iter().copied().collect::<Vec<_>>()
    );
}
