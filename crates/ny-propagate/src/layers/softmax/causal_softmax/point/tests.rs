// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::ShapeBuilder;

fn check_point(layer: &CausalSoftmaxLayer, input: &BoundedTensor) {
    let output = layer
        .propagate_concrete_point_with_deadline(input, None)
        .unwrap();
    assert_eq!(output.shape(), input.shape());
    assert_eq!(output.lower(), output.upper());
    let shape = input.shape();
    let (queries, keys) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let values: Vec<_> = input.lower().iter().copied().collect();
    let actual: Vec<_> = output.lower().iter().copied().collect();
    for row in 0..values.len() / keys {
        let active = layer.active_range(row % queries, keys);
        let shift = active
            .clone()
            .map(|k| values[row * keys + k] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = active
            .clone()
            .map(|k| (values[row * keys + k] as f64 - shift).exp())
            .collect();
        let mass: f64 = weights.iter().sum();
        for k in 0..keys {
            let expected = if active.contains(&k) {
                (weights[k - active.start] / mass) as f32
            } else {
                0.0
            };
            assert_eq!(actual[row * keys + k].to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn concrete_causal_signed_scores_preserve_original_masks_batches_and_layouts() {
    for shape in [&[4, 5][..], &[2, 4, 5][..], &[2, 3, 4, 5][..]] {
        let values = ArrayD::from_shape_fn(IxDyn(shape), |i| {
            let q = i[shape.len() - 2];
            let k = i[shape.len() - 1];
            match k {
                0 => -1000.0,
                1 => 1000.0,
                _ => (q as f32 - k as f32) * 0.25,
            }
        });
        let input = BoundedTensor::concrete(values).unwrap();
        for window in [None, Some(0), Some(1), Some(usize::MAX)] {
            let mut layer = CausalSoftmaxLayer::new(-1);
            layer.window_size = window;
            check_point(&layer, &input);
        }
    }
    // F-order storage must be read by logical coordinates, not its physical
    // memory order. This is a valid rectangular attention profile.
    let values = ArrayD::from_shape_vec(
        IxDyn(&[3, 4]).f(),
        (0..12).map(|i| i as f32 * 0.1 - 0.6).collect(),
    )
    .unwrap();
    assert!(!values.is_standard_layout());
    check_point(
        &CausalSoftmaxLayer::new(-1),
        &BoundedTensor::concrete(values).unwrap(),
    );
}

#[test]
fn concrete_causal_refuses_intervals_nonfinite_and_invalid_profiles() {
    let layer = CausalSoftmaxLayer::new(-1);
    let interval =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[2, 2])), ArrayD::ones(IxDyn(&[2, 2]))).unwrap();
    assert!(layer
        .propagate_concrete_point_with_deadline(&interval, None)
        .is_err());
    assert!(layer
        .propagate_concrete_point_with_deadline(&BoundedTensor::new_conservative(&[2, 2]), None)
        .is_err());
    for shape in [&[2][..], &[3, 2][..], &[1, 1, 1, 1, 1][..]] {
        let point = BoundedTensor::concrete(ArrayD::zeros(IxDyn(shape))).unwrap();
        assert!(layer
            .propagate_concrete_point_with_deadline(&point, None)
            .is_err());
    }
    let point = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2, 2]))).unwrap();
    assert!(CausalSoftmaxLayer::new(10)
        .propagate_concrete_point_with_deadline(&point, None)
        .is_err());
}

#[test]
fn concrete_causal_uses_supplied_deadline_and_polls_inside_rows() {
    let layer = CausalSoftmaxLayer::new(-1);
    let point = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2, 4096]))).unwrap();
    assert!(matches!(
        layer.propagate_concrete_point_with_deadline(&point, Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
    let point = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[192, 192]))).unwrap();
    let mut polls = 0;
    let result = layer.point_with_poll(&point, &mut || {
        polls += 1;
        if polls == 220 {
            Err(NyError::DeadlineExceeded("injected row checkpoint".into()))
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
    assert_eq!(polls, 220);
}

#[test]
fn concrete_causal_preflight_prices_visible_pairs_and_refuses_oversized_buffers() {
    let dense = CausalSoftmaxLayer::point_work_bound(&[2, 1024, 1024], None).unwrap();
    let window = CausalSoftmaxLayer::point_work_bound(&[2, 1024, 1024], Some(7)).unwrap();
    assert!(window < dense && dense < 200_000_000);
    assert!(CausalSoftmaxLayer::point_work_bound(&[usize::MAX, usize::MAX], None).is_err());
    assert!(CausalSoftmaxLayer::point_work_bound(&[8192, 8192], None).is_err());
    assert!(CausalSoftmaxLayer::point_work_bound(&[1, 1, 1, 1, 1], None).is_err());
}
