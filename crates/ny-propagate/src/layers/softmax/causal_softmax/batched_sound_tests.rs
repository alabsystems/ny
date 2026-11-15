// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::BatchedLinearBounds;
use ndarray::{ArrayD, IxDyn};

#[test]
fn causal_batched_sound_composes_negative_one_key_direction() {
    let shape = vec![1, 1];
    let pre =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&shape)), ArrayD::zeros(IxDyn(&shape))).unwrap();
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0),
        ArrayD::zeros(IxDyn(&shape)),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0),
        ArrayD::zeros(IxDyn(&shape)),
        shape.clone(),
        shape,
    )
    .unwrap();
    let result = CausalSoftmaxLayer::new(-1)
        .propagate_linear_batched_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();
    assert!(
        result.lower_b()[[0, 0]] <= -1.0,
        "uncomposed probability published for -p: {:?}",
        result.lower_b()
    );
    assert!(result.upper_b()[[0, 0]] >= -1.0 && result.upper_b()[[0, 0]] < -0.99);
    assert!(result
        .lower_a()
        .iter()
        .chain(result.upper_a())
        .all(|&a| a == 0.0));
}

#[test]
fn causal_batched_sound_keeps_incoming_bias_and_coefficient_error() {
    let shape = vec![1, 1];
    let pre =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&shape)), ArrayD::zeros(IxDyn(&shape))).unwrap();
    let mut bounds = BatchedLinearBounds::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0),
        ArrayD::from_elem(IxDyn(&shape), 0.3),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0),
        ArrayD::from_elem(IxDyn(&shape), 0.3),
        shape.clone(),
        shape,
    )
    .unwrap();
    bounds.set_coeff_err(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.25),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.5),
    );
    let result = CausalSoftmaxLayer::new(-1)
        .propagate_linear_batched_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();
    let (lo, hi) = (
        result.lower_b()[[0, 0]] as f64,
        result.upper_b()[[0, 0]] as f64,
    );
    assert!(lo <= -1.25 + 0.3_f32 as f64 && hi >= -0.5 + 0.3_f32 as f64);
    assert!(lo > -0.96 && hi < -0.19, "lost bias or error: [{lo},{hi}]");
    assert!(
        !result.has_coeff_err(),
        "error must be paid before zeroing A"
    );
}

#[test]
fn causal_batched_signed_multihead_directions_preserve_original_mask_and_shapes() {
    let input_shape = vec![2, 2, 3, 3];
    let output_shape = vec![2, 2, 3, 2];
    let a_shape = vec![2, 2, 3, 2, 3];
    let pre = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&input_shape)),
        ArrayD::zeros(IxDyn(&input_shape)),
    )
    .unwrap();
    let mut la = ArrayD::zeros(IxDyn(&a_shape));
    let mut ua = la.clone();
    let mut lb = ArrayD::zeros(IxDyn(&output_shape));
    let mut ub = lb.clone();
    for b in 0..2 {
        for h in 0..2 {
            for q in 0..3 {
                for o in 0..2 {
                    lb[[b, h, q, o]] = (b + h + q + o) as f32 * 0.1 - 0.25;
                    ub[[b, h, q, o]] = lb[[b, h, q, o]] + 0.125;
                    for k in 0..3 {
                        la[[b, h, q, o, k]] = (k + b + o) as f32 * 0.75 - 1.0;
                        ua[[b, h, q, o, k]] = 0.5 - (k + h + o) as f32 * 0.5;
                    }
                }
            }
        }
    }
    let mut bounds =
        BatchedLinearBounds::new(la, lb, ua, ub, input_shape.clone(), output_shape.clone())
            .unwrap();
    bounds.set_coeff_err(
        ArrayD::from_elem(IxDyn(&a_shape), 0.125),
        ArrayD::from_elem(IxDyn(&a_shape), 0.25),
    );
    for window in [None, Some(0), Some(1), Some(usize::MAX)] {
        let mut layer = CausalSoftmaxLayer::new(-1);
        layer.window_size = window;
        let result = layer
            .propagate_linear_batched_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
            .unwrap();
        assert_eq!(result.lower_a().shape(), a_shape.as_slice());
        assert_eq!(result.lower_b().shape(), output_shape.as_slice());
        assert_eq!(result.input_shape, input_shape);
        assert_eq!(result.output_shape, output_shape);
        for b in 0..2 {
            for h in 0..2 {
                for q in 0..3 {
                    for o in 0..2 {
                        let active = layer.active_range(q, 3);
                        let count = active.len() as f64;
                        let lower = bounds.lower_b()[[b, h, q, o]] as f64
                            + active
                                .clone()
                                .map(|k| bounds.lower_a()[[b, h, q, o, k]] as f64 - 0.125)
                                .sum::<f64>()
                                / count;
                        let upper = bounds.upper_b()[[b, h, q, o]] as f64
                            + active
                                .map(|k| bounds.upper_a()[[b, h, q, o, k]] as f64 + 0.25)
                                .sum::<f64>()
                                / count;
                        assert!(result.lower_b()[[b, h, q, o]] as f64 <= lower);
                        assert!(result.upper_b()[[b, h, q, o]] as f64 >= upper);
                    }
                }
            }
        }
    }
}
