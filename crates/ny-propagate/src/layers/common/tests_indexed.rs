// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

use crate::layers::activations::LinearRelaxation;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{array, ArrayD, IxDyn};

#[test]
fn test_elementwise_indexed_relaxation_uses_neuron_index() {
    fn indexed_relax(_l: f32, _u: f32, i: usize) -> LinearRelaxation {
        let slope = (i + 1) as f32;
        let intercept = i as f32;
        LinearRelaxation::new(slope, intercept, slope, intercept)
    }

    let bounds = LinearBounds::identity(3);
    let pre = BoundedTensor::new(
        array![-1.0_f32, -1.0, -1.0].into_dyn(),
        array![1.0_f32, 1.0, 1.0].into_dyn(),
    )
    .expect("invariant: matching 1D bound shapes");
    let result = crown_elementwise_backward_indexed(&bounds, &pre, indexed_relax)
        .expect("indexed helper should handle identity bounds");

    // Identity incoming bounds should preserve per-neuron slope/intercept exactly.
    assert!((result.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[1, 1]] - 2.0).abs() < 1e-5);
    assert!((result.lower_a[[2, 2]] - 3.0).abs() < 1e-5);
    assert!((result.upper_a[[0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.upper_a[[1, 1]] - 2.0).abs() < 1e-5);
    assert!((result.upper_a[[2, 2]] - 3.0).abs() < 1e-5);
    assert!((result.lower_b[0] - 0.0).abs() < 1e-5);
    assert!((result.lower_b[1] - 1.0).abs() < 1e-5);
    assert!((result.lower_b[2] - 2.0).abs() < 1e-5);
    assert!((result.upper_b[0] - 0.0).abs() < 1e-5);
    assert!((result.upper_b[1] - 1.0).abs() < 1e-5);
    assert!((result.upper_b[2] - 2.0).abs() < 1e-5);
}

#[test]
fn test_batched_indexed_relaxation_uses_neuron_index() {
    fn indexed_relax(_l: f32, _u: f32, i: usize) -> LinearRelaxation {
        let slope = (i + 1) as f32;
        let intercept = i as f32;
        LinearRelaxation::new(slope, intercept, slope, intercept)
    }

    // 2 batches, 1 output, 3 inputs. All coefficients are +1.
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0; 6])
            .expect("invariant: lower_a shape and data length must match"),
        ArrayD::zeros(IxDyn(&[2, 1])),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0; 6])
            .expect("invariant: upper_a shape and data length must match"),
        ArrayD::zeros(IxDyn(&[2, 1])),
        vec![3],
        vec![1],
    );
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0; 6])
            .expect("invariant: lower pre-activation shape must match"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6])
            .expect("invariant: upper pre-activation shape must match"),
    )
    .expect("invariant: matching batched lower/upper shapes");
    let result = crown_elementwise_backward_batched_indexed(&bounds, &pre, indexed_relax)
        .expect("indexed batched helper should handle simple positive coefficients");

    // Per-input slopes (1,2,3) should be applied in every batch row.
    assert!((result.lower_a[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 0, 1]] - 2.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 0, 2]] - 3.0).abs() < 1e-5);
    assert!((result.lower_a[[1, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.lower_a[[1, 0, 1]] - 2.0).abs() < 1e-5);
    assert!((result.lower_a[[1, 0, 2]] - 3.0).abs() < 1e-5);
    assert!((result.upper_a[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.upper_a[[0, 0, 1]] - 2.0).abs() < 1e-5);
    assert!((result.upper_a[[0, 0, 2]] - 3.0).abs() < 1e-5);
    assert!((result.upper_a[[1, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((result.upper_a[[1, 0, 1]] - 2.0).abs() < 1e-5);
    assert!((result.upper_a[[1, 0, 2]] - 3.0).abs() < 1e-5);

    // Bias should accumulate 0 + 1 + 2 = 3 for each batch.
    assert!((result.lower_b[[0, 0]] - 3.0).abs() < 1e-5);
    assert!((result.lower_b[[1, 0]] - 3.0).abs() < 1e-5);
    assert!((result.upper_b[[0, 0]] - 3.0).abs() < 1e-5);
    assert!((result.upper_b[[1, 0]] - 3.0).abs() < 1e-5);
}
