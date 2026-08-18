// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Hermetic regression for the cGAN affine-prefix bound bug.

use ndarray::Array2;
use ny_propagate::layers::{
    BatchNormLayer, ConvTranspose2dLayer, LinearLayer, ReLULayer, ReshapeLayer,
};
use ny_propagate::{BoundPropagation, GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

fn batch_norm(channels: usize, seed: usize) -> BatchNormLayer {
    let gamma = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[channels]), |index| {
        0.5 + ((index[0] * 3 + seed) % 5) as f32 * 0.3
    });
    let beta = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[channels]), |index| {
        ((index[0] + seed) % 3) as f32 * 0.1 - 0.1
    });
    let mean = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[channels]), |index| {
        ((index[0] * 2 + seed) % 4) as f32 * 0.2 - 0.3
    });
    let variance = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[channels]), |index| {
        0.5 + ((index[0] + seed) % 3) as f32 * 0.4
    });
    BatchNormLayer::new(&gamma, &beta, &mean, &variance, 1e-5).expect("batch norm")
}

#[test]
fn crown_ibp_keeps_cgan_affine_prefix_near_exact() {
    const INPUT_DIM: usize = 5;
    let weights = Array2::from_shape_fn((8, INPUT_DIM), |(row, column)| {
        (((row * 7 + column * 3) % 11) as f32 * 0.21 - 1.0)
            * if (row + column) % 2 == 0 { 1.0 } else { -1.0 }
    });
    let kernel = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 2, 2, 2]), |index| {
        (((index[0] * 5 + index[1] * 3 + index[2] * 2 + index[3]) % 7) as f32 * 0.33 - 1.0)
            * if (index[0] + index[1] + index[2] + index[3]) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
    });
    let first_bn = batch_norm(2, 1);
    let second_bn = batch_norm(2, 2);
    let transposed = || {
        ConvTranspose2dLayer::new_full(kernel.clone(), None, (2, 2), (0, 0), (1, 1), (0, 0))
            .expect("transposed convolution")
    };

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(LinearLayer::new(weights.clone(), None).expect("linear")),
    ));
    graph.add_node(GraphNode::new(
        "reshape",
        Layer::Reshape(ReshapeLayer {
            target_shape: vec![2, 2, 2],
        }),
        vec!["linear".into()],
    ));
    graph.add_node(GraphNode::new(
        "first_bn",
        Layer::BatchNorm(first_bn.clone()),
        vec!["reshape".into()],
    ));
    graph.add_node(GraphNode::new(
        "transposed",
        Layer::ConvTranspose2d(transposed()),
        vec!["first_bn".into()],
    ));
    graph.add_node(GraphNode::new(
        "second_bn",
        Layer::BatchNorm(second_bn.clone()),
        vec!["transposed".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["second_bn".into()],
    ));
    graph.set_output("relu");

    let lower = ndarray::Array1::from_elem(INPUT_DIM, 0.29f32);
    let upper = ndarray::Array1::from_elem(INPUT_DIM, 0.31f32);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).expect("input box");
    let collected = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .expect("CROWN-IBP bounds");
    let claimed = collected
        .bounds
        .get("second_bn")
        .expect("second batch-norm bounds");
    let claimed_max_width = claimed
        .lower()
        .iter()
        .zip(claimed.upper())
        .map(|(lower, upper)| upper - lower)
        .fold(0.0f32, f32::max);

    // Every operation before the ReLU is affine. Basis propagation therefore
    // gives its exact real-valued width, up to the bounded f32 point-forward
    // noise covered by the tolerance below.
    let forward = |point: &ndarray::Array1<f32>| {
        let linear = weights.dot(point);
        let reshaped = linear
            .into_shape_with_order(ndarray::IxDyn(&[2, 2, 2]))
            .expect("reshape");
        let bounded = BoundedTensor::concrete(reshaped).expect("point tensor");
        let bounded = first_bn.propagate_ibp(&bounded).expect("first batch norm");
        let bounded = transposed()
            .propagate_ibp(&bounded)
            .expect("transposed convolution");
        let bounded = second_bn
            .propagate_ibp(&bounded)
            .expect("second batch norm");
        ndarray::Array1::from_iter(bounded.lower().iter().copied())
    };
    let center = ndarray::Array1::from_elem(INPUT_DIM, 0.3f32);
    let baseline = forward(&center);
    let mut exact_width = ndarray::Array1::<f64>::zeros(baseline.len());
    for dimension in 0..INPUT_DIM {
        let mut basis = center.clone();
        basis[dimension] += 1.0;
        let displaced = forward(&basis);
        for output in 0..baseline.len() {
            exact_width[output] += 0.02 * f64::from((displaced[output] - baseline[output]).abs());
        }
    }
    let exact_max_width = exact_width.iter().copied().fold(0.0f64, f64::max);
    assert!(
        f64::from(claimed_max_width) <= exact_max_width * 1.05 + 1e-4,
        "affine-prefix CROWN width {claimed_max_width:.4e} exceeds exact width \
         {exact_max_width:.4e} by more than the rounding envelope"
    );
}
