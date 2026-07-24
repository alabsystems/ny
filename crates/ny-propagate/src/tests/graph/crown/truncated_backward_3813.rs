// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use crate::network::SpecCrownRequest;
use crate::*;
use ndarray::{arr2, ArrayD, IxDyn};

fn scalar_bounds_3813(bounds: &BoundedTensor) -> (f32, f32) {
    let lower = bounds
        .lower()
        .iter()
        .next()
        .copied()
        .expect("toy graph truncated CROWN lower bound should contain one element");
    let upper = bounds
        .upper()
        .iter()
        .next()
        .copied()
        .expect("toy graph truncated CROWN upper bound should contain one element");
    (lower, upper)
}

fn build_truncated_conv_graph_3813() -> (GraphNetwork, BoundedTensor) {
    let kernel1 =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.6_f32, -0.3, 0.2, 0.5]).unwrap();
    let kernel2 =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.4_f32, 0.1, -0.2, 0.7]).unwrap();
    let conv1 = Conv2dLayer::with_input_shape(
        kernel1,
        Some(ndarray::arr1(&[0.05_f32])),
        (1, 1),
        (0, 0),
        4,
        4,
    )
    .unwrap();
    let conv2 = Conv2dLayer::with_input_shape(
        kernel2,
        Some(ndarray::arr1(&[-0.1_f32])),
        (1, 1),
        (0, 0),
        3,
        3,
    )
    .unwrap();
    let linear = LinearLayer::new(
        arr2(&[[0.5_f32, -0.4, 0.3, 0.2]]),
        Some(ndarray::arr1(&[0.05_f32])),
    )
    .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Conv2d(conv2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.4_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .unwrap();

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_spec_crown_truncation_contains_full_and_beats_ibp_3813() {
    let (graph, input) = build_truncated_conv_graph_3813();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("toy graph node bounds should collect");
    let spec = arr2(&[[1.0_f32]]);

    let full = SpecCrownRequest::new(&graph, &input, &spec, None)
        .node_bounds(&node_bounds)
        .run()
        .expect("full spec-guided CROWN should succeed on the toy conv graph");
    let truncated = SpecCrownRequest::new(&graph, &input, &spec, None)
        .node_bounds(&node_bounds)
        .truncate_after(2)
        .run()
        .expect("truncated spec-guided CROWN should succeed on the toy conv graph");
    let ibp = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on the toy conv graph");
    let (full_lower, full_upper) = scalar_bounds_3813(&full);
    let (truncated_lower, truncated_upper) = scalar_bounds_3813(&truncated);
    let (ibp_lower, ibp_upper) = scalar_bounds_3813(&ibp);

    assert!(
        truncated_lower <= full_lower + 1e-5,
        "#3813 graph truncated lower must contain full spec-guided CROWN: truncated={}, full={}",
        truncated_lower,
        full_lower,
    );
    assert!(
        full_upper <= truncated_upper + 1e-5,
        "#3813 graph truncated upper must contain full spec-guided CROWN: full={}, truncated={}",
        full_upper,
        truncated_upper,
    );
    assert!(
        ibp_lower <= truncated_lower + 1e-5,
        "#3813 graph truncated lower should stay at least as tight as IBP: ibp={}, truncated={}",
        ibp_lower,
        truncated_lower,
    );
    assert!(
        truncated_upper <= ibp_upper + 1e-5,
        "#3813 graph truncated upper should stay at least as tight as IBP: truncated={}, ibp={}",
        truncated_upper,
        ibp_upper,
    );
}
