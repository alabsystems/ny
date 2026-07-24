// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{Array1, ArrayD, IxDyn};

fn build_first_conv() -> Conv2dLayer {
    let mut kernel = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -0.5;
    kernel[[0, 0, 1, 0]] = 0.3;
    kernel[[0, 0, 1, 1]] = 0.2;
    kernel[[1, 0, 0, 0]] = -0.4;
    kernel[[1, 0, 0, 1]] = 0.8;
    kernel[[1, 0, 1, 0]] = -0.1;
    kernel[[1, 0, 1, 1]] = 0.6;
    let bias = Array1::from_vec(vec![0.1, -0.1]);
    Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 4, 4).unwrap()
}

fn build_second_conv() -> Conv2dLayer {
    let mut kernel = ArrayD::zeros(IxDyn(&[1, 2, 2, 2]));
    kernel[[0, 0, 0, 0]] = 0.5;
    kernel[[0, 0, 0, 1]] = -0.3;
    kernel[[0, 0, 1, 0]] = 0.2;
    kernel[[0, 0, 1, 1]] = 0.4;
    kernel[[0, 1, 0, 0]] = -0.2;
    kernel[[0, 1, 0, 1]] = 0.6;
    kernel[[0, 1, 1, 0]] = -0.1;
    kernel[[0, 1, 1, 1]] = 0.3;
    let bias = Array1::from_vec(vec![0.05]);
    Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 3, 3).unwrap()
}

pub(crate) fn build_avgpool_memory_budget_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(build_first_conv()),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(build_second_conv()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "avgpool",
        Layer::AveragePool(AveragePoolLayer::new((2, 2), (1, 1), (0, 0), false)),
        vec!["conv2".to_string()],
    ));
    graph.set_output("avgpool");

    let center = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32);
    let input = BoundedTensor::from_epsilon(center, 0.3).unwrap();
    (graph, input)
}
