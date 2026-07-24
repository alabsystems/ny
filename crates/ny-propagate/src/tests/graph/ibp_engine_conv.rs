// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::tests::{assert_all_close, crown::helpers::CountingGemmEngine};
use crate::*;
use ndarray::{arr1, ArrayD, IxDyn};

fn build_conv1d_graph_case() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.5, -0.25, 0.75, -0.2, 0.4, 0.1]).unwrap();
    let conv =
        Conv1dLayer::with_input_length(kernel, Some(arr1(&[0.15_f32, -0.05])), 1, 1, 6).unwrap();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv1d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-0.5_f32, -0.25, 0.0, -0.1, -0.2, -0.3])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![0.75_f32, 0.5, 0.4, 0.6, 0.8, 0.7]).unwrap(),
    )
    .unwrap();

    (graph, input)
}

fn build_conv2d_graph_case() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 2, 2]),
        vec![0.5, -0.1, 0.25, 0.75, -0.2, 0.3, 0.4, -0.15],
    )
    .unwrap();
    let conv =
        Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32, -0.2])), (1, 1), (0, 0), 3, 3)
            .unwrap();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![-0.5_f32, -0.25, 0.0, -0.1, -0.2, -0.3, 0.2, -0.4, 0.1],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![0.75_f32, 0.5, 0.4, 0.6, 0.8, 0.7, 0.9, 0.3, 0.5],
        )
        .unwrap(),
    )
    .unwrap();

    (graph, input)
}

fn assert_graph_engine_parity(graph: &GraphNetwork, input: &BoundedTensor, label: &str) {
    let baseline = graph.propagate_ibp(input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_ibp_with_engine(input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        &format!("{label} lower"),
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        &format!("{label} upper"),
    );
    assert!(
        engine.gemm_calls() > 0,
        "{label}: GraphNetwork::propagate_ibp_with_engine should thread GemmEngine"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_propagate_ibp_with_engine_threads_conv1d_nodes_4081() {
    let (graph, input) = build_conv1d_graph_case();
    assert_graph_engine_parity(&graph, &input, "graph conv1d ibp with engine");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_propagate_ibp_with_engine_threads_conv2d_nodes_4081() {
    let (graph, input) = build_conv2d_graph_case();
    assert_graph_engine_parity(&graph, &input, "graph conv2d ibp with engine");
}
