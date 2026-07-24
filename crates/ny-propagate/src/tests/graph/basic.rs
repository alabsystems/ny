// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Basic GraphNetwork construction and routing tests.
use super::super::assert_all_close;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_test_utils::CountingGemmEngine;

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_empty() {
    let graph = GraphNetwork::new();
    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), input.shape());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_single_node() {
    let mut graph = GraphNetwork::new();

    // Single ReLU node
    let relu_node = GraphNode::from_input("relu", Layer::ReLU(ReLULayer));
    graph.add_node(relu_node);
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5, 2.0]).into_dyn(),
        arr1(&[-0.5_f32, 1.5, 3.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // ReLU: max(0, x)
    assert!((output.lower()[[0]] - 0.0).abs() < 1e-5); // max(0, -1) = 0
    assert!((output.upper()[[0]] - 0.0).abs() < 1e-5); // max(0, -0.5) = 0
    assert!((output.lower()[[1]] - 0.5).abs() < 1e-5); // max(0, 0.5) = 0.5
    assert!((output.upper()[[1]] - 1.5).abs() < 1e-5); // max(0, 1.5) = 1.5
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_sequential_chain() {
    // Create a chain: input -> linear -> relu
    let mut graph = GraphNetwork::new();

    // Linear layer: 2 inputs -> 2 outputs
    let weight = arr2(&[[1.0_f32, 0.0], [0.0, -1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));

    // ReLU after linear
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 1.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Linear: [x, y] -> [x, -y]
    // For x in [1, 2], y in [1, 2]:
    //   output[0] in [1, 2]
    //   output[1] in [-2, -1]
    // ReLU: [max(0, 1), max(0, -2)] to [max(0, 2), max(0, -1)]
    //   output[0] in [1, 2]
    //   output[1] in [0, 0]
    assert!((output.lower()[[0]] - 1.0).abs() < 1e-5);
    assert!((output.upper()[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower()[[1]] - 0.0).abs() < 1e-5);
    assert!((output.upper()[[1]] - 0.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_propagate_ibp_with_engine_threads_linear_nodes_3954() {
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-1.0, 2.0]]),
        Some(arr1(&[0.0, 0.1])),
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.25]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();

    let baseline = graph.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "graph ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "graph ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3954 regression: GraphNetwork::propagate_ibp_with_engine should thread GemmEngine to Linear nodes"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_from_sequential() {
    // Build sequential network
    let mut network = Network::new();
    let weight = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Convert to graph
    let graph = GraphNetwork::from_sequential(&network).unwrap();

    assert_eq!(graph.num_nodes(), 2);

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let sequential_output = network.propagate_ibp(&input).unwrap();
    let graph_output = graph.propagate_ibp(&input).unwrap();

    // Should produce identical results
    assert!((sequential_output.lower()[[0]] - graph_output.lower()[[0]]).abs() < 1e-5);
    assert!((sequential_output.lower()[[1]] - graph_output.lower()[[1]]).abs() < 1e-5);
    assert!((sequential_output.upper()[[0]] - graph_output.upper()[[0]]).abs() < 1e-5);
    assert!((sequential_output.upper()[[1]] - graph_output.upper()[[1]]).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_branching() {
    // Create a graph with branching: two linear projections from input, then add
    //        input
    //       /     \
    //    proj_a  proj_b
    //       \     /
    //         add
    let mut graph = GraphNetwork::new();

    // proj_a: identity
    let weight_a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let proj_a = LinearLayer::new(weight_a, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_a", Layer::Linear(proj_a)));

    // proj_b: scale by 2
    let weight_b = arr2(&[[2.0_f32, 0.0], [0.0, 2.0]]);
    let proj_b = LinearLayer::new(weight_b, None).unwrap();
    graph.add_node(GraphNode::from_input("proj_b", Layer::Linear(proj_b)));

    // Add: proj_a + proj_b
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "proj_a",
        "proj_b",
    ));
    graph.set_output("add");

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // proj_a: [1, 2], proj_b: [2, 4]
    // add: [3, 6]
    assert!((output.lower()[[0]] - 3.0).abs() < 1e-5);
    assert!((output.upper()[[0]] - 3.0).abs() < 1e-5);
    assert!((output.lower()[[1]] - 6.0).abs() < 1e-5);
    assert!((output.upper()[[1]] - 6.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_where_ibp_union() {
    // Build a simple ternary Where graph:
    //   x = I(input)
    //   y = -x
    //   cond = relu(x) (condition is ignored by IBP Where relaxation; union bounds)
    //   out = where(cond, x, y)
    let mut graph = GraphNetwork::new();

    let w = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "cond",
        Layer::ReLU(ReLULayer),
        vec!["x".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();

    let out = graph.propagate_ibp(&input).unwrap();

    // x in [-1,2], y in [-2,1] => out in [-2,2]
    assert!((out.lower()[[0]] - (-2.0)).abs() < 1e-5);
    assert!((out.upper()[[0]] - 2.0).abs() < 1e-5);

    // x in [0.5,1.5], y in [-1.5,-0.5] => out in [-1.5,1.5]
    assert!((out.lower()[[1]] - (-1.5)).abs() < 1e-5);
    assert!((out.upper()[[1]] - 1.5).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_topological_sort() {
    let mut graph = GraphNetwork::new();

    // Build: input -> a -> b -> c
    //                    \-> d (b and d in parallel from a)
    let weight = arr2(&[[1.0_f32]]);

    graph.add_node(GraphNode::from_input(
        "a",
        Layer::Linear(LinearLayer::new(weight, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "b",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "c",
        Layer::ReLU(ReLULayer),
        vec!["b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "d",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()],
    ));

    let sorted = graph.topological_sort().unwrap();

    // a must come before b, c, d
    // b must come before c
    let pos_a = sorted.iter().position(|x| x == "a").unwrap();
    let pos_b = sorted.iter().position(|x| x == "b").unwrap();
    let pos_c = sorted.iter().position(|x| x == "c").unwrap();
    let pos_d = sorted.iter().position(|x| x == "d").unwrap();

    assert!(pos_a < pos_b);
    assert!(pos_a < pos_d);
    assert!(pos_b < pos_c);
}

#[ntest::timeout(10000)]
#[test]
fn test_topological_sort_rejects_dangling_unary_input() {
    // Node "b" depends on non-existent node "a" — should error, not emit phantom.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "b",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()], // "a" does not exist in graph
    ));

    let result = graph.topological_sort();
    assert!(
        result.is_err(),
        "Expected InvalidSpec for dangling input 'a'"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("'a'"),
        "Error should name the dangling node 'a'"
    );
    assert!(
        msg.contains("does not exist"),
        "Error should explain the dangling reference"
    );
    assert!(
        msg.contains("referenced by node 'b'"),
        "Error should name the consumer node 'b'"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_topological_sort_rejects_dangling_binary_input() {
    // Node "d" depends on existing "b" and missing "c".
    let mut graph = GraphNetwork::new();
    let weight = arr2(&[[1.0_f32]]);
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::Linear(LinearLayer::new(weight, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "d",
        Layer::ReLU(ReLULayer),
        vec!["b".to_string(), "c".to_string()], // "c" does not exist
    ));

    let result = graph.topological_sort();
    assert!(
        result.is_err(),
        "Expected InvalidSpec for dangling input 'c'"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("'c'"),
        "Error should name the dangling node 'c'"
    );
    assert!(
        msg.contains("referenced by node 'd'"),
        "Error should name the consumer node 'd'"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_topological_sort_cycle_detection_still_works() {
    // Existing cycle test: a -> b -> a should still error.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "a",
        Layer::ReLU(ReLULayer),
        vec!["b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "b",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()],
    ));

    let result = graph.topological_sort();
    assert!(result.is_err(), "Expected cycle detection error");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Cycle"), "Error should mention cycle");
}

/// Scalar input (ndim=0) must return Err, not panic at shape[ndim-1] (#2690).
#[ntest::timeout(10000)]
#[test]
fn test_crown_per_position_scalar_input_returns_err_2690() {
    let graph = GraphNetwork::new();
    let scalar = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[]), 2.0_f32),
    )
    .unwrap();

    let err = graph
        .propagate_crown_per_position(&scalar)
        .expect_err("scalar input (ndim=0) must return InvalidSpec");
    let msg = err.to_string();
    assert!(
        msg.contains("scalar input") || msg.contains("ndim=0"),
        "expected scalar-input diagnostic, got: {msg}"
    );
}

/// Zero-batch input (shape=[0, H]) must return Err, not panic at row(0) (#2690).
#[ntest::timeout(10000)]
#[test]
fn test_crown_per_position_zero_batch_returns_err_2690() {
    let graph = GraphNetwork::new();
    let zero_batch =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[0, 4])), ArrayD::zeros(IxDyn(&[0, 4]))).unwrap();

    let err = graph
        .propagate_crown_per_position(&zero_batch)
        .expect_err("zero-batch input must return InvalidSpec");
    let msg = err.to_string();
    assert!(
        msg.contains("zero-batch") || msg.contains("num_positions=0"),
        "expected zero-batch diagnostic, got: {msg}"
    );
}
