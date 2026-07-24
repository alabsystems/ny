// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn make_pad_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Pad(PadLayer::new(
        vec![(1, 1)],
        PadMode::Constant(0.0),
    )));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("pad+linear network should convert to graph")
}

#[test]
fn graph_pgd_whitelist_rejects_pad_graphs_4096() {
    let graph = make_pad_linear_graph();
    let input = make_interval_input(0.0, 1.0);
    let spec = make_upper_bound_spec(0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("pad graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed pad graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: pad graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

// --- #4096 equivalence tests round 2: elementwise activations ---

fn make_leaky_relu_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::LeakyReLU(LeakyReLULayer::new(0.1)));
    GraphNetwork::from_sequential(&network)
        .expect("single leaky_relu network should convert to graph")
}

fn make_exp_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Exp(ExpLayer));
    GraphNetwork::from_sequential(&network).expect("single exp network should convert to graph")
}

fn make_log_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Log(LogLayer));
    GraphNetwork::from_sequential(&network).expect("single log network should convert to graph")
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_leaky_relu_4096() {
    let graph = make_leaky_relu_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-2.0]).into_dyn(),
        arr1(&[0.5]).into_dyn(),
        arr1(&[3.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_exp_4096() {
    let graph = make_exp_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.0]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr1(&[1.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_log_4096() {
    let graph = make_log_graph();
    let engine = NaiveCpuGemmEngine;
    // Log requires strictly positive inputs.
    let samples = vec![
        arr1(&[0.5]).into_dyn(),
        arr1(&[1.0]).into_dyn(),
        arr1(&[3.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

// --- #4096 equivalence tests round 2: binary operations ---

fn make_sub_graph() -> GraphNetwork {
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.25]))).unwrap();
    let linear_b = LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[-0.5]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::binary(
        "sub",
        Layer::Sub(SubLayer),
        "linear_a",
        "linear_b",
    ));
    graph.set_output("sub");
    graph
}

fn make_mul_binary_graph() -> GraphNetwork {
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[1.0]))).unwrap();
    let linear_b = LinearLayer::new(arr2(&[[0.5]]), Some(arr1(&[0.5]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "linear_a",
        "linear_b",
    ));
    graph.set_output("mul");
    graph
}

fn make_div_graph() -> GraphNetwork {
    // Ensure divisor is strictly positive to avoid division by zero.
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[2.0]))).unwrap();
    let linear_b = LinearLayer::new(arr2(&[[0.5]]), Some(arr1(&[1.0]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::binary(
        "div",
        Layer::Div(DivLayer),
        "linear_a",
        "linear_b",
    ));
    graph.set_output("div");
    graph
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_sub_4096() {
    let graph = make_sub_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.0]).into_dyn(),
        arr1(&[0.5]).into_dyn(),
        arr1(&[2.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_mul_binary_4096() {
    let graph = make_mul_binary_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.0]).into_dyn(),
        arr1(&[0.5]).into_dyn(),
        arr1(&[2.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_div_4096() {
    let graph = make_div_graph();
    let engine = NaiveCpuGemmEngine;
    // Use strictly positive inputs so divisor stays safe.
    let samples = vec![
        arr1(&[0.5]).into_dyn(),
        arr1(&[1.0]).into_dyn(),
        arr1(&[2.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

// --- #4096 equivalence tests round 2: unary-constant operations ---

fn make_add_constant_graph() -> GraphNetwork {
    let constant = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(constant)));
    GraphNetwork::from_sequential(&network)
        .expect("single add_constant network should convert to graph")
}

fn make_mul_constant_graph() -> GraphNetwork {
    let constant = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::MulConstant(MulConstantLayer::new(constant)));
    GraphNetwork::from_sequential(&network)
        .expect("single mul_constant network should convert to graph")
}

fn make_sub_constant_graph() -> GraphNetwork {
    let constant = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.25]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::SubConstant(SubConstantLayer::new(constant)));
    GraphNetwork::from_sequential(&network)
        .expect("single sub_constant network should convert to graph")
}

fn make_div_constant_graph() -> GraphNetwork {
    let constant = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::DivConstant(DivConstantLayer::new(constant)));
    GraphNetwork::from_sequential(&network)
        .expect("single div_constant network should convert to graph")
}

fn make_abs_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Abs(AbsLayer));
    GraphNetwork::from_sequential(&network).expect("single abs network should convert to graph")
}

fn make_sqrt_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));
    GraphNetwork::from_sequential(&network).expect("single sqrt network should convert to graph")
}

fn make_pow_constant_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::PowConstant(PowConstantLayer::square()));
    GraphNetwork::from_sequential(&network)
        .expect("single pow_constant network should convert to graph")
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_add_constant_4096() {
    let graph = make_add_constant_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.0]).into_dyn(),
        arr1(&[0.5]).into_dyn(),
        arr1(&[2.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_mul_constant_4096() {
    let graph = make_mul_constant_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.5]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr1(&[3.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_sub_constant_4096() {
    let graph = make_sub_constant_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-0.5]).into_dyn(),
        arr1(&[1.0]).into_dyn(),
        arr1(&[2.5]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_div_constant_4096() {
    let graph = make_div_constant_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-2.0]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr1(&[4.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_abs_4096() {
    let graph = make_abs_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-2.5]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr1(&[1.5]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_sqrt_4096() {
    let graph = make_sqrt_graph();
    let engine = NaiveCpuGemmEngine;
    // Sqrt requires non-negative inputs.
    let samples = vec![
        arr1(&[0.25]).into_dyn(),
        arr1(&[1.0]).into_dyn(),
        arr1(&[4.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}

#[test]
fn graph_pgd_preserve_leading_axis_matches_sequential_pow_constant_4096() {
    let graph = make_pow_constant_graph();
    let engine = NaiveCpuGemmEngine;
    let samples = vec![
        arr1(&[-1.5]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
        arr1(&[2.0]).into_dyn(),
    ];
    assert_preserve_leading_axis_matches_sequential(&graph, &samples, Some(&engine));
}
