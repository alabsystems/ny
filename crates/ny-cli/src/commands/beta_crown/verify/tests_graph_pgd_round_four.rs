// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph PGD exact-gradient regression tests (#4274).

use super::*;
use ndarray::{arr1, arr2};
use ny_propagate::layers::AddLayer;

// ---------------------------------------------------------------------------
// Helpers: ResNet-style residual graph with all-whitelist layers
// ---------------------------------------------------------------------------

/// Build a small Linear -> ReLU -> Linear + skip -> Add residual graph
/// (all layers on the exact-gradient whitelist). Output dimension = 2.
///
/// Graph:
///   input -> linear1 -> relu -> linear2 --+-> add -> output
///              |                           |
///              +--- (skip) ----------------+
fn make_resnet_style_residual_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.5, -0.5], [0.3, 1.2]]), Some(arr1(&[0.1, -0.1])))
        .expect("valid weights");
    let linear2 = LinearLayer::new(arr2(&[[0.8, 0.2], [-0.4, 0.6]]), Some(arr1(&[0.0, 0.05])))
        .expect("valid weights");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".into()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".into()],
    ));
    graph.add_node(GraphNode::binary(
        "add",
        Layer::Add(AddLayer),
        "linear2",
        "linear1",
    ));
    graph.set_output("add");
    graph
}

/// Build a graph with a Sigmoid (NOT on the exact-gradient whitelist).
fn make_sigmoid_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    network.add_layer(Layer::Sigmoid(SigmoidLayer));
    GraphNetwork::from_sequential(&network).expect("linear+sigmoid should convert to graph")
}

fn make_spec_less_eq(lower: &[f32], upper: &[f32], a: usize, b: usize) -> VnnLibSpec {
    let mut decls = String::new();
    let mut bounds = String::new();
    for (i, (lo, hi)) in lower.iter().zip(upper.iter()).enumerate() {
        decls.push_str(&format!("(declare-const X_{i} Real)\n"));
        bounds.push_str(&format!(
            "(assert (>= X_{i} {lo}))\n(assert (<= X_{i} {hi}))\n"
        ));
    }
    for j in 0..=1 {
        decls.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    bounds.push_str(&format!("(assert (<= Y_{a} Y_{b}))\n"));
    parse_vnnlib(&format!("{decls}{bounds}")).unwrap()
}

// ---------------------------------------------------------------------------
// Test 1: exact-gradient path exercises without error on whitelist graph
// ---------------------------------------------------------------------------

#[test]
fn test_exact_grad_resnet_residual_exercises_path_4274() {
    let graph = make_resnet_style_residual_graph();
    assert!(
        graph_supports_exact_gradients(&graph),
        "residual Linear+ReLU+Add graph should pass exact-gradient whitelist"
    );

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Spec: output[0] <= output[1]
    let spec = make_spec_less_eq(&[-1.0, -1.0], &[1.0, 1.0], 0, 1);

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        3,
        50,
        Default::default(),
        20,
        None,
        None,
        true,
        false,
    );
    assert!(
        result.is_ok(),
        "exact-gradient graph PGD should not error: {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// Test 2: non-whitelist graph falls back cleanly to SPSA
// ---------------------------------------------------------------------------

#[test]
fn test_exact_grad_fallback_on_sigmoid_graph_4274() {
    let graph = make_sigmoid_graph();
    assert!(
        !graph_supports_exact_gradients(&graph),
        "Sigmoid graph should NOT pass exact-gradient whitelist"
    );

    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    let spec = make_upper_bound_spec(-2.0, 2.0, 0.3);

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        2,
        20,
        Default::default(),
        20,
        None,
        None,
        true,
        false,
    );
    assert!(
        result.is_ok(),
        "SPSA fallback on non-whitelist graph should not error"
    );
}

// ---------------------------------------------------------------------------
// Test 3: whitelist function correctness
// ---------------------------------------------------------------------------

#[test]
fn test_graph_supports_exact_gradients_whitelist_4274() {
    // Pure linear: eligible
    let linear = make_single_linear_graph(2.0, 0.5);
    assert!(graph_supports_exact_gradients(&linear));

    // ReLU: eligible
    let relu = make_relu_graph();
    assert!(graph_supports_exact_gradients(&relu));

    // Conv2d: eligible
    let conv2d = make_conv2d_graph();
    assert!(graph_supports_exact_gradients(&conv2d));

    // MaxPool: NOT eligible (non-linear pooling)
    let maxpool = make_maxpool_graph();
    assert!(!graph_supports_exact_gradients(&maxpool));

    // Sigmoid: NOT eligible (S-curve relaxation gap)
    let sigmoid = make_sigmoid_graph();
    assert!(!graph_supports_exact_gradients(&sigmoid));
}

// ---------------------------------------------------------------------------
// Test 4: GPU engine threading through exact gradient path
// ---------------------------------------------------------------------------

#[test]
fn test_exact_grad_threads_gemm_engine_4274() {
    let graph = make_resnet_style_residual_graph();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let spec = make_spec_less_eq(&[-1.0, -1.0], &[1.0, 1.0], 0, 1);

    let engine = CountingGemmEngine::new();
    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        2,
        10,
        Default::default(),
        20,
        None,
        Some(&engine as &dyn GemmEngine),
        true,
        false,
    );
    assert!(
        result.is_ok(),
        "exact-gradient with engine should not error: {:?}",
        result.err()
    );
}
