// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public graph CROWN engine-threading coverage for DAG backward propagation.

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;

use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::*;

fn build_residual_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let linear1 = LinearLayer::new(
        arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]),
        Some(arr1(&[0.1_f32, -0.05])),
    )
    .expect("valid residual-dag linear1");
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let linear2 = LinearLayer::new(
        arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]),
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .expect("valid residual-dag linear2");
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        NETWORK_INPUT,
        "linear2",
    ));
    graph.set_output("residual");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("valid residual-dag input");

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_fixed_slope_with_engine_threads_gemm_on_residual_dag_3860() {
    let (graph, input) = build_residual_dag();
    let baseline = graph
        .propagate_crown_fixed_slope(&input)
        .expect("#3860 baseline fixed-slope graph CROWN should succeed");

    let counting_engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_crown_fixed_slope_with_engine(&input, Some(&counting_engine))
        .expect("#3860 counting-engine fixed-slope graph CROWN should succeed");

    assert_bounds_finite(&with_engine, "graph CROWN fixed-slope with engine output");
    assert!(
        counting_engine.gemm_calls() > 0,
        "#3860 residual DAG fixed-slope path should exercise the caller GemmEngine"
    );
    assert_bounded_tensor_close(
        &baseline,
        &with_engine,
        1e-5,
        "#3860 residual DAG fixed-slope public engine path",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_with_engine_threads_gemm_on_residual_dag_3860() {
    let (graph, input) = build_residual_dag();
    let baseline = graph
        .propagate_crown(&input)
        .expect("#3860 baseline public graph CROWN should succeed");

    let counting_engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_crown_with_engine(&input, Some(&counting_engine))
        .expect("#3860 counting-engine public graph CROWN should succeed");

    assert_bounds_finite(&with_engine, "graph CROWN with engine output");
    assert!(
        counting_engine.gemm_calls() > 0,
        "#3860 public graph CROWN path should exercise the caller GemmEngine"
    );
    assert_bounded_tensor_close(
        &baseline,
        &with_engine,
        1e-5,
        "#3860 residual DAG public engine path",
    );
}
