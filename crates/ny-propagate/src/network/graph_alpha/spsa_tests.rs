// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::bounds::GraphAlphaState;
use crate::layers::{Layer, LinearLayer, ReLULayer, SkipMergeLayer};
use crate::network::core::{GraphNetwork, GraphNode};
use ndarray::{arr1, arr2};
use std::collections::HashMap;

/// Regression test for #2063: SPSA must propagate CROWN failures instead of
/// silently substituting 0.0 objective values.
///
/// Before the fix, `.ok().map(...).unwrap_or(0.0)` swallowed CROWN errors.
/// Now `?` propagates the error so the caller sees the failure.
#[test]
fn spsa_propagates_crown_failure_instead_of_swallowing() {
    // Build a graph where CROWN backward will fail:
    // _input -> linear -> relu -> skip_merge(relu, relu) -> output
    // SkipMerge with 2 inputs has no backward CROWN handler, so
    // propagate_crown_to_node_with_alpha will return Err.
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "merge",
        Layer::SkipMerge(SkipMergeLayer::new()),
        vec!["relu".to_string(), "relu".to_string()],
    ));
    graph.set_output("merge");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let ibp_bounds: HashMap<String, BoundedTensor> = graph
        .collect_crown_ibp_bounds_dag(&input)
        .unwrap_or_default();

    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .alphas
        .insert("relu".to_string(), Array1::from_vec(vec![0.5, 0.5]));
    alpha_state
        .unstable_mask
        .insert("relu".to_string(), Array1::from_vec(vec![true, true]));

    let result = graph.compute_spsa_gradients_dag_for_output_sparse(
        &input,
        &ibp_bounds,
        &alpha_state,
        "merge",
        0.01,
        1,
        None,
        None,
    );

    assert!(
        result.is_err(),
        "SPSA should propagate CROWN failure, not silently use 0.0. Got: {:?}",
        result
    );
}

/// Regression test for #2245: SPSA with num_samples=0 must not produce NaN
/// from division by zero in gradient averaging.
#[test]
fn spsa_zero_samples_does_not_produce_nan() {
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let ibp_bounds: HashMap<String, BoundedTensor> = graph
        .collect_crown_ibp_bounds_dag(&input)
        .unwrap_or_default();

    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .alphas
        .insert("relu".to_string(), Array1::from_vec(vec![0.5, 0.5]));
    alpha_state
        .unstable_mask
        .insert("relu".to_string(), Array1::from_vec(vec![true, true]));

    let result = graph.compute_spsa_gradients_dag_for_output_sparse(
        &input,
        &ibp_bounds,
        &alpha_state,
        "relu",
        0.01,
        0,
        None,
        None,
    );

    let grads = result.expect("SPSA with 0 samples should succeed, not error");
    for (name, grad) in &grads.relu {
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                !g.is_nan(),
                "Gradient for {}[{}] is NaN — division by zero regression (#2245)",
                name,
                i
            );
            assert!(
                !g.is_infinite(),
                "Gradient for {}[{}] is Inf — division by zero regression (#2245)",
                name,
                i
            );
        }
    }
}
