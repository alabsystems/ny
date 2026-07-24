// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for constraint-aware CROWN tests.

use ndarray::{arr1, arr2};
use ny_test_utils::assert_bounded_tensor_close;
use std::collections::HashMap;

use crate::beta_crown::{GraphNeuronConstraint, GraphSplitHistory};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

use super::TOL;

pub(super) fn build_single_relu_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

pub(super) fn build_two_relu_clip_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[-0.2]))).expect("valid linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");
    graph
}

pub(super) fn build_input_bounds() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds")
}

pub(super) fn inactive_relu_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    })
}

pub(super) fn active_relu_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    })
}

pub(super) fn clip_test_history() -> GraphSplitHistory {
    GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.0,
        })
}

pub(super) fn scalar_interval(bounds: &BoundedTensor) -> (f32, f32) {
    let lower = bounds.lower().iter().copied().fold(f32::INFINITY, f32::min);
    let upper = bounds
        .upper()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    (lower, upper)
}

pub(super) fn assert_scalar_bounds(
    bounds: &BoundedTensor,
    expected_lower: f32,
    expected_upper: f32,
    label: &str,
) {
    let (lower, upper) = scalar_interval(bounds);
    assert!(
        (lower - expected_lower).abs() <= TOL,
        "{} lower mismatch: expected {}, got {}",
        label,
        expected_lower,
        lower
    );
    assert!(
        (upper - expected_upper).abs() <= TOL,
        "{} upper mismatch: expected {}, got {}",
        label,
        expected_upper,
        upper
    );
}

pub(super) fn assert_cache_bounds_close(
    lhs: &HashMap<String, std::sync::Arc<BoundedTensor>>,
    rhs: &HashMap<String, std::sync::Arc<BoundedTensor>>,
    label: &str,
) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "{} cache length mismatch: lhs={}, rhs={}",
        label,
        lhs.len(),
        rhs.len()
    );

    for (node_name, lhs_bounds) in lhs {
        let rhs_bounds = rhs
            .get(node_name)
            .unwrap_or_else(|| panic!("{} missing node '{}'", label, node_name));
        assert_bounded_tensor_close(
            lhs_bounds,
            rhs_bounds,
            TOL,
            &format!("{}:{}", label, node_name),
        );
    }
}
