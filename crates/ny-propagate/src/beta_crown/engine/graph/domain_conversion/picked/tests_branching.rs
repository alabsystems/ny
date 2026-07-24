// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use crate::batched_domain::{DomainMetadata, PickedDomains};
use crate::layers::{Layer, ReLULayer};
use crate::{GraphNetwork, GraphNode};

use super::tests::{make_relu_graph, make_simple_picked};
use super::{
    branch_input_split_from_picked, branch_relu_from_picked, select_input_split_dimension,
};

#[test]
fn test_branch_relu_both_feasible() {
    let picked = make_simple_picked(
        &[-1.0, -1.0],
        &[2.0, 2.0],
        &[-1.0, -0.5],
        &[2.0, 1.0],
        "relu0",
        -1.0,
        2.0,
    );
    let graph = make_relu_graph("relu0");
    let layer_names = vec!["relu0".to_string()];

    let (active, inactive, propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu0", 0, 1.0, &layer_names, false)
            .expect("branching should succeed");

    assert!(active.is_some(), "active child should be feasible");
    assert!(inactive.is_some(), "inactive child should be feasible");
    assert!(!propagation_failure, "no propagation failure expected");
}

#[test]
fn test_branch_relu_only_active_feasible() {
    let picked = make_simple_picked(
        &[0.5, 0.0],
        &[2.0, 1.0],
        &[0.5, 0.0],
        &[2.0, 1.0],
        "relu0",
        -1.0,
        2.0,
    );
    let graph = make_relu_graph("relu0");
    let layer_names = vec!["relu0".to_string()];

    let (active, inactive, propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu0", 0, 1.0, &layer_names, false)
            .expect("branching should succeed");

    assert!(active.is_some(), "active child should be feasible");
    assert!(
        inactive.is_none(),
        "inactive child should be infeasible (lower > 0)"
    );
    assert!(!propagation_failure);
}

#[test]
fn test_branch_relu_only_inactive_feasible() {
    let picked = make_simple_picked(
        &[-3.0, 0.0],
        &[-0.5, 1.0],
        &[-3.0, 0.0],
        &[-0.5, 1.0],
        "relu0",
        -1.0,
        2.0,
    );
    let graph = make_relu_graph("relu0");
    let layer_names = vec!["relu0".to_string()];

    let (active, inactive, propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu0", 0, 1.0, &layer_names, false)
            .expect("branching should succeed");

    assert!(
        active.is_none(),
        "active child should be infeasible (upper < 0)"
    );
    assert!(inactive.is_some(), "inactive child should be feasible");
    assert!(!propagation_failure);
}

#[test]
fn test_branch_relu_nan_guard_returns_propagation_failure() {
    let picked = make_simple_picked(
        &[f32::NAN, 0.0],
        &[1.0, 1.0],
        &[f32::NAN, 0.0],
        &[1.0, 1.0],
        "relu0",
        -1.0,
        2.0,
    );
    let graph = make_relu_graph("relu0");
    let layer_names = vec!["relu0".to_string()];

    let (active, inactive, propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu0", 0, 1.0, &layer_names, false)
            .expect("NaN should be handled gracefully, not error");

    assert!(active.is_none());
    assert!(inactive.is_none());
    assert!(
        propagation_failure,
        "NaN bounds must set propagation_failure=true"
    );
}

#[test]
fn test_branch_relu_nonexistent_node_errors() {
    let picked = make_simple_picked(&[0.0], &[1.0], &[-1.0], &[1.0], "relu0", 0.0, 1.0);
    let graph = make_relu_graph("relu0");
    let layer_names = vec!["relu0".to_string()];

    let err = branch_relu_from_picked(
        0,
        &picked,
        &graph,
        "nonexistent_node",
        0,
        1.0,
        &layer_names,
        false,
    );
    assert!(err.is_err(), "nonexistent node should produce error");
}

#[test]
fn test_branch_relu_neuron_idx_out_of_bounds_returns_none() {
    let picked = make_simple_picked(
        &[0.0, -1.0],
        &[1.0, 2.0],
        &[-1.0, 0.0],
        &[1.0, 2.0],
        "relu0",
        -1.0,
        2.0,
    );
    let graph = make_relu_graph("relu0");
    let layer_names = vec!["relu0".to_string()];

    let (active, inactive, propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu0", 99, 1.0, &layer_names, false)
            .expect("out-of-bounds neuron returns None, not error");

    assert!(active.is_none());
    assert!(inactive.is_none());
    assert!(!propagation_failure);
}

#[test]
fn test_branch_input_split_basic() {
    let picked = make_simple_picked(
        &[0.0, -1.0],
        &[4.0, 3.0],
        &[-1.0],
        &[1.0],
        "layer0",
        -1.0,
        2.0,
    );

    let (left, right) =
        branch_input_split_from_picked(0, &picked, 0, false).expect("input split should succeed");

    let left = left.expect("left child should exist");
    let right = right.expect("right child should exist");

    let left_upper_flat: Vec<f32> = left.input_uppers.iter().copied().collect();
    assert_eq!(left_upper_flat[0], 2.0, "left upper[0] should be midpoint");

    let right_lower_flat: Vec<f32> = right.input_lowers.iter().copied().collect();
    assert_eq!(
        right_lower_flat[0], 2.0,
        "right lower[0] should be midpoint"
    );

    assert!(left.layer_lowers.is_empty());
    assert!(right.layer_lowers.is_empty());
}

#[test]
fn test_branch_input_split_zero_width_returns_none() {
    let picked = make_simple_picked(
        &[1.0, -1.0],
        &[1.0, 3.0],
        &[-1.0],
        &[1.0],
        "layer0",
        0.0,
        1.0,
    );

    let (left, right) =
        branch_input_split_from_picked(0, &picked, 0, false).expect("zero-width should not error");

    assert!(
        left.is_none(),
        "zero-width split should produce no children"
    );
    assert!(right.is_none());
}

#[test]
fn test_branch_input_split_dim_out_of_bounds_errors() {
    let picked = make_simple_picked(
        &[0.0, -1.0],
        &[1.0, 1.0],
        &[-1.0],
        &[1.0],
        "layer0",
        0.0,
        1.0,
    );

    let err = branch_input_split_from_picked(0, &picked, 99, false);
    assert!(err.is_err(), "split_dim out of bounds should error");
}

#[test]
fn test_branch_input_split_nan_width_returns_none() {
    let picked = make_simple_picked(
        &[f32::NAN, 0.0],
        &[1.0, 1.0],
        &[-1.0],
        &[1.0],
        "layer0",
        -1.0,
        1.0,
    );

    let (left, right) =
        branch_input_split_from_picked(0, &picked, 0, false).expect("NaN width should not error");
    assert!(left.is_none(), "NaN width should produce no children");
    assert!(right.is_none());
}

#[test]
fn test_branch_input_split_depth_increments() {
    let picked = make_simple_picked(&[0.0], &[4.0], &[-1.0], &[1.0], "layer0", -1.0, 2.0);

    let (left, _right) =
        branch_input_split_from_picked(0, &picked, 0, false).expect("split should succeed");

    let left = left.expect("left child should exist");
    assert_eq!(left.metadata.len(), 1);
}

#[test]
fn test_select_input_split_dimension_picks_widest() {
    let picked = make_simple_picked(
        &[0.0, -2.0],
        &[1.0, 3.0],
        &[-1.0],
        &[1.0],
        "layer0",
        0.0,
        1.0,
    );

    let (dim, midpoint) =
        select_input_split_dimension(&picked, 0).expect("should select dimension");

    assert_eq!(dim, 1, "dim 1 has width 5, should be picked");
    assert!((midpoint - 0.5).abs() < 1e-6);
}

#[test]
fn test_branch_relu_from_picked_with_pre_activation_layer() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear0", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["linear0".to_string()],
    ));
    graph.set_output("relu0");

    let mut layer_lowers = HashMap::new();
    let mut layer_uppers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0, -2.0, 0.5]).unwrap(),
    );
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 3.0, 2.0]).unwrap(),
    );
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.0, 0.0, 0.5]).unwrap(),
    );
    layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 3.0, 2.0]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0, -2.0, 0.5]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 3.0, 2.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![2.0],
        metadata: vec![DomainMetadata::root(-1.0, 2.0).unwrap()],
    };

    let layer_names = vec!["linear0".to_string(), "relu0".to_string()];

    let (active, inactive, propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu0", 1, 1.5, &layer_names, false)
            .expect("branching with pre-act layer should succeed");

    assert!(active.is_some(), "active child feasible (upper=3 >= 0)");
    assert!(
        inactive.is_some(),
        "inactive child feasible (lower=-2 <= 0)"
    );
    assert!(!propagation_failure);

    let active_domain = active.unwrap();
    let input_lower_flat: Vec<f32> = active_domain.input_bounds.lower().iter().copied().collect();
    assert_eq!(input_lower_flat, vec![-1.0, -2.0, 0.5]);
}
