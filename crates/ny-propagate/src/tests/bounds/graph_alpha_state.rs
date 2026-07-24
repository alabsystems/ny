// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_relu_node_positive() {
    // Kills mutant: replace < with ==, >, <= at lines 1014, 1019
    let mut state = GraphAlphaState::new();
    let pre =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();
    let alpha = state.alpha("relu1").unwrap();
    assert_eq!(alpha[[0]], 1.0); // positive region
    assert_eq!(alpha[[1]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_relu_node_negative() {
    // Test negative region
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(
        arr1(&[-3.0, -2.0]).into_dyn(),
        arr1(&[-1.0, -0.5]).into_dyn(),
    )
    .unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();
    let alpha = state.alpha("relu1").unwrap();
    assert_eq!(alpha[[0]], 0.0);
    assert_eq!(alpha[[1]], 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_relu_node_unstable() {
    // Kills mutant: replace >= with < at line 1025, <= with > at line 1029
    // Kills mutant: replace > with ==, <, >= at line 1036, delete - at line 1036
    let mut state = GraphAlphaState::new();
    // l=-1, u=2: u > -l (2 > 1) => alpha = 1
    // l=-3, u=1: u > -l (1 > 3) false => alpha = 0
    let pre =
        BoundedTensor::new(arr1(&[-1.0, -3.0]).into_dyn(), arr1(&[2.0, 1.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();
    let alpha = state.alpha("relu1").unwrap();
    assert_eq!(alpha[[0]], 1.0);
    assert_eq!(alpha[[1]], 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_relu_node_does_something() {
    // Kills mutant: replace add_relu_node with ()
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    assert!(state.alpha("relu1").is_none());
    state.add_relu_node("relu1", &pre, false).unwrap();
    assert!(state.alpha("relu1").is_some());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_get_alpha_returns_correct() {
    // Kills mutant: replace get_alpha -> Option<&Array1<f32>> with None
    // Kills mutant: replace with Some(...) variants
    let mut state = GraphAlphaState::new();
    let pre =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();
    let alpha = state.alpha("relu1");
    assert!(alpha.is_some());
    let alpha = alpha.unwrap();
    assert_eq!(alpha.len(), 2);
    assert_eq!(alpha[[0]], 1.0);
    assert_eq!(alpha[[1]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_get_alpha_returns_none_for_missing() {
    let state = GraphAlphaState::new();
    assert!(state.alpha("nonexistent").is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update_velocity_formula() {
    // Kills mutants: replace - with + or / at line 1077
    // Kills mutants: replace * with + or / at line 1077
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();

    let initial_alpha = state.alpha("relu1").unwrap()[[0]];
    assert_eq!(initial_alpha, 1.0);

    let gradient = arr1(&[1.0]);
    state.update("relu1", &gradient, 0.1, 0.9);

    // vel = 0.9 * 0 - 0.1 * 1.0 = -0.1
    // alpha = 1.0 + (-0.1) = 0.9, clamped
    let expected = (initial_alpha - 0.1).clamp(0.0, 1.0);
    let actual = state.alpha("relu1").unwrap()[[0]];
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual={}, expected={}",
        actual,
        expected
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update_does_something() {
    // Kills mutant: replace update with ()
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();

    let initial_alpha = state.alpha("relu1").unwrap()[[0]];
    assert_eq!(initial_alpha, 1.0);

    let gradient = arr1(&[2.0]); // Large gradient
    state.update("relu1", &gradient, 0.5, 0.0); // No momentum, high LR

    let new_alpha = state.alpha("relu1").unwrap()[[0]];
    // vel = 0 - 0.5 * 2.0 = -1.0, alpha = 1.0 + (-1.0) = 0.0 (clamped)
    assert!(
        (new_alpha - initial_alpha).abs() > 0.01,
        "new_alpha={}, initial={}",
        new_alpha,
        initial_alpha
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update_accumulates() {
    // Kills mutant: replace += with -= or *= at line 1079
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();

    let alpha0 = state.alpha("relu1").unwrap()[[0]];
    assert_eq!(alpha0, 1.0);

    let gradient = arr1(&[0.5]);
    state.update("relu1", &gradient, 0.1, 0.5);
    let alpha1 = state.alpha("relu1").unwrap()[[0]];
    state.update("relu1", &gradient, 0.1, 0.5);
    let alpha2 = state.alpha("relu1").unwrap()[[0]];

    // With momentum and consistent positive gradient, alpha should keep decreasing
    assert!(
        alpha2 < alpha1,
        "alpha2={} should be < alpha1={}",
        alpha2,
        alpha1
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_num_unstable_counts() {
    // Kills mutant: replace num_unstable -> usize with 0 or 1
    let mut state = GraphAlphaState::new();
    // Node 1: 2 unstable
    let pre1 =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    // Node 2: 1 unstable, 1 stable
    let pre2 =
        BoundedTensor::new(arr1(&[-1.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre1, false).unwrap();
    state.add_relu_node("relu2", &pre2, false).unwrap();
    assert_eq!(state.num_unstable(), 3);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_relu_nodes_returns_keys() {
    // Kills mutant: replace relu_nodes with empty iterator
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();
    state.add_relu_node("relu2", &pre, false).unwrap();

    let nodes: Vec<&str> = state.relu_nodes().collect();
    assert_eq!(nodes.len(), 2);
    assert!(nodes.contains(&"relu1"));
    assert!(nodes.contains(&"relu2"));
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_strict_gt_boundary_u_equals_neg_l() {
    // Kills mutant: replace > with >= in GraphAlphaState::add_relu_node heuristic.
    let mut state = GraphAlphaState::new();
    let pre = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    state.add_relu_node("relu1", &pre, false).unwrap();
    let alpha = state.alpha("relu1").unwrap()[[0]];
    assert_eq!(alpha, 0.0);
}
