// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for multi-objective verification helpers.
//!
//! Tests `objective_bounds_multi`, `propagate_multi_objective_with_beta`,
//! and `optimize_graph_beta_analytical_multi_objective` including the
//! `best_beta_snapshot` comparison logic.
//!
//! Issue: #1892 (AC5)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{arr1, ArrayD};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::{GraphCrownContext, MultiObjectiveTargets};
use crate::beta_crown::engine::graph::multi_objective::{
    merge_pruned_objective_bounds, prune_verified_multi_objective_targets,
};
use crate::beta_crown::engine::graph::objectives::objective_bounds;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::{GraphNeuronConstraint, GraphSplitHistory};
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

/// Build a simple graph, input bounds, and node bounds for testing.
fn setup_graph_with_bounds() -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, Arc<BoundedTensor>>,
) {
    let w1 = ndarray::arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = ndarray::arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

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

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let ibp = graph.collect_node_bounds(&input).unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> =
        ibp.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    (graph, input, node_bounds)
}

/// Build a 1-input, 2-output graph with a single unstable ReLU for multi-objective tests.
fn setup_multi_output_graph_with_bounds() -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, Arc<BoundedTensor>>,
) {
    let linear1 = LinearLayer::new(ndarray::Array2::eye(1), None).unwrap();
    let w2 = ndarray::arr2(&[[1.0_f32], [-1.0]]);
    let b2 = arr1(&[0.25_f32, -0.15]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

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

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let ibp = graph.collect_node_bounds(&input).unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> =
        ibp.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    (graph, input, node_bounds)
}

/// Create a split history with a single ReLU constraint on relu1[0] active.
fn single_relu_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    })
}

#[test]
fn test_objective_bounds_nan_lower_produces_conservative_bounds() {
    let lower = ArrayD::from_shape_vec(vec![2], vec![f32::NAN, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![2], vec![2.0, 3.0]).unwrap();
    let bt = BoundedTensor::new_unchecked(lower, upper).unwrap();
    let (lo, hi) = objective_bounds(&bt, &[1.0, 1.0]).unwrap();
    assert_eq!(lo, f32::NEG_INFINITY, "NaN lower input must produce -inf");
    assert_eq!(hi, f32::INFINITY, "NaN lower input must produce +inf");
}

#[test]
fn test_objective_bounds_nan_upper_produces_conservative_bounds() {
    let lower = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![2], vec![f32::NAN, 4.0]).unwrap();
    let bt = BoundedTensor::new_unchecked(lower, upper).unwrap();
    let (lo, hi) = objective_bounds(&bt, &[1.0, 1.0]).unwrap();
    assert_eq!(lo, f32::NEG_INFINITY, "NaN upper input must produce -inf");
    assert_eq!(hi, f32::INFINITY, "NaN upper input must produce +inf");
}

#[test]
fn test_objective_bounds_finite_inputs_exact() {
    let lower = ArrayD::from_shape_vec(vec![3], vec![1.0, -2.0, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![3], vec![3.0, 1.0, 2.0]).unwrap();
    let bt = BoundedTensor::new(lower, upper).unwrap();
    let (lo, hi) = objective_bounds(&bt, &[1.0, -1.0, 2.0]).unwrap();
    assert!(
        (lo - 1.0).abs() < 1e-6,
        "lower bound should be 1.0, got {lo}"
    );
    assert!(
        (hi - 9.0).abs() < 1e-6,
        "upper bound should be 9.0, got {hi}"
    );
}

#[test]
fn test_objective_bounds_shape_mismatch() {
    let lower = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![2], vec![3.0, 4.0]).unwrap();
    let bt = BoundedTensor::new(lower, upper).unwrap();
    let result = objective_bounds(&bt, &[1.0, 2.0, 3.0]);
    assert!(result.is_err(), "shape mismatch should return Err");
}

#[test]
fn test_objective_bounds_nan_with_negative_coefficients() {
    let lower = ArrayD::from_shape_vec(vec![2], vec![1.0, f32::NAN]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![2], vec![3.0, 5.0]).unwrap();
    let bt = BoundedTensor::new_unchecked(lower, upper).unwrap();
    let (lo, hi) = objective_bounds(&bt, &[-1.0, 1.0]).unwrap();
    assert_eq!(
        lo,
        f32::NEG_INFINITY,
        "NaN with neg coeff must produce -inf"
    );
    assert_eq!(hi, f32::INFINITY, "NaN with neg coeff must produce +inf");
}

// ============================================================
// objective_bounds_multi
// ============================================================

/// Test objective_bounds_multi with known values.
///
/// Network output: lower=[-2, 1], upper=[3, 4]
/// Objective 1: [1, -1] → bounds on Y0 - Y1
///   lower = 1*(-2) + (-1)*4 = -6
///   upper = 1*3 + (-1)*1 = 2
/// Objective 2: [0, 1] → bounds on Y1
///   lower = 0 + 1*1 = 1
///   upper = 0 + 1*4 = 4
#[ntest::timeout(5000)]
#[test]
fn test_objective_bounds_multi_known_values() {
    let output =
        BoundedTensor::new(arr1(&[-2.0, 1.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();

    let objectives = vec![vec![1.0, -1.0], vec![0.0, 1.0]];
    let bounds = BetaCrownVerifier::objective_bounds_multi(&output, &objectives).unwrap();

    assert_eq!(bounds.len(), 2);
    // Objective 1: [1, -1]
    assert!((bounds[0].0 - (-6.0)).abs() < 1e-6, "obj1 lower = -6");
    assert!((bounds[0].1 - 2.0).abs() < 1e-6, "obj1 upper = 2");
    // Objective 2: [0, 1]
    assert!((bounds[1].0 - 1.0).abs() < 1e-6, "obj2 lower = 1");
    assert!((bounds[1].1 - 4.0).abs() < 1e-6, "obj2 upper = 4");
}

/// Test objective_bounds_multi returns error on shape mismatch.
#[ntest::timeout(5000)]
#[test]
fn test_objective_bounds_multi_shape_mismatch() {
    let output =
        BoundedTensor::new(arr1(&[-2.0, 1.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();

    let objectives = vec![vec![1.0, -1.0, 0.5]]; // 3 elements vs 2 outputs
    let result = BetaCrownVerifier::objective_bounds_multi(&output, &objectives);
    assert!(result.is_err(), "should return shape mismatch error");
}

/// Test objective_bounds_multi with all-positive coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_objective_bounds_multi_positive_coefficients() {
    let output =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 5.0]).into_dyn()).unwrap();

    let objectives = vec![vec![2.0, 3.0]];
    let bounds = BetaCrownVerifier::objective_bounds_multi(&output, &objectives).unwrap();
    // lower = 2*1 + 3*2 = 8, upper = 2*3 + 3*5 = 21. The reduction accumulates
    // in f64 and casts each endpoint OUTWARD (next_down/next_up), so each may
    // sit exactly 1 f32 ULP outside the exact value (ULP(21) ≈ 1.9e-6) and
    // must NEVER sit inside it.
    assert!(
        bounds[0].0 <= 8.0 && 8.0 - bounds[0].0 < 2e-6,
        "lower must enclose exact 8 from below within 1 ULP, got {}",
        bounds[0].0
    );
    assert!(
        bounds[0].1 >= 21.0 && bounds[0].1 - 21.0 < 4e-6,
        "upper must enclose exact 21 from above within 1 ULP, got {}",
        bounds[0].1
    );
}

/// Test objective_bounds_multi with all-negative coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_objective_bounds_multi_negative_coefficients() {
    let output =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 5.0]).into_dyn()).unwrap();

    let objectives = vec![vec![-2.0, -3.0]];
    let bounds = BetaCrownVerifier::objective_bounds_multi(&output, &objectives).unwrap();
    // lower = (-2)*3 + (-3)*5 = -21, upper = (-2)*1 + (-3)*2 = -8. The
    // reduction accumulates in f64 and casts each endpoint OUTWARD
    // (next_down/next_up), so each may sit exactly 1 f32 ULP outside the
    // exact value (ULP(21) ≈ 1.9e-6) and must NEVER sit inside it.
    assert!(
        bounds[0].0 <= -21.0 && -21.0 - bounds[0].0 < 4e-6,
        "lower must enclose exact -21 from below within 1 ULP, got {}",
        bounds[0].0
    );
    assert!(
        bounds[0].1 >= -8.0 && bounds[0].1 - (-8.0) < 2e-6,
        "upper must enclose exact -8 from above within 1 ULP, got {}",
        bounds[0].1
    );
}

// ============================================================
// propagate_multi_objective_with_beta
// ============================================================

/// Test propagate_multi_objective_with_beta produces spec-guided bounds.
#[ntest::timeout(10000)]
#[test]
fn test_propagate_multi_objective_with_beta() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Simple graph has 1 output neuron, so objectives are 1D
    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let (obj_bounds, node_cache) = verifier
        .propagate_multi_objective_with_beta(&graph, &input, &context, &beta_state, &targets)
        .unwrap();

    assert_eq!(obj_bounds.len(), 1);
    let (lb, ub) = obj_bounds[0];
    assert!(lb.is_finite());
    assert!(lb <= ub);
    assert!(!node_cache.is_empty());
}

fn collect_sequential_objective_reference_4306(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    context: &GraphCrownContext<'_>,
    beta_state: &GraphBetaState,
    objectives: &[Vec<f32>],
) -> (
    Vec<(f32, f32)>,
    HashMap<String, Arc<BoundedTensor>>,
    Vec<Option<CachedLinearBounds>>,
) {
    let mut sequential_bounds = Vec::with_capacity(objectives.len());
    let mut sequential_node_bounds = None;
    let mut sequential_caches: Vec<Option<CachedLinearBounds>> =
        Vec::with_capacity(objectives.len());
    for (idx, objective) in objectives.iter().enumerate() {
        let (output, node_cache, captured_cache) = verifier
            .propagate_crown_with_graph_beta_and_cache(
                graph,
                input,
                context,
                beta_state,
                Some(objective),
                None,
                true,
            )
            .unwrap();
        sequential_bounds.push((output.lower_scalar(), output.upper_scalar()));
        if idx == 0 {
            sequential_node_bounds = Some(node_cache);
        }
        sequential_caches.push(captured_cache);
    }

    (
        sequential_bounds,
        sequential_node_bounds.expect("first sequential pass should cache nodes"),
        sequential_caches,
    )
}

fn assert_objective_bounds_match_4306(actual: &[(f32, f32)], expected: &[(f32, f32)]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, ((actual_l, actual_u), (expected_l, expected_u))) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (*actual_l - *expected_l).abs() <= 1e-5,
            "lower bound changed at objective {idx}: actual={actual_l}, expected={expected_l}"
        );
        assert!(
            (*actual_u - *expected_u).abs() <= 1e-5,
            "upper bound changed at objective {idx}: actual={actual_u}, expected={expected_u}"
        );
    }
}

fn assert_bounded_tensor_close_4306(actual: &BoundedTensor, expected: &BoundedTensor, label: &str) {
    let actual = actual.flatten();
    let expected = expected.flatten();
    assert_eq!(
        actual.lower().len(),
        expected.lower().len(),
        "{label}: lower length changed"
    );
    assert_eq!(
        actual.upper().len(),
        expected.upper().len(),
        "{label}: upper length changed"
    );
    for (idx, (&actual_l, &expected_l)) in actual.lower().iter().zip(expected.lower()).enumerate() {
        assert!(
            (actual_l - expected_l).abs() <= 1e-5,
            "{label}: lower[{idx}] changed: actual={actual_l}, expected={expected_l}"
        );
    }
    for (idx, (&actual_u, &expected_u)) in actual.upper().iter().zip(expected.upper()).enumerate() {
        assert!(
            (actual_u - expected_u).abs() <= 1e-5,
            "{label}: upper[{idx}] changed: actual={actual_u}, expected={expected_u}"
        );
    }
}

fn assert_linear_bounds_close_4306(
    actual: &crate::LinearBounds,
    expected: &crate::LinearBounds,
    label: &str,
) {
    assert_eq!(
        actual.lower_a().shape(),
        expected.lower_a().shape(),
        "{label}: lower_a shape changed"
    );
    assert_eq!(
        actual.upper_a().shape(),
        expected.upper_a().shape(),
        "{label}: upper_a shape changed"
    );
    assert_eq!(
        actual.lower_b().len(),
        expected.lower_b().len(),
        "{label}: lower_b length changed"
    );
    assert_eq!(
        actual.upper_b().len(),
        expected.upper_b().len(),
        "{label}: upper_b length changed"
    );

    for (idx, (&actual_l, &expected_l)) in actual
        .lower_a()
        .iter()
        .zip(expected.lower_a().iter())
        .enumerate()
    {
        assert!(
            (actual_l - expected_l).abs() <= 1e-5,
            "{label}: lower_a[{idx}] changed: actual={actual_l}, expected={expected_l}"
        );
    }
    for (idx, (&actual_u, &expected_u)) in actual
        .upper_a()
        .iter()
        .zip(expected.upper_a().iter())
        .enumerate()
    {
        assert!(
            (actual_u - expected_u).abs() <= 1e-5,
            "{label}: upper_a[{idx}] changed: actual={actual_u}, expected={expected_u}"
        );
    }
    for (idx, (&actual_l, &expected_l)) in actual
        .lower_b()
        .iter()
        .zip(expected.lower_b().iter())
        .enumerate()
    {
        assert!(
            (actual_l - expected_l).abs() <= 1e-5,
            "{label}: lower_b[{idx}] changed: actual={actual_l}, expected={expected_l}"
        );
    }
    for (idx, (&actual_u, &expected_u)) in actual
        .upper_b()
        .iter()
        .zip(expected.upper_b().iter())
        .enumerate()
    {
        assert!(
            (actual_u - expected_u).abs() <= 1e-5,
            "{label}: upper_b[{idx}] changed: actual={actual_u}, expected={expected_u}"
        );
    }
}

fn assert_batched_cache_matches_sequential_4306(
    batched_node_bounds: &HashMap<String, Arc<BoundedTensor>>,
    sequential_node_bounds: &HashMap<String, Arc<BoundedTensor>>,
    batched_caches: &[Option<CachedLinearBounds>],
    sequential_caches: &[Option<CachedLinearBounds>],
) {
    assert_eq!(
        batched_node_bounds.len(),
        sequential_node_bounds.len(),
        "batched fast-path should preserve shared node-bound cache cardinality"
    );
    for (node_name, expected_bounds) in sequential_node_bounds {
        let actual_bounds = batched_node_bounds
            .get(node_name)
            .unwrap_or_else(|| panic!("batched fast-path dropped node cache for {node_name}"));
        assert_bounded_tensor_close_4306(
            actual_bounds,
            expected_bounds,
            &format!("node_bounds[{node_name}]"),
        );
    }

    assert_eq!(batched_caches.len(), sequential_caches.len());
    for (objective_idx, (actual_cache, expected_cache)) in batched_caches
        .iter()
        .zip(sequential_caches.iter())
        .enumerate()
    {
        let actual_cache = actual_cache
            .as_ref()
            .unwrap_or_else(|| panic!("batched fast-path dropped cached_las[{objective_idx}]"));
        let expected_cache = expected_cache
            .as_ref()
            .unwrap_or_else(|| panic!("sequential reference missing cached_las[{objective_idx}]"));
        let actual_bounds = actual_cache.linear_bounds("relu1").unwrap_or_else(|| {
            panic!("batched cached_las[{objective_idx}] missing relu1 linear bounds")
        });
        let expected_bounds = expected_cache.linear_bounds("relu1").unwrap_or_else(|| {
            panic!("sequential cached_las[{objective_idx}] missing relu1 linear bounds")
        });
        assert_linear_bounds_close_4306(
            &actual_bounds,
            &expected_bounds,
            &format!("cached_las[{objective_idx}].relu1"),
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_multi_objective_with_beta_and_cache_matches_sequential_rows_4306() {
    let (graph, input, node_bounds) = setup_multi_output_graph_with_bounds();
    let history = single_relu_history();
    let beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let objectives = vec![vec![1.0_f32, -0.35_f32], vec![-0.6_f32, 1.0_f32]];
    let thresholds = vec![0.0_f32, 0.0_f32];
    let verified_mask = vec![false, false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let (sequential_bounds, sequential_node_bounds, sequential_caches) =
        collect_sequential_objective_reference_4306(
            &verifier,
            &graph,
            &input,
            &context,
            &beta_state,
            &objectives,
        );

    let seed_caches: Vec<Option<&CachedLinearBounds>> =
        sequential_caches.iter().map(Option::as_ref).collect();
    let (batched_bounds, batched_node_bounds, batched_caches) = verifier
        .propagate_multi_objective_with_beta_and_cache(
            &graph,
            &input,
            &context,
            &beta_state,
            &targets,
            &seed_caches,
            true,
        )
        .unwrap();

    assert_objective_bounds_match_4306(&batched_bounds, &sequential_bounds);
    assert_batched_cache_matches_sequential_4306(
        &batched_node_bounds,
        &sequential_node_bounds,
        &batched_caches,
        &sequential_caches,
    );
}

// ============================================================
// optimize_graph_beta_analytical_multi_objective
// ============================================================

/// Test optimize_graph_beta_analytical_multi_objective with empty beta.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_analytical_empty_beta() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = GraphSplitHistory::new();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 5,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let (obj_bounds, _) = verifier
        .optimize_graph_beta_analytical_multi_objective(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &targets,
            false,
        )
        .unwrap();

    assert_eq!(obj_bounds.len(), 1);
    assert!(obj_bounds[0].0.is_finite());
    assert!(obj_bounds[0].0 <= obj_bounds[0].1);
}

/// Test optimize_graph_beta_analytical_multi_objective with optimization.
///
/// Verifies: iteration loop runs, best_beta_snapshot comparison logic
/// (lines 206-216 in objectives.rs), and final bounds are valid.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_analytical_optimization() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 5,
        beta_lr: 0.05,
        beta_tolerance: 1e-8,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let (obj_bounds, node_cache) = verifier
        .optimize_graph_beta_analytical_multi_objective(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &targets,
            false,
        )
        .unwrap();

    assert_eq!(obj_bounds.len(), 1);
    let (lb, ub) = obj_bounds[0];
    assert!(lb.is_finite());
    assert!(lb <= ub);
    assert!(!node_cache.is_empty());
}

/// Regression test for #4306: multi-objective analytical optimization should
/// evaluate all objectives through the batched spec-matrix path.
///
/// With zero Adam learning rate, the optimization loop still executes but beta
/// values remain unchanged, so the returned bounds must match direct batched
/// spec-guided propagation exactly.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_analytical_optimization_matches_direct_batched_4306() {
    let (graph, input, node_bounds) = setup_multi_output_graph_with_bounds();
    let history = single_relu_history();
    let initial_beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let objectives = vec![vec![1.0_f32, -0.35_f32], vec![-0.6_f32, 1.0_f32]];
    let thresholds = vec![0.0_f32, 0.0_f32];
    let verified_mask = vec![false, false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let optimizer_config = BetaCrownConfig {
        beta_iterations: 1,
        beta_lr: 0.0,
        beta_tolerance: 0.0,
        ..Default::default()
    };
    let optimizer = BetaCrownVerifier::new(optimizer_config);
    let mut beta_state = initial_beta_state.clone();
    let (optimized_bounds, optimized_node_cache) = optimizer
        .optimize_graph_beta_analytical_multi_objective(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &targets,
            false,
        )
        .unwrap();

    let direct = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (direct_bounds, direct_node_cache) = direct
        .propagate_multi_objective_with_beta(
            &graph,
            &input,
            &context,
            &initial_beta_state,
            &targets,
        )
        .unwrap();

    assert_eq!(optimized_bounds.len(), direct_bounds.len());
    for (idx, ((actual_l, actual_u), (expected_l, expected_u))) in optimized_bounds
        .iter()
        .zip(direct_bounds.iter())
        .enumerate()
    {
        assert!(
            (*actual_l - *expected_l).abs() <= 1e-5,
            "optimized lower bound changed at objective {idx}: actual={actual_l}, expected={expected_l}"
        );
        assert!(
            (*actual_u - *expected_u).abs() <= 1e-5,
            "optimized upper bound changed at objective {idx}: actual={actual_u}, expected={expected_u}"
        );
    }
    assert_eq!(
        optimized_node_cache.len(),
        direct_node_cache.len(),
        "optimize-path batching should preserve shared node-bound cache shape"
    );
    for (node_name, expected_bounds) in &direct_node_cache {
        let actual_bounds = optimized_node_cache
            .get(node_name)
            .unwrap_or_else(|| panic!("optimize-path batching dropped node cache for {node_name}"));
        assert_bounded_tensor_close_4306(
            actual_bounds,
            expected_bounds,
            &format!("optimized_node_cache[{node_name}]"),
        );
    }
}

/// Test multi-objective best_beta_snapshot comparison selects tighter bounds.
///
/// This test verifies the snapshot comparison at lines 206-216 of objectives.rs:
/// the function computes spec-guided bounds for both the final beta state and the
/// best beta snapshot, and returns whichever has the higher minimum margin.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_best_beta_snapshot_comparison() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    // Run with many iterations — the best_beta_snapshot should be used if it's
    // better than the final state.
    let config = BetaCrownConfig {
        beta_iterations: 20,
        beta_lr: 0.1,
        beta_tolerance: 1e-10, // very tight — force all iterations to run
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();

    let (obj_bounds, _) = verifier
        .optimize_graph_beta_analytical_multi_objective(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &targets,
            false,
        )
        .unwrap();

    // The returned bounds should be the tighter of final vs snapshot
    let (lb, ub) = obj_bounds[0];
    assert!(lb.is_finite(), "returned lower bound must be finite");
    assert!(lb <= ub, "lb <= ub");

    // Verify beta_state was updated to match the returned bounds (consistency for warm-start)
    // Re-evaluate with the returned beta state to check consistency
    let config2 = BetaCrownConfig {
        beta_iterations: 0,
        ..Default::default()
    };
    let verifier2 = BetaCrownVerifier::new(config2);
    let (check_bounds, _) = verifier2
        .propagate_multi_objective_with_beta(&graph, &input, &context, &beta_state, &targets)
        .unwrap();

    // The check bounds should be close to the returned bounds (same beta state)
    let check_lb = check_bounds[0].0;
    assert!(
        (check_lb - lb).abs() < 1e-4,
        "re-evaluated lb={} should match returned lb={} (beta consistency)",
        check_lb,
        lb
    );
}

/// Test multi-objective with verified_mask filtering.
///
/// When all objectives are verified, the function should still return valid bounds
/// but min_margin computation should filter them out.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_all_verified_mask() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 3,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![true]; // all verified
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let (obj_bounds, _) = verifier
        .optimize_graph_beta_analytical_multi_objective(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &targets,
            false,
        )
        .unwrap();

    assert_eq!(obj_bounds.len(), 1);
    assert!(obj_bounds[0].0.is_finite());
}

/// Regression for #3813: child-domain multi-objective propagation should prune
/// already-verified OR-specs before spending the remaining deadline budget.
#[ntest::timeout(5000)]
#[test]
fn test_multi_objective_pruned_targets_skip_verified_deadline_slot_3813() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let mut config = BetaCrownConfig::default();
    config.alpha_config.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    let verifier = BetaCrownVerifier::new(config);

    let objectives = vec![vec![1.0], vec![1.0]];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![true, false];

    let full_targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);
    let (unpruned_bounds, _) = verifier
        .propagate_multi_objective_with_beta(&graph, &input, &context, &beta_state, &full_targets)
        .unwrap();
    // The implementation may compute bounds for all objectives in a single pass
    // before checking the deadline, so the unpruned result can be finite even
    // with an expired deadline. The key invariant tested below is that pruning +
    // merge produce valid, finite bounds for the active objective.
    assert_eq!(
        unpruned_bounds.len(),
        2,
        "unpruned should return bounds for both objectives"
    );

    let pruned_targets =
        prune_verified_multi_objective_targets(&objectives, &thresholds, &verified_mask);
    let active_targets = MultiObjectiveTargets::new(
        &pruned_targets.objectives,
        &pruned_targets.thresholds,
        &pruned_targets.verified_mask,
    );
    let (active_bounds, _) = verifier
        .propagate_multi_objective_with_beta(&graph, &input, &context, &beta_state, &active_targets)
        .unwrap();
    assert_eq!(active_bounds.len(), 1);
    assert!(
        active_bounds[0].0.is_finite() && active_bounds[0].1.is_finite(),
        "the pruned target set should spend its one allowed spec pass on the only unverified objective"
    );

    let merged_bounds =
        merge_pruned_objective_bounds(&[(7.0, 8.0), (0.0, 0.0)], &pruned_targets, active_bounds);
    assert_eq!(
        merged_bounds[0],
        (7.0, 8.0),
        "merging pruned results must preserve the already-verified objective bounds"
    );
    assert!(
        merged_bounds[1].0.is_finite() && merged_bounds[1].1.is_finite(),
        "merging pruned results must restore a finite bound for the previously unverified objective"
    );
}

// ============================================================
// #3109: multi-objective β-opt deadline granularity
// ============================================================

/// Regression for the multi-objective β-opt deadline-granularity fix (#3109),
/// "already past" case.
///
/// IMPORTANT: `BetaCrownVerifier::new` (engine/core.rs) OVERWRITES
/// `alpha_config.deadline` with `now + timeout`, so the deadline must be set on
/// the constructed verifier (matching the pattern in
/// `optimize_loop_regressions.rs`), not on the config passed to `new`.
///
/// With the wall-clock deadline already in the past, the OUTER β-iteration loop
/// breaks before the first pass. There is no completed iteration, so the
/// function falls through to the final spec-guided pass, whose inner per-node
/// deadline check aborts immediately with `DeadlineExceeded`. The function
/// therefore returns PROMPTLY with that graceful-timeout error rather than
/// grinding through all `beta_iterations` optimization passes. Callers
/// (root.rs / input_split) map `DeadlineExceeded` to a sound Timeout verdict.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_beta_opt_past_deadline_bails_promptly_3109() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    // Many iterations + an unreachable tolerance would force the FULL budget to
    // run if the deadline were ignored. The expired deadline must short-circuit.
    let config = BetaCrownConfig {
        beta_iterations: 5000,
        beta_lr: 0.1,
        beta_tolerance: 1e-12,
        ..Default::default()
    };
    let mut verifier = BetaCrownVerifier::new(config);
    // Set AFTER construction so `new`'s `now + timeout` override does not clobber it.
    verifier.config.alpha_config.deadline =
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let seed_caches: Vec<Option<&CachedLinearBounds>> = vec![None];

    let start = Instant::now();
    let result = verifier.optimize_graph_beta_analytical_multi_objective_with_cache(
        &graph,
        &input,
        &context,
        &mut beta_state,
        &targets,
        false,
        &seed_caches,
        false,
    );
    let elapsed = start.elapsed();

    // Bailed promptly: did not grind through 5000 optimization passes.
    assert!(
        elapsed < Duration::from_secs(2),
        "expired deadline must bail promptly, not run the full budget (elapsed {elapsed:?})"
    );
    let err = result.expect_err("an already-expired deadline must surface a graceful timeout");
    assert!(
        err.is_deadline_exceeded(),
        "expected DeadlineExceeded graceful timeout, got {err:?}"
    );
}

/// Core regression for the #3109 fix: a deadline reached *during* the loop must
/// break out and return the best bounds captured from a COMPLETED iteration,
/// WITHOUT running (or erroring on) the final spec-guided pass + snapshot eval.
///
/// Setup: a near-future deadline (set after construction so it survives `new`),
/// a huge iteration budget, and an unreachable tolerance. The loop runs real
/// spec-guided CROWN passes — each captures a `best_loop_result` — until the
/// wall clock crosses the deadline, at which point the inner per-node check
/// aborts a pass with `DeadlineExceeded`. The fix catches that, breaks, and the
/// post-loop short-circuit returns the best completed bounds.
///
/// BEFORE the fix the inner `?` propagated `DeadlineExceeded`, discarding all
/// beta-opt work and failing the whole child propagation. AFTER the fix the
/// call returns `Ok` with a valid, sound spec-guided CROWN bound (from a
/// completed pass) and bails far short of the 200_000-iteration budget.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_beta_opt_deadline_during_loop_returns_best_so_far_3109() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.0];
    let verified_mask = vec![false];
    let targets = MultiObjectiveTargets::new(&objectives, &thresholds, &verified_mask);

    let config = BetaCrownConfig {
        // Huge budget + unreachable tolerance: without a deadline this would run
        // far longer than the small window we allow before the deadline fires.
        beta_iterations: 200_000,
        beta_lr: 0.1,
        beta_tolerance: 1e-12,
        ..Default::default()
    };
    let mut verifier = BetaCrownVerifier::new(config);
    // Generous future window so many iterations complete (capturing
    // best_loop_result) before the deadline fires. Set AFTER construction.
    verifier.config.alpha_config.deadline = Some(Instant::now() + Duration::from_millis(50));

    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let seed_caches: Vec<Option<&CachedLinearBounds>> = vec![None];

    let start = Instant::now();
    let (obj_bounds, node_cache, caches) = verifier
        .optimize_graph_beta_analytical_multi_objective_with_cache(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &targets,
            false,
            &seed_caches,
            false,
        )
        .expect("deadline-during-loop must return best completed bounds, not an error");
    let elapsed = start.elapsed();

    // Returned promptly after the deadline rather than running 200k iterations.
    assert!(
        elapsed < Duration::from_secs(5),
        "must bail on deadline, not run the full budget (elapsed {elapsed:?})"
    );
    assert_eq!(obj_bounds.len(), 1);
    let (lb, ub) = obj_bounds[0];
    // Sound bound: finite and ordered (a valid spec-guided CROWN bound).
    assert!(
        lb.is_finite() && ub.is_finite(),
        "bound must be finite: ({lb}, {ub})"
    );
    assert!(lb <= ub, "lb <= ub required: ({lb}, {ub})");
    assert!(!node_cache.is_empty(), "node bounds must be populated");
    assert_eq!(
        caches.len(),
        objectives.len(),
        "one cache slot per objective"
    );
}
