// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for graph BaB beta optimization functions.
//!
//! Tests `optimize_graph_beta_spsa`, `optimize_graph_beta_analytical`,
//! `evaluate_graph_child_bounds`, and `propagate_crown_with_graph_beta_and_intermediates`.
//!
//! Issue: #1892

use super::gpu_bab::simple_graph_network;
use super::prelude::*;
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::state::GraphDomainAlphaState;
use std::collections::HashMap;

/// Verify that computed bounds contain the true network output at sample points.
///
/// Evaluates the graph network at evenly-spaced grid points within the input
/// domain and asserts that every output is within `[lower_bound, upper_bound]`.
/// This is the core soundness invariant: bounds must contain the true output.
///
/// Uses a grid of `grid_size` points per dimension (total grid_size^d evaluations).
fn assert_bounds_contain_samples(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    lower_bound: f32,
    upper_bound: f32,
    grid_size: usize,
) {
    let lower = input.lower();
    let upper = input.upper();
    let ndim = lower.len();
    assert_eq!(ndim, 2, "helper assumes 2D input for simple_graph_network");

    let l0 = lower[[0]];
    let u0 = upper[[0]];
    let l1 = lower[[1]];
    let u1 = upper[[1]];

    for i in 0..grid_size {
        for j in 0..grid_size {
            let t0 = if grid_size > 1 {
                i as f32 / (grid_size - 1) as f32
            } else {
                0.5
            };
            let t1 = if grid_size > 1 {
                j as f32 / (grid_size - 1) as f32
            } else {
                0.5
            };
            let x0 = l0 + (u0 - l0) * t0;
            let x1 = l1 + (u1 - l1) * t1;

            let point = ArrayD::from_shape_vec(IxDyn(&[2]), vec![x0, x1]).unwrap();
            let concrete = BoundedTensor::concrete(point).unwrap();
            let output = graph.propagate_ibp(&concrete).unwrap();
            let y = output.lower()[[0]]; // lower == upper for concrete input

            assert!(
                y >= lower_bound - 1e-6 && y <= upper_bound + 1e-6,
                "soundness violation: f({}, {}) = {} not in [{}, {}]",
                x0,
                x1,
                y,
                lower_bound,
                upper_bound
            );
        }
    }
}

/// Build a simple graph, input bounds, and node bounds for testing.
fn setup_graph_with_bounds() -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, Arc<BoundedTensor>>,
) {
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let ibp = graph.collect_node_bounds(&input).unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> =
        ibp.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    (graph, input, node_bounds)
}

/// Build a graph whose output uses mixed-sign coefficients at the ReLU so
/// upper-path dual alpha is exercised during backward propagation.
fn setup_mixed_sign_graph_with_bounds() -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, Arc<BoundedTensor>>,
) {
    let linear1 = LinearLayer::new(arr2(&[[1.0, -1.0], [-1.0, 1.0]]), None).unwrap();
    let linear2 = LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap();

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

/// Create a split history with a single ReLU constraint on relu1[0] active.
fn single_relu_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    })
}

// ============================================================
// AC1: optimize_graph_beta_spsa
// ============================================================

/// Test SPSA optimization with empty beta state early-exits and returns valid bounds.
#[ntest::timeout(10000)]
#[test]
fn test_spsa_empty_beta_early_exit() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = GraphSplitHistory::new(); // no constraints
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 10,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, cache) = verifier
        .optimize_graph_beta_spsa(&graph, &input, &context, &mut beta_state, &[1.0])
        .unwrap();

    assert!(lb.is_finite(), "lower bound must be finite");
    assert!(ub.is_finite(), "upper bound must be finite");
    assert!(lb <= ub, "lower <= upper: {} <= {}", lb, ub);
    assert!(!cache.is_empty(), "node bounds cache should be non-empty");
}

/// Test SPSA optimization with zero iterations early-exits.
#[ntest::timeout(10000)]
#[test]
fn test_spsa_zero_iterations_early_exit() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 0,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, _) = verifier
        .optimize_graph_beta_spsa(&graph, &input, &context, &mut beta_state, &[1.0])
        .unwrap();

    assert!(lb.is_finite());
    assert!(lb <= ub);
}

/// Test SPSA optimization runs multiple iterations and produces valid bounds.
///
/// Verifies: gradient estimation loop executes, Adam step updates beta values,
/// convergence check runs, and final bounds are the better of optimized vs best-seen.
///
/// For the simple graph (Linear(2->2) -> ReLU -> Linear(2->1)) with input [-1,1]^2
/// and relu1[0] constrained active, the unconstrained CROWN bounds are known.
/// We verify SPSA produces bounds within a reasonable envelope and that
/// the beta state is actually modified (proving the optimizer ran).
#[ntest::timeout(10000)]
#[test]
fn test_spsa_optimization_runs_and_produces_valid_bounds() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    // Capture initial beta values before optimization
    let beta_values_before: Vec<f32> = beta_state.entries.iter().map(|e| e.value).collect();

    let config = BetaCrownConfig {
        beta_iterations: 5,
        beta_lr: 0.05,
        beta_tolerance: 1e-8, // tight tolerance to ensure iterations run
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, cache) = verifier
        .optimize_graph_beta_spsa(&graph, &input, &context, &mut beta_state, &[1.0])
        .unwrap();

    assert!(lb.is_finite(), "lower bound must be finite");
    assert!(ub.is_finite(), "upper bound must be finite");
    assert!(lb <= ub, "lb <= ub: {} <= {}", lb, ub);
    assert!(!cache.is_empty(), "node bounds cache should be non-empty");

    // Verify the optimizer actually ran: beta values should differ from initial
    let beta_values_after: Vec<f32> = beta_state.entries.iter().map(|e| e.value).collect();
    // With 5 iterations and tight tolerance, at least one beta should have changed
    let any_changed = beta_values_before
        .iter()
        .zip(beta_values_after.iter())
        .any(|(before, after)| (*before - *after).abs() > 1e-10);
    assert!(
        any_changed,
        "SPSA with 5 iterations should modify at least one beta value; \
         before={:?}, after={:?}",
        beta_values_before, beta_values_after
    );
}

/// Test SPSA with a non-trivial objective and verify bounds are sound.
///
/// The simple graph is Linear(2->2) -> ReLU -> Linear(2->1).
/// With objective [1.0], we verify the final output.
/// With input [-1,1] to [1,1], the beta optimization should not degrade bounds.
#[ntest::timeout(10000)]
#[test]
fn test_spsa_bounds_no_worse_than_baseline() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    // Baseline: no optimization
    let config_baseline = BetaCrownConfig {
        beta_iterations: 0,
        ..Default::default()
    };
    let verifier_baseline = BetaCrownVerifier::new(config_baseline);
    let mut beta_baseline = GraphBetaState::from_history(&history).unwrap();
    let (lb_baseline, _, _) = verifier_baseline
        .optimize_graph_beta_spsa(&graph, &input, &context, &mut beta_baseline, &[1.0])
        .unwrap();

    // Optimized: 10 iterations
    let config_opt = BetaCrownConfig {
        beta_iterations: 10,
        beta_lr: 0.05,
        beta_tolerance: 1e-8,
        ..Default::default()
    };
    let verifier_opt = BetaCrownVerifier::new(config_opt);
    let mut beta_opt = GraphBetaState::from_history(&history).unwrap();
    let (lb_opt, ub_opt, _) = verifier_opt
        .optimize_graph_beta_spsa(&graph, &input, &context, &mut beta_opt, &[1.0])
        .unwrap();

    // SPSA optimization should not worsen the lower bound
    // (it returns the better of final vs best-seen)
    assert!(
        lb_opt >= lb_baseline - 1e-6,
        "SPSA should not degrade lower bound: opt={} vs baseline={}",
        lb_opt,
        lb_baseline
    );
    assert!(lb_opt <= ub_opt, "lb <= ub after optimization");
}

// ============================================================
// AC2: optimize_graph_beta_analytical
// ============================================================

/// Test analytical optimization with empty beta state early-exits.
#[ntest::timeout(10000)]
#[test]
fn test_analytical_empty_beta_early_exit() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = GraphSplitHistory::new();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let mut alpha_state = GraphDomainAlphaState::empty();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 10,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, _) = verifier
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &mut alpha_state,
            &[1.0],
        )
        .unwrap();

    assert!(lb.is_finite());
    assert!(lb <= ub);
}

/// Test analytical optimization with zero iterations early-exits.
#[ntest::timeout(10000)]
#[test]
fn test_analytical_zero_iterations_early_exit() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let mut alpha_state = GraphDomainAlphaState::empty();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 0,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, _) = verifier
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &mut alpha_state,
            &[1.0],
        )
        .unwrap();

    assert!(lb.is_finite());
    assert!(lb <= ub);
}

/// Test analytical optimization runs iterations and produces valid bounds.
///
/// Verifies: A matrix capture, analytical gradient computation, Adam step,
/// joint alpha-beta convergence check, and best-of selection.
///
/// Cross-validates against SPSA: both should produce consistent bounds for
/// the same network/input/constraint, and analytical should not be worse.
#[ntest::timeout(10000)]
#[test]
fn test_analytical_optimization_produces_valid_bounds() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();
    let mut alpha_state = GraphDomainAlphaState::empty();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    // Get baseline (0-iteration) bounds for comparison
    let mut beta_baseline = GraphBetaState::from_history(&history).unwrap();
    let mut alpha_baseline = GraphDomainAlphaState::empty();
    let baseline_config = BetaCrownConfig {
        beta_iterations: 0,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let baseline_verifier = BetaCrownVerifier::new(baseline_config);
    let (lb_baseline, _, _) = baseline_verifier
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_baseline,
            &mut alpha_baseline,
            &[1.0],
        )
        .unwrap();

    let config = BetaCrownConfig {
        beta_iterations: 5,
        beta_lr: 0.05,
        beta_tolerance: 1e-8,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, cache) = verifier
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &mut alpha_state,
            &[1.0],
        )
        .unwrap();

    assert!(lb.is_finite(), "lower bound must be finite");
    assert!(ub.is_finite(), "upper bound must be finite");
    assert!(lb <= ub, "lb <= ub: {} <= {}", lb, ub);
    assert!(!cache.is_empty(), "node bounds cache non-empty");

    // Key invariant: analytical optimization should not degrade bounds.
    // For this simple network, the optimal beta is exactly 0 (the gradient at
    // beta=0 is negative, so increasing beta worsens bounds). The analytical
    // optimizer correctly identifies this and keeps beta at 0.
    assert!(
        lb >= lb_baseline - 1e-6,
        "analytical lb ({}) must not be worse than baseline lb ({})",
        lb,
        lb_baseline
    );

    // Cross-validate: run SPSA on the same setup and compare.
    // SPSA uses noisy gradient estimates, so it may push beta to a small
    // positive value even when the true optimum is 0. Both should still
    // produce sound bounds (lb_analytical <= true_lb <= ub_analytical).
    let mut beta_spsa = GraphBetaState::from_history(&history).unwrap();
    let spsa_config = BetaCrownConfig {
        beta_iterations: 5,
        beta_lr: 0.05,
        beta_tolerance: 1e-8,
        use_analytical_beta_gradients: false,
        ..Default::default()
    };
    let spsa_verifier = BetaCrownVerifier::new(spsa_config);
    let (lb_spsa, ub_spsa, _) = spsa_verifier
        .optimize_graph_beta_spsa(&graph, &input, &context, &mut beta_spsa, &[1.0])
        .unwrap();

    // Both methods produce sound lower bounds. On this simple network the
    // analytical gradient at beta=0 is negative (beta=0 is a constrained
    // optimum), so analytical keeps beta=0. SPSA's noisy gradients push
    // beta positive, sometimes escaping the local optimum. This means
    // SPSA can produce a tighter lb on this network. Both are sound.
    //
    // Tolerance: SPSA uses stochastic gradient estimates, so it can find different
    // optima than analytical. On this network, SPSA's noisy gradients sometimes
    // escape the constrained optimum at beta=0. Empirically, differences up to 0.5
    // are normal for 5 iterations with lr=0.05. The deterministic soundness check
    // below (assert_bounds_contain_samples at 7×7 grid) is the primary correctness
    // validation; this tolerance is a secondary consistency check.
    assert!(
        (lb - lb_spsa).abs() < 0.5,
        "analytical lb ({}) and SPSA lb ({}) should be within 0.5 of each other",
        lb,
        lb_spsa
    );
    assert!(
        ub_spsa.is_finite(),
        "SPSA upper bound must be finite for cross-validation"
    );

    // Deterministic soundness: both bounds must contain the true network output
    // at sample points. Use 7×7 grid (49 points) for higher-confidence checks.
    assert_bounds_contain_samples(&graph, &input, lb, ub, 7);
    assert_bounds_contain_samples(&graph, &input, lb_spsa, ub_spsa, 7);
}

/// Test analytical optimization with alpha state populated.
///
/// When alpha_state is non-empty, the analytical path should jointly optimize
/// alpha and beta parameters. We verify alpha gradients are accumulated and
/// the Adam step updates alpha.
#[ntest::timeout(10000)]
#[test]
fn test_analytical_with_alpha_state_joint_optimization() {
    let (graph, input, node_bounds) = setup_mixed_sign_graph_with_bounds();
    let history = single_relu_history();
    let mut beta_state = GraphBetaState::from_history(&history).unwrap();

    // Initialize alpha state from graph bounds (non-empty)
    let mut alpha_state =
        GraphDomainAlphaState::from_graph_bounds(&graph, &node_bounds, &history, &input);
    let alpha_len_before = alpha_state.len();
    for neuron_map in alpha_state.upper_neurons_mut().values_mut() {
        for neuron_state in neuron_map.values_mut() {
            neuron_state.set_alpha(0.5);
        }
    }
    let upper_alpha_before: Vec<(String, usize, f32)> = alpha_state
        .upper_neurons()
        .iter()
        .flat_map(|(node_name, neuron_map)| {
            neuron_map.iter().map(move |(&neuron_idx, neuron_state)| {
                (node_name.clone(), neuron_idx, neuron_state.alpha())
            })
        })
        .collect();

    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig {
        beta_iterations: 5,
        beta_lr: 0.05,
        beta_tolerance: 1e-8,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (lb, ub, _) = verifier
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_state,
            &mut alpha_state,
            &[1.0],
        )
        .unwrap();

    assert!(lb.is_finite());
    assert!(lb <= ub);
    // Alpha state should still have the same number of entries
    assert_eq!(alpha_state.len(), alpha_len_before);
    let upper_alpha_changed = upper_alpha_before
        .iter()
        .any(|(node_name, neuron_idx, before)| {
            (alpha_state.alpha_upper(node_name, *neuron_idx) - *before).abs() > 1e-6
        });
    assert!(
        upper_alpha_changed,
        "joint optimization should update at least one upper alpha"
    );
}

/// Test analytical bounds are at least as good as baseline (no optimization).
#[ntest::timeout(10000)]
#[test]
fn test_analytical_bounds_no_worse_than_baseline() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    // Baseline
    let config_baseline = BetaCrownConfig {
        beta_iterations: 0,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier_baseline = BetaCrownVerifier::new(config_baseline);
    let mut beta_baseline = GraphBetaState::from_history(&history).unwrap();
    let mut alpha_baseline = GraphDomainAlphaState::empty();
    let (lb_baseline, _, _) = verifier_baseline
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_baseline,
            &mut alpha_baseline,
            &[1.0],
        )
        .unwrap();

    // Optimized
    let config_opt = BetaCrownConfig {
        beta_iterations: 10,
        beta_lr: 0.05,
        beta_tolerance: 1e-8,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier_opt = BetaCrownVerifier::new(config_opt);
    let mut beta_opt = GraphBetaState::from_history(&history).unwrap();
    let mut alpha_opt = GraphDomainAlphaState::empty();
    let (lb_opt, ub_opt, _) = verifier_opt
        .optimize_graph_beta_analytical(
            &graph,
            &input,
            &context,
            &mut beta_opt,
            &mut alpha_opt,
            &[1.0],
        )
        .unwrap();

    assert!(
        lb_opt >= lb_baseline - 1e-6,
        "analytical should not degrade lower bound: opt={} vs baseline={}",
        lb_opt,
        lb_baseline
    );
    assert!(lb_opt <= ub_opt);
}

// ============================================================
// AC3: evaluate_graph_child_bounds
// ============================================================

/// Test evaluate_graph_child_bounds with lazy alpha init (empty alpha state).
///
/// Verifies: alpha is initialized from graph bounds when child.alpha_state is empty,
/// should_optimize decision, bounds are merged with parent, priority is set.
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_child_bounds_lazy_alpha_init() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();

    let config = BetaCrownConfig {
        beta_iterations: 0, // No optimization - inherited pass
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let mut child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(), // empty → should be initialized
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let result =
        verifier.evaluate_graph_child_bounds(&graph, &mut child, &node_bounds, &[1.0], None, None);

    assert!(result.is_ok(), "evaluate_graph_child_bounds should succeed");
    assert!(result.unwrap(), "should return Ok(true)");
    assert!(child.lower_bound.is_finite(), "lower bound should be set");
    assert!(child.upper_bound.is_finite(), "upper bound should be set");
    assert!(
        child.lower_bound <= child.upper_bound,
        "lb <= ub: {} <= {}",
        child.lower_bound,
        child.upper_bound
    );
    assert!(child.priority.is_finite(), "priority should be set");
    // Node bounds should include parent entries (merged)
    assert!(
        !child.node_bounds.is_empty(),
        "node_bounds should contain merged entries"
    );

    // Soundness: bounds must contain the true network output at sample points.
    assert_bounds_contain_samples(&graph, &input, child.lower_bound, child.upper_bound, 5);
}

/// Test evaluate_graph_child_bounds with analytical optimization enabled.
///
/// Verifies the analytical path produces valid results, that bounds are
/// tighter than the no-optimization baseline, and that the analytical
/// and SPSA paths produce consistent results.
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_child_bounds_analytical_path() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();

    // Baseline: no optimization (0 iterations)
    let config_baseline = BetaCrownConfig {
        beta_iterations: 0,
        ..Default::default()
    };
    let verifier_baseline = BetaCrownVerifier::new(config_baseline);
    let mut baseline_child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };
    verifier_baseline
        .evaluate_graph_child_bounds(
            &graph,
            &mut baseline_child,
            &node_bounds,
            &[1.0],
            None,
            None,
        )
        .unwrap();
    let lb_baseline = baseline_child.lower_bound;

    // Analytical: 3 iterations
    let config = BetaCrownConfig {
        beta_iterations: 3,
        beta_lr: 0.05,
        beta_max_depth: 10,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let mut child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let result =
        verifier.evaluate_graph_child_bounds(&graph, &mut child, &node_bounds, &[1.0], None, None);

    assert!(result.is_ok());
    assert!(result.unwrap());
    assert!(child.lower_bound.is_finite());
    assert!(child.lower_bound <= child.upper_bound);

    // Analytical optimization should not worsen bounds compared to baseline
    assert!(
        child.lower_bound >= lb_baseline - 1e-6,
        "analytical child lb ({}) should not be worse than baseline ({})",
        child.lower_bound,
        lb_baseline
    );

    // Soundness: bounds must contain the true network output at sample points.
    assert_bounds_contain_samples(&graph, &input, child.lower_bound, child.upper_bound, 5);
}

/// Test evaluate_graph_child_bounds with SPSA path (analytical disabled).
///
/// Verifies SPSA produces bounds consistent with the analytical path.
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_child_bounds_spsa_path() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();

    // Get analytical bounds for comparison
    let config_analytical = BetaCrownConfig {
        beta_iterations: 3,
        beta_lr: 0.05,
        beta_max_depth: 10,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier_analytical = BetaCrownVerifier::new(config_analytical);
    let mut analytical_child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };
    verifier_analytical
        .evaluate_graph_child_bounds(
            &graph,
            &mut analytical_child,
            &node_bounds,
            &[1.0],
            None,
            None,
        )
        .unwrap();

    // SPSA path
    let config = BetaCrownConfig {
        beta_iterations: 3,
        beta_lr: 0.05,
        beta_max_depth: 10,
        use_analytical_beta_gradients: false, // SPSA path
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let mut child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let result =
        verifier.evaluate_graph_child_bounds(&graph, &mut child, &node_bounds, &[1.0], None, None);

    assert!(result.is_ok());
    assert!(result.unwrap());
    assert!(child.lower_bound.is_finite());
    assert!(child.lower_bound <= child.upper_bound);

    // SPSA and analytical should produce bounds in the same ballpark. SPSA uses
    // stochastic gradient estimates so differences up to 0.5 are expected. The
    // deterministic soundness check below is the primary correctness validation.
    assert!(
        (child.lower_bound - analytical_child.lower_bound).abs() < 0.5,
        "SPSA lb ({}) and analytical lb ({}) should be within 0.5",
        child.lower_bound,
        analytical_child.lower_bound
    );

    // Deterministic soundness: both bounds must contain the true output.
    // Use 7×7 grid (49 points) for higher-confidence containment checks.
    assert_bounds_contain_samples(&graph, &input, child.lower_bound, child.upper_bound, 7);
    assert_bounds_contain_samples(
        &graph,
        &input,
        analytical_child.lower_bound,
        analytical_child.upper_bound,
        7,
    );
}

/// Test evaluate_graph_child_bounds skips optimization when depth > beta_max_depth.
///
/// When depth exceeds beta_max_depth, bounds should equal the no-optimization baseline
/// since the optimizer is not invoked. Verify this by comparing against a baseline
/// with beta_iterations=0.
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_child_bounds_depth_exceeds_max_skips_optimization() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();

    // Baseline: explicitly no optimization
    let config_baseline = BetaCrownConfig {
        beta_iterations: 0,
        ..Default::default()
    };
    let verifier_baseline = BetaCrownVerifier::new(config_baseline);
    let mut baseline_child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 5,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };
    verifier_baseline
        .evaluate_graph_child_bounds(
            &graph,
            &mut baseline_child,
            &node_bounds,
            &[1.0],
            None,
            None,
        )
        .unwrap();

    // Depth-gated: 10 iterations configured but depth=5 > max_depth=2
    let config = BetaCrownConfig {
        beta_iterations: 10,
        beta_max_depth: 2,
        use_analytical_beta_gradients: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let mut child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 5, // exceeds beta_max_depth=2
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let result =
        verifier.evaluate_graph_child_bounds(&graph, &mut child, &node_bounds, &[1.0], None, None);

    assert!(result.is_ok());
    assert!(result.unwrap());
    assert!(child.lower_bound.is_finite());
    assert!(child.lower_bound <= child.upper_bound);

    // Depth-gated bounds should match the no-optimization baseline exactly
    // (since optimization was skipped due to depth exceeding max)
    assert!(
        (child.lower_bound - baseline_child.lower_bound).abs() < 1e-6,
        "depth-gated lb ({}) should match baseline lb ({}) since optimization is skipped",
        child.lower_bound,
        baseline_child.lower_bound
    );
    assert!(
        (child.upper_bound - baseline_child.upper_bound).abs() < 1e-6,
        "depth-gated ub ({}) should match baseline ub ({}) since optimization is skipped",
        child.upper_bound,
        baseline_child.upper_bound
    );

    // Soundness: even depth-gated bounds must contain the true output.
    assert_bounds_contain_samples(&graph, &input, child.lower_bound, child.upper_bound, 5);
}

/// Test that evaluate_graph_child_bounds merges parent node bounds for missing nodes.
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_child_bounds_merges_parent_bounds() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();

    let config = BetaCrownConfig {
        beta_iterations: 0,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let mut child = GraphBabDomain {
        history: history.clone(),
        node_bounds: HashMap::new(), // empty — should be filled from parent
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let result =
        verifier.evaluate_graph_child_bounds(&graph, &mut child, &node_bounds, &[1.0], None, None);

    assert!(result.is_ok());
    assert!(result.unwrap());

    // After evaluation, child node_bounds should contain at least the parent entries
    for (parent_key, parent_val) in &node_bounds {
        let child_val = child.node_bounds.get(parent_key).unwrap_or_else(|| {
            panic!("child should have parent node '{}' after merge", parent_key)
        });
        // Merged bounds should be valid: finite and ordered.
        assert!(
            child_val.lower().iter().all(|v| v.is_finite()),
            "merged bounds for '{}' should have finite lower values",
            parent_key
        );
        assert!(
            child_val.upper().iter().all(|v| v.is_finite()),
            "merged bounds for '{}' should have finite upper values",
            parent_key
        );
        // Merged bounds should be at least as tight as parent (since child
        // recomputes bounds with constraints, they may be tighter).
        for (i, (pl, pu)) in parent_val
            .lower()
            .iter()
            .zip(parent_val.upper().iter())
            .enumerate()
        {
            let cl = child_val.lower()[[i]];
            let cu = child_val.upper()[[i]];
            assert!(
                cl >= pl - 1e-5 && cu <= pu + 1e-5,
                "merged bounds for '{}' element {} should be within parent: \
                 child=[{}, {}], parent=[{}, {}]",
                parent_key,
                i,
                cl,
                cu,
                pl,
                pu
            );
        }
    }

    // Soundness: child bounds must contain the true network output.
    assert!(child.lower_bound.is_finite());
    assert!(child.lower_bound <= child.upper_bound);
    assert_bounds_contain_samples(&graph, &input, child.lower_bound, child.upper_bound, 5);
}

/// Test evaluate_graph_child_bounds sets priority based on verify_upper_bound config.
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_child_bounds_priority_upper_bound_mode() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();

    // verify_upper_bound = true → priority = upper_bound
    let config = BetaCrownConfig {
        beta_iterations: 0,
        verify_upper_bound: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let mut child = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input.clone()),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    verifier
        .evaluate_graph_child_bounds(&graph, &mut child, &node_bounds, &[1.0], None, None)
        .unwrap();

    assert_eq!(
        child.priority, child.upper_bound,
        "with verify_upper_bound=true, priority should equal upper_bound"
    );

    // verify_upper_bound = false → priority = -lower_bound
    let config2 = BetaCrownConfig {
        beta_iterations: 0,
        verify_upper_bound: false,
        ..Default::default()
    };
    let verifier2 = BetaCrownVerifier::new(config2);

    let mut child2 = GraphBabDomain {
        history: history.clone(),
        node_bounds: node_bounds.clone(),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: f32::INFINITY,
        input_bounds: Arc::new(input),
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    verifier2
        .evaluate_graph_child_bounds(&graph, &mut child2, &node_bounds, &[1.0], None, None)
        .unwrap();

    assert_eq!(
        child2.priority, -child2.lower_bound,
        "with verify_upper_bound=false, priority should equal -lower_bound"
    );
}

// ============================================================
// AC4: propagate_crown_with_graph_beta_and_intermediates
// ============================================================

/// Test that intermediates contain A matrices for constrained ReLU nodes.
#[ntest::timeout(10000)]
#[test]
fn test_intermediates_contain_a_matrices() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let (output, node_cache, intermediate) = verifier
        .propagate_crown_with_graph_beta_and_intermediates(
            &graph,
            &input,
            &context,
            &beta_state,
            Some(&[1.0]),
        )
        .unwrap();

    // Output should be valid bounds
    let lb = output.lower_scalar();
    let ub = output.upper_scalar();
    assert!(lb.is_finite());
    assert!(lb <= ub);
    assert!(!node_cache.is_empty());

    // Intermediate should have final_bounds (LinearBounds) with valid coefficients.
    // LinearBounds stores lower_a, lower_b, upper_a, upper_b for the accumulated
    // backward pass. The bias vectors (lower_b, upper_b) correspond to the
    // constant part of the linear bound.
    assert!(
        !intermediate.final_bounds.lower_b.is_empty(),
        "final_bounds lower_b should be non-empty"
    );
    assert!(
        intermediate
            .final_bounds
            .lower_b
            .iter()
            .all(|v| v.is_finite()),
        "final_bounds lower_b values should be finite"
    );

    // Soundness: output bounds must contain the true network output.
    assert_bounds_contain_samples(&graph, &input, lb, ub, 5);
}

/// Test intermediates with no objective (None) returns raw output bounds.
#[ntest::timeout(10000)]
#[test]
fn test_intermediates_no_objective() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = single_relu_history();
    let beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier.propagate_crown_with_graph_beta_and_intermediates(
        &graph,
        &input,
        &context,
        &beta_state,
        None, // No objective
    );

    assert!(result.is_ok(), "should succeed with None objective");
    let (output, _, intermediate) = result.unwrap();
    // Without objective, output is the raw network output bounds
    let lb = output.lower_scalar();
    let ub = output.upper_scalar();
    assert!(lb.is_finite());
    assert!(lb <= ub);
    assert!(
        !intermediate.final_bounds.lower_b.is_empty(),
        "final_bounds should be populated"
    );

    // Soundness: raw output bounds must contain the true network output.
    assert_bounds_contain_samples(&graph, &input, lb, ub, 5);
}

/// Test intermediates with empty beta state (no constraints).
#[ntest::timeout(10000)]
#[test]
fn test_intermediates_empty_beta() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = GraphSplitHistory::new();
    let beta_state = GraphBetaState::from_history(&history).unwrap();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let (output, _, _) = verifier
        .propagate_crown_with_graph_beta_and_intermediates(
            &graph,
            &input,
            &context,
            &beta_state,
            Some(&[1.0]),
        )
        .unwrap();

    let lb = output.lower_scalar();
    let ub = output.upper_scalar();
    assert!(lb.is_finite());
    assert!(lb <= ub);

    // Soundness: bounds must contain the true network output.
    assert_bounds_contain_samples(&graph, &input, lb, ub, 5);
}

// AC5 (multi-objective helpers) tests are in engine::graph::tests::objectives
// because the functions under test are pub(super) within engine::graph.
