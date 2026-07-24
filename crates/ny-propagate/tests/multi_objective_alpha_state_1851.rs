// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{arr1, arr2};
use ny_core::Result;
use ny_propagate::beta_crown::{
    GraphDomainAlphaState, GraphNeuronConstraint, GraphSplitHistory, MultiObjectiveGraphBabDomain,
};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{
    BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_child_inherits_parent_alpha_state_1851() -> Result<()> {
    // Simple graph: input -> ReLU
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["_input".to_string()],
    ));
    graph.set_output("relu0");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.8_f32]).into_dyn(),
        arr1(&[1.0_f32, 0.9_f32]).into_dyn(),
    )
    .expect("valid input bounds");

    let mut root =
        MultiObjectiveGraphBabDomain::root(HashMap::new(), vec![(0.0, 1.0)], &input, &[0.0], false)
            .unwrap();

    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    root.set_alpha_state(GraphDomainAlphaState::from_graph_bounds(
        &graph,
        &node_bounds,
        root.history(),
        &input,
    ));

    // Force a non-heuristic value to verify parent->child warm-start.
    let parent_alpha = root
        .alpha_state_mut()
        .neuron_mut("relu0", 0)
        .expect("root alpha state should contain neuron 0");
    parent_alpha.set_alpha(0.37);

    // Split a different neuron so neuron 0 remains unconstrained in child.
    let child = root
        .with_constraint(
            &graph,
            GraphNeuronConstraint::new("relu0".to_string(), 1, true, 0.0)?,
            false,
            &[0.0],
        )?
        .expect("active branch should be feasible");

    let child_alpha = child
        .alpha_state()
        .neuron("relu0", 0)
        .expect("child should inherit alpha for unconstrained neuron 0")
        .alpha();
    assert!(
        (child_alpha - 0.37).abs() < 1e-6,
        "expected inherited alpha=0.37, got {}",
        child_alpha
    );

    // The split neuron is constrained in the child and should not be optimizable.
    assert!(
        child.alpha_state().neuron("relu0", 1).is_none(),
        "constrained neuron should not have optimizable alpha entry"
    );

    // Inactive branch should preserve the same alpha invariants.
    let inactive_child = root
        .with_constraint(
            &graph,
            GraphNeuronConstraint::new("relu0".to_string(), 1, false, 0.0)?,
            false,
            &[0.0],
        )?
        .expect("inactive branch should be feasible");

    let inactive_alpha = inactive_child
        .alpha_state()
        .neuron("relu0", 0)
        .expect("inactive child should inherit alpha for unconstrained neuron 0")
        .alpha();
    assert!(
        (inactive_alpha - 0.37).abs() < 1e-6,
        "expected inherited alpha=0.37 in inactive child, got {}",
        inactive_alpha
    );
    assert!(
        inactive_child.alpha_state().neuron("relu0", 1).is_none(),
        "inactive constrained neuron should not have optimizable alpha entry"
    );

    Ok(())
}

#[ntest::timeout(60000)]
#[test]
fn test_multi_objective_optimized_root_alpha_vs_heuristic_benchmark_1851() {
    // Graph: 2 -> 6 -> ReLU -> 4 -> ReLU -> 2
    // This network has enough unstable ReLUs for alpha initialization differences
    // to affect branch-and-bound behavior in multi-objective mode.
    let w1 = arr2(&[
        [1.2, -0.9],
        [-1.1, 0.8],
        [0.7, 0.6],
        [-0.5, 1.0],
        [0.9, 0.4],
        [-0.8, -0.7],
    ]);
    let b1 = arr1(&[0.1, -0.1, 0.0, 0.05, -0.05, 0.02]);
    let w2 = arr2(&[
        [0.6, -0.4, 0.8, 0.2, -0.3, 0.5],
        [-0.7, 0.9, -0.2, 0.6, 0.4, -0.5],
        [0.3, 0.5, -0.6, 0.7, -0.8, 0.2],
        [-0.4, -0.3, 0.7, -0.6, 0.5, 0.9],
    ]);
    let b2 = arr1(&[0.0, 0.1, -0.1, 0.05]);
    let w3 = arr2(&[[1.0, -0.6, 0.4, 0.8], [-0.5, 0.9, -0.7, 0.3]]);
    let b3 = arr1(&[0.05, -0.02]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("linear1 should build")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("linear2 should build")),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).expect("linear3 should build")),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input bounds");

    // Two objectives over the 2-output head.
    let objectives = vec![vec![1.0_f32, -0.4_f32], vec![-0.3_f32, 1.0_f32]];

    // Estimate objective minima from a point grid so thresholds are tight enough
    // to exercise BaB, while still typically satisfiable.
    const GRID: usize = 41;
    let mut sampled_mins = vec![f32::INFINITY; objectives.len()];
    for i in 0..GRID {
        let x0 = -1.0 + 2.0 * (i as f32 / (GRID - 1) as f32);
        for j in 0..GRID {
            let x1 = -1.0 + 2.0 * (j as f32 / (GRID - 1) as f32);
            let point = BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn())
                .expect("point bounds should build");
            let out = graph
                .propagate_ibp(&point)
                .expect("point propagation should succeed")
                .flatten();
            for (obj_idx, obj) in objectives.iter().enumerate() {
                let mut value = 0.0_f32;
                for (k, coeff) in obj.iter().enumerate() {
                    value += coeff * out.lower()[[k]];
                }
                sampled_mins[obj_idx] = sampled_mins[obj_idx].min(value);
            }
        }
    }
    let thresholds: Vec<f32> = sampled_mins.iter().map(|m| m - 0.02).collect();

    let mut base_config = BetaCrownConfig {
        verify_upper_bound: false,
        timeout: Duration::from_secs(8),
        max_domains: 2_048,
        max_depth: 20,
        batch_size: 32,
        beta_iterations: 5,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        enable_cuts: false,
        ..Default::default()
    };
    base_config.alpha_config.iterations = 12;

    // Assert root α-CROWN state actually differs from heuristic initialization.
    let (initial_bounds, root_alpha) = graph
        .collect_alpha_crown_bounds_dag(&input, &base_config.alpha_config)
        .expect("alpha-crown bounds should succeed");
    let initial_bounds_arc: HashMap<String, Arc<BoundedTensor>> = initial_bounds
        .iter()
        .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
        .collect();
    let root_history = GraphSplitHistory::new();

    let heuristic_alpha = GraphDomainAlphaState::from_graph_bounds(
        &graph,
        &initial_bounds_arc,
        &root_history,
        &input,
    );
    let optimized_alpha = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &initial_bounds_arc,
        &root_history,
        &input,
    );

    let mut max_alpha_delta = 0.0_f32;
    for (node_name, h_map) in heuristic_alpha.neurons() {
        if let Some(o_map) = optimized_alpha.neurons().get(node_name) {
            for (&neuron_idx, h) in h_map {
                if let Some(o) = o_map.get(&neuron_idx) {
                    max_alpha_delta = max_alpha_delta.max((h.alpha() - o.alpha()).abs());
                }
            }
        }
    }
    assert!(
        max_alpha_delta > 1e-4,
        "expected optimized root alpha to differ from heuristic initialization"
    );

    // Run A: heuristic root alpha initialization.
    let mut heuristic_config = base_config.clone();
    heuristic_config.use_alpha_crown = false;
    let heuristic_result = BetaCrownVerifier::new(heuristic_config)
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .expect("heuristic multi-objective verification should complete");

    // Run B: optimized root alpha initialization from α-CROWN.
    let mut optimized_config = base_config;
    optimized_config.use_alpha_crown = true;
    let optimized_result = BetaCrownVerifier::new(optimized_config)
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .expect("optimized multi-objective verification should complete");

    eprintln!(
        "\n=== #1851 multi-objective root-alpha benchmark ===\n\
         sampled_mins={:?}\n\
         thresholds={:?}\n\
         max_alpha_delta={:.6}\n\
         heuristic: result={:?}, explored={}, verified={}\n\
         optimized: result={:?}, explored={}, verified={}",
        sampled_mins,
        thresholds,
        max_alpha_delta,
        heuristic_result.result,
        heuristic_result.domains_explored,
        heuristic_result.domains_verified,
        optimized_result.result,
        optimized_result.domains_explored,
        optimized_result.domains_verified,
    );

    assert!(
        heuristic_result.domains_explored > 0 && optimized_result.domains_explored > 0,
        "both benchmark runs must explore at least one domain"
    );
    assert!(
        heuristic_result.domains_explored != optimized_result.domains_explored
            || heuristic_result.domains_verified != optimized_result.domains_verified
            || std::mem::discriminant(&heuristic_result.result)
                != std::mem::discriminant(&optimized_result.result),
        "expected optimized-root-alpha run to produce observably different end-to-end \
         behavior from heuristic-root-alpha run. \
         heuristic={:?} (explored={}, verified={}), \
         optimized={:?} (explored={}, verified={})",
        heuristic_result.result,
        heuristic_result.domains_explored,
        heuristic_result.domains_verified,
        optimized_result.result,
        optimized_result.domains_explored,
        optimized_result.domains_verified,
    );
}

/// Algorithm audit: verify alpha propagation through depth-2 BaB tree.
///
/// Graph: _input(3 neurons) -> ReLU(3) with 3 unstable neurons.
/// Split neuron 0 at depth 0 -> child. Split neuron 1 at depth 1 -> grandchild.
/// Neuron 2 (never split) should retain the root's custom alpha at depth 2.
///
/// This tests the invariant that `from_parent` warm-start correctly chains
/// through multiple generations: root alpha -> child alpha -> grandchild alpha
/// for unconstrained neurons.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_alpha_depth_2_chain() -> Result<()> {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["_input".to_string()],
    ));
    graph.set_output("relu0");

    // 3 unstable neurons
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.8, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.9, 0.6]).into_dyn(),
    )
    .expect("valid input bounds");

    let mut root =
        MultiObjectiveGraphBabDomain::root(HashMap::new(), vec![(0.0, 1.0)], &input, &[0.0], false)
            .unwrap();

    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    root.set_alpha_state(GraphDomainAlphaState::from_graph_bounds(
        &graph,
        &node_bounds,
        root.history(),
        &input,
    ));

    // Set distinctive alpha values for all 3 neurons.
    root.alpha_state_mut()
        .neuron_mut("relu0", 0)
        .unwrap()
        .set_alpha(0.25);
    root.alpha_state_mut()
        .neuron_mut("relu0", 1)
        .unwrap()
        .set_alpha(0.60);
    root.alpha_state_mut()
        .neuron_mut("relu0", 2)
        .unwrap()
        .set_alpha(0.88);

    // Depth 1: split neuron 0 (active branch)
    let child = root
        .with_constraint(
            &graph,
            GraphNeuronConstraint::new("relu0".to_string(), 0, true, 0.0)?,
            false,
            &[0.0],
        )?
        .expect("depth-1 split should succeed");

    // Child should have alpha for neurons 1 and 2 (not neuron 0, which is now constrained)
    assert!(
        child.alpha_state().neuron("relu0", 0).is_none(),
        "neuron 0 is constrained at depth 1, should not be in alpha state"
    );
    let child_alpha_1 = child
        .alpha_state()
        .neuron("relu0", 1)
        .expect("neuron 1 should be in child alpha state")
        .alpha();
    assert!(
        (child_alpha_1 - 0.60).abs() < 1e-6,
        "child neuron 1 alpha should be 0.60 from parent, got {child_alpha_1}"
    );
    let child_alpha_2 = child
        .alpha_state()
        .neuron("relu0", 2)
        .expect("neuron 2 should be in child alpha state")
        .alpha();
    assert!(
        (child_alpha_2 - 0.88).abs() < 1e-6,
        "child neuron 2 alpha should be 0.88 from parent, got {child_alpha_2}"
    );

    // Depth 2: split neuron 1 (inactive branch)
    let grandchild = child
        .with_constraint(
            &graph,
            GraphNeuronConstraint::new("relu0".to_string(), 1, false, 0.0)?,
            false,
            &[0.0],
        )?
        .expect("depth-2 split should succeed");

    // Grandchild should only have alpha for neuron 2 (neurons 0 and 1 are constrained)
    assert!(
        grandchild.alpha_state().neuron("relu0", 0).is_none(),
        "neuron 0 constrained at depth 1, must not be in grandchild alpha state"
    );
    assert!(
        grandchild.alpha_state().neuron("relu0", 1).is_none(),
        "neuron 1 constrained at depth 2, must not be in grandchild alpha state"
    );
    let grandchild_alpha_2 = grandchild
        .alpha_state()
        .neuron("relu0", 2)
        .expect("neuron 2 should be in grandchild alpha state")
        .alpha();
    assert!(
        (grandchild_alpha_2 - 0.88).abs() < 1e-6,
        "grandchild neuron 2 alpha should be 0.88 from root (through 2 generations), got {grandchild_alpha_2}"
    );
    assert_eq!(grandchild.depth(), 2, "grandchild should be at depth 2");

    Ok(())
}
