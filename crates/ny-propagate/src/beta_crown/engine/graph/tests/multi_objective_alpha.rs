// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective alpha-state regression tests.
//!
//! Regression for #1851: when a multi-objective graph domain branches on a
//! neuron, child domains must keep optimized alpha for unconstrained neurons
//! and drop the newly constrained neuron from alpha optimization state.

use std::collections::HashMap;
use std::time::Duration;

use ndarray::{arr1, arr2};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

fn setup_multi_objective_domain_with_alpha() -> Result<(GraphNetwork, MultiObjectiveGraphBabDomain)>
{
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn())?;

    let mut domain = MultiObjectiveGraphBabDomain::root(
        HashMap::new(),
        vec![(-1.0, 1.0), (-1.5, 0.7)],
        &input_bounds,
        &[0.0, 0.0],
        false,
    )?;

    let mut alpha_state = GraphDomainAlphaState::empty();
    alpha_state.insert("relu".to_string(), 0, AlphaNeuronState::new(0.21));
    alpha_state.insert("relu".to_string(), 1, AlphaNeuronState::new(0.79));
    domain.alpha_state = alpha_state;

    Ok((graph, domain))
}

fn setup_multi_objective_benchmark_graph() -> GraphNetwork {
    let w1 = arr2(&[[1.2, -0.8], [-0.6, 1.1], [0.9, 0.7], [-0.7, 0.4]]);
    let b1 = arr1(&[0.1, -0.05, 0.0, 0.12]);
    let w2 = arr2(&[[0.8, -0.5, 0.6, -0.2], [-0.3, 0.9, -0.4, 0.7]]);
    let b2 = arr1(&[0.05, -0.08]);
    let w3 = arr2(&[[1.0, -0.2], [-0.4, 0.9]]);
    let b3 = arr1(&[0.02, -0.03]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("valid linear2")),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).expect("valid linear3")),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");
    graph
}

fn sample_output_minima(graph: &GraphNetwork, grid: usize) -> Result<[f32; 2]> {
    let mut mins = [f32::INFINITY, f32::INFINITY];
    for i in 0..grid {
        let x0 = -1.0 + 2.0 * (i as f32 / (grid - 1) as f32);
        for j in 0..grid {
            let x1 = -1.0 + 2.0 * (j as f32 / (grid - 1) as f32);
            let point = BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn())?;
            let output = graph.propagate_ibp(&point)?.flatten();
            mins[0] = mins[0].min(output.lower()[[0]]);
            mins[1] = mins[1].min(output.lower()[[1]]);
        }
    }
    Ok(mins)
}

fn assert_flat_bounds_close(label: &str, baseline: &BoundedTensor, captured: &BoundedTensor) {
    let baseline = baseline.flatten();
    let captured = captured.flatten();
    assert_eq!(
        baseline.lower().shape(),
        captured.lower().shape(),
        "{label}: lower-bound shape changed"
    );
    assert_eq!(
        baseline.upper().shape(),
        captured.upper().shape(),
        "{label}: upper-bound shape changed"
    );

    for (idx, (old, new)) in baseline
        .lower()
        .iter()
        .zip(captured.lower().iter())
        .enumerate()
    {
        assert!(
            (*old - *new).abs() <= 1e-6,
            "{label}: lower bound changed at index {idx}: baseline={old}, captured={new}"
        );
    }

    for (idx, (old, new)) in baseline
        .upper()
        .iter()
        .zip(captured.upper().iter())
        .enumerate()
    {
        assert!(
            (*old - *new).abs() <= 1e-6,
            "{label}: upper bound changed at index {idx}: baseline={old}, captured={new}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_root_linear_capture_preserves_spec_bounds_3813() -> Result<()> {
    let graph = setup_multi_objective_benchmark_graph();
    let input = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())?;
    let node_bounds = graph.collect_node_bounds(&input)?;
    let spec_matrix = arr2(&[[1.0_f32, 0.0], [0.0, 1.0_f32]]);

    let baseline = graph.propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
        &input,
        &spec_matrix,
        None,
        &node_bounds,
        None,
    )?;
    let (captured, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
            None,
        )?;

    assert!(
        linear.is_some(),
        "root spec-guided CROWN should capture linear coefficients for warm-start"
    );

    assert_flat_bounds_close("linear-capture path", &baseline, &captured);

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_multi_objective_child_active_preserves_unconstrained_alpha_1851() -> Result<()> {
    let (graph, domain) = setup_multi_objective_domain_with_alpha()?;
    let thresholds = [0.0, 0.0];

    let child = domain
        .with_constraint(
            &graph,
            GraphNeuronConstraint {
                node_name: "relu".to_string(),
                neuron_idx: 0,
                is_active: true,
                score: 0.5,
            },
            false,
            &thresholds,
        )?
        .expect("active branch should be feasible");

    assert!(
        (child.alpha_state.alpha("relu", 1) - 0.79).abs() < 1e-6,
        "unconstrained neuron alpha should be warm-started from parent"
    );
    assert!(
        child.alpha_state.neuron("relu", 0).is_none(),
        "branched active neuron should be constrained and removed from alpha state"
    );
    assert_eq!(
        child.history.is_constrained("relu", 0),
        Some(true),
        "history should record active constraint for branched neuron"
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_multi_objective_child_inactive_preserves_unconstrained_alpha_1851() -> Result<()> {
    let (graph, domain) = setup_multi_objective_domain_with_alpha()?;
    let thresholds = [0.0, 0.0];

    let child = domain
        .with_constraint(
            &graph,
            GraphNeuronConstraint {
                node_name: "relu".to_string(),
                neuron_idx: 0,
                is_active: false,
                score: 0.5,
            },
            false,
            &thresholds,
        )?
        .expect("inactive branch should be feasible");

    assert!(
        (child.alpha_state.alpha("relu", 1) - 0.79).abs() < 1e-6,
        "unconstrained neuron alpha should be warm-started from parent"
    );
    assert!(
        child.alpha_state.neuron("relu", 0).is_none(),
        "branched inactive neuron should be constrained and removed from alpha state"
    );
    assert_eq!(
        child.history.is_constrained("relu", 0),
        Some(false),
        "history should record inactive constraint for branched neuron"
    );

    Ok(())
}

#[ntest::timeout(60000)]
#[test]
fn test_multi_objective_root_alpha_ab_benchmark_1851() -> Result<()> {
    let graph = setup_multi_objective_benchmark_graph();
    let input = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())?;
    let sampled_min = sample_output_minima(&graph, 41)?;

    // Keep thresholds close to sampled minima so heuristic root alpha usually
    // needs branching while optimized root alpha can tighten faster.
    let thresholds = vec![sampled_min[0] - 0.01, sampled_min[1] - 0.01];
    let objectives = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let base_config = BetaCrownConfig {
        verify_upper_bound: false,
        timeout: Duration::from_secs(10),
        max_domains: 2_000,
        max_depth: 20,
        batch_size: 16,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        beta_iterations: 0,
        root_beta_iterations: 0,
        enable_cuts: false,
        ..Default::default()
    };

    let mut optimized_config = base_config.clone();
    optimized_config.use_alpha_crown = true;
    optimized_config.alpha_config.iterations = 30;

    let mut heuristic_config = base_config;
    heuristic_config.use_alpha_crown = false;
    // Both configs use the same fix_interm_bounds (default=true, i.e. IBP intermediates)
    // so the A/B comparison isolates the effect of alpha optimization only.
    // Previously, heuristic used fix_interm_bounds=false (CROWN-IBP intermediates),
    // giving it tighter intermediate bounds and an unfair advantage — causing the
    // optimized path to sometimes verify fewer domains despite better output bounds.

    let optimized = BetaCrownVerifier::new(optimized_config)
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)?;
    let heuristic = BetaCrownVerifier::new(heuristic_config)
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)?;

    eprintln!(
        "\n=== #1851 multi-objective root-alpha A/B ===\n\
         optimized-root-alpha: {:?}, explored={}, verified={}\n\
         heuristic-root-alpha: {:?}, explored={}, verified={}",
        optimized.result,
        optimized.domains_explored,
        optimized.domains_verified,
        heuristic.result,
        heuristic.domains_explored,
        heuristic.domains_verified
    );

    assert!(
        heuristic.domains_explored >= 2,
        "heuristic-root-alpha baseline should exercise BaB branching (explored={})",
        heuristic.domains_explored
    );

    // Regression for #1975: alpha-optimized intermediate bounds must be
    // forwarded to the spec-guided CROWN step (via `bootstrap.initial_node_bounds`
    // in compute_root_objective_bounds), not discarded. With the forwarding in
    // place the optimized path obtains the same intermediate bounds as the
    // heuristic PLUS root alpha output optimization.
    //
    // Reference: alpha-beta-CROWN optimized_bounds.py:610 passes
    // interm_bounds=interm_bounds when fix_interm_bounds=True.
    //
    // Invariants we can soundly assert here:
    //
    // 1. Soundness: a tighter root bound must never turn an unverifiable
    //    property into a (false) Verified — the optimized path must agree with
    //    the heuristic path's correct `Unknown`/safe verdict and never report a
    //    spurious success. `domains_verified` itself is a cut-generation
    //    accounting counter ("contributes to cut generation"), NOT a monotone
    //    quality metric: a tighter root bound changes the BoundImpact branching
    //    trajectory, so the two configs explore structurally different BaB trees
    //    (here optimized explores 43 leaves, heuristic 73) and the per-trajectory
    //    leaf-verification counts are not comparable. Asserting
    //    `optimized.domains_verified >= heuristic.domains_verified` is therefore
    //    an over-strong, non-invariant claim and was removed (classification C).
    //
    // 2. The optimized path must still flow the forwarded bounds into the
    //    spec-guided CROWN leaves and verify at least one subdomain (the #1975
    //    bug previously starved it of bounds and collapsed it toward zero).
    assert!(
        !matches!(optimized.result, BabVerificationStatus::Verified)
            || matches!(heuristic.result, BabVerificationStatus::Verified),
        "optimized-root-alpha must not report a spurious Verified verdict when the \
         heuristic path (identical intermediate bounds) cannot verify: \
         optimized={:?}, heuristic={:?}",
        optimized.result,
        heuristic.result,
    );
    assert!(
        optimized.domains_verified >= 1,
        "optimized-root-alpha verified zero domains, indicating alpha-optimized \
         intermediate bounds are not reaching the spec-guided CROWN leaves (#1975). \
         explored={}",
        optimized.domains_explored,
    );

    // Both paths should verify at least 1 domain on this small benchmark
    assert!(
        optimized.domains_verified >= 1,
        "optimized-root-alpha failed to verify any domains (explored={})",
        optimized.domains_explored,
    );

    Ok(())
}
