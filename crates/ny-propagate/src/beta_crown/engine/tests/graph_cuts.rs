// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;
use std::sync::atomic::Ordering;

fn build_two_relu_graph_with_bounds() -> (
    GraphNetwork,
    std::collections::HashMap<String, Arc<BoundedTensor>>,
) {
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::GraphNode;
    use ndarray::{arr1, arr2, Array1};
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut graph = GraphNetwork::new();

    let linear1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
        Some(arr1(&[0.1_f32, -0.1])),
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.3], [0.3, 1.0]]),
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();
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

    let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();

    let bounds_linear1 = BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -0.5]).into_dyn(),
        Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
    )
    .unwrap();
    node_bounds.insert("linear1".to_string(), Arc::new(bounds_linear1));

    let bounds_linear2 = BoundedTensor::new(
        Array1::from_vec(vec![-0.8, -0.3]).into_dyn(),
        Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
    )
    .unwrap();
    node_bounds.insert("linear2".to_string(), Arc::new(bounds_linear2));

    (graph, node_bounds)
}

// GCP-CROWN: Graph Cuts Tests
// =========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_from_history() {
    // Test creating a graph cutting plane from a verified domain's split history
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let cut = GraphCuttingPlane::from_verified_domain(&history).unwrap();
    assert!(cut.is_some());
    let cut = cut.unwrap();
    assert_eq!(cut.terms.len(), 2);
    assert_eq!(cut.terms[0].node_name, "relu1");
    assert_eq!(cut.terms[0].neuron_idx, 0);
    // Active -> positive
    assert_eq!(cut.terms[0].coefficient, 1.0);
    assert_eq!(cut.terms[1].node_name, "relu2");
    assert_eq!(cut.terms[1].neuron_idx, 1);
    // Inactive -> negative
    assert_eq!(cut.terms[1].coefficient, -1.0);
    // (1 active) - 1 = 0
    assert_eq!(cut.bias, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_rejects_genbab_constraints() {
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_genbab_constraint(
        crate::beta_crown::GenBabConstraint::new("gelu1".to_string(), 3, -0.25, true, 0.0).unwrap(),
    );

    let cut = GraphCuttingPlane::from_verified_domain(&history).unwrap();
    assert!(
        cut.is_none(),
        "Graph cuts must reject histories with GenBaB constraints"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_empty_history() {
    // Empty history should not produce a cut
    let history = GraphSplitHistory::new();
    let cut = GraphCuttingPlane::from_verified_domain(&history).unwrap();
    assert!(cut.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_branching_records_split_score() {
    let linear1 = LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.set_output("relu1");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let initial_bounds = graph.collect_node_bounds(&input_bounds).unwrap();
    let domain = GraphBabDomain::root(initial_bounds, -1.0, 1.0, &input_bounds, false).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let relu_nodes = vec!["relu1".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
    assert!(!unstable.is_empty(), "Expected unstable ReLU neurons");

    let (node_name, neuron_idx, score) = verifier
        .select_graph_branch(&graph, &domain, &unstable)
        .unwrap();
    assert!(score > 0.0, "Split score should be positive");

    let constraint = GraphNeuronConstraint {
        node_name,
        neuron_idx,
        is_active: true,
        score,
    };
    let child = domain
        .with_constraint(&graph, constraint, false)
        .unwrap()
        .expect("Expected feasible child domain");
    assert_eq!(child.history.constraints.len(), 1);
    assert!(
        (child.history.constraints[0].score - score).abs() < 1e-6,
        "Constraint score should be preserved in history"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_paths_skip_relu_with_empty_inputs_2098() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let mut graph = GraphNetwork::new();
    // Construct directly via struct literal to bypass the debug_assert! in
    // GraphNode::new (#2481) — this test intentionally creates a malformed
    // node with empty inputs to verify that cut computation skips it (#2098).
    graph.add_node(GraphNode {
        name: "relu_bad".to_string(),
        layer: Layer::ReLU(ReLULayer),
        inputs: vec![],
    });
    graph.set_output("relu_bad");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>> =
        std::collections::HashMap::new();

    let cut = GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: "relu_bad".to_string(),
            neuron_idx: 0,
            coefficient: -2.0,
        }],
        bias: 0.0,
        lambda: 1.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Proactive),
    };

    // If relu_bad incorrectly fell back to _input with unstable [-1, 1], this
    // term would contribute -2.0 (negative coeff picks z_max=1) and yield a
    // non-zero gradient.
    let mut cut_pool = GraphCutPool::new(8);
    assert!(cut_pool.add_cut(cut), "cut should be inserted");
    verifier.compute_graph_cut_gradients(&graph, &mut cut_pool, &node_bounds, &input_bounds);

    let grad = cut_pool.cuts[0].lambda_grad;
    assert!(
        grad.abs() < 1e-6,
        "malformed ReLU cut terms must be skipped in gradients, expected 0, got {grad}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_basic() {
    let mut pool = GraphCutPool::new(10);
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);

    // Add a cut
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let added = pool.add_from_verified_domain(&history).unwrap();
    assert!(added);
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.total_generated, 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_min_depth() {
    // Pool with min_depth=2 should reject single-constraint histories
    let mut pool = GraphCutPool::with_min_depth(10, 2);

    // Single constraint - should be rejected
    let mut history1 = GraphSplitHistory::new();
    history1.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let added1 = pool.add_from_verified_domain(&history1).unwrap();
    assert!(!added1);
    assert_eq!(pool.len(), 0);

    // Two constraints - should be accepted
    let mut history2 = GraphSplitHistory::new();
    history2.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history2.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    let added2 = pool.add_from_verified_domain(&history2).unwrap();
    assert!(added2);
    assert_eq!(pool.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_redundancy() {
    // Create a cut
    let mut history1 = GraphSplitHistory::new();
    history1.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history1.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    let cut = GraphCuttingPlane::from_verified_domain(&history1)
        .unwrap()
        .unwrap();

    // A domain with the same constraints should make the cut redundant
    assert!(cut.is_redundant_for(&history1));

    // A domain with different constraints should not make it redundant
    let mut history2 = GraphSplitHistory::new();
    history2.add_constraint(GraphNeuronConstraint {
        node_name: "relu3".to_string(),
        neuron_idx: 2,
        is_active: true,
        score: 0.0,
    });
    assert!(!cut.is_redundant_for(&history2));
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_relevant_cuts() {
    let mut pool = GraphCutPool::with_min_depth(10, 2);

    // Add a cut
    let mut history1 = GraphSplitHistory::new();
    history1.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history1.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    pool.add_from_verified_domain(&history1).unwrap();

    // For an unrelated domain, the cut should be relevant
    let relevant = pool.relevant_cuts_for(&GraphSplitHistory::new());
    assert_eq!(relevant.len(), 1);

    // For a domain with the same constraints, the cut should be redundant
    let relevant2 = pool.relevant_cuts_for(&history1);
    assert_eq!(relevant2.len(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_usage_metadata_updates() {
    let mut pool = GraphCutPool::with_min_depth(10, 2);

    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    assert!(pool.add_from_verified_domain(&history).unwrap());
    assert_eq!(pool.cuts.len(), 1);

    let cut = &pool.cuts[0];
    assert_eq!(cut.metadata.created_iter.load(Ordering::Relaxed), 0);
    assert_eq!(cut.metadata.cut_kind(), CutKind::Verified);
    assert_eq!(cut.metadata.use_count.load(Ordering::Relaxed), 0);

    let relevant = pool.relevant_cuts_for(&GraphSplitHistory::new());
    assert_eq!(relevant.len(), 1);
    assert_eq!(cut.metadata.last_used_iter.load(Ordering::Relaxed), 1);
    assert_eq!(cut.metadata.use_count.load(Ordering::Relaxed), 1);
    assert_eq!(cut.metadata.created_iter.load(Ordering::Relaxed), 0);

    let relevant = pool.relevant_cuts_for(&GraphSplitHistory::new());
    assert_eq!(relevant.len(), 1);
    assert_eq!(cut.metadata.last_used_iter.load(Ordering::Relaxed), 2);
    assert_eq!(cut.metadata.use_count.load(Ordering::Relaxed), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_adam_update() {
    let mut cut = GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 1.0,
        lambda: 0.0,
        lambda_grad: 1.0, // positive gradient
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // After several Adam updates with positive gradient, lambda should increase
    for t in 1..=10 {
        cut.lambda_grad = 1.0;
        cut.update_lambda_adam(0.1, 0.9, 0.999, 1e-8, t);
    }

    assert!(
        cut.lambda > 0.0,
        "Lambda should increase with positive gradient"
    );
    assert!(cut.lambda >= 0.0, "Lambda must be non-negative");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_proactive_fraction_cap() {
    let config = BetaCrownConfig {
        max_cuts: 4,
        cut_proactive_fraction: 0.25,
        ..Default::default()
    };

    let mut pool = GraphCutPool::from_config(&config);

    let make_proactive_cut = |node_name: &str, neuron_idx| GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: node_name.to_string(),
            neuron_idx,
            coefficient: 1.0,
        }],
        bias: 0.0,
        lambda: 0.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 0,
        metadata: CutMetadata::new(0, CutKind::Proactive),
    };

    assert!(pool.add_cut(make_proactive_cut("relu1", 1)));
    assert!(!pool.add_cut(make_proactive_cut("relu1", 2)));
    assert_eq!(pool.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_eviction_prefers_low_score() {
    let config = BetaCrownConfig {
        max_cuts: 2,
        cut_eviction_policy: CutEvictionPolicy::UtilityWeighted,
        cut_score_weights: CutScoreWeights {
            w_lambda: 0.0,
            w_recent: 0.0,
            w_usage: 1.0,
            w_contrib: 0.0,
            w_depth: 0.0,
            lambda_cap: 1.0,
            contrib_cap: 1.0,
            tau_iters: 1.0,
            verified_bonus: 0.0,
            near_miss_bonus: 0.0,
            proactive_bonus: 0.0,
        },
        ..Default::default()
    };

    let mut pool = GraphCutPool::from_config(&config);

    let make_cut = |node_name: &str, neuron_idx| GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: node_name.to_string(),
            neuron_idx,
            coefficient: 1.0,
        }],
        bias: 0.0,
        lambda: 0.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    assert!(pool.add_cut(make_cut("relu1", 1)));
    assert!(pool.add_cut(make_cut("relu1", 2)));

    pool.cuts[0].metadata.use_count.store(10, Ordering::Relaxed);
    pool.cuts[1].metadata.use_count.store(1, Ordering::Relaxed);

    assert!(pool.add_cut(make_cut("relu1", 3)));

    let indices: Vec<usize> = pool
        .cuts
        .iter()
        .map(|cut| cut.terms[0].neuron_idx)
        .collect();
    assert!(indices.contains(&1));
    assert!(indices.contains(&3));
    assert!(!indices.contains(&2));
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_pool_proactive_evicts_proactive_first() {
    let config = BetaCrownConfig {
        max_cuts: 2,
        cut_proactive_fraction: 0.5,
        cut_eviction_policy: CutEvictionPolicy::UtilityWeighted,
        cut_score_weights: CutScoreWeights {
            w_lambda: 0.0,
            w_recent: 0.0,
            w_usage: 1.0,
            w_contrib: 0.0,
            w_depth: 0.0,
            lambda_cap: 1.0,
            contrib_cap: 1.0,
            tau_iters: 1.0,
            verified_bonus: 0.0,
            near_miss_bonus: 0.0,
            proactive_bonus: 0.0,
        },
        ..Default::default()
    };

    let mut pool = GraphCutPool::from_config(&config);

    let make_cut = |node_name: &str, neuron_idx, kind| GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: node_name.to_string(),
            neuron_idx,
            coefficient: 1.0,
        }],
        bias: 0.0,
        lambda: 0.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, kind),
    };

    assert!(pool.add_cut(make_cut("relu1", 1, CutKind::Proactive)));
    assert!(pool.add_cut(make_cut("relu1", 2, CutKind::Verified)));

    assert!(pool.add_cut(make_cut("relu1", 3, CutKind::Proactive)));

    let mut proactive = Vec::new();
    let mut verified = Vec::new();
    for cut in &pool.cuts {
        match cut.metadata.cut_kind() {
            CutKind::Proactive => proactive.push(cut.terms[0].neuron_idx),
            CutKind::Verified => verified.push(cut.terms[0].neuron_idx),
            CutKind::NearMiss => {}
        }
    }

    assert_eq!(proactive.len(), 1);
    assert_eq!(verified.len(), 1);
    assert_eq!(verified[0], 2);
    assert_eq!(proactive[0], 3);
}

#[ntest::timeout(10000)]
#[test]
fn test_proactive_cuts_generation() {
    // Test that proactive cuts are generated correctly from a simple graph
    let (graph, node_bounds) = build_two_relu_graph_with_bounds();

    // Generate proactive cuts
    let mut cut_pool = GraphCutPool::new(100);
    let cuts_generated = cut_pool
        .generate_proactive_cuts(&graph, &node_bounds, 50)
        .unwrap();

    // Verify cuts were generated
    assert!(
        cuts_generated > 0,
        "Should generate proactive cuts for unstable neurons"
    );
    assert_eq!(
        cut_pool.len(),
        cuts_generated,
        "Cut pool length should match cuts generated"
    );

    // Verify cut properties
    for cut in &cut_pool.cuts {
        // All proactive cuts have source_depth 0
        assert_eq!(
            cut.source_depth, 0,
            "Proactive cuts should have source_depth 0"
        );
        // All cuts should have small initial lambda
        assert!(
            cut.lambda >= 0.0 && cut.lambda <= 0.1,
            "Proactive cuts should have small initial lambda"
        );
        // All cuts should have non-empty terms
        assert!(!cut.terms.is_empty(), "Cut should have terms");
    }

    // Verify we have both single-neuron and pairwise cuts
    let single_neuron_cuts = cut_pool.cuts.iter().filter(|c| c.terms.len() == 1).count();
    let _pairwise_cuts = cut_pool.cuts.iter().filter(|c| c.terms.len() == 2).count();

    assert!(
        single_neuron_cuts > 0,
        "Should have single-neuron indicator cuts"
    );
    // Pairwise cuts are generated when there are multiple ReLU nodes
    // (may be 0 if only single-neuron cuts are generated due to max_cuts limit)
}

/// Regression test for #554: proactive cut generation must be deterministic
/// across multiple invocations. Previously, HashMap iteration order caused
/// nondeterministic pairwise cut selection.
#[ntest::timeout(10000)]
#[test]
fn test_proactive_cuts_deterministic_ordering_554() {
    let (graph, node_bounds) = build_two_relu_graph_with_bounds();

    // Generate proactive cuts multiple times and verify identical output.
    let mut reference_terms: Option<Vec<Vec<(String, usize)>>> = None;

    for _ in 0..5 {
        let mut pool = GraphCutPool::new(100);
        let count = pool
            .generate_proactive_cuts(&graph, &node_bounds, 50)
            .unwrap();
        assert!(count > 0, "Should generate cuts");

        let terms: Vec<Vec<(String, usize)>> = pool
            .cuts
            .iter()
            .map(|c| {
                c.terms
                    .iter()
                    .map(|t| (t.node_name.clone(), t.neuron_idx))
                    .collect()
            })
            .collect();

        if let Some(ref expected) = reference_terms {
            assert_eq!(
                &terms, expected,
                "Proactive cut ordering must be deterministic (#554)"
            );
        } else {
            reference_terms = Some(terms);
        }
    }
}

/// Regression test for #554: pairwise cuts must use bias=1.0
/// ("at least one active"), not 0.5.
#[ntest::timeout(10000)]
#[test]
fn test_proactive_pairwise_cuts_bias_554() {
    let (graph, node_bounds) = build_two_relu_graph_with_bounds();

    let mut pool = GraphCutPool::new(100);
    let count = pool
        .generate_proactive_cuts(&graph, &node_bounds, 50)
        .unwrap();
    assert!(
        count > 0,
        "Must generate cuts for bias assertions to be meaningful"
    );

    for cut in &pool.cuts {
        if cut.terms.len() == 2 {
            // Pairwise cuts encode "at least one active": z1 + z2 >= 1.0
            assert_eq!(
                cut.bias, 1.0,
                "Pairwise proactive cuts must have bias=1.0 (#554)"
            );
        } else if cut.terms.len() == 1 {
            // Single-neuron cuts use midpoint bias 0.5
            assert_eq!(
                cut.bias, 0.5,
                "Single-neuron proactive cuts must have bias=0.5"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_warmup_inherits_parent_values() {
    // Test that child domains inherit β values from parent via warmup

    // Create a parent split history with two constraints
    let history1 = GraphSplitHistory::new();
    let history1 = history1.with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let history1 = history1.with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    // Create parent β state and set non-default values
    let mut parent_beta = GraphBetaState::from_history(&history1).unwrap();
    parent_beta.entries[0].value = 0.5; // Optimized value
    parent_beta.entries[0].m = 0.1; // Adam momentum
    parent_beta.entries[0].v = 0.01;
    parent_beta.entries[1].value = 0.3;
    parent_beta.entries[1].m = 0.05;

    // Create child history with one more constraint
    let history2 = history1.with_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 2,
        is_active: true,
        score: 0.0,
    });

    // Create child β state with warmup
    let child_beta = GraphBetaState::from_history_with_warmup(
        &history2,
        &parent_beta,
        GraphBetaState::DEFAULT_BETA_INIT,
    )
    .unwrap();

    // Verify child has 3 entries
    assert_eq!(child_beta.entries.len(), 3);

    // Verify existing constraints inherit parent values
    let entry0 = child_beta.entry("relu1", 0).unwrap();
    assert_eq!(entry0.value, 0.5, "Should inherit parent β value");
    assert_eq!(entry0.m, 0.1, "Should inherit parent Adam momentum m");
    assert_eq!(entry0.v, 0.01, "Should inherit parent Adam momentum v");
    assert_eq!(entry0.sign, 1.0, "Active constraint has sign +1");

    let entry1 = child_beta.entry("relu1", 1).unwrap();
    assert_eq!(entry1.value, 0.3, "Should inherit parent β value");
    assert_eq!(entry1.m, 0.05, "Should inherit parent Adam momentum m");
    assert_eq!(entry1.sign, -1.0, "Inactive constraint has sign -1");

    // Verify new constraint has default initialization
    let entry2 = child_beta.entry("relu2", 2).unwrap();
    assert_eq!(
        entry2.value,
        GraphBetaState::DEFAULT_BETA_INIT,
        "New constraint should have default β"
    );
    assert_eq!(entry2.m, 0.0, "New constraint should have zero momentum m");
    assert_eq!(entry2.v, 0.0, "New constraint should have zero momentum v");
    assert_eq!(entry2.sign, 1.0, "Active constraint has sign +1");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_cut_authority_quarantine_with_objective() {
    // Defense in depth: bypass public validation and supply an objective plus a
    // non-empty cut pool. The graph finalizer must still leave proof bounds
    // unchanged.
    use crate::beta_crown::domain::GraphCrownContext;

    let (graph, _) = build_two_relu_graph_with_bounds();
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    // Collect proper IBP bounds for all graph nodes
    let ibp_bounds = graph.collect_node_bounds(&input_bounds).unwrap();
    let node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>> = ibp_bounds
        .iter()
        .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
        .collect();

    // Build a cut pool with a non-trivial cut
    let mut pool = GraphCutPool::new(10);
    let mut cut_history = GraphSplitHistory::new();
    cut_history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    cut_history.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    pool.add_from_verified_domain(&cut_history).unwrap();
    // Set lambda high so cut contribution would be visible if applied
    pool.cuts[0].lambda = 10.0;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: true,
        ..Default::default()
    });

    let objective = vec![1.0_f32, 0.0];

    // Compute objective bounds with a cut pool. The fixture's cut IS relevant
    // to the empty history (non-empty `relevant_cuts_for`), so any surviving
    // cut→bound path would show up in the comparison below.
    let empty_history = GraphSplitHistory::new();
    assert!(
        !pool.relevant_cuts_for(&empty_history).is_empty(),
        "fixture must expose a relevant cut so the authority gate is exercised"
    );
    let context_with_cuts =
        GraphCrownContext::new(&empty_history, Some(&pool), Some(&node_bounds), None);
    let (bounds_with_quarantined_cuts, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input_bounds,
            &context_with_cuts,
            None,
            Some(&objective),
        )
        .unwrap();

    // Compute the same objective bounds without any cuts for comparison.
    let empty_history2 = GraphSplitHistory::new();
    let context_no_cuts = GraphCrownContext::new(&empty_history2, None, Some(&node_bounds), None);
    let (bounds_no_cuts, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input_bounds,
            &context_no_cuts,
            None,
            Some(&objective),
        )
        .unwrap();

    let lower_with = bounds_with_quarantined_cuts.lower().as_slice().unwrap();
    let lower_without = bounds_no_cuts.lower().as_slice().unwrap();
    for (i, (&with, &without)) in lower_with.iter().zip(lower_without.iter()).enumerate() {
        assert_eq!(
            with, without,
            "objective lower[{i}] changed despite the cut-authority quarantine"
        );
    }
}

/// Regression test for #2575: GraphCuttingPlane Adam must not produce NaN/Inf
/// when beta1=1.0.
#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_adam_beta1_one_no_div_by_zero_2575() {
    let mut cut = GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 1.0,
        lambda: 0.0,
        lambda_grad: 1.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // beta1=1.0 would cause division by zero without .max(f32::EPSILON) guard
    cut.update_lambda_adam(0.1, 1.0, 0.999, 1e-8, 1);

    assert!(
        cut.lambda.is_finite(),
        "lambda should be finite with beta1=1.0, got {}",
        cut.lambda
    );
}

/// Regression test for #2575: GraphCuttingPlane Adam must not produce NaN/Inf
/// when beta2=1.0.
#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_adam_beta2_one_no_div_by_zero_2575() {
    let mut cut = GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 1.0,
        lambda: 0.0,
        lambda_grad: 1.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // beta2=1.0 would cause division by zero without .max(f32::EPSILON) guard
    cut.update_lambda_adam(0.1, 0.9, 1.0, 1e-8, 1);

    assert!(
        cut.lambda.is_finite(),
        "lambda should be finite with beta2=1.0, got {}",
        cut.lambda
    );
}

/// Regression test for #2575: GraphCuttingPlane Adam must not produce NaN/Inf
/// when t=0 (missing t.max(1) guard).
#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_adam_t_zero_no_div_by_zero_2575() {
    let mut cut = GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 1.0,
        lambda: 0.0,
        lambda_grad: 1.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // t=0 would cause division by zero (1.0 - beta^0 = 0.0) without t.max(1) guard
    cut.update_lambda_adam(0.1, 0.9, 0.999, 1e-8, 0);

    assert!(
        cut.lambda.is_finite(),
        "lambda should be finite with t=0, got {}",
        cut.lambda
    );
}

/// Regression test for #2598: GraphCuttingPlane Adam must reset state on NaN gradient.
#[ntest::timeout(10000)]
#[test]
fn test_graph_cutting_plane_nan_lambda_guard_2598() {
    let mut cut = GraphCuttingPlane {
        terms: vec![GraphCutTerm {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 1.0,
        lambda: 0.0,
        lambda_grad: 1.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // Normal step
    cut.update_lambda_adam(0.1, 0.9, 0.999, 1e-8, 1);
    assert!(
        cut.lambda > 0.0,
        "lambda should be positive after normal step"
    );

    // Inject NaN gradient
    cut.lambda_grad = f32::NAN;
    cut.update_lambda_adam(0.1, 0.9, 0.999, 1e-8, 2);
    assert_eq!(
        cut.lambda, 0.0,
        "lambda should reset to 0.0 on NaN gradient"
    );
    assert_eq!(cut.lambda_m, 0.0, "lambda_m should reset on NaN");
    assert_eq!(cut.lambda_v, 0.0, "lambda_v should reset on NaN");
}
