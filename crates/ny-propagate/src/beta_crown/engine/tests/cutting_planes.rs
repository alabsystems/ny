// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;
use std::sync::atomic::Ordering;

fn make_test_network(layers: Vec<Layer>) -> Network {
    Network {
        layers,
        gpu_crown_cache: std::sync::Mutex::new(None),
    }
}

// Quarantined cutting-plane data structures and research arithmetic.
//
// These low-level tests preserve experimental code for a future certified
// fold. They do not grant certificate authority or establish soundness of the
// legacy post-concretization scalar.
// =========================================================================

fn make_basic_cut(kind: CutKind, bias: f32) -> CuttingPlane {
    CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias,
        lambda: 0.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, kind),
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_from_history() {
    // Create a split history with some constraints
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 2,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Generate cut from history
    let cut = CuttingPlane::from_verified_domain(&history).unwrap();
    assert!(cut.is_some());

    let cut = cut.unwrap();
    assert_eq!(cut.terms.len(), 3);
    assert_eq!(cut.source_depth, 3);

    // Check term signs
    assert_eq!(cut.terms[0].coefficient, 1.0); // active -> positive
    assert_eq!(cut.terms[1].coefficient, -1.0); // inactive -> negative
    assert_eq!(cut.terms[2].coefficient, 1.0); // active -> positive
}

#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_empty_history() {
    let history = SplitHistory::new();
    let cut = CuttingPlane::from_verified_domain(&history).unwrap();
    assert!(cut.is_none());
}

#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_redundancy() {
    // Create a cut from some constraints
    let mut history1 = SplitHistory::new();
    history1.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history1.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let cut = CuttingPlane::from_verified_domain(&history1)
        .unwrap()
        .unwrap();

    // A domain with the same constraints should find the cut redundant
    assert!(cut.is_redundant_for(&history1));

    // A domain with different constraints should not find it redundant
    let mut history2 = SplitHistory::new();
    history2.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: false,
        score: 0.0, // Different!
    });
    assert!(!cut.is_redundant_for(&history2));

    // A domain with partial constraints
    let mut history3 = SplitHistory::new();
    history3.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    // Only one constraint matches
    assert!(!cut.is_redundant_for(&history3));
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_basic() {
    let mut pool = CutPool::new(10);
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);

    // Add cut from history
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    assert!(pool.add_from_verified_domain(&history).unwrap());
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.total_generated, 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_rejects_shallow() {
    let mut pool = CutPool::new(10);

    // Single-constraint history (too shallow, min_depth=2)
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Should not add cut from shallow domain
    assert!(!pool.add_from_verified_domain(&history).unwrap());
    assert_eq!(pool.len(), 0);
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_capacity() {
    let mut pool = CutPool::new(2); // Small capacity

    // Add first cut
    let mut history1 = SplitHistory::new();
    for i in 0..3 {
        history1.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: i,
            is_active: true,
            score: 0.0,
        });
    }
    assert!(pool.add_from_verified_domain(&history1).unwrap());

    // Add second cut
    let mut history2 = SplitHistory::new();
    for i in 3..6 {
        history2.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: i,
            is_active: false,
            score: 0.0,
        });
    }
    assert!(pool.add_from_verified_domain(&history2).unwrap());

    // Third cut should evict an older cut (pool full)
    let mut history3 = SplitHistory::new();
    for i in 6..9 {
        history3.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: i,
            is_active: true,
            score: 0.0,
        });
    }
    assert!(pool.add_from_verified_domain(&history3).unwrap());
    assert_eq!(pool.len(), 2);
    assert_eq!(pool.total_generated, 3);
    assert_eq!(pool.cuts_evicted_total, 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_proactive_cap() {
    let mut pool = CutPool::new(5);
    pool.cut_proactive_fraction = 0.2;

    assert!(pool.add_cut(make_basic_cut(CutKind::Proactive, 0.1)));
    assert!(!pool.add_cut(make_basic_cut(CutKind::Proactive, 0.2)));

    let proactive_count = pool
        .cuts
        .iter()
        .filter(|cut| cut.metadata.cut_kind() == CutKind::Proactive)
        .count();
    assert_eq!(proactive_count, 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_fifo_evicts_oldest() {
    let mut pool = CutPool::new(2);
    pool.eviction_policy = CutEvictionPolicy::Fifo;

    assert!(pool.add_cut(make_basic_cut(CutKind::Verified, 1.0)));
    assert!(pool.add_cut(make_basic_cut(CutKind::Verified, 2.0)));

    pool.cuts[0]
        .metadata
        .created_iter
        .store(0, Ordering::Relaxed);
    pool.cuts[1]
        .metadata
        .created_iter
        .store(10, Ordering::Relaxed);

    assert!(pool.add_cut(make_basic_cut(CutKind::Verified, 3.0)));

    let has_oldest = pool.cuts.iter().any(|cut| (cut.bias - 1.0).abs() < 1e-6);
    let has_second = pool.cuts.iter().any(|cut| (cut.bias - 2.0).abs() < 1e-6);
    let has_new = pool.cuts.iter().any(|cut| (cut.bias - 3.0).abs() < 1e-6);

    assert!(!has_oldest);
    assert!(has_second);
    assert!(has_new);
    assert_eq!(pool.cuts_evicted_total, 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_relevant_cuts() {
    let mut pool = CutPool::new(10);

    // Add a cut with specific constraints
    let mut history1 = SplitHistory::new();
    history1.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history1.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    pool.add_from_verified_domain(&history1).unwrap();

    // Domain with no constraints -> cut is relevant
    let empty_history = SplitHistory::new();
    let relevant = pool.relevant_cuts_for(&empty_history);
    assert_eq!(relevant.len(), 1);

    // Domain with matching constraints -> cut is redundant
    let relevant = pool.relevant_cuts_for(&history1);
    assert_eq!(relevant.len(), 0);
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_usage_metadata_updates() {
    let mut pool = CutPool::new(10);

    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    assert!(pool.add_from_verified_domain(&history).unwrap());
    assert_eq!(pool.cuts.len(), 1);

    let cut = &pool.cuts[0];
    assert_eq!(cut.metadata.created_iter.load(Ordering::Relaxed), 0);
    assert_eq!(cut.metadata.use_count.load(Ordering::Relaxed), 0);

    let relevant = pool.relevant_cuts_for(&SplitHistory::new());
    assert_eq!(relevant.len(), 1);
    assert_eq!(cut.metadata.last_used_iter.load(Ordering::Relaxed), 1);
    assert_eq!(cut.metadata.use_count.load(Ordering::Relaxed), 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_adam_step() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let mut cut = CuttingPlane::from_verified_domain(&history)
        .unwrap()
        .unwrap();
    assert_eq!(cut.lambda, 0.0);

    // Set gradient and take step
    cut.lambda_grad = 1.0;
    let config = AdaptiveOptConfig::default();
    cut.gradient_step_adam(&config, 1);

    // Lambda should have increased (gradient ascent)
    assert!(cut.lambda > 0.0);
    // Lambda should be non-negative (projected)
    assert!(cut.lambda >= 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_strengthening_keeps_high_scores() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint::new(0, 0, true, 0.1).unwrap());
    history.add_constraint(NeuronConstraint::new(0, 1, true, 0.2).unwrap());
    history.add_constraint(NeuronConstraint::new(0, 2, false, 0.8).unwrap());
    history.add_constraint(NeuronConstraint::new(0, 3, false, 0.9).unwrap());

    let beta_state = BetaState::from_history(&history).unwrap();
    let strengthened = CuttingPlane::from_verified_domain_strengthened(&history, &beta_state, 0.5)
        .unwrap()
        .unwrap();

    assert_eq!(strengthened.history.constraints.len(), 2);
    assert_eq!(strengthened.cut.source_depth, 2);
    assert_eq!(strengthened.dropped_constraints, 2);

    let kept: Vec<_> = strengthened
        .history
        .constraints
        .iter()
        .map(|c| c.neuron_idx)
        .collect();
    assert_eq!(kept, vec![2, 3]);
}

#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_strengthening_keeps_beta_positive() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint::new(1, 0, true, 0.05).unwrap());
    history.add_constraint(NeuronConstraint::new(1, 1, false, 0.9).unwrap());
    history.add_constraint(NeuronConstraint::new(1, 2, true, 0.2).unwrap());

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 0.4);

    let strengthened = CuttingPlane::from_verified_domain_strengthened(&history, &beta_state, 0.5)
        .unwrap()
        .unwrap();

    assert_eq!(strengthened.history.constraints.len(), 2);
    assert_eq!(strengthened.dropped_constraints, 1);

    let kept: Vec<_> = strengthened
        .history
        .constraints
        .iter()
        .map(|c| c.neuron_idx)
        .collect();
    assert_eq!(kept, vec![0, 1]);
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_proactive_cuts_generation() {
    // Test that proactive cuts are generated correctly for sequential networks
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2, Array1};

    // Linear -> ReLU -> Linear -> ReLU
    let linear1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
        Some(arr1(&[0.1, -0.1])),
    )
    .unwrap();
    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.3], [0.3, 1.0]]),
        Some(arr1(&[0.0, 0.0])),
    )
    .unwrap();

    let network = make_test_network(vec![
        Layer::Linear(linear1),
        Layer::ReLU(ReLULayer),
        Layer::Linear(linear2),
        Layer::ReLU(ReLULayer),
    ]);

    // Create layer bounds with unstable neurons (crossing zero)
    // layer_bounds[0] = output of layer 0 (Linear1)
    // layer_bounds[1] = output of layer 1 (ReLU1)
    // layer_bounds[2] = output of layer 2 (Linear2)
    // layer_bounds[3] = output of layer 3 (ReLU2)

    let bounds_linear1 = BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -0.5]).into_dyn(), // Some neurons crossing zero
        Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
    )
    .unwrap();

    let bounds_relu1 = BoundedTensor::new(
        Array1::from_vec(vec![0.0, 0.0]).into_dyn(), // ReLU output non-negative
        Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
    )
    .unwrap();

    let bounds_linear2 = BoundedTensor::new(
        Array1::from_vec(vec![-0.8, -0.3]).into_dyn(), // Some neurons crossing zero
        Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
    )
    .unwrap();

    let bounds_relu2 = BoundedTensor::new(
        Array1::from_vec(vec![0.0, 0.0]).into_dyn(),
        Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
    )
    .unwrap();

    let layer_bounds = vec![bounds_linear1, bounds_relu1, bounds_linear2, bounds_relu2];

    // Generate proactive cuts
    let mut cut_pool = CutPool::new(100);
    let cuts_generated = cut_pool
        .generate_proactive_cuts(&network, &layer_bounds, 50)
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
        // All cuts should reference valid layer indices
        for term in &cut.terms {
            assert!(
                term.layer_idx < network.layers.len(),
                "Cut term should reference valid layer index"
            );
        }
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_quarantined_cut_config_fields_remain_inspectable() {
    let config = BetaCrownConfig {
        enable_cuts: true,
        max_cuts: 500,
        min_cut_depth: 3,
        ..Default::default()
    };

    assert!(config.enable_cuts);
    assert_eq!(config.max_cuts, 500);
    assert_eq!(config.min_cut_depth, 3);
    assert!(config.validate().is_err());
    assert!(!config.cut_proof_authority_enabled());
}

#[ntest::timeout(5000)]
#[test]
fn test_quarantined_legacy_cut_gradient_arithmetic() {
    // Characterize the matching legacy gradient only; it has no proof authority.
    // d(lb)/d(lambda) = bias - constraint_min (for lambda * (bias - constraint_min))
    let mut cut = CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
            neuron_idx: 0,
            coefficient: 1.0, // positive -> use z_min
        }],
        bias: 0.5,
        lambda: 1.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // Create layer bounds: neuron 0 has [-1, 2] → unstable, z ∈ [0, 1]
    let lower = arr1(&[-1.0]).into_dyn();
    let upper = arr1(&[2.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();
    let _layer_bounds = [Arc::new(bounds)];

    // With ReLU indicators:
    // coeff=1 (positive) → use z_min = 0
    // constraint_min = 1 * 0 = 0
    // Expected gradient: bias - constraint_min = 0.5 - 0 = 0.5
    // (positive gradient means increasing lambda increases lower bound)

    let constraint_min = 0.0; // coeff=1 * z_min=0
    let expected_grad = cut.bias - constraint_min; // 0.5 - 0 = 0.5

    // Update gradient
    cut.lambda_grad = expected_grad;
    assert!((cut.lambda_grad - 0.5).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_quarantined_legacy_cut_optimizer_arithmetic() {
    // Characterize the research optimizer independently of proof bounds.
    let config = AdaptiveOptConfig {
        lr_lambda: Some(0.1),
        ..Default::default()
    };

    let mut cut = CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 2.0, // Positive bias means cut can contribute positively
        lambda: 0.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // Simulate gradient: bias - constraint_min = 2.0 - 0.0 = 2.0 (positive)
    // (Assuming unstable neuron with z_min=0)
    // Positive gradient means increasing lambda increases lower bound
    cut.lambda_grad = 2.0;

    // Take Adam step
    let initial_lambda = cut.lambda;
    cut.gradient_step_adam(&config, 1);

    // Lambda should increase (gradient ascent with positive gradient)
    assert!(
        cut.lambda > initial_lambda,
        "Lambda should increase with positive gradient"
    );
    assert!(cut.lambda >= 0.0, "Lambda should stay non-negative");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_input_split_residual_add() {
    // Graph: y = x + relu(x)
    let w = arr2(&[[1.0f32]]);
    let id = LinearLayer::new(w, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("id", Layer::Linear(id)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["id".to_string()],
    ));
    graph.add_node(GraphNode::binary("add", Layer::Add(AddLayer), "id", "relu"));
    graph.set_output("add");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 32,
        max_depth: 8,
        timeout: Duration::from_secs(1),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0], -1.1)
        .unwrap();
    assert!(matches!(result.result, BabVerificationStatus::Verified));
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_simple() {
    // Graph: y = relu(x)
    // Input: [-1, 1]
    // Output: [0, 1] for ReLU
    // Objective: y >= -0.5 (should verify since min output is 0)
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 32,
        max_depth: 8,
        timeout: Duration::from_secs(1),
        ..Default::default()
    });

    // Verify: y >= -0.5 (i.e., 1*y > -0.5)
    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], -0.5)
        .unwrap();
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "Expected Verified, got {:?}",
        result.result
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_detects_violation_relu_input() {
    // Regression test: a false property must not return Verified.
    // Graph: y = relu(x), x ∈ [-1, 1], property y > 0.5 (false).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
        .unwrap();
    assert!(
        matches!(
            result.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "Expected PotentialViolation, got {:?}",
        result.result
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_domain_limit_returns_unknown_1860() {
    let graph = super::gpu_bab::simple_graph_network();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 1,
        max_depth: 100,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
        .unwrap();

    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Domain limit 1 reached"),
                "expected domain limit reason, got {reason}",
            );
        }
        other => panic!("expected Unknown with domain-limit reason, got {other:?}"),
    }
    assert_eq!(
        result.domains_explored, 1,
        "max_domains=1 should stop after exploring the unresolved root domain",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_depth_limit_returns_unknown_1860() {
    let graph = super::gpu_bab::simple_graph_network();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 100,
        max_depth: 1,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
        .unwrap();

    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Max depth 1 reached"),
                "expected max-depth reason, got {reason}",
            );
        }
        other => panic!("expected Unknown with max-depth reason, got {other:?}"),
    }
    assert_eq!(result.max_depth_reached, 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_supports_conv2d() {
    // Conv2d models should use the ReLU-splitting path (no forced fallback),
    // so a violation should still be detected.
    // Graph: y = relu(conv(x)), x ∈ [-1,1], conv is identity, property y > 0.5 (false).
    let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0f32);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.set_output("relu");

    let input =
        BoundedTensor::new(arr3(&[[[-1.0]]]).into_dyn(), arr3(&[[[1.0]]]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
        .unwrap();
    assert!(
        matches!(
            result.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "Expected PotentialViolation, got {:?}",
        result.result
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_two_layer() {
    // Graph: y = relu(relu(x))
    // Input: [-2, 2]
    // After first ReLU: [0, 2]
    // After second ReLU: [0, 2]
    // Verify: y >= -1 (should pass since output is always >= 0)
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["relu1".to_string()],
    ));
    graph.set_output("relu2");

    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 10,
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], -1.0)
        .unwrap();
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "Expected Verified, got {:?}",
        result.result
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_beta_crown_relu_split_residual() {
    // Graph: y = x + relu(x) (residual connection)
    // Input: [-1, 1]
    // When x >= 0: y = x + x = 2x, range [0, 2]
    // When x < 0: y = x + 0 = x, range [-1, 0]
    // Total output range: [-1, 2]
    // Verify: y > -1.5 (should pass since min is -1)
    let w = arr2(&[[1.0f32]]);
    let id = LinearLayer::new(w, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("id", Layer::Linear(id)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["id".to_string()],
    ));
    graph.add_node(GraphNode::binary("add", Layer::Add(AddLayer), "id", "relu"));
    graph.set_output("add");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 10,
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], -1.5)
        .unwrap();
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "Expected Verified, got {:?}",
        result.result
    );
}

// #1861 Regression Tests: Verify that unresolved/violated domains prevent false Verified.
// ======================================================================================

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_violated_domain_must_not_return_verified_1861() {
    // Regression test for #1861: multi-objective path was silently dropping
    // violated domains and still returning Verified when queue emptied.
    //
    // Network: y = relu(x), x ∈ [-1, 1]
    // Output range: [0, 1]
    // Two objectives with the SAME threshold so both need y > 0.5:
    //   obj[0] = [1.0] -> y > 0.5 (FALSE: min output is 0)
    //   obj[1] = [1.0] -> y > 0.5 (FALSE: min output is 0)
    // The property is false, so the verifier must NOT return Verified.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    let objectives = vec![vec![1.0], vec![1.0]];
    let thresholds = vec![0.5, 0.5];
    let result = verifier
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .unwrap();
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "Bug #1861: multi-objective returned Verified despite violated domains. Got {:?}",
        result.result,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_depth_limit_returns_unknown_1861() {
    // Regression test for #1861: multi-objective path was silently dropping
    // depth-limited domains without flagging them as unresolved.
    //
    // Network: x ∈ R^2, y = W2 * relu(W1 * x)
    //   W1 = [[1, -1], [-1, 1]], b1 = [0, 0]
    //   W2 = [[1, 1]]
    //
    // Input: [-1, 1]^2
    // With max_depth=0 (no splitting), the initial CROWN bounds are too
    // loose to verify tight properties. Before fix: depth-limited domains
    // were silently dropped. After fix: returns Unknown.
    let w1 = arr2(&[[1.0f32, -1.0], [-1.0, 1.0]]);
    let b1 = arr1(&[0.0, 0.0]);
    let l1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let w2 = arr2(&[[1.0f32, 1.0]]);
    let l2 = LinearLayer::new(w2, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(l1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(l2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 0, // Zero depth — no splitting allowed
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    // Tight threshold: y > 0.5. True output range includes 0 (both relu
    // outputs zero), so root CROWN bounds are too loose. With max_depth=0
    // the verifier can't split and must return Unknown.
    let objectives = vec![vec![1.0]];
    let thresholds = vec![0.5];
    let result = verifier
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .unwrap();
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "Bug #1861: multi-objective returned Verified despite depth-limited domains. Got {:?}",
        result.result,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gcp_crown_scored_verification_rejects_cut_authority() {
    // The legacy research implementation remains unit-testable below, but it
    // must not enter certificate-bearing verification.
    //
    // Network: Linear -> ReLU -> Linear -> ReLU -> Linear
    // Multiple ReLU layers create opportunities for cuts.
    let w1 = arr2(&[[1.0f32], [-1.0]]);
    let b1 = arr1(&[0.0, 0.0]);
    let l1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let w2 = arr2(&[[1.0f32, 0.5], [0.5, 1.0]]);
    let b2 = arr1(&[0.0, 0.0]);
    let l2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let w3 = arr2(&[[1.0f32, -1.0]]);
    let l3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(l1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(l2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(l3));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    // Any request for cut proof authority must fail before BaB starts.
    let verifier_with_cuts = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: true,
        max_cuts: 100,
        min_cut_depth: 1,
        max_domains: 128,
        max_depth: 16,
        beta_iterations: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    let error = verifier_with_cuts
        .verify(&network, &input, -2.0)
        .expect_err("cut-enabled verification must be quarantined");
    assert!(
        error
            .to_string()
            .contains("cut proof authority is quarantined"),
        "unexpected validation error: {error}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_quarantined_legacy_lambda_optimizer_arithmetic() {
    // Characterize the research optimizer's positive-gradient behavior only.
    let config = AdaptiveOptConfig {
        lr_lambda: Some(0.1),
        ..Default::default()
    };

    let mut cut = CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 1.0,
        lambda: 0.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    };

    // Simulate multiple optimization iterations with positive gradient
    for t in 1..=10 {
        // Gradient = bias - constraint_min = 1.0 - 0.0 = 1.0 (positive)
        // This means increasing lambda increases lower bound
        cut.lambda_grad = 1.0;
        cut.gradient_step_adam(&config, t);
    }

    // Lambda should have increased significantly
    assert!(
        cut.lambda > 0.5,
        "Lambda should increase with positive gradient, got {}",
        cut.lambda
    );

    // Lambda should stay non-negative
    assert!(cut.lambda >= 0.0, "Lambda must be non-negative");
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_eviction_prefers_low_score() {
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

    let mut pool = CutPool::from_config(&config);

    let make_cut = |neuron_idx| CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
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

    assert!(pool.add_cut(make_cut(1)));
    assert!(pool.add_cut(make_cut(2)));

    pool.cuts[0].metadata.use_count.store(10, Ordering::Relaxed);
    pool.cuts[1].metadata.use_count.store(1, Ordering::Relaxed);

    assert!(pool.add_cut(make_cut(3)));

    let indices: Vec<usize> = pool
        .cuts
        .iter()
        .map(|cut| cut.terms[0].neuron_idx)
        .collect();
    assert!(indices.contains(&1));
    assert!(indices.contains(&3));
    assert!(!indices.contains(&2));
}

#[ntest::timeout(5000)]
#[test]
fn test_cut_pool_proactive_fraction_cap() {
    let config = BetaCrownConfig {
        max_cuts: 4,
        cut_proactive_fraction: 0.25,
        ..Default::default()
    };

    let mut pool = CutPool::from_config(&config);

    let make_proactive_cut = |neuron_idx| CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
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

    assert!(pool.add_cut(make_proactive_cut(1)));
    assert!(!pool.add_cut(make_proactive_cut(2)));
    assert_eq!(pool.len(), 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_try_add_strengthened_cut_no_constraints() {
    let config = BetaCrownConfig {
        enable_biccos_constraint_strengthening: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let layer_bounds = vec![input.clone()];
    let domain = BabDomain::root(layer_bounds, 0.0, 1.0).unwrap();

    let mut cut_pool = CutPool::new(4);
    let added = verifier
        .try_add_strengthened_cut(
            &mut cut_pool,
            &network,
            &input,
            0.0,
            &domain.layer_bounds,
            &domain,
            None,
        )
        .unwrap();

    assert!(!added, "No constraints should yield no strengthened cut");
    assert!(cut_pool.is_empty());
}

/// Regression test for #2575: CuttingPlane Adam must not produce NaN/Inf
/// when beta1=1.0, beta2=1.0, or t=0 (division-by-zero guard).
#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_adam_div_by_zero_guard_2575() {
    let make_cut = || {
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        });
        let mut cut = CuttingPlane::from_verified_domain(&history)
            .unwrap()
            .unwrap();
        cut.lambda_grad = 1.0;
        cut
    };

    // beta1=1.0: bias_correction1 = 1.0 - 1.0^t = 0.0
    let mut cut = make_cut();
    cut.gradient_step_adam(
        &AdaptiveOptConfig {
            beta1: 1.0,
            bias_correction: true,
            ..Default::default()
        },
        1,
    );
    assert!(
        cut.lambda.is_finite(),
        "lambda should be finite with beta1=1.0, got {}",
        cut.lambda
    );

    // beta2=1.0: bias_correction2 = 1.0 - 1.0^t = 0.0
    let mut cut = make_cut();
    cut.gradient_step_adam(
        &AdaptiveOptConfig {
            beta2: 1.0,
            bias_correction: true,
            ..Default::default()
        },
        1,
    );
    assert!(
        cut.lambda.is_finite(),
        "lambda should be finite with beta2=1.0, got {}",
        cut.lambda
    );

    // t=0: bias_correction = 1.0 - beta^0 = 0.0
    let mut cut = make_cut();
    cut.gradient_step_adam(&AdaptiveOptConfig::default(), 0);
    assert!(
        cut.lambda.is_finite(),
        "lambda should be finite with t=0, got {}",
        cut.lambda
    );
}

/// Regression test for #2598: CuttingPlane Adam must reset state on NaN gradient,
/// and evaluate() must return 0.0 (not NaN) for corrupted cuts.
#[ntest::timeout(5000)]
#[test]
fn test_cutting_plane_nan_lambda_guard_2598() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut cut = CuttingPlane::from_verified_domain(&history)
        .unwrap()
        .unwrap();

    // Normal step to establish non-zero state
    cut.lambda_grad = 1.0;
    cut.gradient_step_adam(&AdaptiveOptConfig::default(), 1);
    assert!(
        cut.lambda > 0.0,
        "lambda should be positive after normal step"
    );

    // Inject NaN gradient — should trigger m/v NaN guard
    cut.lambda_grad = f32::NAN;
    cut.gradient_step_adam(&AdaptiveOptConfig::default(), 2);
    assert_eq!(
        cut.lambda, 0.0,
        "lambda should reset to 0.0 on NaN gradient"
    );
    assert_eq!(cut.lambda_m, 0.0, "lambda_m should reset on NaN");
    assert_eq!(cut.lambda_v, 0.0, "lambda_v should reset on NaN");

    // Evaluate with NaN lambda should return 0.0, not NaN
    cut.lambda = f32::NAN;
    let result = cut.evaluate(&[(1.0, 2.0)]);
    assert_eq!(result, 0.0, "evaluate() should return 0.0 for NaN lambda");
}

// #2422: Cut contribution broadcast guard tests
// =========================================================================

/// Helper: compute bounds with and without cuts for a given output layer.
/// Returns (baseline_bounds, bounds_with_cuts).
fn bounds_with_and_without_cuts_2422(
    output_weights: Array2<f32>,
    output_bounds: BoundedTensor,
) -> (BoundedTensor, BoundedTensor) {
    let w1 = arr2(&[[1.0f32, 0.5], [-0.5, 1.0]]);
    let l1 = LinearLayer::new(w1, None).unwrap();
    let l2 = LinearLayer::new(output_weights, None).unwrap();
    let network = make_test_network(vec![
        Layer::Linear(l1),
        Layer::ReLU(ReLULayer),
        Layer::Linear(l2),
    ]);
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let lb0 =
        BoundedTensor::new(arr1(&[-1.5, -1.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap();
    let lb1 =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> =
        vec![Arc::new(lb0), Arc::new(lb1), Arc::new(output_bounds)];
    let history = SplitHistory::new();
    let beta_state = BetaState::from_history(&history).unwrap();
    let alpha_state = DomainAlphaState::empty();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let empty_pool = CutPool::new(10);
    let baseline = verifier
        .compute_bounds_with_alpha_beta(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &empty_pool,
            None,
        )
        .unwrap();

    let mut cut_pool = CutPool::new(10);
    cut_pool.add_cut(CuttingPlane {
        terms: vec![CutTerm {
            layer_idx: 1,
            neuron_idx: 0,
            coefficient: 1.0,
        }],
        bias: 0.5,
        lambda: 1.0,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 1,
        metadata: CutMetadata::new(0, CutKind::Verified),
    });
    let verifier_cuts = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: true,
        ..Default::default()
    });
    let with_cuts = verifier_cuts
        .compute_bounds_with_alpha_beta(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            None,
        )
        .unwrap();
    (baseline, with_cuts)
}

/// Defense in depth: even if validation is bypassed, quarantined cuts cannot
/// alter multi-output certificate bounds.
#[ntest::timeout(5000)]
#[test]
fn test_quarantined_cut_authority_does_not_modify_multi_output_2422() {
    let w2 = arr2(&[[1.0f32, -0.3], [0.3, 1.0]]); // 2→2: multi-output
    let lb2 =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap();
    let (baseline, with_cuts) = bounds_with_and_without_cuts_2422(w2, lb2);

    assert_eq!(
        baseline.lower().len(),
        2,
        "output should be multi-dimensional"
    );
    for (i, (base, cut_val)) in baseline
        .lower()
        .iter()
        .zip(with_cuts.lower().iter())
        .enumerate()
    {
        assert_eq!(
            *base, *cut_val,
            "#2422: lower[{i}] differs — scalar cut was broadcast to multi-output"
        );
    }
    for (i, (base, cut_val)) in baseline
        .upper()
        .iter()
        .zip(with_cuts.upper().iter())
        .enumerate()
    {
        assert_eq!(
            *base, *cut_val,
            "#2422: upper[{i}] differs — cut affected multi-output bounds"
        );
    }
}

/// Defense in depth: even the legacy scalar-output seam remains fail-closed
/// when validation is bypassed.
#[ntest::timeout(5000)]
#[test]
fn test_quarantined_cut_authority_does_not_modify_scalar_output_2422() {
    let w2 = arr2(&[[1.0f32, -1.0]]); // 2→1: scalar output
    let lb2 = BoundedTensor::new(arr1(&[-1.5]).into_dyn(), arr1(&[1.5]).into_dyn()).unwrap();
    let (baseline, with_cuts) = bounds_with_and_without_cuts_2422(w2, lb2);

    assert_eq!(baseline.lower().len(), 1, "output should be scalar");
    assert_eq!(
        with_cuts.lower()[[0]],
        baseline.lower()[[0]],
        "quarantined cuts changed a scalar lower bound: base={}, cut={}",
        baseline.lower()[[0]],
        with_cuts.lower()[[0]]
    );
    assert_eq!(
        baseline.upper()[[0]],
        with_cuts.upper()[[0]],
        "upper bound should be unchanged by cuts"
    );
}

// =========================================================================
