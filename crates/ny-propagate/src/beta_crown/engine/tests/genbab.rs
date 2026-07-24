// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;

// GenBaB Phase 1: NeuronSplit and GeneralSplitHistory tests
// =========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_layer_ref_display() {
    assert_eq!(format!("{}", LayerRef::Index(5)), "layer_5");
    assert_eq!(format!("{}", LayerRef::Name("relu_1".into())), "relu_1");
}

#[ntest::timeout(10000)]
#[test]
fn test_neuron_split_relu_active() {
    let split = NeuronSplit::relu_active(LayerRef::Index(2), 5);
    assert_eq!(split.layer, LayerRef::Index(2));
    assert_eq!(split.neuron_idx, 5);
    assert_eq!(split.lower_bound, Some(0.0));
    assert_eq!(split.upper_bound, None);
    assert!(split.is_relu_split());
    assert_eq!(split.branching_point(), Some(0.0));
    assert_eq!(split.score, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_neuron_split_relu_inactive() {
    let split = NeuronSplit::relu_inactive(LayerRef::Name("relu_0".into()), 3);
    assert_eq!(split.layer, LayerRef::Name("relu_0".into()));
    assert_eq!(split.neuron_idx, 3);
    assert_eq!(split.lower_bound, None);
    assert_eq!(split.upper_bound, Some(0.0));
    assert!(split.is_relu_split());
    assert_eq!(split.branching_point(), Some(0.0));
    assert_eq!(split.score, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_neuron_split_at_point() {
    // GeLU upper branch at -0.5 (x >= -0.5)
    let upper = NeuronSplit::at_point(LayerRef::Name("gelu_1".into()), 7, -0.5, true).unwrap();
    assert_eq!(upper.lower_bound, Some(-0.5));
    assert_eq!(upper.upper_bound, None);
    assert!(!upper.is_relu_split());
    assert_eq!(upper.branching_point(), Some(-0.5));
    assert_eq!(upper.score, 0.0);

    // GeLU lower branch at -0.5 (x <= -0.5)
    let lower = NeuronSplit::at_point(LayerRef::Name("gelu_1".into()), 7, -0.5, false).unwrap();
    assert_eq!(lower.lower_bound, None);
    assert_eq!(lower.upper_bound, Some(-0.5));
    assert!(!lower.is_relu_split());
    assert_eq!(lower.branching_point(), Some(-0.5));
    assert_eq!(lower.score, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_neuron_split_from_constraint() {
    let constraint = NeuronConstraint::new(3, 2, true, 1.25).expect("valid constraint");
    let split = NeuronSplit::from_constraint(&constraint).expect("valid split");
    assert_eq!(*split.layer(), LayerRef::Index(3));
    assert_eq!(split.neuron_idx(), 2);
    assert!(split.is_relu_split());
    assert_eq!(split.lower_bound(), Some(0.0));
    assert_eq!(split.score(), 1.25);
}

#[ntest::timeout(10000)]
#[test]
fn test_neuron_split_from_graph_constraint() {
    let constraint =
        GraphNeuronConstraint::new("relu_5".into(), 10, false, 2.0).expect("valid constraint");
    let split = NeuronSplit::from_graph_constraint(&constraint).expect("valid split");
    assert_eq!(*split.layer(), LayerRef::Name("relu_5".into()));
    assert_eq!(split.neuron_idx(), 10);
    assert!(split.is_relu_split());
    assert_eq!(split.upper_bound(), Some(0.0));
    assert_eq!(split.score(), 2.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_basic() {
    let mut history = GeneralSplitHistory::new();
    assert_eq!(history.depth(), 0);

    history.add_split(NeuronSplit::relu_active(LayerRef::Index(1), 0));
    assert_eq!(history.depth(), 1);

    history.add_split(NeuronSplit::at_point(LayerRef::Name("gelu".into()), 5, -0.5, true).unwrap());
    assert_eq!(history.depth(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_with_split() {
    let history = GeneralSplitHistory::new();
    let new_history = history.with_split(NeuronSplit::relu_active(LayerRef::Index(0), 0));
    assert_eq!(history.depth(), 0);
    assert_eq!(new_history.depth(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_get_neuron_bounds() {
    let mut history = GeneralSplitHistory::new();
    let layer = LayerRef::Name("gelu_0".into());

    // Initially no bounds
    assert_eq!(history.neuron_bounds(&layer, 3), (None, None));

    // Add lower bound split
    history.add_split(NeuronSplit::at_point(layer.clone(), 3, -0.5, true).unwrap());
    assert_eq!(history.neuron_bounds(&layer, 3), (Some(-0.5), None));

    // Add upper bound split (further restricts)
    history.add_split(NeuronSplit::at_point(layer.clone(), 3, 0.5, false).unwrap());
    assert_eq!(history.neuron_bounds(&layer, 3), (Some(-0.5), Some(0.5)));

    // Different neuron still has no bounds
    assert_eq!(history.neuron_bounds(&layer, 4), (None, None));
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_from_graph_history() {
    let mut graph_history = GraphSplitHistory::new();
    graph_history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_0".into(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    graph_history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".into(),
        neuron_idx: 5,
        is_active: false,
        score: 0.0,
    });
    graph_history.add_genbab_constraint(
        crate::beta_crown::GenBabConstraint::new("gelu_2".into(), 3, -0.5, true, 1.25).unwrap(),
    );

    let general_history =
        GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general_history.depth(), 3);
    assert_eq!(
        general_history.splits()[0].layer(),
        &LayerRef::Name("relu_0".into())
    );
    assert_eq!(general_history.splits()[0].lower_bound(), Some(0.0));
    assert_eq!(
        general_history.splits()[1].layer(),
        &LayerRef::Name("relu_1".into())
    );
    assert_eq!(general_history.splits()[1].upper_bound(), Some(0.0));
    assert_eq!(
        general_history.splits()[2].layer(),
        &LayerRef::Name("gelu_2".into())
    );
    assert_eq!(general_history.splits()[2].lower_bound(), Some(-0.5));
    assert!((general_history.splits()[2].score() - 1.25).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_range_merge() {
    let mut graph_history = GraphSplitHistory::new();
    let lower =
        crate::beta_crown::GenBabConstraint::new("gelu_3".into(), 1, -0.25, true, 0.8).unwrap();
    let upper =
        crate::beta_crown::GenBabConstraint::new("gelu_3".into(), 1, 0.5, false, 0.8).unwrap();
    graph_history.add_genbab_constraints_for_split([lower, upper]);

    assert_eq!(graph_history.depth(), 1);
    assert_eq!(graph_history.genbab_constraints.len(), 2);

    let general_history =
        GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general_history.depth(), 1);
    assert_eq!(
        general_history.splits()[0].layer(),
        &LayerRef::Name("gelu_3".into())
    );
    assert_eq!(general_history.splits()[0].lower_bound(), Some(-0.25));
    assert_eq!(general_history.splits()[0].upper_bound(), Some(0.5));
    assert!((general_history.splits()[0].score() - 0.8).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_no_merge_across_splits() {
    let mut graph_history = GraphSplitHistory::new();
    let upper =
        crate::beta_crown::GenBabConstraint::new("gelu_4".into(), 2, -0.2, true, 0.3).unwrap();
    let lower =
        crate::beta_crown::GenBabConstraint::new("gelu_4".into(), 2, 0.1, false, 0.3).unwrap();
    graph_history.add_genbab_constraint(upper);
    graph_history.add_genbab_constraint(lower);

    assert_eq!(graph_history.depth(), 2);

    let general_history =
        GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general_history.depth(), 2);
    assert_eq!(general_history.splits()[0].lower_bound(), Some(-0.2));
    assert_eq!(general_history.splits()[0].upper_bound(), None);
    assert_eq!(general_history.splits()[1].lower_bound(), None);
    assert_eq!(general_history.splits()[1].upper_bound(), Some(0.1));
}

// =========================================================================
// GenBaB Integration Tests: End-to-end verification with GeLU
// =========================================================================

/// Integration test: Verify a simple GeLU network using GenBaB branching.
/// This tests the full BaB pipeline with GenBaB for general nonlinearities.
#[ntest::timeout(10000)]
#[test]
fn test_genbab_gelu_verification_trivial() {
    use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;
    use crate::layers::GELULayer;

    // Build: Linear -> GeLU -> Linear
    // Input: [0.5, 1.5] -> always positive, GeLU ≈ identity for x > 0
    // Output should be clearly positive

    let w1 = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]); // Identity
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[1.0_f32, 1.0]]); // Sum outputs
    let b2 = arr1(&[0.1_f32]); // Positive bias
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    // Input bounds: [0.5, 1.5] (strictly positive)
    let input = BoundedTensor::new(
        arr1(&[0.5_f32, 0.5]).into_dyn(),
        arr1(&[1.5_f32, 1.5]).into_dyn(),
    )
    .unwrap();

    // Configure verifier with GenBaB
    let genbab_config = NonlinearBranchingConfig::default();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(genbab_config),
        max_domains: 100,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Objective: maximize single output (coefficient = 1.0)
    let objective = [1.0_f32];

    // Threshold well below expected output (output should be ~0.5 + 0.5 + 0.1 = 1.1)
    let result = verifier
        .verify_graph_relu_split(&graph, &input, &objective, 0.0)
        .unwrap();

    // Should verify since GeLU(x) ≈ x for x > 0, and sum + bias > 0
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "GeLU network with positive inputs should verify for threshold=0"
    );
}

/// Build a Linear1(2→4) → ReLU → Linear2(4→2) → GeLU graph for splitting tests.
///
/// Weights from the multi-objective benchmark (#1851). Linear2 uses small weights
/// (0.04-0.18) to keep GeLU pre-activation ranges narrow (~[-0.3, 0.5]).
fn build_relu_gelu_splitting_graph() -> GraphNetwork {
    use crate::layers::GELULayer;
    use crate::layers::ReLULayer;

    let w1 = arr2(&[[1.2_f32, -0.8], [-0.6, 1.1], [0.9, 0.7], [-0.7, 0.4]]);
    let b1 = arr1(&[0.1_f32, -0.05, 0.0, 0.12]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let w2 = arr2(&[[0.16_f32, -0.10, 0.12, -0.04], [-0.06, 0.18, -0.08, 0.14]]);
    let b2 = arr1(&[0.01_f32, -0.02]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear2".to_string()],
    ));
    graph.set_output("gelu");
    graph
}

/// Integration test: GenBaB verifier requires splitting on a ReLU+GeLU network.
///
/// Network: Linear1(2→4) → ReLU → Linear2(4→2) → GeLU
/// Input domain: [-1, 1]², objective: c = [1, 1] (sum both GeLU outputs).
///
/// Mathematical justification (verified numerically on 1000×1000 grid):
///   True minimum of c·f(x) ≈ 0.00395, attained near x ≈ (-0.085, -0.003).
///   Root α-CROWN lower bound ≈ -0.10 (empirical, from verifier output).
///   Threshold = -0.05: below true min (property holds), but above root CROWN
///   bound, so the initial CROWN pass cannot verify → BaB splitting is required.
///
/// Recalibrated in #3261: original single-layer Identity→GeLU test was unable
/// to require splitting (IBP∩CROWN gives exact bounds for single-layer networks).
/// This 4-ReLU-neuron network creates cross-layer relaxation gaps that force
/// multi-domain exploration. Empirical: ~2120 domains explored, 2 verified.
#[ntest::timeout(10000)]
#[test]
fn test_genbab_gelu_verification_needs_splitting() {
    use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;

    let graph = build_relu_gelu_splitting_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let genbab_config = NonlinearBranchingConfig {
        num_candidates: 4,
        ..Default::default()
    };
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(genbab_config),
        max_domains: 2100,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // See doc comment for threshold justification: true min ≈ 0.004 > -0.05 > root CROWN ≈ -0.10.
    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0_f32, 1.0], -0.05)
        .unwrap();

    assert!(
        result.domains_explored >= 2,
        "Root CROWN bound < -0.05 should require BaB splitting \
         (got domains_explored={}, expected >= 2)",
        result.domains_explored,
    );
    assert!(
        result.domains_verified >= 1,
        "At least one sub-domain should verify after ReLU splitting \
         (got domains_verified={}, explored={})",
        result.domains_verified,
        result.domains_explored,
    );
    // Note: BabVerificationStatus::Verified not asserted here (P1 1090 request).
    // With max_domains=2100, BaB explores ~2120 domains but only verifies ~2.
    // The relaxation gap is too large for full convergence at this budget.
    // This test is specifically for BaB splitting behavior, not full verification.
    // The trivial test (line 232) already asserts Verified for GenBaB GELU.
}

/// Regression test for #3262: GeLU-only network (no ReLU) must not return
/// Unknown "No unstable ReLU neurons" when GenBaB is configured. The BaB
/// loop must route to GenBaB processing before checking for ReLU neurons.
#[ntest::timeout(10000)]
#[test]
fn test_genbab_gelu_only_network_does_not_terminate_early_3262() {
    use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;
    use crate::layers::GELULayer;

    // Pure GeLU network: Linear(2→2) → GeLU → Linear(2→1)
    let w1 = arr2(&[[1.5_f32, -0.8], [-0.6, 1.3]]);
    let b1 = arr1(&[0.1_f32, -0.05]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let w2 = arr2(&[[1.0_f32, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let genbab_config = NonlinearBranchingConfig {
        num_candidates: 4,
        ..Default::default()
    };
    // batch_size=1 forces the sequential path (process_sequential_domains)
    // where the #3262 fix lives. Default batch_size=64 takes the parallel
    // path which already had the GenBaB-first check before this fix.
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(genbab_config),
        max_domains: 100,
        timeout: Duration::from_secs(5),
        batch_size: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], -5.0)
        .unwrap();

    // With threshold -5.0 (very loose), root CROWN bounds easily verify.
    // The key regression check: must NOT short-circuit with "No unstable
    // ReLU neurons" before GenBaB gets a chance to process the domain.
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "GeLU-only network with GenBaB (sequential path) and loose threshold \
         should verify, not terminate early with Unknown (#3262). Got: {:?}",
        result.result,
    );
}
