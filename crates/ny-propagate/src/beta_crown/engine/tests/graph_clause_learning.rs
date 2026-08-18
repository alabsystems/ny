// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Loop-level oracles for GRAPH-engine BaB conflict-clause learning (v2 port,
//! `conflict_clauses_graph`). Store-level discriminating oracles — purity
//! guard, record+prune round trip, gate-off inertness — live next to the
//! store in `conflict_clauses_graph::tests`; here we drive the REAL graph BaB
//! loops end to end and assert the A/B contract of the gate:
//!
//!   gate OFF (default) => the disabled store no-ops both entry points, the
//!   loop is byte-identical to baseline;
//!   gate ON => region-inclusion pruning may close domains early but must
//!   NEVER change a verdict (it only closes domains already proven safe by a
//!   same-run certificate).

use super::prelude::*;
use crate::beta_crown::biccos_q_stage0::{
    reset_test_observations as reset_biccos_q_stage0_observations,
    set_test_gate_override as set_biccos_q_stage0_gate_override,
    test_observations as biccos_q_stage0_observations,
};
use crate::beta_crown::conflict_clause_replay::{
    reset_test_runtime_observations, set_test_biccos_q_stage1_gate_override,
    set_test_runtime_gate_override, test_runtime_observations,
};
use crate::beta_crown::conflict_clauses::set_test_gate_override;
use crate::beta_crown::conflict_clauses_graph::{reset_test_store_mutations, test_store_mutations};

struct ClauseGateReset;

impl Drop for ClauseGateReset {
    fn drop(&mut self) {
        set_test_gate_override(None);
        set_test_runtime_gate_override(None);
        set_test_biccos_q_stage1_gate_override(None);
        set_biccos_q_stage0_gate_override(None);
    }
}

/// Single-objective graph loop (`verify_graph_relu_split_with_bounds`):
/// gate ON vs OFF must produce the identical final verdict across thresholds
/// covering Verified, in-between, and PotentialViolation.
#[ntest::timeout(30000)]
#[test]
fn test_graph_relu_split_with_bounds_gate_on_vs_off_identical_verdicts() {
    use crate::beta_crown::domain::GraphPrecomputedBounds;

    // simple_graph_network: 2 -> Linear -> ReLU -> Linear -> 1 with two
    // unstable ReLUs over [-1, 1]^2 — the loop actually branches.
    let graph = super::gpu_bab::simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = || BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 200,
        max_depth: 10,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };

    for threshold in [-10.0f32, -0.5, 100.0] {
        set_test_gate_override(Some(false));
        let verifier_off = BetaCrownVerifier::new(config());
        let (node_bounds, output_bounds) = verifier_off
            .compute_initial_graph_bounds(&graph, &input, None)
            .unwrap();
        let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);
        let off = verifier_off
            .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], threshold, &precomputed)
            .expect("gate-off verify must succeed");

        set_test_gate_override(Some(true));
        let verifier_on = BetaCrownVerifier::new(config());
        let on = verifier_on
            .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], threshold, &precomputed)
            .expect("gate-on verify must succeed");
        set_test_gate_override(None);

        assert_eq!(
            std::mem::discriminant(&off.result),
            std::mem::discriminant(&on.result),
            "graph clause learning changed the verdict at threshold {threshold}: \
             OFF={:?} ON={:?}",
            off.result,
            on.result
        );
        // Spot-check the two decided variants exactly: a sound pruner may only
        // close domains an OFF-run certificate already implies safe.
        if matches!(
            off.result,
            BabVerificationStatus::Verified | BabVerificationStatus::PotentialViolation { .. }
        ) {
            assert_eq!(off.result, on.result);
        }
        // Pruning can only SKIP bound work, never add it.
        assert!(
            on.domains_explored <= off.domains_explored,
            "gate ON must not explore more domains than OFF at threshold {threshold} \
             (ON={} OFF={})",
            on.domains_explored,
            off.domains_explored
        );
    }
}

/// Multi-objective graph loop: gate ON vs OFF must produce the identical
/// final verdict (disjunctive lane, both objectives must verify).
#[ntest::timeout(30000)]
#[test]
fn test_graph_multi_objective_gate_on_vs_off_identical_verdicts() {
    let graph = super::gpu_bab::simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = || BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 200,
        max_depth: 10,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let objectives = vec![vec![1.0f32], vec![-1.0f32]];

    for thresholds in [[-10.0f32, -10.0], [-0.5f32, -0.5]] {
        set_test_gate_override(Some(false));
        let off = BetaCrownVerifier::new(config())
            .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
            .expect("gate-off multi-objective verify must succeed");

        set_test_gate_override(Some(true));
        let on = BetaCrownVerifier::new(config())
            .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
            .expect("gate-on multi-objective verify must succeed");
        set_test_gate_override(None);

        assert_eq!(
            std::mem::discriminant(&off.result),
            std::mem::discriminant(&on.result),
            "multi-objective clause learning changed the verdict at thresholds {thresholds:?}: \
             OFF={:?} ON={:?}",
            off.result,
            on.result
        );
        if matches!(
            off.result,
            BabVerificationStatus::Verified | BabVerificationStatus::PotentialViolation { .. }
        ) {
            assert_eq!(off.result, on.result);
        }
    }
}

/// The production shared multi-objective clause source is lower-bound-only.
/// Even with both exact gates armed, an upper-bound-configured verifier must
/// fail closed at multi-objective ingress before ordinary store construction,
/// replay construction, source offers, proof attempts, or store mutation.
#[ntest::timeout(30000)]
#[test]
fn test_graph_multi_objective_upper_mode_has_zero_clause_authority() {
    let _gate_reset = ClauseGateReset;
    set_test_gate_override(Some(true));
    set_test_runtime_gate_override(Some(true));
    set_test_biccos_q_stage1_gate_override(Some(true));
    reset_test_runtime_observations();
    reset_test_store_mutations();

    let graph = super::gpu_bab::simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        batch_size: 4,
        max_domains: 200,
        max_depth: 10,
        timeout: Duration::from_secs(10),
        ..Default::default()
    });
    let objectives = vec![vec![1.0f32]];
    let thresholds = vec![1.0f32];

    let error = verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&NaiveCpuGemmEngine),
            None,
        )
        .expect_err("upper-mode multi-objective verification must fail closed");
    assert!(
        error
            .to_string()
            .contains("requires sign-normalized lower-bound objectives"),
        "unexpected upper-mode refusal: {error}"
    );

    let observations = test_runtime_observations();
    assert_eq!(observations.from_env_calls, 0);
    assert_eq!(observations.source_offers, 0);
    assert_eq!(observations.proof_attempts, 0);
    assert_eq!(
        observations.stage1_gate_reads, 0,
        "upper mode must refuse before the subordinate Stage-1 gate is read"
    );
    assert_eq!(observations.stage1_source_offers, 0);
    assert_eq!(observations.stage1_proof_attempts, 0);
    assert_eq!(
        test_store_mutations(),
        0,
        "upper-mode production caller must not mutate an ordinary or replay store"
    );
}

/// Exercise the production lower-bound source seam with all three exact gates.
///
/// Largest-bound-width deliberately branches irrelevant neuron 2 first because
/// its root interval is twice as wide. Neurons 0 and 1 are then fixed inactive,
/// closing a depth-three child. With β optimization disabled, Stage 0 retains
/// the semantically first two constraints and drops irrelevant neuron 2;
/// replaying that two-literal candidate proves `y=0 > -0.5` on the enlarged
/// region. This distinguishes the full offer → rank → replay → seal → store
/// path from direct runtime unit tests.
#[ntest::timeout(30000)]
#[test]
fn test_biccos_q_stage1_lower_mode_production_seam_accepts_replay() {
    let _gate_reset = ClauseGateReset;
    set_test_gate_override(Some(true));
    set_test_runtime_gate_override(Some(true));

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                None,
            )
            .expect("valid identity linear"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(
            LinearLayer::new(arr2(&[[-1.0, -1.0, 0.0]]), None).expect("valid output linear"),
        ),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-1.0, -1.0, -2.0]).into_dyn(),
        arr1(&[1.0, 1.0, 2.0]).into_dyn(),
    )
    .expect("valid input");
    let objectives = vec![vec![1.0f32]];
    let thresholds = vec![-0.5f32];
    let run = || {
        BetaCrownVerifier::new(BetaCrownConfig {
            verify_upper_bound: false,
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            branching_heuristic: BranchingHeuristic::LargestBoundWidth,
            beta_iterations: 0,
            batch_size: 4,
            max_domains: 32,
            max_depth: 6,
            timeout: Duration::from_secs(10),
            ..Default::default()
        })
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&NaiveCpuGemmEngine),
            None,
        )
        .expect("lower-mode fixture verification")
    };

    // Parent replay stays armed while Stage 1 is dark. The completed-child
    // source must still fall through to the established replay offer.
    set_test_biccos_q_stage1_gate_override(Some(false));
    reset_test_runtime_observations();
    reset_test_store_mutations();
    let stage1_off = run();
    let off_observations = test_runtime_observations();
    assert_eq!(off_observations.from_env_calls, 1);
    assert_eq!(off_observations.stage1_gate_reads, 1);
    assert_eq!(off_observations.stage1_source_offers, 0);
    assert_eq!(off_observations.stage1_proposals, 0);
    assert_eq!(off_observations.stage1_proof_attempts, 0);
    assert_eq!(off_observations.stage1_accepts, 0);
    assert!(
        off_observations.source_offers > 0,
        "Stage-1-off verified children must preserve established replay fallback"
    );

    set_test_biccos_q_stage1_gate_override(Some(true));
    reset_test_runtime_observations();
    reset_test_store_mutations();
    let stage1_on = run();
    let on_observations = test_runtime_observations();
    assert_eq!(on_observations.from_env_calls, 1);
    assert_eq!(on_observations.stage1_gate_reads, 1);
    assert!(on_observations.stage1_source_offers > 0);
    assert!(on_observations.stage1_proposals > 0);
    assert!(on_observations.stage1_proof_attempts > 0);
    assert!(
        on_observations.stage1_accepts > 0,
        "fixture must replay-certify and insert a ranked Stage-1 candidate: \
         {on_observations:?}"
    );
    assert!(
        test_store_mutations() > 0,
        "accepted Stage 1 must cross the production replay-token store boundary"
    );
    assert_eq!(
        stage1_off.result, stage1_on.result,
        "Stage 1 may prune only replay-certified regions"
    );
}

/// The Stage-0 observer is wired at the completed verified-child wave seam,
/// but it owns no solver-state handle. Exact gate ON vs OFF must therefore
/// preserve every result field except elapsed wall time, including bit-exact
/// output bounds, while the guarded observation counters prove the source
/// interception was genuinely exercised.
#[ntest::timeout(30000)]
#[test]
fn test_biccos_q_stage0_observes_verified_wave_without_solver_changes() {
    let _gate_reset = ClauseGateReset;
    set_test_gate_override(Some(false));
    set_test_runtime_gate_override(Some(false));

    // y = ReLU(x) - ReLU(-x) = x over [-1, 1]. The root lower bound does not
    // clear -0.5, while the x>=0 child does. This deterministically produces a
    // verified child and an unverified sibling in the shared executor wave.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0], [-1.0]]), None).expect("valid first linear")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, -1.0]]), None).expect("valid output linear")),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    let input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("valid input");
    let objectives = vec![vec![1.0f32]];
    let thresholds = vec![-0.5f32];
    let config = || BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        batch_size: 4,
        max_domains: 32,
        max_depth: 4,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let run = || {
        BetaCrownVerifier::new(config())
            .verify_graph_relu_split_multi_objective_with_engine(
                &graph,
                &input,
                &objectives,
                &thresholds,
                Some(&NaiveCpuGemmEngine),
                None,
            )
            .expect("fixture verification")
    };

    reset_biccos_q_stage0_observations();
    set_biccos_q_stage0_gate_override(Some(false));
    let off = run();
    let off_observations = biccos_q_stage0_observations();
    assert_eq!(off_observations.from_env_calls, 1);
    assert_eq!(off_observations.wave_observations, 0);
    assert_eq!(off_observations.source_offers, 0);

    reset_biccos_q_stage0_observations();
    set_biccos_q_stage0_gate_override(Some(true));
    let on = run();
    let on_observations = biccos_q_stage0_observations();
    assert_eq!(on_observations.from_env_calls, 1);
    assert!(on_observations.wave_observations > 0);
    assert!(
        on_observations.source_offers > 0 && on_observations.plans_emitted > 0,
        "fixture must reach the verified-child Stage-0 source seam: {on_observations:?}"
    );

    assert_eq!(off.result, on.result);
    assert_eq!(off.domains_explored, on.domains_explored);
    assert_eq!(off.max_depth_reached, on.max_depth_reached);
    assert_eq!(off.cuts_generated, on.cuts_generated);
    assert_eq!(off.domains_verified, on.domains_verified);
    let bound_bits = |result: &crate::beta_crown::BetaCrownResult| {
        result.output_bounds.as_ref().map(|bounds| {
            bounds
                .lower()
                .iter()
                .chain(bounds.upper().iter())
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(bound_bits(&off), bound_bits(&on));
}
