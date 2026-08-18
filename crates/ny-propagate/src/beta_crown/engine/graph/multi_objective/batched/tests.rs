// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU-batched BaB domain processing.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, arr2};
use ny_core::{GemmEngine, NaiveCpuGemmEngine, NyError};
use ny_tensor::BoundedTensor;

use super::children::collect_multi_objective_children;
use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::{
    GraphBabDomain, MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain, NodeBoundsMap,
};
use crate::beta_crown::engine::domain_results::{
    GraphDomainResult, MultiObjectiveGraphDomainResult,
};
use crate::beta_crown::engine::graph::adaptive_microbatch::MicrobatchRefusalReason;
use crate::beta_crown::engine::graph::domain_batch::{
    GraphDomainBatchExecutor, MultiObjectiveBatchRequest, SingleObjectiveBatchRequest,
};
use crate::beta_crown::engine::graph::multi_objective::batched::children::MultiObjectiveChildCreationResult;
use crate::beta_crown::engine::graph::multi_objective::selective_root_alpha::ChildContinuationStateProvenance;
use crate::beta_crown::BetaCrownConfig;
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

use super::super::super::super::BetaCrownVerifier;

fn build_single_linear_graph_4280() -> (GraphNetwork, Arc<BoundedTensor>) {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
        .expect("single-output linear layer should construct");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.set_output("linear1");

    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())
            .expect("positive input bounds should construct"),
    );

    (graph, input_bounds)
}

struct AllocationRefusingEngine;

impl GemmEngine for AllocationRefusingEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        Err(NyError::GpuMemoryExceeded {
            required_bytes: 2,
            budget_bytes: 1,
        })
    }
}

fn make_multi_objective_root_domain_4280(
    graph: &GraphNetwork,
    input_bounds: &Arc<BoundedTensor>,
    threshold: f32,
    seed_bounds: (f32, f32),
) -> MultiObjectiveGraphBabDomain {
    let node_bounds = graph
        .collect_node_bounds(input_bounds)
        .expect("single-linear graph bounds should collect");
    MultiObjectiveGraphBabDomain::root(
        node_bounds,
        vec![seed_bounds],
        input_bounds.as_ref(),
        &[threshold],
        false,
    )
    .expect("root multi-objective domain should construct")
}

#[test]
fn test_auto_enlarge_off_executor_setup_error_keeps_legacy_fallback_1993() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    graph.set_output("relu1");

    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input bounds should be valid"),
    );
    let initial_bounds = graph
        .collect_node_bounds(&input_bounds)
        .expect("node bounds should be collectable");
    let root = GraphBabDomain::root(initial_bounds, -1.0, 1.0, &input_bounds, false)
        .expect("root domain with finite bounds should not fail");

    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    verifier.config.alpha_config.deadline = None;
    assert!(!verifier.config.auto_enlarge_batch_size);
    let objective = vec![1.0];
    let relu_nodes = vec!["relu1".to_string(), "missing_relu".to_string()];

    // Includes one valid ReLU node to force child creation and one missing node
    // to force BatchedDomains setup failure.
    let results = GraphDomainBatchExecutor::execute_single_objective(
        &verifier,
        SingleObjectiveBatchRequest {
            graph: &graph,
            domains: &[&root],
            relu_nodes: &relu_nodes,
            objective: &objective,
            threshold: 0.0,
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            split_depth: 1,
            retry_refusals: false,
        },
    )
    .expect("legacy execution keeps its internal fallback");

    assert!(
        matches!(results.as_slice(), [GraphDomainResult::PropagationFailure]),
        "setup failure must map to PropagationFailure, got: {results:?}"
    );
}

#[test]
fn shared_single_objective_executor_caps_each_parent_at_max_depth() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");
    let input = BoundedTensor::new(
        arr1(&[-1.0, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0, 1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();
    let bounds = graph.collect_node_bounds(&input).unwrap();
    let root = GraphBabDomain::root(bounds, -1.0, 1.0, &input, false).unwrap();
    let mut deep = root.clone();
    deep.depth = 3;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        max_depth: 4,
        beta_iterations: 0,
        ..Default::default()
    });
    let domains = [&root, &deep];
    let relu_nodes = ["relu".to_string()];
    let results = GraphDomainBatchExecutor::execute_single_objective(
        &verifier,
        SingleObjectiveBatchRequest {
            graph: &graph,
            domains: &domains,
            relu_nodes: &relu_nodes,
            objective: &[1.0, 0.0, 0.0, 0.0],
            threshold: 0.0,
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            split_depth: 4,
            retry_refusals: false,
        },
    )
    .expect("legacy execution keeps its internal fallback");

    assert_eq!(results.len(), 2);
    for (parent, result) in domains.into_iter().zip(results) {
        let GraphDomainResult::Children(children) = result else {
            panic!("mixed-depth shared executor must produce children");
        };
        assert!(!children.is_empty());
        assert!(children
            .iter()
            .all(|(child, _)| child.depth <= verifier.config.max_depth));
        let expected_depth = verifier.config.max_depth.min(parent.depth + 4);
        assert!(children
            .iter()
            .all(|(child, _)| child.depth == expected_depth));
    }
}

#[test]
fn test_auto_enlarge_off_executor_never_surfaces_retry_refusal() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".into()],
    ));
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
        vec!["relu".into()],
    ));
    graph.set_output("output");
    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap(),
    );
    let root = GraphBabDomain::root(
        graph.collect_node_bounds(&input_bounds).unwrap(),
        0.0,
        1.0,
        input_bounds.as_ref(),
        false,
    )
    .unwrap();
    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    // This is an executor-unit refusal probe, not a wall-clock verifier run.
    // A live authoritative deadline deliberately selects the pollable CPU
    // linear-backward path, so clear it here to force the engine dispatch whose
    // structured allocation refusal the adaptive caller must observe.
    verifier.config.alpha_config.deadline = None;
    assert!(!verifier.config.auto_enlarge_batch_size);
    let results = GraphDomainBatchExecutor::execute_single_objective(
        &verifier,
        SingleObjectiveBatchRequest {
            graph: &graph,
            domains: &[&root],
            relu_nodes: &["relu".to_string()],
            objective: &[1.0],
            threshold: 0.5,
            engine: &AllocationRefusingEngine,
            cut_pool: None,
            split_depth: 1,
            retry_refusals: false,
        },
    )
    .expect("gate-off executor must retain its internal fallback/error mapping");
    assert_eq!(results.len(), 1);

    let adaptive = GraphDomainBatchExecutor::execute_single_objective(
        &verifier,
        SingleObjectiveBatchRequest {
            graph: &graph,
            domains: &[&root],
            relu_nodes: &["relu".to_string()],
            objective: &[1.0],
            threshold: 0.5,
            engine: &AllocationRefusingEngine,
            cut_pool: None,
            split_depth: 1,
            retry_refusals: true,
        },
    );
    assert!(
        matches!(adaptive, Err(MicrobatchRefusalReason::DeviceAllocation)),
        "adaptive executor must surface a retryable allocation refusal, got {adaptive:?}"
    );
}

#[test]
fn test_multi_objective_parent_lookup_failure_returns_propagation_failure_1993() {
    let domains_with_unstable: Vec<MultiObjDomainWithUnstable<'_>> = Vec::new();
    let child_creation_results: Vec<MultiObjectiveChildCreationResult> = vec![(7, Vec::new())];
    let mut quick_results = HashMap::new();

    let (children, parent_lookup) = collect_multi_objective_children(
        &domains_with_unstable,
        child_creation_results,
        &mut quick_results,
    );

    assert!(children.is_empty(), "no children should be collected");
    assert!(
        parent_lookup.is_empty(),
        "parent lookup should remain empty on lookup failure"
    );
    assert!(
        matches!(
            quick_results.get(&7),
            Some(MultiObjectiveGraphDomainResult::PropagationFailure)
        ),
        "missing parent lookup must map to PropagationFailure"
    );
}

#[test]
fn collect_multi_objective_children_moves_cache_payload_without_clone() {
    let (graph, input_bounds) = build_single_linear_graph_4280();
    let parent = make_multi_objective_root_domain_4280(&graph, &input_bounds, 0.5, (-1.0, 1.0));
    let mut child = parent.clone();
    let mut cache = CachedLinearBounds::default();
    cache
        .lower_a
        .insert("linear1".to_string(), arr2(&[[1.25_f32]]));
    cache
        .lower_b
        .insert("linear1".to_string(), arr1(&[-0.5_f32]));
    child
        .set_cached_las(vec![Some(cache)])
        .expect("one objective requires one cache slot");
    child.history.add_constraint(
        GraphNeuronConstraint::new(
            "collect_move_owned_history_allocation_probe".to_string(),
            0,
            true,
            1.0,
        )
        .expect("finite synthetic history constraint"),
    );
    let source_history_vec = child.history.constraints.as_ptr();
    let source_history_name = child.history.constraints[0].node_name().as_ptr();
    let source_payload = child.cached_las()[0]
        .as_ref()
        .expect("source cache should exist")
        .lower_a["linear1"]
        .as_ptr();
    let expected_a_bits = child.cached_las()[0]
        .as_ref()
        .expect("source cache should exist")
        .lower_a["linear1"][[0, 0]]
    .to_bits();
    let expected_b_bits = child.cached_las()[0]
        .as_ref()
        .expect("source cache should exist")
        .lower_b["linear1"][0]
        .to_bits();

    let domains_with_unstable: Vec<MultiObjDomainWithUnstable<'_>> = vec![(0, &parent, Vec::new())];
    let child_creation_results: Vec<MultiObjectiveChildCreationResult> =
        vec![(0, vec![(0, child, true, Default::default())])];
    let mut quick_results = HashMap::new();

    let (children, parent_lookup) = collect_multi_objective_children(
        &domains_with_unstable,
        child_creation_results,
        &mut quick_results,
    );

    assert!(quick_results.is_empty());
    let looked_up_parent = *parent_lookup.get(&0).expect("parent lookup should exist");
    assert_eq!(
        std::ptr::from_ref(looked_up_parent),
        std::ptr::from_ref(&parent)
    );
    assert_eq!(children.len(), 1);
    let moved_child = &children[0].1;
    assert_eq!(moved_child.history.constraints.as_ptr(), source_history_vec);
    assert_eq!(
        moved_child.history.constraints[0].node_name().as_ptr(),
        source_history_name,
        "collection must move the domain rather than deep-clone owned history"
    );
    let moved_cache = moved_child.cached_las()[0]
        .as_ref()
        .expect("moved cache should exist");
    assert_eq!(moved_cache.lower_a["linear1"].as_ptr(), source_payload);
    assert_eq!(
        moved_cache.lower_a["linear1"][[0, 0]].to_bits(),
        expected_a_bits
    );
    assert_eq!(moved_cache.lower_b["linear1"][0].to_bits(), expected_b_bits);
}

#[test]
fn test_execute_multi_objective_no_unstable_recomputes_verified_bounds_4280() {
    let (graph, input_bounds) = build_single_linear_graph_4280();
    let domain = make_multi_objective_root_domain_4280(&graph, &input_bounds, 0.5, (0.0, 1.0));
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let relu_nodes: Vec<String> = Vec::new();
    let objectives = vec![vec![1.0_f32]];
    let thresholds = vec![0.5_f32];

    let results = GraphDomainBatchExecutor::execute_multi_objective(
        &verifier,
        MultiObjectiveBatchRequest {
            bab_round: 0,
            graph: &graph,
            domains: &[&domain],
            relu_nodes: &relu_nodes,
            objectives: &objectives,
            thresholds: &thresholds,
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            selective_root_alpha_candidate: None,
        },
    );

    assert!(
        matches!(
            results.as_slice(),
            [MultiObjectiveGraphDomainResult::NoUnstable {
                all_verified: true,
                any_violated: false,
            }]
        ),
        "no-unstable executor path should recompute tight verified bounds, got {results:?}"
    );
}

#[test]
fn test_execute_multi_objective_no_unstable_recomputes_violated_bounds_4280() {
    let (graph, input_bounds) = build_single_linear_graph_4280();
    let domain = make_multi_objective_root_domain_4280(&graph, &input_bounds, 3.0, (2.0, 4.0));
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let relu_nodes: Vec<String> = Vec::new();
    let objectives = vec![vec![1.0_f32]];
    let thresholds = vec![3.0_f32];

    let results = GraphDomainBatchExecutor::execute_multi_objective(
        &verifier,
        MultiObjectiveBatchRequest {
            bab_round: 0,
            graph: &graph,
            domains: &[&domain],
            relu_nodes: &relu_nodes,
            objectives: &objectives,
            thresholds: &thresholds,
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            selective_root_alpha_candidate: None,
        },
    );

    assert!(
        matches!(
            results.as_slice(),
            [MultiObjectiveGraphDomainResult::NoUnstable {
                all_verified: false,
                any_violated: true,
            }]
        ),
        "no-unstable executor path should preserve violation detection, got {results:?}"
    );
}

#[test]
fn test_execute_multi_objective_no_unstable_preserves_deadline_outcome() {
    let (graph, input_bounds) = build_single_linear_graph_4280();
    let domain = make_multi_objective_root_domain_4280(&graph, &input_bounds, 0.5, (0.0, 1.0));
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: std::time::Duration::ZERO,
        ..Default::default()
    });
    let results = GraphDomainBatchExecutor::execute_multi_objective(
        &verifier,
        MultiObjectiveBatchRequest {
            bab_round: 0,
            graph: &graph,
            domains: &[&domain],
            relu_nodes: &[],
            objectives: &[vec![1.0_f32]],
            thresholds: &[0.5_f32],
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            selective_root_alpha_candidate: None,
        },
    );

    assert!(
        matches!(
            results.as_slice(),
            [MultiObjectiveGraphDomainResult::DeadlineExpired]
        ),
        "NoUnstable CROWN must preserve the typed deadline, got {results:?}"
    );
}

#[test]
fn test_execute_multi_objective_expired_branch_wave_is_typed_deadline() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    graph.set_output("relu1");
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input");
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("root bounds should collect");
    let domain =
        MultiObjectiveGraphBabDomain::root(node_bounds, vec![(-1.0, 1.0)], &input, &[0.0], false)
            .expect("root multi-objective domain");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: std::time::Duration::ZERO,
        ..Default::default()
    });
    let results = GraphDomainBatchExecutor::execute_multi_objective(
        &verifier,
        MultiObjectiveBatchRequest {
            bab_round: 0,
            graph: &graph,
            domains: &[&domain],
            relu_nodes: &["relu1".to_string()],
            objectives: &[vec![1.0_f32]],
            thresholds: &[0.0_f32],
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            selective_root_alpha_candidate: None,
        },
    );

    assert!(
        matches!(
            results.as_slice(),
            [MultiObjectiveGraphDomainResult::DeadlineExpired]
        ),
        "expired branch creation must not become PropagationFailure, got {results:?}"
    );
}

// ---------------------------------------------------------------------------
// Domain-batched single-pass adapter: SOUNDNESS gate.
//
// Adversarial check of the new dense-spec domain-batched adapter over children
// with distinct split histories / base bounds / partial verified masks. See the
// "SOUNDNESS FINDING" note below for why we gate on (1) batch-vs-batch-of-1
// equivalence + verified-latch and (2) an independent concrete-sampling
// soundness floor, rather than bit-equivalence to the looser per-child path.
// ---------------------------------------------------------------------------

/// Build a multi-ReLU graph with a 2-dim output (2 objectives possible).
///
/// 2 inputs -> linear1(4) -> relu1 -> linear2(2) -> relu2 -> linear3(2).
fn build_equiv_parity_graph() -> GraphNetwork {
    let w1 = arr2(&[[1.2_f32, -0.8], [-0.6, 1.1], [0.9, 0.7], [-0.7, 0.4]]);
    let b1 = arr1(&[0.1_f32, -0.05, 0.0, 0.12]);
    let w2 = arr2(&[[0.8_f32, -0.5, 0.6, -0.2], [-0.3, 0.9, -0.4, 0.7]]);
    let b2 = arr1(&[0.05_f32, -0.08]);
    let w3 = arr2(&[[1.0_f32, -0.2], [-0.4, 0.9]]);
    let b3 = arr1(&[0.02_f32, -0.03]);

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

/// Build a root multi-objective domain over the two objectives `Y_0` and `Y_1`.
fn build_equiv_root(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> MultiObjectiveGraphBabDomain {
    let node_bounds = graph
        .collect_node_bounds(input)
        .expect("root node bounds should collect");
    // Seed the root with a certified parent enclosure projected directly from
    // the already-collected output IBP bounds. The fixture only needs sound,
    // non-vacuous parent values for the verified latch; running constrained
    // CROWN here would make setup depend on the finite dense-ReLU policy that
    // the selective-wrapper test exercises separately.
    let output = node_bounds
        .get(graph.output_node.as_str())
        .expect("root output bounds should be present");
    let obj_bounds = BetaCrownVerifier::objective_bounds_multi(output, objectives)
        .expect("root objective bounds should compute");
    MultiObjectiveGraphBabDomain::root(node_bounds, obj_bounds, input, thresholds, false)
        .expect("root multi-objective domain should construct")
}

// SOUNDNESS FINDING (documented for reviewers).
//
// The plan assumed the per-child single-pass path
// (`propagate_multi_objective_with_beta_and_cache` -> `backward_crown_constrained`)
// and the dense-spec batched primitive
// (`propagate_crown_with_batched_domains_full_specs` -> the batched backward core)
// compute identical bounds modulo GEMM reassociation. They do NOT: they are two
// different CROWN backward implementations and diverge for non-empty split
// histories (and whenever per-domain alpha state is present). Empirically the
// batched core matches the canonical direct CROWN oracle
// (`propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline`) while the
// per-child `backward_crown_constrained` is strictly LOOSER. Both are sound
// (their lower bounds are valid lower bounds), so swapping in the batched
// primitive is a sound change; it is just not bit-equivalent to the old path.
//
// The gate below therefore proves the two properties that actually matter:
//   (1) ADAPTER FAITHFULNESS + NO CROSS-DOMAIN CONTAMINATION — batching N
//       children together yields the SAME per-child bounds as running each child
//       through the adapter individually (batch of 1), and the verified-latch
//       keeps parent bounds for already-verified objectives.
//   (2) INDEPENDENT SOUNDNESS — every batched lower bound is <= the concrete
//       sampled true minimum over the child's sub-domain (the disqualifying-
//       failure guard for verify_upper=false).

/// Run the adapter over a single child (batch of 1) and return its merged
/// per-objective bounds. Used as the per-domain isolation reference.
fn adapter_single_child(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    child: &MultiObjectiveGraphBabDomain,
    relu_nodes: &[String],
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Vec<(f32, f32)> {
    let out = verifier
        .batched_single_pass_multi_objective_children(
            graph,
            &[child],
            relu_nodes,
            objectives,
            thresholds,
            &NaiveCpuGemmEngine,
            false,
        )
        .expect("single-child adapter should not fall back");
    out[0]
        .as_ref()
        .map(|(obj_bounds, _, _, _, _, _, _)| obj_bounds.clone())
        .unwrap_or_else(|e| panic!("single-child adapter errored: {e:?}"))
}

#[test]
fn selective_wrapper_off_and_private_w_cutoff_preserve_h_but_hard_deadline_is_terminal() {
    let graph = build_equiv_parity_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid input");
    let objectives = vec![vec![1.0_f32, -0.35], vec![-0.6, 1.0]];
    let thresholds = vec![0.0_f32, 0.0];
    let relu_nodes = vec!["relu1".to_string(), "relu2".to_string()];
    let root = build_equiv_root(&graph, &input, &objectives, &thresholds);
    let child = root
        .with_constraint(
            &graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active: true,
                score: 1.0,
            },
            false,
            &thresholds,
        )
        .expect("child construction should not error")
        .expect("child should be feasible");
    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    // H and the gate-off wrapper are the no-deadline reference. The two calls
    // below that pass an authoritative deadline still install their own live
    // or expired finite scope and therefore exercise refusal/expiry policy.
    verifier.config.alpha_config.deadline = None;

    let established = verifier
        .batched_single_pass_multi_objective_children(
            &graph,
            &[&child],
            &relu_nodes,
            &objectives,
            &thresholds,
            &NaiveCpuGemmEngine,
            false,
        )
        .expect("H should evaluate");
    let off = verifier
        .batched_selective_root_alpha_multi_objective_children(
            &graph,
            &[&child],
            &relu_nodes,
            &objectives,
            &thresholds,
            &NaiveCpuGemmEngine,
            false,
            None,
            None,
        )
        .expect("gate-off wrapper should evaluate H");
    super::batched_dense_specs::reset_batched_single_pass_dispatch_count_for_test();
    let private_cutoff_expired = verifier
        .batched_selective_root_alpha_multi_objective_children(
            &graph,
            &[&child],
            &relu_nodes,
            &objectives,
            &thresholds,
            &NaiveCpuGemmEngine,
            false,
            Some(root.alpha_state()),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(4)),
        )
        .expect("private W cutoff must retain completed H");
    assert_eq!(
        super::batched_dense_specs::batched_single_pass_dispatch_count_for_test(),
        1,
        "private cutoff must run H exactly once and decline W before dispatch"
    );
    super::batched_dense_specs::reset_batched_single_pass_dispatch_count_for_test();
    let hard_expired = verifier.batched_selective_root_alpha_multi_objective_children(
        &graph,
        &[&child],
        &relu_nodes,
        &objectives,
        &thresholds,
        &NaiveCpuGemmEngine,
        false,
        Some(root.alpha_state()),
        Some(std::time::Instant::now()),
    );
    assert!(
        matches!(
            hard_expired,
            Err(super::batched_dense_specs::BatchedMultiObjectiveAdapterError::DeadlineExpired)
        ),
        "optional W must not mask the verifier's hard deadline"
    );
    assert_eq!(
        super::batched_dense_specs::batched_single_pass_dispatch_count_for_test(),
        0,
        "an expired hard authority must refuse before launching H or W"
    );

    let extract = |results: &[super::batched_dense_specs::BatchedChildResult]| {
        let result = results[0].as_ref().expect("child result should be sound");
        assert_eq!(
            result.6,
            ChildContinuationStateProvenance::Established,
            "optional W refusal must retain H provenance"
        );
        result
            .0
            .iter()
            .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect::<Vec<_>>()
    };
    let established_bits = extract(&established);
    assert_eq!(extract(&off), established_bits, "gate-off H bounds drifted");
    assert_eq!(
        extract(&private_cutoff_expired),
        established_bits,
        "private-cutoff-cancelled W changed completed H bounds"
    );
}

/// Independent soundness floor: sample concrete points in the child's constrained
/// input box, evaluate the network exactly (degenerate-box IBP is exact at a
/// point), enforce the child's ReLU split constraints at each point, and return
/// the per-objective minimum observed value — a valid upper estimate of the true
/// minimum over the sub-domain (any sound lower bound must be <= this).
fn sampled_objective_minimums(
    graph: &GraphNetwork,
    parent_node_bounds: &NodeBoundsMap,
    child: &MultiObjectiveGraphBabDomain,
    objectives: &[Vec<f32>],
) -> Vec<f32> {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (_fwd, constrained_input) = verifier
        .compute_constrained_forward_bounds_from_view(
            graph,
            child.input_bounds.as_ref(),
            &child.history,
            Some(parent_node_bounds.into()),
            None,
        )
        .expect("constrained forward bounds should succeed");
    let ci = constrained_input.flatten();
    let lo = ci.lower();
    let hi = ci.upper();
    let n = lo.len();
    let steps = 7usize;
    let total = steps.pow(n as u32);
    let mut mins = vec![f32::INFINITY; objectives.len()];
    let output_node = graph.output_node.as_str();
    for combo in 0..total {
        let mut pt = Vec::with_capacity(n);
        let mut c = combo;
        for d in 0..n {
            let t = (c % steps) as f32 / (steps - 1) as f32;
            c /= steps;
            pt.push(lo[[d]] + t * (hi[[d]] - lo[[d]]));
        }
        let point = BoundedTensor::new(
            ndarray::Array1::from(pt.clone()).into_dyn(),
            ndarray::Array1::from(pt).into_dyn(),
        )
        .expect("degenerate sample box should construct");
        let nb = graph
            .collect_node_bounds(&point)
            .expect("concrete forward should succeed");

        // Enforce ALL of the child's ReLU split constraints at this concrete
        // point. A split constraint is on the relu node's PRE-ACTIVATION (its
        // first input): active => preact[idx] >= 0, inactive => preact[idx] <= 0.
        // Points violating any constraint are outside this child's sub-domain and
        // must not contribute to the true-minimum estimate (#soundness).
        let mut in_domain = true;
        for constraint in &child.history.constraints {
            let relu_node = graph
                .nodes
                .get(&constraint.node_name)
                .expect("constrained relu node present");
            let pre_name = relu_node.inputs.first().expect("relu has an input");
            let preact = if pre_name == crate::NETWORK_INPUT {
                point.flatten()
            } else {
                nb.get(pre_name).expect("pre-activation present").flatten()
            };
            let v = preact.lower()[[constraint.neuron_idx]];
            if (constraint.is_active && v < 0.0) || (!constraint.is_active && v > 0.0) {
                in_domain = false;
                break;
            }
        }
        if !in_domain {
            continue;
        }

        let out = nb
            .get(output_node)
            .expect("output node bounds present")
            .flatten();
        for (oi, obj) in objectives.iter().enumerate() {
            let mut v = 0.0_f32;
            for (k, &coeff) in obj.iter().enumerate() {
                v += coeff * out.lower()[[k]];
            }
            if v < mins[oi] {
                mins[oi] = v;
            }
        }
    }
    mins
}

#[test]
fn test_dense_spec_adapter_matches_direct_crown_oracle_and_is_sound() {
    let graph = build_equiv_parity_graph();
    let objectives = vec![vec![1.0_f32, -0.35], vec![-0.6, 1.0]];
    let thresholds = vec![0.0_f32, 0.0];
    let relu_nodes = vec!["relu1".to_string(), "relu2".to_string()];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    // Two distinct roots with DIFFERENT input bounds => distinct base node_bounds.
    let input_a = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid input_a");
    let input_b = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.8]).into_dyn(),
        arr1(&[0.9_f32, 0.6]).into_dyn(),
    )
    .expect("valid input_b");
    let root_a = build_equiv_root(&graph, &input_a, &objectives, &thresholds);
    let root_b = build_equiv_root(&graph, &input_b, &objectives, &thresholds);

    // Children with DISTINCT split histories (=> non-empty beta_state):
    //   child0: root_a, relu1[0] active
    //   child1: root_a, relu1[1] inactive
    //   child2: root_b, relu2[0] active  (different base bounds)
    let make_child = |root: &MultiObjectiveGraphBabDomain,
                      node: &str,
                      idx: usize,
                      active: bool|
     -> MultiObjectiveGraphBabDomain {
        let constraint = GraphNeuronConstraint {
            node_name: node.to_string(),
            neuron_idx: idx,
            is_active: active,
            score: 1.0,
        };
        root.with_constraint(&graph, constraint, false, &thresholds)
            .expect("with_constraint should not error")
            .expect("child should be feasible")
    };

    let mut child0 = make_child(&root_a, "relu1", 0, true);
    let child1 = make_child(&root_a, "relu1", 1, false);
    let mut child2 = make_child(&root_b, "relu2", 0, true);

    // Sanity: non-empty beta_state from the split history.
    assert!(
        !child0.beta_state.is_empty(),
        "child0 must carry a non-empty beta_state from its split history"
    );

    // Give one child a PARTIALLY-VERIFIED mask: objective 0 verified, with a
    // distinctive parent bound that the verified-latch must preserve verbatim.
    child0.verified = vec![true, false];
    child0.objective_bounds[0] = (7.0, 8.0);

    // And another partially-verified child latching objective 1.
    child2.verified = vec![false, true];
    child2.objective_bounds[1] = (-9.0, -8.0);

    let children = [&child0, &child1, &child2];
    let parents = [
        &root_a.node_bounds,
        &root_a.node_bounds,
        &root_b.node_bounds,
    ];

    // REFERENCE A (per-domain isolation): run each child through the adapter
    // INDIVIDUALLY (batch of 1). Batching N children together must reproduce
    // these element-wise — the core adversarial test for cross-domain
    // contamination / mis-indexing introduced by the optimization.
    let isolated: Vec<Vec<(f32, f32)>> = children
        .iter()
        .map(|child| {
            adapter_single_child(
                &verifier,
                &graph,
                child,
                &relu_nodes,
                &objectives,
                &thresholds,
            )
        })
        .collect();

    // REFERENCE B (independent soundness floor): concrete sampling per child/obj.
    let sampled_mins: Vec<Vec<f32>> = children
        .iter()
        .zip(parents.iter())
        .map(|(child, parent_nb)| sampled_objective_minimums(&graph, parent_nb, child, &objectives))
        .collect();

    // ACTUAL: the new domain-batched adapter over ALL children at once.
    let batchable: Vec<&MultiObjectiveGraphBabDomain> = children.to_vec();
    let actual = verifier
        .batched_single_pass_multi_objective_children(
            &graph,
            &batchable,
            &relu_nodes,
            &objectives,
            &thresholds,
            &NaiveCpuGemmEngine,
            false,
        )
        .expect("adapter should not request whole-batch fallback for this batch");

    assert_eq!(actual.len(), children.len(), "one result per child");

    const TOL: f32 = 1e-5;
    for (ci, ((child_result, iso_bounds), child)) in actual
        .iter()
        .zip(isolated.iter())
        .zip(children.iter())
        .enumerate()
    {
        let (obj_bounds, _node_cache, _beta, _alpha, _cached_las, _pruned, _provenance) =
            child_result
                .as_ref()
                .unwrap_or_else(|e| panic!("child {ci} adapter result errored: {e:?}"));
        assert_eq!(
            obj_bounds.len(),
            iso_bounds.len(),
            "child {ci}: objective count mismatch"
        );
        for (oi, ((batched_l, batched_u), (iso_l, iso_u))) in
            obj_bounds.iter().zip(iso_bounds.iter()).enumerate()
        {
            // (1) NO CROSS-DOMAIN CONTAMINATION: all-at-once batch must equal the
            // batch-of-1 result for this child/objective (covers both the active
            // CROWN rows and the verified-latch slots).
            assert!(
                (batched_l - iso_l).abs() <= TOL && (batched_u - iso_u).abs() <= TOL,
                "child {ci} obj {oi}: all-at-once batch diverged from batch-of-1 \
                 (cross-domain contamination?): batched=({batched_l},{batched_u}) \
                 isolated=({iso_l},{iso_u})"
            );

            // (2) VERIFIED-LATCH: for already-verified objectives the adapter must
            // keep the parent's bound verbatim (no fresh CROWN).
            if child.verified[oi] {
                let (parent_l, parent_u) = child.objective_bounds[oi];
                assert!(
                    (batched_l - parent_l).abs() <= TOL && (batched_u - parent_u).abs() <= TOL,
                    "child {ci} obj {oi} (verified-latch): adapter must keep parent bound \
                     ({parent_l},{parent_u}), got ({batched_l},{batched_u})"
                );
                continue;
            }

            // (3) ONE-SIDED SOUNDNESS guard (the disqualifying-failure guard for
            // verify_upper=false => lower drives 'verified'). The batched lower
            // bound MUST NOT exceed the concrete sampled true minimum. A lower
            // bound above the true value would falsely verify. MUST fail loudly.
            let true_min = sampled_mins[ci][oi];
            assert!(
                *batched_l <= true_min + TOL,
                "UNSOUND: child {ci} obj {oi}: batched lower {batched_l} exceeds sampled \
                 true minimum {true_min} (+{TOL})"
            );
        }
    }
}

/// #w5-bab-throughput: union spec pruning parity. With `prune_specs_to_union`
/// the adapter seeds the dense backward with only the union of unverified
/// objective rows; CROWN backward is row-independent, so every child's merged
/// per-objective bounds must be IDENTICAL to the full-matrix call — verified
/// entries keep the parent's latched bounds verbatim, active entries get the
/// same fresh rows.
#[test]
fn test_dense_spec_adapter_union_pruning_matches_full_matrix_w5() {
    let graph = build_equiv_parity_graph();
    let objectives = vec![vec![1.0_f32, -0.35], vec![-0.6, 1.0]];
    let thresholds = vec![0.0_f32, 0.0];
    let relu_nodes = vec!["relu1".to_string(), "relu2".to_string()];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid input");
    let root = build_equiv_root(&graph, &input, &objectives, &thresholds);

    let constraint = GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    };
    let mut child = root
        .with_constraint(&graph, constraint, false, &thresholds)
        .expect("with_constraint should not error")
        .expect("child should be feasible");

    // Objective 0 verified with a distinctive latched parent bound: the union
    // spec matrix must exclude its row AND the merge must preserve it verbatim.
    child.verified = vec![true, false];
    child.objective_bounds[0] = (7.0, 8.0);

    let batch = [&child];
    let full = verifier
        .batched_single_pass_multi_objective_children(
            &graph,
            &batch,
            &relu_nodes,
            &objectives,
            &thresholds,
            &NaiveCpuGemmEngine,
            false,
        )
        .expect("full-matrix adapter call should not fall back");
    let pruned = verifier
        .batched_single_pass_multi_objective_children(
            &graph,
            &batch,
            &relu_nodes,
            &objectives,
            &thresholds,
            &NaiveCpuGemmEngine,
            true,
        )
        .expect("union-pruned adapter call should not fall back");

    let full_bounds = full[0]
        .as_ref()
        .map(|(b, _, _, _, _, _, _)| b.clone())
        .expect("full-matrix child result should be Ok");
    let pruned_bounds = pruned[0]
        .as_ref()
        .map(|(b, _, _, _, _, _, _)| b.clone())
        .expect("union-pruned child result should be Ok");

    assert_eq!(
        full_bounds, pruned_bounds,
        "#w5: union pruning must reproduce the full-matrix per-objective bounds exactly"
    );
    assert_eq!(
        pruned_bounds[0],
        (7.0, 8.0),
        "#w5: verified-latch must keep the parent's bound verbatim under union pruning"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// #kfsb-multi (barrier 2): wave-batched kFSB branch selection tests.
// ═════════════════════════════════════════════════════════════════════════════

mod kfsb_multi_tests {
    use std::collections::HashMap;
    use std::mem::size_of_val;
    use std::sync::Arc;

    use ndarray::{arr1, arr2};
    use ny_core::NaiveCpuGemmEngine;

    use super::super::batched_multi::with_kfsb_final_publication_now;
    use super::super::children::KfsbCertEffect;
    use super::super::kfsb_multi::{
        adaptive_depth_authority_identity, adaptive_depth_proxy_recommended_rank,
        adaptive_depth_shadow_budget_available, adaptive_depth_shadow_deadline,
        append_layer_quota_candidates, claim_adaptive_depth_attempt, clear_shadow_cached_las,
        complete_clip_decision_scoring_deadline, depth_two_lookahead_score,
        kfsb_f64_shadow_budget_available, kfsb_f64_shadow_deadline, kfsb_f64_shadow_objective,
        materialize_kfsb_candidates_with_completeness, pick_kfsb_candidate,
        pick_kfsb_candidate_subset_original_order, rank_adaptive_depth_candidates,
        rank_kfsb_candidate_portfolio, resolve_adaptive_depth_authority_candidate,
        resolve_adaptive_depth_commit_enabled, resolve_adaptive_depth_select_enabled,
        resolve_adaptive_depth_shadow_enabled, resolve_kfsb_cached_la_enabled,
        resolve_kfsb_f64_shadow_enabled, select_complete_depth_two_lookahead,
        select_depth_two_frontier_worst_slot, select_depth_two_root_portfolio,
        select_kfsb_straggler, AdaptiveDepthPrivatePeakDecline, AdaptiveDepthPrivatePeakLedger,
        AdaptiveDepthShadowCapture, DepthTwoLookaheadBudget, DepthTwoLookaheadCapture,
        DepthTwoLookaheadOverlayPlan, DepthTwoLookaheadSideScore, DomainPrep, KfsbF64ShadowCapture,
        SideSlot,
    };
    use crate::batched_domain::CachedLinearBounds;
    use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
    use crate::beta_crown::config::{
        DepthTwoBranchLookaheadConfig, DepthTwoBranchLookaheadMode, KfsbReduceOp,
    };
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::branching::kfsb_shared::GraphKfsbCandidate;
    use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
    use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier};
    use crate::{
        BoundedTensor, GraphNetwork, GraphNode, Layer, LinearBounds, LinearLayer, ReLULayer,
    };

    /// input x in [-1,1] -> linear1: (n0 = x  [unstable], n1 = x + 3 [stable
    /// active]) -> relu1 -> linear2: sum -> output. Splitting n0 tightens both
    /// children (active: out = 2x+3 >= 3; inactive: out = x+3 >= 2) while
    /// "splitting" the stable n1 leaves the active child at the root bound and
    /// makes the inactive child INFEASIBLE (l = 2 > 0).
    fn kfsb_fixture() -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        let linear1 =
            LinearLayer::new(arr2(&[[1.0], [1.0]]), Some(arr1(&[0.0, 3.0]))).expect("linear1");
        let linear2 = LinearLayer::new(arr2(&[[1.0, 1.0]]), None).expect("linear2");
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

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input bounds");
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        // Threshold 0.0 with a wide seed keeps objective 0 unverified and
        // unviolated, so the selector sees exactly one straggler.
        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-10.0, 10.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("root domain");
        (graph, domain)
    }

    /// Two-row variant whose first row is reusable but not terminal.
    ///
    /// Both rows describe the same scalar output. Row 0 has the worse root
    /// proof margin and is therefore the KFSB straggler; either side of the
    /// useful split proves it above zero. Row 1's threshold is deliberately
    /// unreachable by a lower-bound proof (the two child minima are 3 and 2),
    /// so a sound certificate can only produce `RowVerified`, never
    /// `ChildComplete`.
    fn kfsb_partial_receipt_fixture() -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        let (graph, seed) = kfsb_fixture();
        let input = seed.input_bounds();
        let node_bounds = graph.collect_node_bounds(input).expect("node bounds");
        let mut domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-10.0, 10.0), (-5.0, 10.0)],
            input,
            &[0.0, 4.0],
            false,
        )
        .expect("partial-receipt root domain");
        let mut row_zero = CachedLinearBounds::default();
        row_zero
            .lower_a
            .insert("relu1".to_string(), arr2(&[[1.0_f32, 1.0_f32]]));
        row_zero
            .upper_a
            .insert("relu1".to_string(), arr2(&[[1.0_f32, 1.0_f32]]));
        row_zero
            .lower_b
            .insert("relu1".to_string(), arr1(&[0.0_f32]));
        row_zero
            .upper_b
            .insert("relu1".to_string(), arr1(&[0.0_f32]));
        let mut row_one = CachedLinearBounds::default();
        row_one
            .lower_a
            .insert("relu1".to_string(), arr2(&[[1.0_f32, 1.0_f32]]));
        row_one
            .upper_a
            .insert("relu1".to_string(), arr2(&[[1.0_f32, 1.0_f32]]));
        row_one
            .lower_b
            .insert("relu1".to_string(), arr1(&[0.0_f32]));
        row_one
            .upper_b
            .insert("relu1".to_string(), arr1(&[0.0_f32]));
        domain
            .set_cached_las(vec![Some(row_zero), Some(row_one)])
            .expect("two objectives require two full-spec cache slots");
        (graph, domain)
    }

    /// Three genuinely unstable candidates for the adaptive-depth observer.
    fn adaptive_depth_fixture() -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        let linear1 = LinearLayer::new(
            arr2(&[[1.0_f32], [-1.0], [0.5]]),
            Some(arr1(&[0.0_f32, 0.0, 0.0])),
        )
        .expect("linear1");
        // Both non-root candidates have negative objective coefficients, so
        // either can carry BaBSR lower-bound relaxation loss. This lets the
        // child-specific interval cache, rather than a fixed coefficient sign,
        // decide the second split.
        let linear2 = LinearLayer::new(arr2(&[[1.0_f32, -0.75, -0.4]]), None).expect("linear2");
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

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input bounds");
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-10.0, 10.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("root domain");
        (graph, domain)
    }

    /// Four unstable candidates exercise the production depth-four wave cap.
    fn adaptive_depth_four_candidate_fixture() -> (GraphNetwork, MultiObjectiveGraphBabDomain) {
        let linear1 = LinearLayer::new(
            arr2(&[[1.0_f32], [-1.0], [0.5], [-0.5]]),
            Some(arr1(&[0.0_f32, 0.0, 0.0, 0.0])),
        )
        .expect("linear1");
        let linear2 =
            LinearLayer::new(arr2(&[[1.0_f32, -0.75, -0.4, -0.2]]), None).expect("linear2");
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

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input bounds");
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-10.0, 10.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("root domain");
        (graph, domain)
    }

    fn kfsb_verifier(reduce_op: KfsbReduceOp) -> BetaCrownVerifier {
        BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::Kfsb,
            fsb_candidates: 2,
            kfsb_reduce_op: reduce_op,
            beta_iterations: 0,
            ..Default::default()
        })
    }

    #[test]
    fn kfsb_cached_la_gate_is_exact_and_default_off() {
        assert!(!resolve_kfsb_cached_la_enabled(None));
        assert!(!resolve_kfsb_cached_la_enabled(Some("")));
        assert!(!resolve_kfsb_cached_la_enabled(Some("0")));
        assert!(!resolve_kfsb_cached_la_enabled(Some("true")));
        assert!(resolve_kfsb_cached_la_enabled(Some("1")));
    }

    #[test]
    fn kfsb_f64_shadow_gate_and_budget_fail_closed() {
        assert!(!resolve_kfsb_f64_shadow_enabled(None));
        assert!(!resolve_kfsb_f64_shadow_enabled(Some("")));
        assert!(!resolve_kfsb_f64_shadow_enabled(Some("0")));
        assert!(!resolve_kfsb_f64_shadow_enabled(Some("true")));
        assert!(resolve_kfsb_f64_shadow_enabled(Some("1")));
        let objective = [1.0_f32, -2.0, 0.0];
        assert!(matches!(
            kfsb_f64_shadow_objective(&objective, false),
            std::borrow::Cow::Borrowed(_)
        ));
        let negated = kfsb_f64_shadow_objective(&objective, true);
        assert_eq!(
            negated
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [-1.0_f32, 2.0, -0.0]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "verify-upper shadow must score lower(-c), exactly matching -upper(c)"
        );

        let now = std::time::Instant::now();
        assert!(kfsb_f64_shadow_deadline(now, None).is_some());
        assert!(
            kfsb_f64_shadow_deadline(now, now.checked_add(std::time::Duration::from_secs(14)))
                .is_none()
        );
        let admitted =
            kfsb_f64_shadow_deadline(now, now.checked_add(std::time::Duration::from_secs(16)))
                .expect("five-second shadow plus ten-second reserve");
        assert!(kfsb_f64_shadow_budget_available(
            now,
            admitted,
            now.checked_add(std::time::Duration::from_secs(16))
        ));
        assert!(!kfsb_f64_shadow_budget_available(
            admitted,
            admitted,
            now.checked_add(std::time::Duration::from_secs(16))
        ));
    }

    #[test]
    fn kfsb_f64_shadow_respects_earlier_call_scoped_authority() {
        let now = std::time::Instant::now();
        let configured_deadline = now.checked_add(std::time::Duration::from_mins(1));
        let call_scoped_deadline = now.checked_add(std::time::Duration::from_secs(14));

        assert!(
            kfsb_f64_shadow_deadline(now, configured_deadline).is_some(),
            "the looser configured deadline would admit the observer"
        );
        assert!(
            kfsb_f64_shadow_deadline(now, call_scoped_deadline).is_none(),
            "the effective call-scoped deadline must preserve post-BaB authority"
        );
    }

    #[test]
    fn kfsb_f64_shadow_streams_exact_post_f32_top_three() {
        let candidate = |node: &str, idx: usize, main: f32| GraphKfsbCandidate {
            node_name: node.to_string(),
            neuron_idx: idx,
            main_score: main,
            backup_score: 0.0,
        };
        let prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 4,
            depth_two_lookahead_candidates: Some(vec![4]),
            attribution_diag: None,
            candidates: vec![
                candidate("n0", 0, 0.0),
                candidate("n1", 1, 0.0),
                candidate("n2", 2, 0.0),
                candidate("n3", 3, 0.0),
                candidate("paper-only", 4, 100.0),
            ],
            sides: vec![
                [SideSlot::Sim(0), SideSlot::Sim(1)],
                [SideSlot::Sim(2), SideSlot::Sim(3)],
                [SideSlot::Sim(4), SideSlot::Sim(5)],
                [SideSlot::Sim(6), SideSlot::Sim(7)],
                [SideSlot::Sim(8), SideSlot::Sim(9)],
            ],
        };
        let values = vec![
            Some(0.1),
            Some(0.1),
            Some(0.4),
            Some(0.4),
            Some(0.3),
            Some(0.3),
            Some(0.2),
            Some(0.2),
            Some(100.0),
            Some(100.0),
        ];
        let mut capture =
            KfsbF64ShadowCapture::new(0, &prep, values.len()).expect("four-candidate capture");
        for sim_index in 0..values.len() {
            capture.record(sim_index, HashMap::new(), &prep, &values, KfsbReduceOp::Min);
        }

        assert!(capture.complete());
        assert_eq!(
            capture
                .top
                .iter()
                .map(|candidate| candidate.candidate_index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3],
            "the observer must retain the legacy post-simulation top three and ignore paper-only roots"
        );
    }

    #[test]
    fn kfsb_f64_shadow_portfolio_uses_historical_near_tie_contract() {
        let candidate = |node: &str, main: f32| GraphKfsbCandidate {
            node_name: node.to_string(),
            neuron_idx: 0,
            main_score: main,
            backup_score: 0.0,
        };
        let candidates = vec![
            candidate("exact-lead", 1.0),
            candidate("historical-main", 2.0),
            candidate("third", 0.0),
            candidate("fourth", 0.0),
        ];
        let exact_lead = f32::from_bits(1.0_f32.to_bits() + 4);
        let values = vec![
            (exact_lead, exact_lead),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.25, 0.25),
        ];

        let ranked = rank_kfsb_candidate_portfolio(&candidates, &values, KfsbReduceOp::Min, 3);
        assert_eq!(
            ranked
                .iter()
                .map(|(candidate_index, _)| *candidate_index)
                .collect::<Vec<_>>(),
            vec![1, 0, 2],
            "top-three validation must repeat the authoritative 1e-6/main-score picker"
        );
    }

    #[test]
    fn kfsb_f64_shadow_f64_pick_preserves_original_first_seen_order() {
        let candidate = |node: &str| GraphKfsbCandidate {
            node_name: node.to_string(),
            neuron_idx: 0,
            main_score: 1.0,
            backup_score: 0.0,
        };
        let candidates = vec![
            candidate("original-first"),
            candidate("f32-rank-first"),
            candidate("third"),
        ];
        let near_lead = f32::from_bits(1.0_f32.to_bits() + 4);
        let ranked_order_values = vec![
            (1, (near_lead, near_lead)),
            (0, (1.0, 1.0)),
            (2, (0.5, 0.5)),
        ];

        let (winner, _, _) = pick_kfsb_candidate_subset_original_order(
            &candidates,
            &ranked_order_values,
            KfsbReduceOp::Min,
        )
        .expect("three-candidate subset");
        assert_eq!(
            winner, 0,
            "a near-tied f64 subset must use original candidate order, not f32 rank order"
        );
    }

    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_f64_shadow_is_one_shot_and_cannot_change_selection() {
        let run = |f64_gate: &'static str, scalar_gate: &'static str| {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_KFSB_F64_SHADOW", f64_gate),
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", scalar_gate),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
                    ("NY_MO_KFSB_K", "3"),
                    ("NY_MO_KFSB_REDUCE", "min"),
                    ("NY_BAB_CHAIN_WIDE", "1"),
                ],
                || {
                    let (graph, domain) = adaptive_depth_fixture();
                    let verifier = kfsb_verifier(KfsbReduceOp::Min);
                    let unstable = vec![
                        ("relu1".to_string(), 0),
                        ("relu1".to_string(), 1),
                        ("relu1".to_string(), 2),
                    ];
                    let wave = vec![(41usize, &domain, unstable)];
                    let committed = verifier.select_graph_branch_kfsb_multi_batched(
                        0,
                        &graph,
                        &wave,
                        &["relu1".to_string()],
                        &[vec![1.0]],
                        &[0.0],
                        &NaiveCpuGemmEngine,
                    );
                    let signature = committed
                        .get(&41)
                        .expect("candidate must commit")
                        .iter()
                        .map(|(child, is_active, _)| {
                            let constraint =
                                child.history().iter_all().next().expect("split constraint");
                            match constraint {
                                crate::beta_crown::branching::GraphConstraint::Relu(neuron) => (
                                    neuron.node_name.clone(),
                                    neuron.neuron_idx,
                                    neuron.is_active,
                                    *is_active,
                                ),
                                other => panic!("unexpected constraint: {other:?}"),
                            }
                        })
                        .collect::<Vec<_>>();
                    let fired = verifier
                        .kfsb_f64_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let scalar_fired = verifier
                        .adaptive_depth_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed);
                    (signature, fired, scalar_fired)
                },
            )
        };

        let (off_signature, off_fired, off_scalar_fired) = run("0", "0");
        let (on_signature, on_fired, on_scalar_fired) = run("1", "0");
        assert_eq!(
            on_signature, off_signature,
            "observation-only f64 telemetry cannot alter the committed split or children"
        );
        assert!(!off_fired, "gate-off path must not claim the one-shot");
        assert!(!off_scalar_fired);
        assert!(on_fired, "gate-on path must claim exactly one attempt");
        assert!(!on_scalar_fired);

        let (both_signature, both_f64_fired, both_scalar_fired) = run("1", "1");
        assert_eq!(
            both_signature, off_signature,
            "simultaneous observers must preserve the committed split and children",
        );
        assert!(both_f64_fired, "f64 must retain its simultaneous one-shot");
        assert!(
            both_scalar_fired,
            "the scalar observer must retain its simultaneous one-shot",
        );
    }

    #[test]
    fn adaptive_depth_shadow_gate_and_budget_fail_closed() {
        assert!(!resolve_adaptive_depth_shadow_enabled(None));
        assert!(!resolve_adaptive_depth_shadow_enabled(Some("")));
        assert!(!resolve_adaptive_depth_shadow_enabled(Some("0")));
        assert!(!resolve_adaptive_depth_shadow_enabled(Some("true")));
        assert!(resolve_adaptive_depth_shadow_enabled(Some("1")));

        assert!(!resolve_adaptive_depth_select_enabled(None));
        assert!(!resolve_adaptive_depth_select_enabled(Some("")));
        assert!(!resolve_adaptive_depth_select_enabled(Some("0")));
        assert!(!resolve_adaptive_depth_select_enabled(Some("true")));
        assert!(resolve_adaptive_depth_select_enabled(Some("1")));
        assert!(!resolve_adaptive_depth_commit_enabled(None));
        assert!(!resolve_adaptive_depth_commit_enabled(Some("")));
        assert!(!resolve_adaptive_depth_commit_enabled(Some("0")));
        assert!(!resolve_adaptive_depth_commit_enabled(Some("true")));
        assert!(resolve_adaptive_depth_commit_enabled(Some("1")));
        let fired = std::sync::atomic::AtomicBool::new(false);
        assert!(claim_adaptive_depth_attempt(&fired));
        assert!(fired.load(std::sync::atomic::Ordering::Acquire));
        assert!(
            !claim_adaptive_depth_attempt(&fired),
            "the observer must claim its deterministic one-shot exactly once"
        );

        let now = std::time::Instant::now();
        assert!(adaptive_depth_shadow_deadline(now, None).is_some());
        assert!(adaptive_depth_shadow_deadline(
            now,
            now.checked_add(std::time::Duration::from_secs(5))
        )
        .is_none());
        assert!(adaptive_depth_shadow_deadline(
            now,
            now.checked_add(std::time::Duration::from_secs(7))
        )
        .is_some());

        let shadow_deadline = now
            .checked_add(std::time::Duration::from_secs(1))
            .expect("shadow deadline");
        assert!(adaptive_depth_shadow_budget_available(
            now,
            shadow_deadline,
            None
        ));
        assert!(!adaptive_depth_shadow_budget_available(
            shadow_deadline,
            shadow_deadline,
            None
        ));
        let distant_shadow = now
            .checked_add(std::time::Duration::from_secs(10))
            .expect("distant shadow deadline");
        assert!(adaptive_depth_shadow_budget_available(
            now,
            distant_shadow,
            now.checked_add(std::time::Duration::from_secs(6))
        ));
        assert!(
            !adaptive_depth_shadow_budget_available(
                now,
                distant_shadow,
                now.checked_add(std::time::Duration::from_secs(5))
            ),
            "optional work may not start at the exact five-second reserve boundary"
        );
    }

    #[test]
    fn typed_depth_two_budget_is_admitted_once_at_entry_and_fails_closed() {
        let now = std::time::Instant::now();
        assert!(
            DepthTwoLookaheadBudget::admit(now, now.checked_add(std::time::Duration::from_secs(5)))
                .is_none(),
            "a typed wave may not start without the full five-second authority reserve"
        );
        let budget =
            DepthTwoLookaheadBudget::admit(now, now.checked_add(std::time::Duration::from_secs(7)))
                .expect("one second of optional work plus the reserve is admissible");
        assert!(budget.available_at(now));
        assert!(
            !budget.available_at(
                now.checked_add(std::time::Duration::from_secs(1))
                    .expect("private deadline")
            ),
            "later phases must observe the entry-created deadline, not reset it"
        );
        assert!(
            !budget.available_at(
                now.checked_add(std::time::Duration::from_secs(2))
                    .expect("expired private deadline")
            ),
            "expired typed work declines to the historical winner"
        );
    }

    #[test]
    fn typed_depth_two_expired_overlay_append_is_transactional() {
        let (graph, domain) = adaptive_depth_fixture();
        let candidate = |neuron_idx| GraphKfsbCandidate {
            node_name: "relu1".to_string(),
            neuron_idx,
            main_score: 1.0 - neuron_idx as f32,
            backup_score: 0.0,
        };
        let mut prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![candidate(0)],
            sides: vec![[SideSlot::Infeasible, SideSlot::Infeasible]],
        };
        let plan = DepthTwoLookaheadOverlayPlan {
            selected: vec![candidate(0), candidate(1)],
        };
        let budget = DepthTwoLookaheadBudget::expired_at(std::time::Instant::now());
        let mut sims: Vec<Option<MultiObjectiveGraphBabDomain>> = Vec::new();
        let mut owners = Vec::new();
        let verifier = kfsb_verifier(KfsbReduceOp::Min);
        assert!(!verifier.append_depth_two_lookahead_overlay(
            &graph,
            &domain,
            &[0.0],
            0,
            &mut prep,
            plan,
            budget,
            &mut sims,
            &mut owners,
        ));
        assert_eq!(prep.candidates.len(), 1);
        assert_eq!(prep.sides.len(), 1);
        assert!(prep.depth_two_lookahead_candidates.is_none());
        assert!(sims.is_empty());
        assert!(owners.is_empty());
    }

    #[test]
    fn typed_depth_two_recurrence_balances_both_children_in_f64() {
        let finite = DepthTwoLookaheadSideScore::Finite;
        let balanced =
            depth_two_lookahead_score(-2.0, finite(5.0), finite(5.0), 0.5).expect("balanced");
        let lopsided =
            depth_two_lookahead_score(-2.0, finite(9.0), finite(1.0), 0.5).expect("lopsided");
        assert!(
            balanced > lopsided,
            "product-over-sum bonus must prefer two strong child outcomes"
        );
        assert_eq!(
            depth_two_lookahead_score(-2.0, finite(5.0), finite(5.0), 0.0),
            Some(-2.0),
            "lambda zero is exactly the one-step score"
        );
        assert_eq!(
            depth_two_lookahead_score(
                1.0,
                DepthTwoLookaheadSideScore::Infeasible,
                finite(4.0),
                0.5,
            ),
            Some(3.0),
            "one empty side uses the finite-side limit"
        );
        assert_eq!(
            depth_two_lookahead_score(
                f64::INFINITY,
                DepthTwoLookaheadSideScore::Infeasible,
                DepthTwoLookaheadSideScore::Infeasible,
                0.5,
            ),
            Some(f64::INFINITY)
        );
        assert!(
            depth_two_lookahead_score(
                0.0,
                DepthTwoLookaheadSideScore::Infeasible,
                DepthTwoLookaheadSideScore::Infeasible,
                0.5,
            )
            .is_none(),
            "all-infeasible infinity requires the matching exact one-step state"
        );
        assert!(
            depth_two_lookahead_score(0.0, finite(f64::MAX), finite(f64::MAX), 0.5)
                .is_some_and(f64::is_finite),
            "algebraically stable recurrence must not overflow a finite portfolio"
        );
        assert!(depth_two_lookahead_score(0.0, finite(-1.0), finite(2.0), 0.5).is_none());
        assert!(depth_two_lookahead_score(0.0, finite(1.0), finite(2.0), f64::NAN).is_none());
    }

    #[test]
    fn typed_depth_two_portfolio_is_complete_and_ties_keep_historical_winner() {
        let tied = [(0, Some(1.0)), (1, Some(3.0)), (2, Some(3.0))];
        assert_eq!(
            select_complete_depth_two_lookahead(&tied, 3, 2),
            Some((2, 3.0)),
            "an exact tie must retain the historical one-step root"
        );
        assert_eq!(
            select_complete_depth_two_lookahead(&tied, 3, 0),
            Some((1, 3.0)),
            "otherwise the first deterministic maximum wins"
        );
        assert!(select_complete_depth_two_lookahead(&tied[..2], 3, 0).is_none());
        assert!(select_complete_depth_two_lookahead(
            &[(0, Some(1.0)), (1, None), (2, Some(3.0))],
            3,
            0,
        )
        .is_none());
        assert!(select_complete_depth_two_lookahead(
            &[(0, Some(1.0)), (0, Some(2.0)), (2, Some(3.0))],
            3,
            0,
        )
        .is_none());
        assert!(select_complete_depth_two_lookahead(&tied, 3, 9).is_none());
    }

    #[test]
    fn typed_depth_two_frontier_target_is_explicit_worst_then_stable_slot() {
        assert_eq!(
            select_depth_two_frontier_worst_slot([(4, -0.1), (2, -0.8), (1, -0.8), (0, f32::NAN),]),
            Some(1)
        );
        assert_eq!(select_depth_two_frontier_worst_slot([(0, f32::NAN)]), None);
    }

    #[test]
    fn typed_depth_two_upper_mode_fails_closed_without_mutating_legacy_straggler() {
        crate::tests::with_serialized_env_vars_removed(
            &[
                "NY_MO_KFSB",
                "NY_BRANCH_KFSB_CHILDSIM",
                "NY_MO_ADAPTIVE_DEPTH_SHADOW",
                "NY_MO_ADAPTIVE_DEPTH_SELECT",
                "NY_MO_ADAPTIVE_DEPTH_COMMIT",
                "NY_MO_KFSB_F64_SHADOW",
                "NY_MO_KFSB_REDUCE",
                "NY_MO_KFSB_K",
            ],
            || {
                let bounds = [(-9.0, 1.0), (-1.0, 8.0), (-4.0, 3.0)];
                let verified = [false, false, false];
                assert_eq!(
                    select_kfsb_straggler(&bounds, &verified, false),
                    Some((0, -9.0)),
                    "lower-bound mode chooses the smallest lower bound"
                );
                assert_eq!(
                    select_kfsb_straggler(&bounds, &verified, true),
                    Some((1, -8.0)),
                    "upper-bound mode chooses the highest raw upper via normalized -upper"
                );
                let upper = BetaCrownVerifier::new(BetaCrownConfig {
                    branching_heuristic: BranchingHeuristic::Kfsb,
                    verify_upper_bound: true,
                    use_kfsb_multi_branching: false,
                    depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                        mode: DepthTwoBranchLookaheadMode::Select,
                        ..Default::default()
                    },
                    ..Default::default()
                });
                assert!(
                    !upper.kfsb_multi_wave_enabled_at_round(0),
                    "phase-1 typed advice must decline rather than score a highest-upper target \
                     with the historical smallest-lower prep"
                );
            },
        );
    }

    #[test]
    fn typed_depth_two_missing_babsr_entry_cannot_fabricate_exact_portfolio() {
        let unstable = (0..15)
            .map(|neuron_idx| ("relu".to_string(), neuron_idx))
            .collect::<Vec<_>>();
        let (mut scored, complete) =
            materialize_kfsb_candidates_with_completeness(&unstable, |(_, neuron_idx)| {
                (*neuron_idx != 14).then_some((-1.0 - *neuron_idx as f32, 0.0))
            });
        scored.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
        });
        assert!(!complete);
        assert_eq!(
            scored[0].neuron_idx, 14,
            "the historical zero-fill would make the missing entry look best"
        );
        assert!(
            select_depth_two_root_portfolio(&scored, complete, 15).is_none(),
            "typed advice must decline even when zero-filling appears to make an exact total"
        );

        let (mut complete_scores, complete) =
            materialize_kfsb_candidates_with_completeness(&unstable, |(_, neuron_idx)| {
                Some((-1.0 - *neuron_idx as f32, 0.0))
            });
        complete_scores.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
        });
        assert!(complete);
        assert_eq!(
            select_depth_two_root_portfolio(&complete_scores, complete, 15)
                .expect("complete score map")
                .len(),
            15
        );
    }

    #[test]
    fn typed_depth_two_capture_is_hard_capped_at_two_times_fifteen() {
        let candidates = (0..17)
            .map(|neuron_idx| GraphKfsbCandidate {
                node_name: "relu".to_string(),
                neuron_idx,
                main_score: (17 - neuron_idx) as f32,
                backup_score: 0.0,
            })
            .collect();
        let sides = (0..17)
            .map(|candidate| {
                [
                    SideSlot::Sim(2 * candidate),
                    SideSlot::Sim(2 * candidate + 1),
                ]
            })
            .collect();
        let mut prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 2,
            depth_two_lookahead_candidates: Some((2..17).collect()),
            attribution_diag: None,
            candidates,
            sides,
        };
        let capture =
            DepthTwoLookaheadCapture::new(0, &prep, 34, 15).expect("complete 15-root overlay");
        assert_eq!(capture.planned_slot_count(), 30);
        assert!(DepthTwoLookaheadCapture::new(0, &prep, 33, 15).is_none());
        assert!(DepthTwoLookaheadCapture::new(0, &prep, 34, 16).is_none());
        prep.depth_two_lookahead_candidates
            .as_mut()
            .expect("paper mapping")[14] = 15;
        assert!(
            DepthTwoLookaheadCapture::new(0, &prep, 34, 15).is_none(),
            "duplicate paper identities must fail closed"
        );
    }

    #[test]
    fn typed_depth_two_advice_crosses_only_revalidated_first_level_identity() {
        let (_graph, domain) = adaptive_depth_fixture();
        let prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: Some(vec![0]),
            attribution_diag: None,
            candidates: vec![GraphKfsbCandidate {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                main_score: 1.0,
                backup_score: 0.0,
            }],
            sides: vec![[SideSlot::Sim(0), SideSlot::Infeasible]],
        };
        let winner = select_complete_depth_two_lookahead(&[(0, Some(2.0))], 1, 0)
            .expect("complete advice")
            .0;
        let identity =
            adaptive_depth_authority_identity(0, 41, &prep, winner).expect("root identity");
        let values = [Some(0.25)];
        let sims = [Some(domain)];
        assert_eq!(
            resolve_adaptive_depth_authority_candidate(
                &identity,
                0,
                41,
                &prep,
                &values,
                &sims,
                KfsbReduceOp::Min,
            )
            .map(|(candidate, _)| candidate),
            Some(0)
        );
        assert!(
            resolve_adaptive_depth_authority_candidate(
                &identity,
                0,
                41,
                &prep,
                &values,
                &[None],
                KfsbReduceOp::Min,
            )
            .is_none(),
            "missing authoritative first-level child must decline private advice"
        );
    }

    #[test]
    fn adaptive_depth_authority_identity_and_sim_preflight_fail_closed() {
        let (_graph, domain) = adaptive_depth_fixture();
        let candidate = GraphKfsbCandidate {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            main_score: 1.25,
            backup_score: -0.5,
        };
        let prep = DomainPrep {
            slot: 0,
            straggler: 0,
            cached_score_candidates: 0,
            legacy_candidates_len: 1,
            depth_two_lookahead_candidates: None,
            attribution_diag: None,
            candidates: vec![candidate],
            sides: vec![[SideSlot::Sim(0), SideSlot::Infeasible]],
        };
        let values = [Some(2.0)];
        let sims = [Some(domain)];

        let mut identity =
            adaptive_depth_authority_identity(0, 17, &prep, 0).expect("valid identity");
        assert_eq!(
            resolve_adaptive_depth_authority_candidate(
                &identity,
                0,
                17,
                &prep,
                &values,
                &sims,
                KfsbReduceOp::Min,
            )
            .map(|(idx, _)| idx),
            Some(0)
        );

        identity.candidate_index = 1;
        assert!(resolve_adaptive_depth_authority_candidate(
            &identity,
            0,
            17,
            &prep,
            &values,
            &sims,
            KfsbReduceOp::Min,
        )
        .is_none());
        identity = adaptive_depth_authority_identity(0, 17, &prep, 0).expect("valid identity");
        assert!(resolve_adaptive_depth_authority_candidate(
            &identity,
            0,
            18,
            &prep,
            &values,
            &sims,
            KfsbReduceOp::Min,
        )
        .is_none());
        identity.node_name = "wrong".to_string();
        assert!(resolve_adaptive_depth_authority_candidate(
            &identity,
            0,
            17,
            &prep,
            &values,
            &sims,
            KfsbReduceOp::Min,
        )
        .is_none());
        identity = adaptive_depth_authority_identity(0, 17, &prep, 0).expect("valid identity");
        assert!(resolve_adaptive_depth_authority_candidate(
            &identity,
            0,
            17,
            &prep,
            &values,
            &[None],
            KfsbReduceOp::Min,
        )
        .is_none());
        assert!(resolve_adaptive_depth_authority_candidate(
            &identity,
            0,
            17,
            &prep,
            &[Some(f32::NAN)],
            &sims,
            KfsbReduceOp::Min,
        )
        .is_none());
    }

    #[test]
    fn adaptive_depth_shadow_ranking_is_unique_and_deterministic() {
        let candidate = |node: &str, idx: usize, main: f32| GraphKfsbCandidate {
            node_name: node.to_string(),
            neuron_idx: idx,
            main_score: main,
            backup_score: 0.0,
        };
        let candidates = vec![
            candidate("b", 0, 2.0),
            candidate("a", 1, 1.0),
            candidate("b", 0, 9.0), // duplicate key, better exact score below
            candidate("c", 2, 3.0),
        ];
        let values = vec![(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (2.0, 2.0)];
        let ranked = rank_adaptive_depth_candidates(&candidates, &values, KfsbReduceOp::Min);
        let indices: Vec<usize> = ranked.into_iter().map(|(idx, _)| idx).collect();
        assert_eq!(indices, vec![2, 3, 1]);
    }

    #[test]
    fn adaptive_depth_proxy_advice_can_differ_but_ties_preserve_history() {
        assert_eq!(
            adaptive_depth_proxy_recommended_rank(&[0.1, 0.9, 0.2]),
            Some(1),
            "the bounded proxy must be capable of recommending a nonhistorical root",
        );
        assert_eq!(
            adaptive_depth_proxy_recommended_rank(&[0.5, 0.5, 0.5]),
            Some(0),
            "ties must preserve historical rank zero",
        );
        assert_eq!(
            adaptive_depth_proxy_recommended_rank(&[0.1, f32::NAN, 0.2]),
            None,
            "malformed advice must fail closed",
        );
    }

    #[test]
    fn adaptive_depth_capture_is_fixed_size_and_retains_only_scalars() {
        let sides = [
            [SideSlot::Sim(0), SideSlot::Sim(1)],
            [SideSlot::Sim(2), SideSlot::Sim(3)],
            [SideSlot::Sim(4), SideSlot::Sim(5)],
        ];
        let mut capture =
            AdaptiveDepthShadowCapture::from_candidate_indices(0, &sides, &[0, 1, 2], 1_000_000)
                .expect("valid fixed capture");
        let small = AdaptiveDepthShadowCapture::from_candidate_indices(0, &sides, &[0, 1, 2], 6)
            .expect("valid fixed capture");

        assert_eq!(AdaptiveDepthShadowCapture::slot_capacity(), 128);
        assert_eq!(capture.planned_slot_count(), 6);
        assert_eq!(capture.captured_score_count(), 0);
        assert_eq!(size_of_val(&capture), size_of_val(&small));
        for sim_index in 0..6 {
            assert!(capture.contains_sim(sim_index));
            assert!(capture.insert_proxy_score(sim_index, sim_index as f32 / 10.0));
        }
        assert!(!capture.contains_sim(999_999));
        assert!(!capture.insert_proxy_score(999_999, 0.0));
        assert!(
            !capture.insert_proxy_score(0, 1.0),
            "a side is captured once"
        );
        assert!(!capture.insert_proxy_score(1, f32::NAN));
        assert_eq!(capture.captured_score_count(), 6);
    }
    #[test]
    fn adaptive_depth_peak_ledger_rejects_overflow_and_coexisting_peak() {
        let mut overflow = AdaptiveDepthPrivatePeakLedger::new(usize::MAX);
        assert_eq!(
            overflow.admit([usize::MAX, 1]),
            Err(AdaptiveDepthPrivatePeakDecline::ArithmeticOverflow),
        );
        assert_eq!(overflow.admitted_peak_bytes(), 0);

        let mut coexistence = AdaptiveDepthPrivatePeakLedger::new(100);
        assert_eq!(coexistence.admit([40]), Ok(40));
        assert_eq!(coexistence.admit([40, 40]), Ok(80));
        assert_eq!(
            coexistence.admit([40, 40, 21]),
            Err(AdaptiveDepthPrivatePeakDecline::PeakCapExceeded),
            "components that fit alone must still be rejected when live together",
        );
        assert_eq!(
            coexistence.admitted_peak_bytes(),
            80,
            "a refused stage must not publish a larger admitted peak",
        );
    }

    #[test]
    fn adaptive_depth_base_select_is_specific_to_each_child_fixpoint() {
        let (graph, domain) = adaptive_depth_fixture();
        let make_child = |is_active| {
            domain
                .with_constraint(
                    &graph,
                    GraphNeuronConstraint {
                        node_name: "relu1".to_string(),
                        neuron_idx: 0,
                        is_active,
                        score: 1.0,
                    },
                    false,
                    &[0.0],
                )
                .expect("root split")
                .expect("both root sides feasible")
        };
        let mut active = make_child(true);
        let mut inactive = make_child(false);

        // Model the constrained-forward caches returned by the one-step kFSB
        // simulations: candidate 1 has the dominant relaxation gap in the
        // active child, while candidate 2 dominates in the inactive child.
        // Candidate 0 remains interval-unstable but is excluded by history.
        active.node_bounds.insert(
            "linear1".to_string(),
            Arc::new(
                BoundedTensor::new(
                    arr1(&[-1.0_f32, -8.0, -0.1]).into_dyn(),
                    arr1(&[1.0_f32, 8.0, 0.1]).into_dyn(),
                )
                .expect("active fixpoint"),
            ),
        );
        inactive.node_bounds.insert(
            "linear1".to_string(),
            Arc::new(
                BoundedTensor::new(
                    arr1(&[-1.0_f32, -0.1, -8.0]).into_dyn(),
                    arr1(&[1.0_f32, 0.1, 8.0]).into_dyn(),
                )
                .expect("inactive fixpoint"),
            ),
        );

        let verifier = kfsb_verifier(KfsbReduceOp::Min);
        let relus = ["relu1".to_string()];
        let active_pick = verifier
            .select_adaptive_depth_base_candidate(&graph, &active, &relus, &[1.0])
            .expect("active child scoring")
            .expect("active child candidate");
        let inactive_pick = verifier
            .select_adaptive_depth_base_candidate(&graph, &inactive, &relus, &[1.0])
            .expect("inactive child scoring")
            .expect("inactive child candidate");

        assert_eq!(active_pick.neuron_idx, 1);
        assert_eq!(inactive_pick.neuron_idx, 2);
        assert_ne!(
            active_pick.neuron_idx, inactive_pick.neuron_idx,
            "depth-2 baseSelect must consume each child's own refreshed bounds"
        );
    }

    #[test]
    fn adaptive_depth_babsr_scoring_rejects_expired_private_deadline() {
        let (graph, domain) = adaptive_depth_fixture();
        let verifier = kfsb_verifier(KfsbReduceOp::Min);
        let error = verifier
            .compute_graph_babsr_scores_from_bounds_until(
                &graph,
                domain.node_bounds(),
                domain.input_bounds(),
                KfsbReduceOp::Min,
                Some(&[1.0]),
                None,
                std::time::Instant::now(),
            )
            .expect_err("expired private scoring deadline must fail closed");

        assert!(error.is_deadline_exceeded());
    }

    #[test]
    fn adaptive_depth_shadow_cache_clear_drops_every_objective_cache() {
        let (_graph, mut domain) = adaptive_depth_fixture();
        let cached = LinearBounds::new(
            arr2(&[[1.0_f32, 0.0, 0.0]]),
            arr1(&[0.0]),
            arr2(&[[1.0_f32, 0.0, 0.0]]),
            arr1(&[0.0]),
        )
        .expect("cached bounds");
        let mut cached_map = HashMap::new();
        cached_map.insert("relu1".to_string(), cached);
        domain
            .set_cached_las(vec![Some(CachedLinearBounds::from_linear_bounds_map(
                cached_map,
            ))])
            .expect("cache shape");
        assert!(domain.cached_las()[0].is_some());
        assert!(clear_shadow_cached_las(&mut domain));
        assert!(domain.cached_las().iter().all(Option::is_none));
    }

    /// The scalar observer leaves committed histories, bounds, masks, and depth
    /// exact.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_shadow_on_off_preserves_committed_children() {
        type Snapshot = Vec<(
            bool,
            Vec<GraphNeuronConstraint>,
            Vec<(u32, u32)>,
            Vec<bool>,
            usize,
        )>;

        let (graph, domain) = adaptive_depth_fixture();
        let run = |gate: &str| -> Snapshot {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", gate),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
                    ("NY_MO_KFSB_REDUCE", "min"),
                ],
                || {
                    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: 3,
                        beta_iterations: 0,
                        ..Default::default()
                    });
                    let unstable = vec![
                        ("relu1".to_string(), 0),
                        ("relu1".to_string(), 1),
                        ("relu1".to_string(), 2),
                    ];
                    let wave = vec![(17usize, &domain, unstable)];
                    let committed = verifier.select_graph_branch_kfsb_multi_batched(
                        0,
                        &graph,
                        &wave,
                        &["relu1".to_string()],
                        &[vec![1.0]],
                        &[0.0],
                        &NaiveCpuGemmEngine,
                    );
                    committed
                        .get(&17)
                        .expect("scored domain commits")
                        .iter()
                        .map(|(child, active, _)| {
                            (
                                *active,
                                child.history().constraints.clone(),
                                child
                                    .objective_bounds()
                                    .iter()
                                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                                    .collect(),
                                child.verified().to_vec(),
                                child.depth(),
                            )
                        })
                        .collect()
                },
            )
        };

        assert_eq!(run("0"), run("1"));
    }

    /// SELECT is now an observer gate only and cannot publish a root or child.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_select_is_advice_only_and_cannot_publish_children() {
        type Snapshot = Vec<(bool, Vec<GraphNeuronConstraint>, usize)>;

        let (graph, domain) = adaptive_depth_fixture();
        let run = |selection: &str, reduce: &str| -> Snapshot {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", selection),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
                    ("NY_MO_KFSB_CERT_REUSE", "0"),
                    ("NY_MO_KFSB_REDUCE", reduce),
                ],
                || {
                    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: 3,
                        beta_iterations: 0,
                        batch_size: 4,
                        min_batch_fill_ratio: 1.0,
                        max_relu_split_depth: 2,
                        ..Default::default()
                    });
                    let unstable = (0..3).map(|index| ("relu1".to_string(), index)).collect();
                    let wave = vec![(17usize, &domain, unstable)];
                    verifier
                        .select_graph_branch_kfsb_multi_batched(
                            0,
                            &graph,
                            &wave,
                            &["relu1".to_string()],
                            &[vec![1.0]],
                            &[0.0],
                            &NaiveCpuGemmEngine,
                        )
                        .get(&17)
                        .expect("historical selection commits")
                        .iter()
                        .map(|(child, active, _)| {
                            (*active, child.history().constraints.clone(), child.depth())
                        })
                        .collect()
                },
            )
        };

        for reduce in ["min", "max"] {
            assert_eq!(
                run("1", reduce),
                run("0", reduce),
                "advice-only SELECT must preserve historical children"
            );
        }
    }
    /// Advice-only SELECT may observe any historical horizon but must never
    /// alter one- or four-decision covers.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_selection_observes_without_overriding_depth_one_or_four() {
        type Snapshot = Vec<(
            bool,
            Vec<GraphNeuronConstraint>,
            Vec<(u32, u32)>,
            Vec<bool>,
            usize,
        )>;

        fn run(
            graph: &GraphNetwork,
            domain: &MultiObjectiveGraphBabDomain,
            parent_index: usize,
            unstable_count: usize,
            production_depth: usize,
            shadow: &str,
            selection: &str,
        ) -> (Snapshot, bool) {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", shadow),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", selection),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
                    ("NY_MO_KFSB_CERT_REUSE", "0"),
                    ("NY_MO_KFSB_REDUCE", "min"),
                ],
                || {
                    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: unstable_count,
                        beta_iterations: 0,
                        batch_size: if production_depth == 4 { 16 } else { 64 },
                        min_batch_fill_ratio: 1.0,
                        max_relu_split_depth: production_depth,
                        ..Default::default()
                    });
                    let unstable = (0..unstable_count)
                        .map(|index| ("relu1".to_string(), index))
                        .collect();
                    let wave = vec![(parent_index, domain, unstable)];
                    let committed = verifier.select_graph_branch_kfsb_multi_batched(
                        0,
                        graph,
                        &wave,
                        &["relu1".to_string()],
                        &[vec![1.0]],
                        &[0.0],
                        &NaiveCpuGemmEngine,
                    );
                    let snapshot = committed
                        .get(&parent_index)
                        .expect("scored domain commits")
                        .iter()
                        .map(|(child, root_phase, _)| {
                            (
                                *root_phase,
                                child.history().constraints.clone(),
                                child
                                    .objective_bounds()
                                    .iter()
                                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                                    .collect(),
                                child.verified().to_vec(),
                                child.depth(),
                            )
                        })
                        .collect();
                    (
                        snapshot,
                        verifier
                            .adaptive_depth_shadow_fired
                            .load(std::sync::atomic::Ordering::Relaxed),
                    )
                },
            )
        }

        let (depth_one_graph, depth_one_domain) = adaptive_depth_fixture();
        let (depth_four_graph, depth_four_domain) = adaptive_depth_four_candidate_fixture();
        for (graph, domain, parent_index, unstable_count, production_depth, expected_leaves) in [
            (&depth_one_graph, &depth_one_domain, 51, 3, 1, 2),
            (&depth_four_graph, &depth_four_domain, 53, 4, 4, 16),
        ] {
            let (control, control_fired) = run(
                graph,
                domain,
                parent_index,
                unstable_count,
                production_depth,
                "0",
                "0",
            );
            assert!(!control_fired);
            assert_eq!(control.len(), expected_leaves);

            let (observed, observed_fired) = run(
                graph,
                domain,
                parent_index,
                unstable_count,
                production_depth,
                "0",
                "1",
            );
            assert!(
                observed_fired,
                "SELECT observer must claim its one-shot at either horizon"
            );
            assert_eq!(
                observed, control,
                "depth {production_depth} must not override"
            );

            let (shadowed, shadowed_fired) = run(
                graph,
                domain,
                parent_index,
                unstable_count,
                production_depth,
                "1",
                "1",
            );
            assert!(shadowed_fired, "explicit SHADOW may run the mismatch");
            assert_eq!(
                shadowed, control,
                "depth {production_depth} shadow observation must not override"
            );
        }
    }

    /// COMMIT is also an observer gate until true bounded second-child
    /// propagation exists; it cannot publish a replay cover or UNSAT verdict.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_commit_is_advice_only_and_cannot_publish_children_or_unsat() {
        type Snapshot = Vec<(
            bool,
            Vec<GraphNeuronConstraint>,
            Vec<(u32, u32)>,
            Vec<bool>,
            usize,
        )>;

        let (graph, domain) = adaptive_depth_fixture();
        let run = |commit: &str| -> Snapshot {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", commit),
                    ("NY_MO_KFSB_CERT_REUSE", "0"),
                    ("NY_MO_KFSB_REDUCE", "min"),
                ],
                || {
                    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: 3,
                        beta_iterations: 0,
                        batch_size: 4,
                        min_batch_fill_ratio: 1.0,
                        max_relu_split_depth: 2,
                        ..Default::default()
                    });
                    let unstable = (0..3).map(|index| ("relu1".to_string(), index)).collect();
                    let wave = vec![(17usize, &domain, unstable)];
                    verifier
                        .select_graph_branch_kfsb_multi_batched(
                            0,
                            &graph,
                            &wave,
                            &["relu1".to_string()],
                            &[vec![1.0]],
                            &[0.0],
                            &NaiveCpuGemmEngine,
                        )
                        .get(&17)
                        .expect("historical selection commits")
                        .iter()
                        .map(|(child, active, effect)| {
                            assert!(
                                !matches!(effect, KfsbCertEffect::ParentComplete(_)),
                                "advice-only M28 cannot publish an UNSAT close"
                            );
                            (
                                *active,
                                child.history().constraints.clone(),
                                child
                                    .objective_bounds()
                                    .iter()
                                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                                    .collect(),
                                child.verified().to_vec(),
                                child.depth(),
                            )
                        })
                        .collect()
                },
            )
        };

        assert_eq!(
            run("1"),
            run("0"),
            "advice-only COMMIT must preserve historical children and verdicts"
        );
    }
    /// Arming the legacy COMMIT observer on a depth-four wave preserves every
    /// common-plan leaf.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_commit_preserves_depth_four_control_exactly() {
        type Snapshot = Vec<(
            bool,
            Vec<GraphNeuronConstraint>,
            Vec<(u32, u32)>,
            Vec<bool>,
            usize,
        )>;

        let (graph, domain) = adaptive_depth_four_candidate_fixture();
        let run = |commit: &str| -> Snapshot {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", commit),
                    ("NY_MO_KFSB_CERT_REUSE", "0"),
                    ("NY_MO_KFSB_REDUCE", "min"),
                ],
                || {
                    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: 4,
                        beta_iterations: 0,
                        batch_size: 16,
                        min_batch_fill_ratio: 1.0,
                        max_relu_split_depth: 4,
                        ..Default::default()
                    });
                    let unstable = (0..4).map(|index| ("relu1".to_string(), index)).collect();
                    let wave = vec![(23usize, &domain, unstable)];
                    verifier
                        .select_graph_branch_kfsb_multi_batched(
                            0,
                            &graph,
                            &wave,
                            &["relu1".to_string()],
                            &[vec![1.0]],
                            &[0.0],
                            &NaiveCpuGemmEngine,
                        )
                        .get(&23)
                        .expect("scored domain commits")
                        .iter()
                        .map(|(child, root_phase, _)| {
                            (
                                *root_phase,
                                child.history().constraints.clone(),
                                child
                                    .objective_bounds()
                                    .iter()
                                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                                    .collect(),
                                child.verified().to_vec(),
                                child.depth(),
                            )
                        })
                        .collect()
                },
            )
        };

        let control = run("0");
        let armed = run("1");
        assert_eq!(control.len(), 16);
        assert!(control
            .iter()
            .all(|(_, history, _, _, depth)| history.len() == 4 && *depth == 4));
        assert_eq!(armed, control);
    }

    /// An ineligible COMMIT horizon must not consume the verifier-lifetime
    /// observer receipt needed by the next eligible depth-two wave.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_commit_claims_first_eligible_wave_on_same_verifier() {
        crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "1"),
                ("NY_MO_KFSB_CERT_REUSE", "0"),
                ("NY_MO_KFSB_REDUCE", "min"),
            ],
            || {
                let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                    branching_heuristic: BranchingHeuristic::Kfsb,
                    fsb_candidates: 4,
                    beta_iterations: 0,
                    batch_size: 16,
                    min_batch_fill_ratio: 1.0,
                    max_relu_split_depth: 4,
                    ..Default::default()
                });

                let (depth_four_graph, depth_four_domain) = adaptive_depth_four_candidate_fixture();
                let depth_four_wave = vec![(
                    23usize,
                    &depth_four_domain,
                    (0..4).map(|index| ("relu1".to_string(), index)).collect(),
                )];
                let depth_four = verifier.select_graph_branch_kfsb_multi_batched(
                    0,
                    &depth_four_graph,
                    &depth_four_wave,
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &NaiveCpuGemmEngine,
                );
                assert_eq!(depth_four.get(&23).map(Vec::len), Some(16));
                assert!(
                    !verifier
                        .adaptive_depth_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "the ineligible depth-four wave must leave the receipt untouched",
                );

                let (depth_two_graph, depth_two_domain) = adaptive_depth_fixture();
                let depth_two_wave: Vec<_> = (100usize..104)
                    .map(|parent_index| {
                        (
                            parent_index,
                            &depth_two_domain,
                            (0..3).map(|index| ("relu1".to_string(), index)).collect(),
                        )
                    })
                    .collect();
                let depth_two = verifier.select_graph_branch_kfsb_multi_batched(
                    5,
                    &depth_two_graph,
                    &depth_two_wave,
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &NaiveCpuGemmEngine,
                );
                assert_eq!(depth_two.len(), depth_two_wave.len());
                assert!(depth_two.values().all(|children| children.len() == 4));
                assert!(
                    verifier
                        .adaptive_depth_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "the next eligible depth-two wave must claim the same verifier receipt",
                );
            },
        );
    }

    /// Adding an ineligible COMMIT gate must not alter M27 SHADOW or SELECT
    /// output/state behavior on the historical first captured parent.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_ineligible_commit_preserves_shadow_select_interactions() {
        type Snapshot = Vec<(
            bool,
            Vec<GraphNeuronConstraint>,
            Vec<(u32, u32)>,
            Vec<bool>,
            usize,
        )>;

        let (graph, domain) = adaptive_depth_four_candidate_fixture();
        let run = |shadow: &str, selection: &str, commit: &str| {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", shadow),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", selection),
                    ("NY_MO_ADAPTIVE_DEPTH_COMMIT", commit),
                    ("NY_MO_KFSB_CERT_REUSE", "0"),
                    ("NY_MO_KFSB_REDUCE", "min"),
                ],
                || {
                    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: 4,
                        beta_iterations: 0,
                        batch_size: 16,
                        min_batch_fill_ratio: 1.0,
                        max_relu_split_depth: 4,
                        ..Default::default()
                    });
                    let unstable = (0..4).map(|index| ("relu1".to_string(), index)).collect();
                    let wave = vec![(41usize, &domain, unstable)];
                    let committed = verifier.select_graph_branch_kfsb_multi_batched(
                        0,
                        &graph,
                        &wave,
                        &["relu1".to_string()],
                        &[vec![1.0]],
                        &[0.0],
                        &NaiveCpuGemmEngine,
                    );
                    let snapshot: Snapshot = committed
                        .get(&41)
                        .expect("depth-four domain commits")
                        .iter()
                        .map(|(child, root_phase, _)| {
                            (
                                *root_phase,
                                child.history().constraints.clone(),
                                child
                                    .objective_bounds()
                                    .iter()
                                    .map(|(lower, upper)| (lower.to_bits(), upper.to_bits()))
                                    .collect(),
                                child.verified().to_vec(),
                                child.depth(),
                            )
                        })
                        .collect();
                    (
                        snapshot,
                        verifier
                            .adaptive_depth_shadow_fired
                            .load(std::sync::atomic::Ordering::Relaxed),
                    )
                },
            )
        };

        let shadow = run("1", "0", "0");
        assert!(shadow.1);
        assert_eq!(run("1", "0", "1"), shadow);

        let selected = run("0", "1", "0");
        assert!(selected.1);
        assert_eq!(run("0", "1", "1"), selected);
    }

    #[test]
    fn kfsb_winner_oracle_separates_balanced_and_one_sided_candidates() {
        let candidates = vec![
            GraphKfsbCandidate {
                node_name: "balanced".to_string(),
                neuron_idx: 0,
                main_score: 2.0,
                backup_score: -2.0,
            },
            GraphKfsbCandidate {
                node_name: "one_sided".to_string(),
                neuron_idx: 1,
                main_score: 1.0,
                backup_score: -1.0,
            },
        ];
        let child_values = [(2.0, 2.5), (-1.0, 10.0)];
        let min_pick =
            pick_kfsb_candidate(&candidates, child_values.iter().copied(), KfsbReduceOp::Min)
                .expect("Min should pick a candidate");
        let max_pick =
            pick_kfsb_candidate(&candidates, child_values.iter().copied(), KfsbReduceOp::Max)
                .expect("Max should pick a candidate");

        assert_eq!(min_pick.0, 0, "Min rewards the balanced worst child");
        assert_eq!(max_pick.0, 1, "Max rewards the one-sided closing child");
    }

    /// Winner-parity regression: the historical fixed-slope proxy ranks n0,
    /// while the exact cached lower-A row from the preceding CROWN pass ranks
    /// n1.  With k=1 the candidate filter makes that prescore difference
    /// observable in the committed (advisory-only) split.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_cached_la_gate_uses_captured_objective_coefficients() {
        let linear1 = LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), Some(arr1(&[0.0_f32, 0.0])))
            .expect("linear1");
        // Historical proxy lA at relu1 is [-1.0, -0.1], hence n0 ranks first.
        let linear2 = LinearLayer::new(arr2(&[[-1.0_f32, -0.1]]), None).expect("linear2");
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

        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect("input bounds");
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let mut domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-10.0, 10.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("root domain");

        // Exact captured objective lA reverses the ranking: n1 has magnitude
        // 10 while n0 has magnitude 0.1.  Upper rows/biases are populated only
        // to construct the same validated cache type used by real CROWN.
        let captured = LinearBounds::new(
            arr2(&[[-0.1_f32, -10.0]]),
            arr1(&[0.0]),
            arr2(&[[-0.1_f32, -10.0]]),
            arr1(&[0.0]),
        )
        .expect("captured lA");
        let mut captured_map = HashMap::new();
        captured_map.insert("relu1".to_string(), captured);
        domain.cached_las[0] = Some(Arc::new(CachedLinearBounds::from_linear_bounds_map(
            captured_map,
        )));

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::Kfsb,
            fsb_candidates: 1,
            beta_iterations: 0,
            ..Default::default()
        });
        let unstable = vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)];
        let relu_nodes = ["relu1".to_string()];
        let objectives = [vec![1.0]];
        let thresholds = [0.0];
        let engine = NaiveCpuGemmEngine;
        let selected = |cached_gate: &str| {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_KFSB_CACHED_LA", cached_gate),
                    ("NY_MO_KFSB_REDUCE", "min"),
                ],
                || {
                    let wave = vec![(0usize, &domain, unstable.clone())];
                    let committed = verifier.select_graph_branch_kfsb_multi_batched(
                        0,
                        &graph,
                        &wave,
                        &relu_nodes,
                        &objectives,
                        &thresholds,
                        &engine,
                    );
                    let children = committed.get(&0).expect("candidate must commit");
                    let constraint = children
                        .first()
                        .expect("winner has a feasible child")
                        .0
                        .history()
                        .iter_all()
                        .next()
                        .expect("child has split constraint");
                    match constraint {
                        crate::beta_crown::branching::GraphConstraint::Relu(neuron) => {
                            neuron.neuron_idx
                        }
                        other => panic!("unexpected constraint: {other:?}"),
                    }
                },
            )
        };

        assert_eq!(selected("0"), 0, "gate-off keeps historical proxy ranking");
        assert_eq!(selected("1"), 1, "gate-on uses captured objective lA");
    }

    /// Min reduce: child evaluation separates the candidates — n0's worst
    /// child (out = x+3, lb 2) beats n1's surviving child (the root-shaped
    /// active domain, CROWN lb ~1.5), so the committed winner must be n0 with
    /// BOTH children present, carrying the split constraint in their history.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_multi_min_reduce_picks_child_evaluated_winner() {
        crate::tests::with_serialized_env_vars_removed(&["NY_MO_KFSB_REDUCE"], || {
            let (graph, domain) = kfsb_fixture();
            let verifier = kfsb_verifier(KfsbReduceOp::Min);
            let unstable = vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)];
            let wave = vec![(7usize, &domain, unstable)];
            let engine = NaiveCpuGemmEngine;

            let committed = verifier.select_graph_branch_kfsb_multi_batched(
                0,
                &graph,
                &wave,
                &["relu1".to_string()],
                &[vec![1.0]],
                &[0.0],
                &engine,
            );

            let children = committed.get(&7).expect("wave domain must be resolved");
            assert_eq!(children.len(), 2, "both children of the winner commit");
            for (child, is_active, _) in children {
                let constraint = child
                    .history()
                    .iter_all()
                    .next()
                    .expect("committed child carries the split constraint");
                match &constraint {
                    crate::beta_crown::branching::GraphConstraint::Relu(nc) => {
                        assert_eq!(nc.node_name, "relu1");
                        assert_eq!(
                            nc.neuron_idx, 0,
                            "Min reduce must pick the genuinely-tightening split n0"
                        );
                        assert_eq!(nc.is_active, *is_active);
                    }
                    other => panic!("unexpected constraint kind: {other:?}"),
                }
            }
        });
    }

    /// Regression for the selector -> process boundary: a terminal simulated
    /// parent-cover certificate must bypass split-leaf construction, Complete
    /// Clip, and both dense/scalar child evaluators. Sending this cover onward
    /// would create an empty objective matrix and turn a sound proof into
    /// PropagationFailure.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_certified_children_bypass_empty_target_propagation() {
        crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_KFSB", "1"),
                ("NY_MO_KFSB_CERT_REUSE", "1"),
                ("NY_MO_KFSB_REDUCE", "min"),
            ],
            || {
                let (graph, domain) = kfsb_fixture();
                let mut verifier = BetaCrownVerifier::new(BetaCrownConfig {
                    branching_heuristic: BranchingHeuristic::Kfsb,
                    fsb_candidates: 2,
                    kfsb_reduce_op: KfsbReduceOp::Min,
                    use_kfsb_multi_branching: true,
                    beta_iterations: 0,
                    ..Default::default()
                });
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
                verifier.config.alpha_config.deadline = Some(deadline);
                let engine = NaiveCpuGemmEngine;
                let wave = vec![(
                    0usize,
                    &domain,
                    vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)],
                )];
                let committed = verifier.select_graph_branch_kfsb_multi_batched(
                    0,
                    &graph,
                    &wave,
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &engine,
                );
                let selected = committed.get(&0).expect("parent cover must commit");
                assert_eq!(selected.len(), 1, "terminal proof skips both split leaves");
                let (cover, _, effect) = &selected[0];
                let KfsbCertEffect::ParentComplete(receipt) = effect else {
                    panic!("terminal fast path needs typed ParentComplete, got {effect:?}");
                };
                assert_eq!(receipt.row, 0);
                assert!(matches!(
                    &receipt.scope,
                    super::super::children::KfsbCertScope::ParentCover
                ));
                assert_eq!(receipt.lower_bits, cover.objective_bounds()[0].0.to_bits());
                assert_eq!(receipt.authority_deadline, deadline);
                assert_eq!(
                    cover.history().exact_provenance_identity(),
                    domain.history().exact_provenance_identity()
                );
                assert_eq!(cover.depth(), domain.depth());
                assert!(cover.all_verified());
                assert!(cover.cached_las()[0].is_none());
                assert!(
                    cover.node_bounds().is_empty(),
                    "verified shell retains no node cache"
                );
                assert!(std::ptr::eq(cover.input_bounds(), domain.input_bounds()));

                let results = verifier.process_graph_domains_batched_gpu_multi_objective(
                    0,
                    &graph,
                    &[&domain],
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &engine,
                    None,
                    None,
                );
                let MultiObjectiveGraphDomainResult::Children(children) = &results[0] else {
                    panic!(
                        "certified cover must survive as Children, got {:?}",
                        results[0]
                    );
                };
                assert_eq!(children.len(), 1);
                assert!(children.iter().all(|(child, verified)| {
                    *verified
                        && child.all_verified()
                        && child.lower_bound() > 0.0
                        && child.history().exact_provenance_identity()
                            == domain.history().exact_provenance_identity()
                        && child.depth() == domain.depth()
                        && child.node_bounds().is_empty()
                        && std::ptr::eq(child.input_bounds(), domain.input_bounds())
                }));
            },
        );
    }

    /// Gate-off preserves the historical exhaustive winner split. In
    /// particular, an all-verifying simulated pair alone cannot collapse the
    /// parent without typed certificate authority.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_terminal_parent_cover_gate_off_commits_normal_split() {
        crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_KFSB", "1"),
                ("NY_MO_KFSB_CERT_REUSE", "0"),
                ("NY_MO_KFSB_REDUCE", "min"),
            ],
            || {
                let (graph, domain) = kfsb_fixture();
                let mut verifier = BetaCrownVerifier::new(BetaCrownConfig {
                    branching_heuristic: BranchingHeuristic::Kfsb,
                    fsb_candidates: 2,
                    kfsb_reduce_op: KfsbReduceOp::Min,
                    use_kfsb_multi_branching: true,
                    beta_iterations: 0,
                    ..Default::default()
                });
                verifier.config.alpha_config.deadline =
                    Some(std::time::Instant::now() + std::time::Duration::from_secs(20));
                let wave = vec![(
                    0usize,
                    &domain,
                    vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)],
                )];
                let committed = verifier.select_graph_branch_kfsb_multi_batched(
                    0,
                    &graph,
                    &wave,
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &NaiveCpuGemmEngine,
                );

                let children = committed.get(&0).expect("winner split must commit");
                assert_eq!(children.len(), 2);
                assert!(children.iter().all(|(child, _, effect)| {
                    matches!(effect, KfsbCertEffect::None)
                        && child.depth() == domain.depth() + 1
                        && child.history().exact_provenance_identity()
                            != domain.history().exact_provenance_identity()
                }));
            },
        );
    }

    /// A non-terminal KFSB receipt must remain attached to the exact child,
    /// prune only its certified row, and survive both child-evaluation routes.
    /// This covers the selector -> typed tuple -> dense/scalar evaluator ->
    /// objective/cache merge -> final parent assembly dataflow.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_partial_row_receipt_survives_dense_and_scalar_assembly() {
        crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_KFSB", "1"),
                ("NY_MO_KFSB_K", "2"),
                ("NY_MO_KFSB_CERT_REUSE", "1"),
                ("NY_MO_KFSB_REDUCE", "min"),
                ("NY_MO_GPU_BETA", "0"),
                ("NY_BAB_DROP_VIOLATED_CHILD", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
            ],
            || {
                let objectives = vec![vec![1.0_f32], vec![1.0_f32]];
                let thresholds = vec![0.0_f32, 4.0_f32];
                let relu_nodes = vec!["relu1".to_string()];
                let engine = NaiveCpuGemmEngine;
                let make_verifier = |beta_iterations| {
                    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig {
                        branching_heuristic: BranchingHeuristic::Kfsb,
                        fsb_candidates: 2,
                        kfsb_reduce_op: KfsbReduceOp::Min,
                        use_kfsb_multi_branching: true,
                        beta_iterations,
                        ..Default::default()
                    });
                    verifier.config.alpha_config.deadline =
                        Some(std::time::Instant::now() + std::time::Duration::from_secs(20));
                    verifier
                };
                let root_phase = |child: &MultiObjectiveGraphBabDomain| match child
                    .history()
                    .iter_all()
                    .next()
                    .expect("committed child has a root constraint")
                {
                    crate::beta_crown::branching::GraphConstraint::Relu(neuron) => neuron.is_active,
                    other => panic!("unexpected root constraint: {other:?}"),
                };

                // Establish the fixture's exact typed authority before testing
                // either consumer. This prevents a full two-row recomputation
                // from accidentally making the process-level assertions pass.
                let (selector_graph, selector_domain) = kfsb_partial_receipt_fixture();
                let selector_verifier = make_verifier(0);
                let selector_caches = selector_domain
                    .cached_las()
                    .iter()
                    .map(|cache| Arc::clone(cache.as_ref().expect("fixture has full-spec cache")))
                    .collect::<Vec<_>>();
                let selector_wave = vec![(0usize, &selector_domain, vec![("relu1".into(), 0)])];
                let committed = selector_verifier.select_graph_branch_kfsb_multi_batched(
                    0,
                    &selector_graph,
                    &selector_wave,
                    &relu_nodes,
                    &objectives,
                    &thresholds,
                    &engine,
                );
                let selected_children = committed.get(&0).expect("KFSB candidate must commit");
                assert_eq!(selected_children.len(), 2);
                let mut certified_lower_by_phase = HashMap::new();
                for (child, is_active, effect) in selected_children {
                    let KfsbCertEffect::RowVerified(receipt) = effect else {
                        panic!("partial fixture must produce RowVerified, got {effect:?}");
                    };
                    assert_eq!(receipt.row, 0);
                    assert_eq!(child.verified(), &[true, false]);
                    assert!(!child.all_verified());
                    assert!(child.cached_las().iter().all(Option::is_some));
                    for (row, parent_cache) in selector_caches.iter().enumerate() {
                        let child_cache = child.cached_las()[row]
                            .as_ref()
                            .expect("committed child retains full-spec cache");
                        assert!(Arc::ptr_eq(parent_cache, child_cache));
                        assert_eq!(
                            child_cache.lower_a["relu1"]
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            parent_cache.lower_a["relu1"]
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                    let scorer_rows = child
                        .cached_las()
                        .iter()
                        .map(Option::as_deref)
                        .collect::<Option<Vec<_>>>()
                        .expect("committed child exposes every cached spec row");
                    let decisions = crate::beta_crown::engine::graph::propagation::batched::interm_refine::complete_clip_decisions_from_cached_las(
                        &selector_graph,
                        selector_domain.node_bounds(),
                        child.node_bounds(),
                        child.history(),
                        &scorer_rows,
                        1,
                        None,
                    )
                    .expect("retained cache must be directly consumable by Complete-Clip");
                    assert_eq!(decisions.get("relu1"), Some(&vec![0]));
                    assert_eq!(root_phase(child), *is_active);
                    assert_eq!(
                        child.objective_bounds()[receipt.row].0.to_bits(),
                        receipt.lower_bits
                    );
                    assert!(certified_lower_by_phase
                        .insert(*is_active, receipt.lower_bits)
                        .is_none());
                }
                assert_eq!(certified_lower_by_phase.len(), 2);

                // Exercise the union-pruned dense adapter directly. Its returned
                // full vector must retain the receipt-installed row bit-for-bit
                // while evaluating only row 1.
                let selected_refs: Vec<&MultiObjectiveGraphBabDomain> = selected_children
                    .iter()
                    .map(|(child, _, _)| child)
                    .collect();
                let dense_results = selector_verifier
                    .batched_single_pass_multi_objective_children(
                        &selector_graph,
                        &selected_refs,
                        &relu_nodes,
                        &objectives,
                        &thresholds,
                        &engine,
                        true,
                    )
                    .expect("partial children are dense-adapter compatible");
                assert_eq!(dense_results.len(), selected_children.len());
                for ((selected, _, _), dense_result) in selected_children.iter().zip(dense_results)
                {
                    let (bounds, _, _, _, active_cached_las, pruned, _) =
                        dense_result.expect("remaining row must propagate");
                    assert_eq!(pruned.active_indices, vec![1]);
                    assert_eq!(active_cached_las.len(), 1);
                    assert_eq!(
                        bounds[0].0.to_bits(),
                        selected.objective_bounds()[0].0.to_bits(),
                        "dense merge must retain the certified lower bit-for-bit"
                    );
                    assert!(bounds[1].0 < thresholds[1]);
                }

                // beta_iterations=0 selects the ordinary dense adapter on this
                // non-convolution graph; beta_iterations=1 selects the exact
                // per-child analytical-beta route. Both must publish the same
                // receipt-bearing row and a still-open second row.
                for beta_iterations in [0, 1] {
                    let (graph, domain) = kfsb_partial_receipt_fixture();
                    let certified_parent_cache = Arc::clone(
                        domain.cached_las()[0]
                            .as_ref()
                            .expect("fixture has certified-row cache"),
                    );
                    let verifier = make_verifier(beta_iterations);
                    let results = verifier.process_graph_domains_batched_gpu_multi_objective(
                        0,
                        &graph,
                        &[&domain],
                        &relu_nodes,
                        &objectives,
                        &thresholds,
                        &engine,
                        None,
                        None,
                    );
                    let MultiObjectiveGraphDomainResult::Children(children) = &results[0] else {
                        panic!(
                            "partial receipt must survive beta_iterations={beta_iterations}, got {:?}",
                            results[0]
                        );
                    };
                    assert_eq!(children.len(), 2);
                    for (child, all_verified) in children {
                        let phase = root_phase(child);
                        assert!(!*all_verified);
                        assert_eq!(child.verified(), &[true, false]);
                        assert!(!child.all_verified());
                        assert_eq!(
                            child.objective_bounds()[0].0.to_bits(),
                            certified_lower_by_phase[&phase],
                            "child evaluation must not overwrite the receipt row"
                        );
                        assert!(child.objective_bounds()[1].0 < thresholds[1]);
                        let retained_cache = child.cached_las()[0]
                            .as_ref()
                            .expect("certified row stays available to next-wave Complete-Clip");
                        assert!(Arc::ptr_eq(&certified_parent_cache, retained_cache));
                        assert_eq!(
                            retained_cache.lower_a["relu1"]
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            certified_parent_cache.lower_a["relu1"]
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(child.cached_las().len(), thresholds.len());
                    }
                }
            },
        );
    }

    /// Receipt authority is strict through the final parent publication, not
    /// merely when the simulated scalar is captured or the child is partitioned.
    /// The test clock affects only that final boundary; all proof-producing and
    /// child-evaluation work must complete under the real live deadline first.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_partial_row_receipt_late_publication_is_deadline_expired() {
        crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_KFSB", "1"),
                ("NY_MO_KFSB_K", "2"),
                ("NY_MO_KFSB_CERT_REUSE", "1"),
                ("NY_MO_KFSB_REDUCE", "min"),
                ("NY_MO_GPU_BETA", "0"),
                ("NY_BAB_DROP_VIOLATED_CHILD", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
            ],
            || {
                let (graph, domain) = kfsb_partial_receipt_fixture();
                let objectives = vec![vec![1.0_f32], vec![1.0_f32]];
                let thresholds = vec![0.0_f32, 4.0_f32];
                let relu_nodes = vec!["relu1".to_string()];
                let engine = NaiveCpuGemmEngine;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
                let mut verifier = BetaCrownVerifier::new(BetaCrownConfig {
                    branching_heuristic: BranchingHeuristic::Kfsb,
                    fsb_candidates: 2,
                    kfsb_reduce_op: KfsbReduceOp::Min,
                    use_kfsb_multi_branching: true,
                    beta_iterations: 0,
                    ..Default::default()
                });
                verifier.config.alpha_config.deadline = Some(deadline);

                let results = with_kfsb_final_publication_now(deadline, || {
                    verifier.process_graph_domains_batched_gpu_multi_objective(
                        0,
                        &graph,
                        &[&domain],
                        &relu_nodes,
                        &objectives,
                        &thresholds,
                        &engine,
                        None,
                        None,
                    )
                });
                assert!(
                    matches!(
                        results.as_slice(),
                        [MultiObjectiveGraphDomainResult::DeadlineExpired]
                    ),
                    "a parent still depending on a partial receipt must not publish at its strict deadline: {results:?}"
                );
            },
        );
    }

    /// αβ-CROWN parity (end-to-end): the configured Max reduction must remain
    /// authoritative in the multi-objective lane. Here n1's inactive half is
    /// infeasible (+inf), so Max selects n1 and commits its one feasible child.
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_multi_lane_honors_configured_max() {
        crate::tests::with_serialized_env_vars_removed(&["NY_MO_KFSB_REDUCE"], || {
            let (graph, domain) = kfsb_fixture();
            let verifier = kfsb_verifier(KfsbReduceOp::Max); // cifar100 preset value
            let unstable = vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)];
            let wave = vec![(3usize, &domain, unstable)];
            let engine = NaiveCpuGemmEngine;

            let committed = verifier.select_graph_branch_kfsb_multi_batched(
                0,
                &graph,
                &wave,
                &["relu1".to_string()],
                &[vec![1.0]],
                &[0.0],
                &engine,
            );

            let children = committed.get(&3).expect("wave domain must be resolved");
            assert_eq!(children.len(), 1, "n1 has one feasible child");
            for (child, is_active, _) in children {
                let constraint = child.history().iter_all().next().expect("constraint");
                match &constraint {
                    crate::beta_crown::branching::GraphConstraint::Relu(nc) => {
                        assert_eq!(nc.node_name, "relu1");
                        assert_eq!(nc.neuron_idx, 1, "configured Max must select n1");
                        assert_eq!(nc.is_active, *is_active);
                    }
                    other => panic!("unexpected constraint kind: {other:?}"),
                }
            }
        });
    }

    /// The configured reduce op is authoritative, while the measurement-only
    /// environment override can still bind in either direction.
    #[test]
    fn kfsb_multi_reduce_op_honors_config_and_override() {
        use super::super::kfsb_multi::resolve_kfsb_multi_reduce_op;
        assert_eq!(
            resolve_kfsb_multi_reduce_op(KfsbReduceOp::Max, None),
            KfsbReduceOp::Max
        );
        assert_eq!(
            resolve_kfsb_multi_reduce_op(KfsbReduceOp::Min, None),
            KfsbReduceOp::Min
        );
        assert_eq!(
            resolve_kfsb_multi_reduce_op(KfsbReduceOp::Max, Some("mean")),
            KfsbReduceOp::Max
        );
        assert_eq!(
            resolve_kfsb_multi_reduce_op(KfsbReduceOp::Max, Some("")),
            KfsbReduceOp::Max
        );
        assert_eq!(
            resolve_kfsb_multi_reduce_op(KfsbReduceOp::Max, Some("min")),
            KfsbReduceOp::Min
        );
        assert_eq!(
            resolve_kfsb_multi_reduce_op(KfsbReduceOp::Min, Some("max")),
            KfsbReduceOp::Max
        );
    }

    #[test]
    fn complete_clip_decision_deadline_preserves_full_authority_reserve() {
        let now = std::time::Instant::now();
        let reserve = std::time::Duration::from_secs(5);
        let slack = std::time::Duration::from_millis(1);

        assert_eq!(
            complete_clip_decision_scoring_deadline(now, now + reserve),
            None,
            "the exact reserve boundary has no private-work budget"
        );
        assert_eq!(
            complete_clip_decision_scoring_deadline(
                now,
                (now + reserve)
                    .checked_sub(slack)
                    .expect("one millisecond fits within the authority reserve"),
            ),
            None,
            "less than the reserve must refuse private work"
        );
        assert_eq!(
            complete_clip_decision_scoring_deadline(now, now + reserve + slack),
            Some(now + slack),
            "the scorer deadline must be authority minus the full reserve"
        );
    }

    /// Gate default-OFF: without NY_MO_KFSB=1 the wave selector never arms,
    /// whatever the heuristic/candidate config — the batched lane stays
    /// byte-identical to the advisory path.
    #[test]
    fn kfsb_multi_gate_is_default_off() {
        crate::tests::with_serialized_env_vars_removed(&["NY_MO_KFSB"], || {
            let verifier = kfsb_verifier(KfsbReduceOp::Max);
            assert!(!verifier.kfsb_multi_wave_enabled());
        });
    }

    /// #kfsb-multi tri-state arming (config opt-in + env kill switch):
    /// (1) `config.use_kfsb_multi_branching = true` + Kfsb + candidates>0 and NO
    ///     env ⇒ ARMED (the cifar100-preset default-on path);
    /// (2) config false and no env ⇒ OFF (byte-identical to the advisory path);
    /// (3) `NY_MO_KFSB=0` force-DISARMS even with the config armed (kill switch).
    #[test]
    fn kfsb_multi_gate_tri_state_arming() {
        crate::tests::with_env_edits(|env| {
            env.remove("NY_MO_KFSB");
            let armed = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                fsb_candidates: 2,
                use_kfsb_multi_branching: true,
                beta_iterations: 0,
                ..Default::default()
            });
            // (1) config-armed, env unset ⇒ on.
            assert!(
                armed.kfsb_multi_wave_enabled(),
                "config.use_kfsb_multi_branching=true + Kfsb + candidates>0 must arm the wave lane"
            );

            // (2) config false, env unset ⇒ off.
            let disarmed = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                fsb_candidates: 2,
                use_kfsb_multi_branching: false,
                beta_iterations: 0,
                ..Default::default()
            });
            assert!(
                !disarmed.kfsb_multi_wave_enabled(),
                "config.use_kfsb_multi_branching=false with no env must keep the wave lane off"
            );

            // (3) KILL SWITCH: NY_MO_KFSB=0 forces off despite the config arming.
            env.set("NY_MO_KFSB", "0");
            assert!(
                !armed.kfsb_multi_wave_enabled(),
                "NY_MO_KFSB=0 must force the wave lane off (kill switch) despite config arming"
            );
        });
    }

    #[test]
    fn typed_depth_two_gate_uses_canonical_first_five_rounds_and_kill_switch() {
        crate::tests::with_env_edits(|env| {
            for name in [
                "NY_MO_KFSB",
                "NY_BRANCH_KFSB_CHILDSIM",
                "NY_MO_ADAPTIVE_DEPTH_SHADOW",
                "NY_MO_ADAPTIVE_DEPTH_SELECT",
                "NY_MO_ADAPTIVE_DEPTH_COMMIT",
                "NY_MO_KFSB_F64_SHADOW",
                "NY_MO_KFSB_REDUCE",
                "NY_MO_KFSB_K",
            ] {
                env.remove(name);
            }
            let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                use_kfsb_multi_branching: false,
                depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                    mode: DepthTwoBranchLookaheadMode::Select,
                    ..Default::default()
                },
                ..Default::default()
            });
            assert!(
                !verifier.kfsb_multi_wave_enabled(),
                "legacy round-agnostic gate remains unchanged"
            );
            assert!(verifier.kfsb_multi_wave_enabled_at_round(0));
            assert!(verifier.kfsb_multi_wave_enabled_at_round(4));
            assert!(!verifier.kfsb_multi_wave_enabled_at_round(5));
            env.set("NY_MO_KFSB_REDUCE", "max");
            assert!(
                !verifier.kfsb_multi_wave_enabled_at_round(0),
                "unsupported Max reduction must not OR-arm a declined typed experiment"
            );
            env.remove("NY_MO_KFSB_REDUCE");

            env.set("NY_MO_KFSB_K", "0");
            assert!(
                !verifier.kfsb_multi_wave_enabled_at_round(0),
                "a zero effective candidate budget must decline before Select can OR-arm"
            );
            env.remove("NY_MO_KFSB_K");

            env.set("NY_MO_ADAPTIVE_DEPTH_SHADOW", "1");
            env.set("NY_MO_ADAPTIVE_DEPTH_SELECT", "0");
            env.set("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0");
            env.set("NY_MO_KFSB_F64_SHADOW", "0");
            assert!(
                verifier.kfsb_multi_wave_enabled_at_round(0),
                "advice-only SHADOW must not suppress typed Select arming"
            );

            env.set("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0");
            env.set("NY_MO_ADAPTIVE_DEPTH_SELECT", "1");
            assert!(
                verifier.kfsb_multi_wave_enabled_at_round(0),
                "advice-only legacy SELECT must not suppress typed Select arming"
            );

            env.set("NY_MO_ADAPTIVE_DEPTH_SELECT", "0");
            env.set("NY_MO_ADAPTIVE_DEPTH_COMMIT", "1");
            assert!(
                verifier.kfsb_multi_wave_enabled_at_round(0),
                "advice-only COMMIT must not suppress typed Select arming"
            );

            env.set("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0");
            env.set("NY_MO_KFSB_F64_SHADOW", "1");
            assert!(
                verifier.kfsb_multi_wave_enabled_at_round(0),
                "the precision observer must not suppress typed Select arming"
            );
            for name in [
                "NY_MO_ADAPTIVE_DEPTH_SHADOW",
                "NY_MO_ADAPTIVE_DEPTH_SELECT",
                "NY_MO_ADAPTIVE_DEPTH_COMMIT",
                "NY_MO_KFSB_F64_SHADOW",
            ] {
                env.remove(name);
            }
            let shadow_only = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                fsb_candidates: 2,
                use_kfsb_multi_branching: false,
                depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                    mode: DepthTwoBranchLookaheadMode::Shadow,
                    ..Default::default()
                },
                ..Default::default()
            });
            assert!(
                !shadow_only.kfsb_multi_wave_enabled_at_round(0),
                "typed Shadow must not replace the historical advisory selector by OR-arming kFSB"
            );
            let piggyback_shadow = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                fsb_candidates: 2,
                use_kfsb_multi_branching: true,
                depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                    mode: DepthTwoBranchLookaheadMode::Shadow,
                    ..Default::default()
                },
                ..Default::default()
            });
            assert!(
                piggyback_shadow.kfsb_multi_wave_enabled_at_round(0),
                "Shadow remains observable when the historical kFSB lane is independently armed"
            );
            let zero_candidates = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                fsb_candidates: 0,
                use_kfsb_multi_branching: false,
                depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                    mode: DepthTwoBranchLookaheadMode::Select,
                    ..Default::default()
                },
                ..Default::default()
            });
            assert!(
                !zero_candidates.kfsb_multi_wave_enabled_at_round(0),
                "typed selection needs a nonempty historical fallback prefix"
            );
            let invalid = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                    mode: DepthTwoBranchLookaheadMode::Select,
                    candidates: 16,
                    ..Default::default()
                },
                ..Default::default()
            });
            assert!(
                !invalid.kfsb_multi_wave_enabled_at_round(0),
                "unchecked invalid policies must also fail closed at runtime"
            );
            env.set("NY_MO_KFSB", "0");
            assert!(!verifier.kfsb_multi_wave_enabled_at_round(0));
        });
    }

    /// An admitted typed receipt owns its wave. Legacy observer flags cannot
    /// change the typed result or consume either verifier-lifetime one-shot;
    /// both receipts remain available after typed top-round eligibility ends.
    #[ntest::timeout(30000)]
    #[test]
    fn typed_depth_two_priority_defers_legacy_observers_to_later_round() {
        type Snapshot = Vec<(bool, Vec<GraphNeuronConstraint>, usize)>;

        let (graph, domain) = adaptive_depth_fixture();
        let make_verifier = || {
            BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::Kfsb,
                fsb_candidates: 3,
                use_kfsb_multi_branching: true,
                beta_iterations: 0,
                depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig {
                    mode: DepthTwoBranchLookaheadMode::Select,
                    candidates: 3,
                    ..Default::default()
                },
                ..Default::default()
            })
        };
        let make_wave = |parent_index| {
            vec![(
                parent_index,
                &domain,
                (0..3).map(|index| ("relu1".to_string(), index)).collect(),
            )]
        };
        let snapshot = |children: &super::super::kfsb_multi::KfsbMultiChildren| -> Snapshot {
            children
                .iter()
                .map(|(child, root_phase, _)| {
                    (
                        *root_phase,
                        child.history().constraints.clone(),
                        child.depth(),
                    )
                })
                .collect()
        };

        let control = crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "0"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "0"),
                ("NY_MO_KFSB_F64_SHADOW", "0"),
                ("NY_MO_KFSB_K", "3"),
                ("NY_MO_KFSB_REDUCE", "min"),
            ],
            || {
                let verifier = make_verifier();
                let committed = verifier.select_graph_branch_kfsb_multi_batched(
                    0,
                    &graph,
                    &make_wave(71),
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &NaiveCpuGemmEngine,
                );
                snapshot(committed.get(&71).expect("typed control commits"))
            },
        );

        crate::tests::with_serialized_env_vars(
            &[
                ("NY_MO_ADAPTIVE_DEPTH_SHADOW", "1"),
                ("NY_MO_ADAPTIVE_DEPTH_SELECT", "1"),
                ("NY_MO_ADAPTIVE_DEPTH_COMMIT", "1"),
                ("NY_MO_KFSB_F64_SHADOW", "1"),
                ("NY_MO_KFSB_K", "3"),
                ("NY_MO_KFSB_REDUCE", "min"),
            ],
            || {
                let verifier = make_verifier();
                let typed = verifier.select_graph_branch_kfsb_multi_batched(
                    0,
                    &graph,
                    &make_wave(71),
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &NaiveCpuGemmEngine,
                );
                assert_eq!(
                    snapshot(typed.get(&71).expect("typed armed commits")),
                    control,
                    "legacy observers must not change typed committed children",
                );
                assert!(
                    !verifier
                        .adaptive_depth_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "typed priority must not consume the scalar one-shot",
                );
                assert!(
                    !verifier
                        .kfsb_f64_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "typed priority must not consume the f64 one-shot",
                );

                assert!(
                    verifier.kfsb_multi_wave_enabled_at_round(5),
                    "the historical kFSB lane must dispatch the later non-typed wave",
                );
                let later = verifier.select_graph_branch_kfsb_multi_batched(
                    5,
                    &graph,
                    &make_wave(72),
                    &["relu1".to_string()],
                    &[vec![1.0]],
                    &[0.0],
                    &NaiveCpuGemmEngine,
                );
                assert!(later.contains_key(&72));
                assert!(
                    verifier
                        .adaptive_depth_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "the scalar one-shot must remain claimable after typed rounds",
                );
                assert!(
                    verifier
                        .kfsb_f64_shadow_fired
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "the f64 one-shot must remain claimable after typed rounds",
                );
            },
        );
    }

    /// Stratified layer quota: each unstable ReLU layer's top-1 main-score
    /// candidate joins the eval set exactly once, deduplicated against the
    /// top-k selection.
    #[test]
    fn kfsb_multi_layer_quota_appends_top1_per_layer() {
        let cand = |layer: &str, idx: usize, main: f32| GraphKfsbCandidate {
            node_name: layer.to_string(),
            neuron_idx: idx,
            main_score: main,
            backup_score: 0.0,
        };
        // Sorted by main desc, spanning two layers.
        let scored = vec![
            cand("Relu_31", 109, 5.0),
            cand("Relu_31", 7, 4.0),
            cand("Relu_13", 3478, 3.0),
            cand("Relu_13", 12, 2.0),
        ];
        // Top-1 selection saw only Relu_31's best.
        let mut candidates = vec![cand("Relu_31", 109, 5.0)];
        append_layer_quota_candidates(&scored, &mut candidates);
        assert_eq!(
            candidates.len(),
            2,
            "quota adds exactly the missing layer's top-1"
        );
        assert_eq!(candidates[1].node_name, "Relu_13");
        assert_eq!(candidates[1].neuron_idx, 3478);

        // Idempotent: a second pass adds nothing.
        let mut again = candidates.clone();
        append_layer_quota_candidates(&scored, &mut again);
        assert_eq!(again.len(), 2);
    }
}

/// #cone-delta KFSB shim wiring: a candidate child of a freshly bounded
/// parent carries `delta = [candidate's pre-activation node]`, and
/// `graph_bab_domain_shim` transfers that delta verbatim onto the shim the
/// dense-spec batched primitive scores — the 2K-per-domain amplification
/// case. A root (never-bounded) domain's shim keeps the delta-unknown
/// sentinel so the forward fails closed to full-history seeding.
#[test]
fn test_kfsb_shim_carries_candidate_delta_pre_node_cone_delta() {
    let linear1 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5], [0.25, 0.75]]),
        Some(arr1(&[0.0_f32, 0.1])),
    )
    .expect("linear1");
    let linear2 =
        LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), Some(arr1(&[0.0_f32]))).expect("linear2");
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

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.75]).into_dyn(),
    )
    .expect("input box");
    let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
    let mut parent =
        MultiObjectiveGraphBabDomain::root(node_bounds, vec![(0.0, 1.0)], &input, &[0.0], false)
            .expect("root domain");

    // Root: delta unknown (sentinel) — and the shim transfers it verbatim.
    let root_shim = super::batched_dense_specs::graph_bab_domain_shim(&parent);
    assert_eq!(
        root_shim.delta_pre_nodes(),
        &[crate::NETWORK_INPUT.to_string()],
        "never-bounded root shim keeps the delta-unknown sentinel"
    );

    // Simulate the bounding pass fixpointing the parent map.
    parent.delta_pre_nodes.clear();

    // KFSB candidate child: exactly one new constraint on relu1.
    let candidate = parent
        .with_constraint(
            &graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active: true,
                score: 1.0,
            },
            false,
            &[0.0],
        )
        .expect("with_constraint")
        .expect("feasible candidate");
    assert_eq!(
        candidate.delta_pre_nodes(),
        &["linear1".to_string()],
        "candidate delta = its pre-activation node only"
    );

    let shim = super::batched_dense_specs::graph_bab_domain_shim(&candidate);
    assert_eq!(
        shim.delta_pre_nodes(),
        &["linear1".to_string()],
        "shim transfers the candidate's delta verbatim"
    );
    assert_eq!(shim.history().constraints.len(), 1);
}
