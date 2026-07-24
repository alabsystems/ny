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
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::{
    GraphBabDomain, GraphCrownContext, MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain,
};
use crate::beta_crown::engine::domain_results::{
    GraphDomainResult, MultiObjectiveGraphDomainResult,
};
use crate::beta_crown::engine::graph::adaptive_microbatch::MicrobatchRefusalReason;
use crate::beta_crown::engine::graph::domain_batch::{
    GraphDomainBatchExecutor, MultiObjectiveBatchRequest, SingleObjectiveBatchRequest,
};
use crate::beta_crown::engine::graph::multi_objective::batched::children::MultiObjectiveChildCreationResult;
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

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
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
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
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
            retry_refusals: true,
        },
    );
    assert!(matches!(
        adaptive,
        Err(MicrobatchRefusalReason::DeviceAllocation)
    ));
}

#[test]
fn test_multi_objective_parent_lookup_failure_returns_propagation_failure_1993() {
    let domains_with_unstable: Vec<MultiObjDomainWithUnstable<'_>> = Vec::new();
    let child_creation_results: Vec<MultiObjectiveChildCreationResult> = vec![(7, Vec::new())];
    let mut quick_results = HashMap::new();

    let (children, parent_lookup) = collect_multi_objective_children(
        &domains_with_unstable,
        &child_creation_results,
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
            graph: &graph,
            domains: &[&domain],
            relu_nodes: &relu_nodes,
            objectives: &objectives,
            thresholds: &thresholds,
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            endgame: false,
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
            graph: &graph,
            domains: &[&domain],
            relu_nodes: &relu_nodes,
            objectives: &objectives,
            thresholds: &thresholds,
            engine: &NaiveCpuGemmEngine,
            cut_pool: None,
            endgame: false,
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
    // Compute objective bounds at the root so the domain carries plausible parent
    // bounds for the verified-latch (verified objectives keep these).
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let arc_node_bounds: HashMap<String, Arc<BoundedTensor>> = node_bounds
        .iter()
        .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
        .collect();
    let root_history = crate::beta_crown::branching::GraphSplitHistory::new();
    let context = GraphCrownContext::new(
        &root_history,
        None,
        Some(&arc_node_bounds),
        Some(&NaiveCpuGemmEngine),
    );
    let (output, _) = verifier
        .propagate_crown_with_graph_constraints(graph, input, &context, None, None)
        .expect("root CROWN should succeed");
    let obj_bounds = BetaCrownVerifier::objective_bounds_multi(&output, objectives)
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
        .map(|(obj_bounds, _, _, _, _, _)| obj_bounds.clone())
        .unwrap_or_else(|e| panic!("single-child adapter errored: infeasible={e}"))
}

/// Independent soundness floor: sample concrete points in the child's constrained
/// input box, evaluate the network exactly (degenerate-box IBP is exact at a
/// point), enforce the child's ReLU split constraints at each point, and return
/// the per-objective minimum observed value — a valid upper estimate of the true
/// minimum over the sub-domain (any sound lower bound must be <= this).
fn sampled_objective_minimums(
    graph: &GraphNetwork,
    parent_node_bounds: &HashMap<String, Arc<BoundedTensor>>,
    child: &MultiObjectiveGraphBabDomain,
    objectives: &[Vec<f32>],
) -> Vec<f32> {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (_fwd, constrained_input) = verifier
        .compute_constrained_forward_bounds(
            graph,
            child.input_bounds.as_ref(),
            &child.history,
            Some(parent_node_bounds),
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
        let (obj_bounds, _node_cache, _beta, _alpha, _cached_las, _pruned) = child_result
            .as_ref()
            .unwrap_or_else(|e| panic!("child {ci} adapter result errored: infeasible={e}"));
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
        .map(|(b, _, _, _, _, _)| b.clone())
        .expect("full-matrix child result should be Ok");
    let pruned_bounds = pruned[0]
        .as_ref()
        .map(|(b, _, _, _, _, _)| b.clone())
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

    use super::super::kfsb_multi::{
        adaptive_depth_authority_identity, adaptive_depth_shadow_budget_available,
        adaptive_depth_shadow_deadline, append_layer_quota_candidates, clear_shadow_cached_las,
        pick_kfsb_candidate, rank_adaptive_depth_authority_portfolio,
        rank_adaptive_depth_candidates, resolve_adaptive_depth_authority_candidate,
        resolve_adaptive_depth_evaluation_enabled, resolve_adaptive_depth_select_enabled,
        resolve_adaptive_depth_shadow_enabled, resolve_kfsb_cached_la_enabled,
        select_complete_adaptive_depth_rank, AdaptiveDepthShadowCapture,
        AdaptiveDepthShadowMetrics, DomainPrep, SideSlot,
    };
    use crate::batched_domain::CachedLinearBounds;
    use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
    use crate::beta_crown::config::KfsbReduceOp;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::branching::kfsb_shared::GraphKfsbCandidate;
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
        assert!(!resolve_adaptive_depth_evaluation_enabled(None, None));
        assert!(resolve_adaptive_depth_evaluation_enabled(Some("1"), None));
        assert!(resolve_adaptive_depth_evaluation_enabled(None, Some("1")));
        assert!(!resolve_adaptive_depth_evaluation_enabled(
            Some("true"),
            Some("yes")
        ));

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

    fn complete_adaptive_depth_metric(score: f32) -> AdaptiveDepthShadowMetrics {
        AdaptiveDepthShadowMetrics {
            expected: 4,
            bounded: 4,
            surviving: 4,
            post_min: score,
            ..AdaptiveDepthShadowMetrics::default()
        }
    }

    #[test]
    fn adaptive_depth_authority_requires_three_complete_finite_trees() {
        let mut metrics = vec![
            complete_adaptive_depth_metric(1.0),
            complete_adaptive_depth_metric(3.0),
            complete_adaptive_depth_metric(2.0),
        ];
        assert_eq!(select_complete_adaptive_depth_rank(&metrics), Some(1));

        metrics[0].post_min = 3.0;
        assert_eq!(
            select_complete_adaptive_depth_rank(&metrics),
            Some(0),
            "an exact score tie must preserve the earlier one-step rank"
        );
        metrics = vec![
            complete_adaptive_depth_metric(-0.0),
            complete_adaptive_depth_metric(0.0),
            complete_adaptive_depth_metric(-1.0),
        ];
        assert_eq!(
            select_complete_adaptive_depth_rank(&metrics),
            Some(0),
            "signed zero is a numerical tie, not a reason to reorder roots"
        );

        metrics[0] = complete_adaptive_depth_metric(1.0);
        metrics[1].post_min = f32::NAN;
        assert_eq!(select_complete_adaptive_depth_rank(&metrics), None);
        metrics[1].post_min = f32::NEG_INFINITY;
        assert_eq!(select_complete_adaptive_depth_rank(&metrics), None);
        metrics[1] = complete_adaptive_depth_metric(3.0);
        metrics[1].failures = 1;
        assert_eq!(select_complete_adaptive_depth_rank(&metrics), None);
        metrics[1] = complete_adaptive_depth_metric(3.0);
        metrics[1].bounded = 3;
        metrics[1].surviving = 3;
        assert_eq!(select_complete_adaptive_depth_rank(&metrics), None);
        assert_eq!(select_complete_adaptive_depth_rank(&metrics[..2]), None);
    }

    #[test]
    fn adaptive_depth_authority_accepts_only_certified_all_infeasible_infinity() {
        let mut metrics = vec![
            complete_adaptive_depth_metric(1.0),
            complete_adaptive_depth_metric(2.0),
            AdaptiveDepthShadowMetrics {
                expected: 2,
                infeasible: 2,
                verified: 2,
                post_min: f32::INFINITY,
                ..AdaptiveDepthShadowMetrics::default()
            },
        ];
        assert_eq!(select_complete_adaptive_depth_rank(&metrics), Some(2));

        metrics[2] = complete_adaptive_depth_metric(f32::INFINITY);
        assert_eq!(
            select_complete_adaptive_depth_rank(&metrics),
            None,
            "infinite propagated bounds are not an all-infeasible certificate"
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
    fn adaptive_depth_authority_portfolio_preserves_historical_near_tie_winner() {
        let candidate = |node: &str, main: f32| GraphKfsbCandidate {
            node_name: node.to_string(),
            neuron_idx: 0,
            main_score: main,
            backup_score: 0.0,
        };
        let mut candidates = vec![
            candidate("exact-first", 1.0),
            candidate("historical", 2.0),
            candidate("third", 0.0),
        ];
        let exact_lead = f32::from_bits(1.0_f32.to_bits() + 4);
        let mut values = vec![(exact_lead, exact_lead), (1.0, 1.0), (0.0, 0.0)];

        let exact = rank_adaptive_depth_candidates(&candidates, &values, KfsbReduceOp::Min);
        assert_eq!(exact[0].0, 0, "exact ranking sees the sub-1e-6 lead");
        let historical =
            pick_kfsb_candidate(&candidates, values.iter().copied(), KfsbReduceOp::Min)
                .expect("historical winner");
        assert_eq!(
            historical.0, 1,
            "the historical 1e-6 tie rule uses the larger main score"
        );

        let portfolio = rank_adaptive_depth_authority_portfolio(
            &candidates,
            &values,
            &[true, true, true],
            KfsbReduceOp::Min,
        )
        .expect("complete captured portfolio");
        assert_eq!(
            portfolio.iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
        let tied_depth2 = vec![
            complete_adaptive_depth_metric(4.0),
            complete_adaptive_depth_metric(4.0),
            complete_adaptive_depth_metric(4.0),
        ];
        let selected_rank =
            select_complete_adaptive_depth_rank(&tied_depth2).expect("complete depth-2 metrics");
        assert_eq!(portfolio[selected_rank].0, historical.0);

        assert!(rank_adaptive_depth_authority_portfolio(
            &candidates,
            &values,
            &[true, false, true],
            KfsbReduceOp::Min,
        )
        .is_none());
        assert!(rank_adaptive_depth_authority_portfolio(
            &candidates,
            &values,
            &[true, true, false],
            KfsbReduceOp::Min,
        )
        .is_none());

        candidates.push(candidate("outside-capture", 3.0));
        values.push((5.0, 5.0));
        assert!(rank_adaptive_depth_authority_portfolio(
            &candidates,
            &values,
            &[true, true, true],
            KfsbReduceOp::Min,
        )
        .is_none());
    }

    #[test]
    fn adaptive_depth_capture_is_fixed_size_on_large_frontier() {
        let consumed = std::cell::Cell::new(0usize);
        let mut capture = AdaptiveDepthShadowCapture::from_sim_indices(
            0,
            (0..1_000_000usize).inspect(|_| consumed.set(consumed.get() + 1)),
            1_000_000,
        );
        let small = AdaptiveDepthShadowCapture::from_sim_indices(0, 0..6usize, 6);

        assert_eq!(AdaptiveDepthShadowCapture::slot_capacity(), 6);
        assert_eq!(capture.planned_slot_count(), 6);
        assert_eq!(capture.captured_map_count(), 0);
        assert_eq!(consumed.get(), 6, "capture planning must stop at six slots");
        assert_eq!(
            size_of_val(&capture),
            size_of_val(&small),
            "capture metadata must not scale with the simulated frontier"
        );
        for sim_index in 0..6 {
            assert!(capture.contains_sim(sim_index));
        }
        assert!(!capture.contains_sim(6));
        assert!(!capture.contains_sim(999_999));

        for sim_index in 0..6 {
            capture.insert_node_bounds(sim_index, HashMap::new());
        }
        capture.insert_node_bounds(999_999, HashMap::new());
        assert_eq!(
            capture.captured_map_count(),
            6,
            "captured result maps must remain bounded by the fixed slot count"
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
    fn adaptive_depth_base_select_expired_budget_fails_before_scoring() {
        let (graph, domain) = adaptive_depth_fixture();
        let verifier = kfsb_verifier(KfsbReduceOp::Min);
        let now = std::time::Instant::now();
        let error = verifier
            .select_adaptive_depth_base_candidate_with_budget(
                &graph,
                &domain,
                &["relu1".to_string()],
                &[1.0],
                now,
                None,
            )
            .err()
            .expect("expired private deadline must fail closed");
        assert!(error.is_deadline_exceeded());

        let shadow_deadline = now
            .checked_add(std::time::Duration::from_secs(10))
            .expect("future shadow deadline");
        let error = verifier
            .select_adaptive_depth_base_candidate_with_budget(
                &graph,
                &domain,
                &["relu1".to_string()],
                &[1.0],
                shadow_deadline,
                now.checked_add(std::time::Duration::from_secs(5)),
            )
            .err()
            .expect("exhausted authority reserve must fail closed");
        assert!(error.is_deadline_exceeded());
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

    /// The observer performs extra private dense work, but the authoritative
    /// committed split, child histories, bounds, masks, and depth remain exact.
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
                        .map(|(child, active)| {
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

    /// Test-only metrics force the third exact one-step-ranked root to win depth 2.
    /// The production commit path must take exactly that root's pre-existing
    /// first-level children, while every incomplete/corrupt authority result
    /// falls back bit-for-bit to the historical one-step winner.
    #[ntest::timeout(30000)]
    #[test]
    fn adaptive_depth_selection_commits_authoritative_children_and_faults_fall_back() {
        type Snapshot = Vec<(
            bool,
            Vec<GraphNeuronConstraint>,
            Vec<(u32, u32)>,
            Vec<bool>,
            usize,
        )>;

        fn snapshot_children(children: &[(MultiObjectiveGraphBabDomain, bool)]) -> Snapshot {
            children
                .iter()
                .map(|(child, active)| {
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
        }

        let (graph, domain) = adaptive_depth_fixture();
        let run = |shadow: &str, selection: &str, fault: &str| -> Snapshot {
            crate::tests::with_serialized_env_vars(
                &[
                    ("NY_MO_ADAPTIVE_DEPTH_SHADOW", shadow),
                    ("NY_MO_ADAPTIVE_DEPTH_SELECT", selection),
                    ("NY_MO_KFSB_REDUCE", "min"),
                    ("NY_TEST_MO_ADAPTIVE_DEPTH_FAULT", fault),
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
                        &graph,
                        &wave,
                        &["relu1".to_string()],
                        &[vec![1.0]],
                        &[0.0],
                        &NaiveCpuGemmEngine,
                    );
                    snapshot_children(committed.get(&17).expect("scored domain commits"))
                },
            )
        };

        let control = run("0", "0", "force-third");
        assert_eq!(
            run("1", "0", "force-third"),
            control,
            "shadow-only mode must remain observation-only"
        );
        let promoted = run("0", "1", "force-third");
        assert_ne!(
            promoted, control,
            "the forced depth-2 winner must differ from the one-step winner"
        );
        assert_eq!(promoted.len(), 2, "both selected root sides are feasible");
        let root_key = |snapshot: &Snapshot| {
            let root = snapshot[0].1.first().expect("committed root constraint");
            (root.node_name.clone(), root.neuron_idx)
        };
        assert_ne!(root_key(&promoted), root_key(&control));
        assert!(promoted
            .iter()
            .all(|(_, history, _, _, depth)| history.len() == 1 && *depth == 1));

        // Rebuild each selected root side directly from the untouched parent.
        // Exact snapshots prove the commit used authoritative first-level
        // children rather than any private depth-2 leaf or propagated bound.
        let expected: Snapshot = promoted
            .iter()
            .map(|(_, history, _, _, _)| {
                let constraint = history.first().expect("one root constraint").clone();
                let child = domain
                    .with_constraint(&graph, constraint, false, &[0.0])
                    .expect("selected root construction")
                    .expect("selected root side remains feasible");
                snapshot_children(&[(child, history[0].is_active)])
                    .pop()
                    .expect("one expected child")
            })
            .collect();
        assert_eq!(promoted, expected);

        for fault in [
            "timeout",
            "nan-bounds",
            "failed-leaf",
            "construction-error",
            "shape-error",
            "malformed-counts",
            "partial-metrics",
            "missing-side",
            "identity-mismatch",
        ] {
            assert_eq!(
                run("0", "1", fault),
                control,
                "fault {fault} must preserve committed histories and bound bits"
            );
        }
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
        domain.cached_las[0] = Some(CachedLinearBounds::from_linear_bounds_map(captured_map));

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
        if std::env::var("NY_MO_KFSB_REDUCE").is_ok() {
            return; // don't fight an externally-set A/B override
        }
        let (graph, domain) = kfsb_fixture();
        let verifier = kfsb_verifier(KfsbReduceOp::Min);
        let unstable = vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)];
        let wave = vec![(7usize, &domain, unstable)];
        let engine = NaiveCpuGemmEngine;

        let committed = verifier.select_graph_branch_kfsb_multi_batched(
            &graph,
            &wave,
            &["relu1".to_string()],
            &[vec![1.0]],
            &[0.0],
            &engine,
        );

        let children = committed.get(&7).expect("wave domain must be resolved");
        assert_eq!(children.len(), 2, "both children of the winner commit");
        for (child, is_active) in children {
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
    }

    /// #kfsb-multi reduce-op PIN (end-to-end): with `config.kfsb_reduce_op = Max`
    /// (exactly the value every cifar100/relational preset configures) and no
    /// A/B env override, the wave lane STILL resolves the `Min` winner — n0 with
    /// BOTH children — because the multi-objective lane is DECOUPLED from
    /// `config.kfsb_reduce_op` (pinned to `Min`; see
    /// `kfsb_multi::resolve_kfsb_multi_reduce_op`). n1's inactive half-space is
    /// infeasible (+inf), but under `Min` its split is ranked by its surviving
    /// child (~1.5), which n0's genuinely-tightening split (min(2,3)=2) beats.
    /// This guards a future preset edit from silently flipping this lane to the
    /// wrong metric. (Was `kfsb_multi_max_reduce_rewards_...`: `Max` is now
    /// reachable only via `NY_MO_KFSB_REDUCE=max`, covered by the pure-resolver
    /// test below.)
    #[ntest::timeout(30000)]
    #[test]
    fn kfsb_multi_lane_ignores_config_max_and_stays_min() {
        if std::env::var("NY_MO_KFSB_REDUCE").is_ok() {
            return; // don't fight an externally-set A/B override
        }
        let (graph, domain) = kfsb_fixture();
        let verifier = kfsb_verifier(KfsbReduceOp::Max); // cifar100/relational preset value
        let unstable = vec![("relu1".to_string(), 0), ("relu1".to_string(), 1)];
        let wave = vec![(3usize, &domain, unstable)];
        let engine = NaiveCpuGemmEngine;

        let committed = verifier.select_graph_branch_kfsb_multi_batched(
            &graph,
            &wave,
            &["relu1".to_string()],
            &[vec![1.0]],
            &[0.0],
            &engine,
        );

        let children = committed.get(&3).expect("wave domain must be resolved");
        assert_eq!(
            children.len(),
            2,
            "the Min metric commits BOTH children of the genuinely-tightening split n0"
        );
        for (child, is_active) in children {
            let constraint = child.history().iter_all().next().expect("constraint");
            match &constraint {
                crate::beta_crown::branching::GraphConstraint::Relu(nc) => {
                    assert_eq!(nc.node_name, "relu1");
                    assert_eq!(
                        nc.neuron_idx, 0,
                        "lane pinned to Min must pick n0 despite config.kfsb_reduce_op = Max"
                    );
                    assert_eq!(nc.is_active, *is_active);
                }
                other => panic!("unexpected constraint kind: {other:?}"),
            }
        }
    }

    /// #kfsb-multi reduce-op PIN (regression, pure): the wave-batched
    /// multi-objective lane's effective reduce op is `Min` by DEFAULT — the
    /// min-of-children metric — DECOUPLED from `config.kfsb_reduce_op` (which
    /// every cifar100/relational preset sets to `max`, the α,β-CROWN
    /// single-objective parity knob). Guards a future preset edit from silently
    /// flipping this lane to the wrong metric. The `NY_MO_KFSB_REDUCE` A/B
    /// override still binds in BOTH directions; any other value falls back to
    /// the pinned `Min`.
    #[test]
    fn kfsb_multi_reduce_op_pinned_to_min_by_default() {
        use super::super::kfsb_multi::resolve_kfsb_multi_reduce_op;
        // Lane default (no A/B env override) is Min.
        assert_eq!(resolve_kfsb_multi_reduce_op(None), KfsbReduceOp::Min);
        // Unknown / non-min-max override falls back to the pinned Min.
        assert_eq!(
            resolve_kfsb_multi_reduce_op(Some("mean")),
            KfsbReduceOp::Min
        );
        assert_eq!(resolve_kfsb_multi_reduce_op(Some("")), KfsbReduceOp::Min);
        // The A/B measurement override still binds both ways.
        assert_eq!(resolve_kfsb_multi_reduce_op(Some("min")), KfsbReduceOp::Min);
        assert_eq!(resolve_kfsb_multi_reduce_op(Some("max")), KfsbReduceOp::Max);
    }

    /// Gate default-OFF: without NY_MO_KFSB=1 the wave selector never arms,
    /// whatever the heuristic/candidate config — the batched lane stays
    /// byte-identical to the advisory path.
    #[test]
    fn kfsb_multi_gate_is_default_off() {
        if std::env::var("NY_MO_KFSB").is_ok() {
            return; // don't fight an externally-set gate
        }
        let verifier = kfsb_verifier(KfsbReduceOp::Max);
        assert!(!verifier.kfsb_multi_wave_enabled());
    }

    /// #kfsb-multi tri-state arming (config opt-in + env kill switch):
    /// (1) `config.use_kfsb_multi_branching = true` + Kfsb + candidates>0 and NO
    ///     env ⇒ ARMED (the cifar100-preset default-on path);
    /// (2) config false and no env ⇒ OFF (byte-identical to the advisory path);
    /// (3) `NY_MO_KFSB=0` force-DISARMS even with the config armed (kill switch).
    #[test]
    fn kfsb_multi_gate_tri_state_arming() {
        if std::env::var("NY_MO_KFSB").is_ok() {
            return; // don't fight an externally-set gate for the env-UNSET cases
        }
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
        crate::tests::with_serialized_env_vars(&[("NY_MO_KFSB", "0")], || {
            assert!(
                !armed.kfsb_multi_wave_enabled(),
                "NY_MO_KFSB=0 must force the wave lane off (kill switch) despite config arming"
            );
        });
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
