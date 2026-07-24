// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

use super::*;
use crate::beta_crown::config::{BetaCrownConfig, InputClipType};
use crate::beta_crown::engine::graph::input_split::grouped_semantics::disjunctive_domain_verified;
use crate::beta_crown::engine::graph::input_split::shared::{
    extract_obj_bounds, graph_spec_ibp_fallback,
};
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::{Layer, LinearLayer};
use crate::{GraphNetwork, GraphNode};

fn build_disjunctive_batch_graph_4353() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid output linear")),
    ));
    graph.set_output("out");
    graph
}

fn unresolved_multi_obj_domain_4353() -> MultiObjInputDomain {
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite parent bounds");
    MultiObjInputDomain {
        input_bounds: Arc::new(input_bounds),
        obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
        linear_bounds: None,
        depth: 0,
        priority: 1.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn disjunctive_baseline_gemm_calls_4353(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
) -> usize {
    let left_child = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn())
        .expect("finite left child");
    let right_child = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite right child");
    let engine = CountingGemmEngine::new();
    let baseline_verified: Vec<bool> = [&left_child, &right_child]
        .into_iter()
        .map(|child| {
            let (bounds, _) =
                graph_spec_ibp_fallback(graph, child, spec_matrix, Some(&engine), None)
                    .expect("per-child grouped IBP fallback should succeed");
            let obj_bounds = extract_obj_bounds(&bounds, thresholds.len()).unwrap();
            disjunctive_domain_verified(&obj_bounds, thresholds, clause_sizes)
        })
        .collect();
    assert_eq!(
        baseline_verified,
        vec![false, true],
        "baseline grouped split children should leave exactly one unresolved domain"
    );
    engine.gemm_calls()
}

fn assert_queued_grouped_child_4353(queue: &mut BinaryHeap<MultiObjInputDomain>) {
    let child = queue
        .pop()
        .expect("one unresolved grouped child should be queued");
    assert!(child.needs_bounding);
    assert_eq!(child.depth, 1);
    assert_eq!(child.obj_bounds, vec![(-1.0, 1.0), (-1.0, 1.0)]);
    assert!(child.linear_bounds.is_none());
    assert!(child.node_bounds_override.is_none());
    assert_eq!(child.input_bounds.lower()[[0]], -1.0);
    assert_eq!(child.input_bounds.upper()[[0]], 0.0);
}

#[test]
fn test_process_disjunctive_domain_batch_reorder_batches_ibp_prescreen_4353() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        input_split_ibp_enhancement: true,
        enable_relaxed_clip: false,
        ..Default::default()
    });
    let graph = build_disjunctive_batch_graph_4353();
    let spec_matrix = arr2(&[[1.0_f32], [0.5_f32]]);
    let thresholds = [-0.1_f32, -0.1_f32];
    let clause_sizes = [1usize, 1usize];
    let baseline_calls =
        disjunctive_baseline_gemm_calls_4353(&graph, &spec_matrix, &thresholds, &clause_sizes);

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;
    let batched_engine = CountingGemmEngine::new();
    let warm_alpha_telemetry = WarmAlphaTelemetry::new(false);

    let result = process_disjunctive_domain_batch(
        &verifier,
        &graph,
        vec![unresolved_multi_obj_domain_4353()],
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        Some(&batched_engine),
        &|_input, _node_bounds| -> Result<MultiObjBounds> {
            panic!("reorder batched grouped prescreen should not call compute_bounds")
        },
        None,
        &warm_alpha_telemetry,
        None,
        Duration::from_secs(1),
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("batched disjunctive processing should not error");

    assert!(result.is_none());
    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(queue.len(), 1);
    assert_queued_grouped_child_4353(&mut queue);
    assert!(
        batched_engine.gemm_calls() < baseline_calls,
        "batched grouped process path should reduce GEMM dispatches: batched={}, baseline={}",
        batched_engine.gemm_calls(),
        baseline_calls
    );
}

/// Verify that `push_batched_relaxed_survivors` checks grouped disjunctive
/// verification on concretized post-clip bounds, not just box infeasibility.
///
/// Without #4367, children whose linear-bound concretization exceeds the grouped
/// thresholds would be pushed to the queue instead of being counted as verified.
///
/// Tests `push_batched_relaxed_survivors` directly with pre-built FlatPendingChild
/// objects to isolate the grouped verification logic from the pipeline.
#[test]
fn test_batched_relaxed_clip_checks_grouped_verification_after_clip_4367() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 1,
        ..Default::default()
    });

    // Two clauses, each with one row.
    let thresholds = [0.5_f32, 0.5_f32];
    let clause_sizes = [1usize, 1usize];
    let shape = &[1usize]; // 1-dim input

    // Linear bounds: 2 rows (one per threshold), 1 col.
    // lb_row = 5.0 * x_lower + 1.0
    let linear_bounds = LinearBounds::new(
        arr2(&[[5.0_f32], [5.0_f32]]),
        arr1(&[1.0_f32, 1.0_f32]),
        arr2(&[[5.0_f32], [5.0_f32]]),
        arr1(&[1.0_f32, 1.0_f32]),
    )
    .expect("valid linear bounds");

    // Two survivors: one where concretization exceeds threshold, one where it doesn't.
    //
    // Child A [0.0, 1.0]: lb = 5*0 + 1 = 1.0 > 0.5 → should be verified
    // Child B [-0.5, 0.0]: lb = 5*(-0.5) + 1 = -1.5 < 0.5 → should NOT be verified
    //   (after clip, lb tightens toward threshold but doesn't exceed it)
    let survivors = vec![
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(linear_bounds.clone()),
            depth: 1,
            priority: 1.0,
            inherited_alpha_state: None,
        },
        FlatPendingChild {
            flat_lower: arr1(&[-0.5_f32]).into_dyn(),
            flat_upper: arr1(&[0.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(linear_bounds),
            depth: 1,
            priority: 1.0,
            inherited_alpha_state: None,
        },
    ];

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("push_batched_relaxed_survivors should not error");

    // Child A: concretized lb = 1.0 > 0.5 for both rows → both clauses satisfied → verified
    assert_eq!(
        domains_verified_by_clip, 1,
        "child A should be verified by post-clip grouped concretization"
    );
    assert_eq!(lifecycle.domains_verified, 1);
    // Child B: remains in queue
    assert_eq!(
        queue.len(),
        1,
        "child B should be pushed to queue (concretized lb < threshold)"
    );
}

/// Build contradictory linear bounds (row 0: x≤0.2, row 1: x≥0.8 → empty box).
fn lb_infeasible_4366() -> LinearBounds {
    LinearBounds::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.2_f32, 0.8_f32]),
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.2_f32, 0.8_f32]),
    )
    .expect("valid linear bounds")
}

/// Build bounds with large positive coefficients (lb = 10*x + 5 >> threshold).
fn lb_verified_4366() -> LinearBounds {
    LinearBounds::new(
        arr2(&[[10.0_f32], [10.0_f32]]),
        arr1(&[5.0_f32, 5.0_f32]),
        arr2(&[[10.0_f32], [10.0_f32]]),
        arr1(&[5.0_f32, 5.0_f32]),
    )
    .expect("valid linear bounds")
}

/// Build mild bounds that don't verify (lb = 0.1*x - 10 < threshold).
fn lb_survive_4366() -> LinearBounds {
    LinearBounds::new(
        arr2(&[[0.1_f32], [0.1_f32]]),
        arr1(&[-10.0_f32, -10.0_f32]),
        arr2(&[[0.1_f32], [0.1_f32]]),
        arr1(&[-10.0_f32, -10.0_f32]),
    )
    .expect("valid linear bounds")
}

#[test]
fn test_flat_reorder_survivor_routes_preserve_parent_alpha_f8() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 1,
        ..Default::default()
    });
    let seed = Arc::new(GraphAlphaState::new());
    let survivors = vec![
        // Direct no-linear-bounds route.
        FlatPendingChild {
            flat_lower: arr1(&[-1.0_f32]).into_dyn(),
            flat_upper: arr1(&[0.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: None,
            depth: 11,
            priority: 1.0,
            inherited_alpha_state: Some(Arc::clone(&seed)),
        },
        // Batched relaxed-clip survivor route.
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_survive_4366()),
            depth: 12,
            priority: 2.0,
            inherited_alpha_state: Some(Arc::clone(&seed)),
        },
    ];
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        &[1],
        &[0.0, 0.0],
        &[2],
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("flat survivor routes should succeed");

    assert_eq!(domains_verified_by_clip, 0);
    assert_eq!(queue.len(), 2);
    let mut depths = Vec::new();
    while let Some(child) = queue.pop() {
        depths.push(child.depth);
        assert!(child.needs_bounding);
        let carried = child
            .inherited_alpha_state
            .as_ref()
            .expect("every flat survivor route must carry alpha");
        assert!(Arc::ptr_eq(carried, &seed));
    }
    depths.sort_unstable();
    assert_eq!(depths, vec![11, 12], "both survivor routes exercised");
}

#[test]
fn test_flat_fallback_survivors_preserve_parent_alpha_disabled_and_complete_f8() {
    use super::super::push_survivors::push_fallback_survivors;

    let graph = build_disjunctive_batch_graph_4353();
    let seed = Arc::new(GraphAlphaState::new());
    for (enable_relaxed_clip, input_clip_type) in [
        (false, InputClipType::Relaxed),
        (true, InputClipType::Complete),
    ] {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_relaxed_clip,
            input_clip_type,
            relaxed_clip_iterations: 1,
            ..Default::default()
        });
        let survivors = vec![FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_survive_4366()),
            depth: 7,
            priority: 1.0,
            inherited_alpha_state: Some(Arc::clone(&seed)),
        }];
        let mut queue = BinaryHeap::new();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut domains_verified_by_clip = 0usize;

        push_fallback_survivors(
            &verifier,
            &graph,
            survivors,
            &[1],
            &[0.0, 0.0],
            None,
            &mut queue,
            &mut lifecycle,
            &mut domains_verified_by_clip,
        )
        .expect("fallback survivor route should succeed");

        let child = queue.pop().expect("fallback survivor must be queued");
        assert!(queue.is_empty());
        assert!(child.needs_bounding);
        let carried = child
            .inherited_alpha_state
            .as_ref()
            .expect("fallback survivor must carry alpha");
        assert!(Arc::ptr_eq(carried, &seed));
    }
}

fn build_three_child_survivors_4366() -> Vec<FlatPendingChild> {
    vec![
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_infeasible_4366()),
            depth: 1,
            priority: 1.0,
            inherited_alpha_state: None,
        },
        FlatPendingChild {
            flat_lower: arr1(&[0.5_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_verified_4366()),
            depth: 2,
            priority: 2.0,
            inherited_alpha_state: None,
        },
        FlatPendingChild {
            flat_lower: arr1(&[0.0_f32]).into_dyn(),
            flat_upper: arr1(&[1.0_f32]).into_dyn(),
            obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            linear_bounds: Some(lb_survive_4366()),
            depth: 3,
            priority: 3.0,
            inherited_alpha_state: None,
        },
    ]
}

/// Regression for #4366 batched clip: three children with distinct dispositions.
/// Child A: infeasible by clip. Child B: verified by grouped check. Child C: queued.
///
/// The infeasibility of child A comes from COMBINING its two rows (x<=0.2 AND
/// x>=0.8 -> empty), i.e. a single conjunctive clause of two rows — so this uses
/// `clause_sizes = [2]`, not `[1, 1]`. Under the clause-aware clip
/// (#disj-cross-clause-clip-unsat) `[1, 1]` would be two INDEPENDENT OR clauses,
/// each individually feasible, and child A would correctly become a survivor
/// (the lsnc false-unsat pattern) — the disjunctive survivor path is covered by
/// `test_batched_clip_disjoint_or_clause_survives_disj`.
#[test]
fn test_batched_clip_three_children_mixed_dispositions_4366() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        ..Default::default()
    });

    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [2usize];
    let shape = &[1usize];
    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        build_three_child_survivors_4366(),
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("batched clip with 3 children should not error");

    assert_eq!(
        domains_verified_by_clip, 2,
        "children A (infeasible) and B (grouped-verified) should both be counted"
    );
    assert_eq!(lifecycle.domains_verified, 2);
    assert_eq!(queue.len(), 1, "only child C should remain in queue");
    let queued = queue.pop().expect("one child in queue");
    assert_eq!(queued.depth, 3, "queued child should be child C (depth=3)");
    assert!(queued.needs_bounding, "queued child needs bounding");
}

/// #disj-cross-clause-clip-unsat plumbing (flat path): the SAME two rows as
/// `lb_infeasible_4366` (x<=0.2, x>=0.8) but as TWO independent OR clauses
/// (`clause_sizes = [1, 1]`) are each individually feasible over [0, 1], so the
/// child must NOT be clip-verified (this is exactly the lsnc false-unsat that
/// the historical cross-clause clip produced). The child survives to the queue,
/// carrying the UNION box, which encloses both [0, 0.2] and [0.8, 1] (i.e. all
/// of [0, 1]) so no counterexample is discarded.
#[test]
fn test_batched_clip_disjoint_or_clause_survives_disj() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        ..Default::default()
    });

    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize]; // two OR clauses, one row each
    let shape = &[1usize];

    let survivors = vec![FlatPendingChild {
        flat_lower: arr1(&[0.0_f32]).into_dyn(),
        flat_upper: arr1(&[1.0_f32]).into_dyn(),
        obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
        linear_bounds: Some(lb_infeasible_4366()),
        depth: 1,
        priority: 1.0,
        inherited_alpha_state: None,
    }];

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("push_batched_relaxed_survivors should not error");

    assert_eq!(
        domains_verified_by_clip, 0,
        "two individually-feasible OR clauses must not be clip-verified"
    );
    assert_eq!(lifecycle.domains_verified, 0);
    assert_eq!(queue.len(), 1, "the survivor must reach the queue");
    let queued = queue.pop().expect("one survivor queued");
    // Union box encloses [0, 0.2] ∪ [0.8, 1] -> essentially [0, 1].
    assert!(
        queued.input_bounds.lower()[[0]] <= 1e-6,
        "union lower must reach 0, got {}",
        queued.input_bounds.lower()[[0]]
    );
    assert!(
        queued.input_bounds.upper()[[0]] >= 1.0 - 1e-6,
        "union upper must reach 1, got {}",
        queued.input_bounds.upper()[[0]]
    );
}

/// Regression: the joint multi-spec clip can find infeasibility from combined
/// constraints that no single row can prove alone.
///
/// Two 1D constraints: row 0 says x ≤ 0.3 (lA=[1], lb=-0.3, thresh=0),
/// row 1 says x ≥ 0.7 (lA=[-1], lb=0.7, thresh=0).
/// Neither row alone makes [0, 1] infeasible, but together they prove the child
/// box is empty (must be ≤0.3 AND ≥0.7).
///
/// Part of #4367 acceptance criteria: "a regression proves the grouped path can
/// eliminate a child through combined multi-spec clipping that no single row
/// proves alone."
#[test]
fn test_joint_multispec_clip_finds_combined_infeasibility_4367() {
    use super::super::push_survivors::push_batched_relaxed_survivors;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 3,
        ..Default::default()
    });

    // ONE conjunctive clause of two rows: BOTH rows must hold for a
    // counterexample, so combining them proves the child empty. (This is the
    // #4367 "combined infeasibility" capability; expressing it as `[1, 1]` would
    // be two INDEPENDENT OR clauses — each individually feasible — which under
    // the clause-aware clip (#disj-cross-clause-clip-unsat) is correctly NOT
    // infeasible.)
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [2usize];
    let shape = &[1usize];

    // Row 0: lA=1, lb=-0.3 → 1*x - 0.3 > 0 → x > 0.3 (clips upper to ~0.3)
    // Row 1: lA=-1, lb=0.7 → -1*x + 0.7 > 0 → x < 0.7 (clips lower to ~0.7)
    // Together (AND): x ≤ 0.3 AND x ≥ 0.7 → empty box.
    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.3_f32, 0.7_f32]),
        arr2(&[[1.0_f32], [-1.0_f32]]),
        arr1(&[-0.3_f32, 0.7_f32]),
    )
    .expect("valid linear bounds");

    let survivors = vec![FlatPendingChild {
        flat_lower: arr1(&[0.0_f32]).into_dyn(),
        flat_upper: arr1(&[1.0_f32]).into_dyn(),
        obj_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
        linear_bounds: Some(linear_bounds),
        depth: 1,
        priority: 1.0,
        inherited_alpha_state: None,
    }];

    let mut queue = BinaryHeap::new();
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut domains_verified_by_clip = 0usize;

    push_batched_relaxed_survivors(
        &verifier,
        survivors,
        shape,
        &thresholds,
        &clause_sizes,
        &mut queue,
        &mut lifecycle,
        &mut domains_verified_by_clip,
    )
    .expect("push_batched_relaxed_survivors should not error");

    // The joint multi-spec clip should find the child infeasible from combined
    // constraints. Either infeasible_after_clip is true, or the post-clip
    // grouped verification catches it.
    assert_eq!(
        domains_verified_by_clip, 1,
        "joint multi-spec clip should verify child from combined constraints"
    );
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(
        queue.len(),
        0,
        "no children should be queued — combined constraints prove the child infeasible"
    );
}
