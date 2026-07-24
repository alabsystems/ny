// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use ndarray::arr2;
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use super::{
    build_dag_concat_graph_4384, build_reference_bounds_graph_3870, dag_concat_input_4384,
    reference_bounds_input_3870,
};
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::config::{BetaCrownConfig, PhaseBudgetConfig};
use crate::beta_crown::engine::graph::input_split::shared::{
    compute_crown_or_ibp_bounds, compute_crown_or_ibp_bounds_with_node_bounds,
    graph_spec_ibp_fallback, graph_spec_ibp_root_screen_with_deadline,
};
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::BetaCrownVerifier;

fn reference_root_threshold(configured_graph: &crate::GraphNetwork, input: &BoundedTensor) -> f32 {
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);
    let (ibp_bounds, _) =
        graph_spec_ibp_fallback(configured_graph, input, &spec_matrix, None, None)
            .expect("IBP reference bounds should compute");
    let (crown_bounds, _) = compute_crown_or_ibp_bounds(
        configured_graph,
        input,
        &spec_matrix,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
    )
    .expect("CROWN reference bounds should compute");

    let ibp_upper = ibp_bounds.upper_scalar();
    let crown_upper = crown_bounds.upper_scalar();
    assert!(
        crown_upper + 1e-6 < ibp_upper,
        "reference graph must distinguish capped IBP from full root CROWN"
    );
    f32::midpoint(crown_upper, ibp_upper)
}

fn reference_forward_root_threshold(
    configured_graph: &crate::GraphNetwork,
    input: &BoundedTensor,
) -> f32 {
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);
    let (ibp_bounds, _) =
        graph_spec_ibp_fallback(configured_graph, input, &spec_matrix, None, None)
            .expect("IBP reference bounds should compute");
    let forward_node_bounds = configured_graph
        .collect_forward_linear_bounds_dag_with_engine(input, None)
        .expect("forward-linear node bounds should compute");
    let (forward_bounds, _) = compute_crown_or_ibp_bounds(
        configured_graph,
        input,
        &spec_matrix,
        None,
        Some(&forward_node_bounds),
        None,
        None,
        None,
        None,
        false,
    )
    .expect("forward+crown reference bounds should compute");

    let ibp_upper = ibp_bounds.upper_scalar();
    let forward_upper = forward_bounds.upper_scalar();
    assert!(
        forward_upper + 1e-6 < ibp_upper,
        "reference graph must distinguish forward+crown from plain IBP"
    );
    f32::midpoint(forward_upper, ibp_upper)
}

#[ntest::timeout(10000)]
#[test]
fn graph_spec_ibp_root_screen_respects_expired_deadline_4207() {
    let graph = build_reference_bounds_graph_3870();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        ..Default::default()
    });
    let configured_graph = verifier.configured_graph_for_crown(&graph);
    let input = reference_bounds_input_3870();
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);

    let err = graph_spec_ibp_root_screen_with_deadline(
        &configured_graph,
        &input,
        &spec_matrix,
        None,
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
        ),
    )
    .expect_err("expired warmup deadline should skip the root IBP screen");

    assert!(
        matches!(err, NyError::DeadlineExceeded(_)),
        "expected DeadlineExceeded, got {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn verify_graph_input_split_root_crown_uses_warmup_deadline_4207() {
    let graph = build_reference_bounds_graph_3870();
    let input = reference_bounds_input_3870();
    let objective = [1.0_f32, -0.35_f32];

    let baseline_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: true,
        use_alpha_crown: false,
        input_split_ibp_enhancement: false,
        enable_cuts: false,
        ..Default::default()
    });
    let configured_graph = baseline_verifier.configured_graph_for_crown(&graph);
    let threshold = reference_root_threshold(&configured_graph, &input);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: true,
        use_alpha_crown: false,
        input_split_ibp_enhancement: false,
        enable_cuts: false,
        max_domains: 0,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        phase_budget: PhaseBudgetConfig {
            initial_bounds_fraction: 0.0,
            ..Default::default()
        },
        ..Default::default()
    });

    let result = verifier
        .verify_graph_input_split(&graph, &input, &objective, threshold)
        .expect("capped warmup should fall back to IBP and defer root decision");

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "root CROWN should use the warmup deadline cap; got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_explored, 0,
        "with max_domains=0 the verifier should stop immediately after the root early-exit check"
    );
}

#[ntest::timeout(10000)]
#[test]
fn verify_graph_input_split_root_forward_bounds_preserve_selected_mode_4354() {
    let graph = build_reference_bounds_graph_3870();
    let input = reference_bounds_input_3870();
    let objective = [1.0_f32, -0.35_f32];

    let baseline_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_forward_bounds: true,
        input_split_ibp_enhancement: false,
        enable_cuts: false,
        ..Default::default()
    });
    let configured_graph = baseline_verifier.configured_graph_for_crown(&graph);
    let threshold = reference_forward_root_threshold(&configured_graph, &input);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_forward_bounds: true,
        input_split_ibp_enhancement: false,
        enable_cuts: false,
        max_domains: 0,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_input_split(&graph, &input, &objective, threshold)
        .expect("forward+crown root decision should succeed");

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "forward+crown should keep the selected intermediate bounds instead of falling back to IBP; got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_explored, 1,
        "verified root should stop before the BaB loop starts"
    );
}

#[ntest::timeout(10000)]
#[test]
fn verify_graph_input_split_root_forward_bounds_respects_warmup_deadline_4354() {
    let graph = build_reference_bounds_graph_3870();
    let input = reference_bounds_input_3870();
    let objective = [1.0_f32, -0.35_f32];

    let baseline_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_forward_bounds: true,
        input_split_ibp_enhancement: false,
        enable_cuts: false,
        ..Default::default()
    });
    let configured_graph = baseline_verifier.configured_graph_for_crown(&graph);
    let threshold = reference_forward_root_threshold(&configured_graph, &input);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_forward_bounds: true,
        input_split_ibp_enhancement: false,
        enable_cuts: false,
        max_domains: 0,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        phase_budget: PhaseBudgetConfig {
            initial_bounds_fraction: 0.0,
            ..Default::default()
        },
        ..Default::default()
    });

    let result = verifier
        .verify_graph_input_split(&graph, &input, &objective, threshold)
        .expect("expired warmup should skip forward-linear root reuse");

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "forward+crown warmup should honor the deadline cap instead of running uncapped; got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_explored, 0,
        "with max_domains=0 the verifier should stop immediately after the root early-exit check"
    );
}

#[ntest::timeout(10000)]
#[test]
fn compute_crown_or_ibp_bounds_ibp_enhancement_deadline_returns_conservative_bounds_4207() {
    let graph = build_reference_bounds_graph_3870();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        input_split_ibp_enhancement: true,
        ..Default::default()
    });
    let configured_graph = verifier.configured_graph_for_crown(&graph);
    let input = reference_bounds_input_3870();
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);

    let (deadline_bounds, _) = compute_crown_or_ibp_bounds(
        &configured_graph,
        &input,
        &spec_matrix,
        None,
        None,
        None,
        None,
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
        ),
        None,
        true,
    )
    .expect("expired warmup deadline should return a sound fallback");

    assert!(
        deadline_bounds.lower_scalar() == f32::NEG_INFINITY,
        "deadline fallback should not trigger an uncapped node-bounds pass; got lower={}",
        deadline_bounds.lower_scalar()
    );
    assert!(
        deadline_bounds.upper_scalar() == f32::INFINITY,
        "deadline fallback should not trigger an uncapped node-bounds pass; got upper={}",
        deadline_bounds.upper_scalar()
    );
}

/// When cached alpha_node_bounds exist and the IBP-enhancement deadline expires,
/// the fallback should reuse those cached bounds for plain IBP rather than
/// returning conservative `[-INF, +INF]`. This exercises the cache-hit branch
/// at shared.rs lines 249-259 (#4207, #4208).
#[ntest::timeout(10000)]
#[test]
fn compute_crown_or_ibp_bounds_ibp_enhancement_deadline_cache_hit_returns_ibp_bounds_4207() {
    let graph = build_reference_bounds_graph_3870();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        input_split_ibp_enhancement: true,
        ..Default::default()
    });
    let configured_graph = verifier.configured_graph_for_crown(&graph);
    let input = reference_bounds_input_3870();
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);

    // Pre-compute node bounds with no deadline to serve as cache
    let alpha_node_bounds = configured_graph
        .collect_node_bounds(&input)
        .expect("uncapped IBP node bounds should compute");

    // Call with cached bounds AND an expired deadline: should hit cache-reuse path
    let (cached_bounds, _) = compute_crown_or_ibp_bounds_with_node_bounds(
        &configured_graph,
        &input,
        &spec_matrix,
        None,
        Some(&alpha_node_bounds),
        None,
        None,
        None,
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
        ),
        None,
        true,
    )
    .expect("expired deadline with cached bounds should return IBP fallback, not error");

    // Cache-hit path should produce finite bounds (tighter than conservative [-INF, +INF])
    assert!(
        cached_bounds.lower_scalar().is_finite(),
        "cache-hit deadline fallback should produce finite lower bound; got {}",
        cached_bounds.lower_scalar()
    );
    assert!(
        cached_bounds.upper_scalar().is_finite(),
        "cache-hit deadline fallback should produce finite upper bound; got {}",
        cached_bounds.upper_scalar()
    );

    // Verify it matches direct IBP-with-those-bounds (the fallback should route to
    // graph_spec_ibp_fallback with the cached map)
    let (direct_ibp, _) = graph_spec_ibp_fallback(
        &configured_graph,
        &input,
        &spec_matrix,
        None,
        Some(&alpha_node_bounds),
    )
    .expect("direct IBP with cached bounds should compute");

    let tol = 1e-6;
    assert!(
        (cached_bounds.lower_scalar() - direct_ibp.lower_scalar()).abs() < tol,
        "cache-hit bounds lower should match direct IBP: {} vs {}",
        cached_bounds.lower_scalar(),
        direct_ibp.lower_scalar()
    );
    assert!(
        (cached_bounds.upper_scalar() - direct_ibp.upper_scalar()).abs() < tol,
        "cache-hit bounds upper should match direct IBP: {} vs {}",
        cached_bounds.upper_scalar(),
        direct_ibp.upper_scalar()
    );
}

/// #4384: `ibp_enhancement=true` on a DAG graph with Concat must complete
/// without error. The shape-mismatch skip guard in `merge_reference_bound_maps`
/// silently discards nodes where IBP and warmup CROWN produce different shapes,
/// keeping the (wider) warmup bounds for those nodes.
#[ntest::timeout(10000)]
#[test]
fn compute_crown_or_ibp_bounds_dag_concat_ibp_enhancement_completes_4384() {
    let graph = build_dag_concat_graph_4384();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        input_split_ibp_enhancement: true,
        ..Default::default()
    });
    let configured_graph = verifier.configured_graph_for_crown(&graph);
    let input = dag_concat_input_4384();
    let spec_matrix = arr2(&[[1.0_f32]]);

    // Collect warmup node bounds (simulates the initial alpha-CROWN pass).
    let alpha_node_bounds = configured_graph
        .collect_node_bounds(&input)
        .expect("warmup IBP node bounds should compute on DAG");

    // Call with ibp_enhancement=true: this triggers a fresh IBP forward pass
    // inside compute_crown_or_ibp_bounds_with_node_bounds, then merges with
    // alpha_node_bounds via build_input_split_reference_bounds →
    // merge_reference_bound_maps. If shapes disagree, the skip guard (#4384)
    // keeps the wider warmup bounds and continues.
    let (bounds, _linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        &configured_graph,
        &input,
        &spec_matrix,
        None,
        Some(&alpha_node_bounds),
        None, // child_node_bounds
        None, // alpha_state
        None, // mul_binary_alphas
        None, // deadline
        None, // crown_backward_layers
        true, // ibp_enhancement
    )
    .expect("ibp_enhancement on DAG concat graph must not error (#4384)");

    assert!(
        bounds.lower_scalar().is_finite(),
        "DAG ibp_enhancement lower bound should be finite; got {}",
        bounds.lower_scalar()
    );
    assert!(
        bounds.upper_scalar().is_finite(),
        "DAG ibp_enhancement upper bound should be finite; got {}",
        bounds.upper_scalar()
    );

    // Baseline: ibp_enhancement=false on the same graph.
    let (baseline, _) = compute_crown_or_ibp_bounds_with_node_bounds(
        &configured_graph,
        &input,
        &spec_matrix,
        None,
        Some(&alpha_node_bounds),
        None,
        None,
        None,
        None,
        None,
        false, // ibp_enhancement off
    )
    .expect("baseline CROWN on DAG should compute");

    // IBP-enhanced bounds should be at least as tight as plain CROWN (or
    // equal if the shape-mismatch skip discards all IBP reference bounds).
    assert!(
        bounds.lower_scalar() >= baseline.lower_scalar() - 1e-6,
        "ibp_enhancement lower {} should be >= baseline lower {}",
        bounds.lower_scalar(),
        baseline.lower_scalar()
    );
}
