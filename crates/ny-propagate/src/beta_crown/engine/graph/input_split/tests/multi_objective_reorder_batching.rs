// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::super::batching::bound_deferred_multi_obj_domains_batch;
use super::super::shared::extract_obj_bounds;
use super::super::shared::multi_obj_domain_priority;
use super::*;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::BabVerificationStatus;
use crate::BranchingHeuristic;

fn assert_obj_bounds_close(label: &str, actual: &[(f32, f32)], expected: &[(f32, f32)]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: number of objective rows changed"
    );

    for (idx, ((actual_l, actual_u), (expected_l, expected_u))) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (*actual_l - *expected_l).abs() <= 1e-6,
            "{label}: lower bound changed at objective {idx}: actual={actual_l}, expected={expected_l}"
        );
        assert!(
            (*actual_u - *expected_u).abs() <= 1e-6,
            "{label}: upper bound changed at objective {idx}: actual={actual_u}, expected={expected_u}"
        );
    }
}

fn direct_multi_obj_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    root_node_bounds: &HashMap<String, BoundedTensor>,
    node_bounds_override: Option<&HashMap<String, BoundedTensor>>,
) -> (BoundedTensor, Option<LinearBounds>) {
    compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input,
        spec_matrix,
        None,
        Some(root_node_bounds),
        node_bounds_override,
        None,
        None,
        None,
        None,
        false,
    )
    .expect("independent direct call should succeed")
}

fn deferred_multi_obj_domain(
    input_bounds: BoundedTensor,
    node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
) -> MultiObjInputDomain {
    MultiObjInputDomain {
        input_bounds: Arc::new(input_bounds),
        obj_bounds: vec![(-1.0, 1.0); 2],
        linear_bounds: None,
        depth: 1,
        priority: 1.0,
        needs_bounding: true,
        node_bounds_override,
        inherited_alpha_state: None,
    }
}

fn assert_deferred_domain_matches_direct(
    label: &str,
    domain: &MultiObjInputDomain,
    expected_bounds: &BoundedTensor,
    expected_linear: Option<LinearBounds>,
) {
    let expected_obj_bounds = extract_obj_bounds(expected_bounds, domain.obj_bounds.len()).unwrap();
    assert_obj_bounds_close(label, &domain.obj_bounds, &expected_obj_bounds);

    match (&domain.linear_bounds, expected_linear) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, &expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "{label}: linear bound availability diverged: batched={} direct={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

#[test]
fn test_bound_deferred_multi_obj_domains_batch_matches_independent_calls_4116() {
    let graph = build_complete_clip_override_graph();
    let root_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid root input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32], [0.5_f32]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let node_bounds_override = build_complete_clip_override_bounds();

    let child_a = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[0.6_f32]).into_dyn())
        .expect("valid child_a");
    let child_b = BoundedTensor::new(arr1(&[-0.4_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid child_b");
    let child_override =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid child_override");

    let expected_a =
        direct_multi_obj_bounds(&graph, &child_a, &spec_matrix, &root_node_bounds, None);
    let expected_b =
        direct_multi_obj_bounds(&graph, &child_b, &spec_matrix, &root_node_bounds, None);
    let expected_override = direct_multi_obj_bounds(
        &graph,
        &child_override,
        &spec_matrix,
        &root_node_bounds,
        Some(node_bounds_override.as_ref()),
    );

    let mut domains = vec![
        deferred_multi_obj_domain(child_a, None),
        deferred_multi_obj_domain(child_b, None),
        deferred_multi_obj_domain(child_override, Some(node_bounds_override)),
    ];

    bound_deferred_multi_obj_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        None,
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
        None,
        0,
    )
    .expect("deferred multi-objective batch should match independent calls");

    assert_deferred_domain_matches_direct("child_a", &domains[0], &expected_a.0, expected_a.1);
    assert_deferred_domain_matches_direct("child_b", &domains[1], &expected_b.0, expected_b.1);
    assert_deferred_domain_matches_direct(
        "override child",
        &domains[2],
        &expected_override.0,
        expected_override.1,
    );

    for (idx, domain) in domains.iter().enumerate() {
        assert!(
            !domain.needs_bounding,
            "deferred domain {idx} should be marked bounded after the batch rebound pass"
        );
        assert!(
            domain.node_bounds_override.is_none(),
            "deferred domain {idx} should consume any queued node-bounds override"
        );
    }
}

#[test]
fn test_bound_deferred_multi_obj_domains_batch_applies_parent_floor_4354() {
    let graph = build_complete_clip_override_graph();
    let root_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid root input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32], [1.0_f32]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let child_input = BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn())
        .expect("valid child input");
    let parent_obj_bounds = vec![(0.1_f32, 1.0_f32), (0.2_f32, 1.0_f32)];
    let direct_bounds =
        direct_multi_obj_bounds(&graph, &child_input, &spec_matrix, &root_node_bounds, None);
    let unguarded_obj_bounds = extract_obj_bounds(&direct_bounds.0, spec_matrix.nrows()).unwrap();
    let mut domains = vec![MultiObjInputDomain {
        input_bounds: Arc::new(child_input),
        obj_bounds: parent_obj_bounds.clone(),
        linear_bounds: None,
        depth: 1,
        priority: 1.0,
        needs_bounding: true,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }];

    assert!(
        unguarded_obj_bounds
            .iter()
            .zip(parent_obj_bounds.iter())
            .all(|((new_l, _), (parent_l, _))| new_l < parent_l),
        "fixture should exercise the monotonicity guard: direct child bounds={unguarded_obj_bounds:?}, parent={parent_obj_bounds:?}"
    );

    bound_deferred_multi_obj_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        None,
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
        None,
        0,
    )
    .expect("deferred multi-objective batch should apply the parent lower-bound floor");

    assert_obj_bounds_close(
        "monotonicity-guarded child",
        &domains[0].obj_bounds,
        &[(0.1_f32, 0.5_f32), (0.2_f32, 0.5_f32)],
    );
    assert_eq!(
        domains[0].priority,
        multi_obj_domain_priority(&domains[0].obj_bounds, &thresholds),
        "deferred rebound should recompute priority from the clamped per-spec bounds"
    );
    assert!(!domains[0].needs_bounding);
}

#[test]
fn test_bound_deferred_multi_obj_domains_batch_override_path_applies_parent_floor_4354() {
    let graph = build_complete_clip_override_graph();
    let root_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid root input");
    let root_node_bounds = graph
        .collect_node_bounds(&root_input)
        .expect("root node bounds should succeed");
    let spec_matrix = arr2(&[[1.0_f32], [1.0_f32]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let node_bounds_override = build_complete_clip_override_bounds();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid child input");
    let parent_obj_bounds = vec![(0.1_f32, 1.0_f32), (0.15_f32, 1.0_f32)];
    let direct_bounds = direct_multi_obj_bounds(
        &graph,
        &child_input,
        &spec_matrix,
        &root_node_bounds,
        Some(node_bounds_override.as_ref()),
    );
    let unguarded_obj_bounds = extract_obj_bounds(&direct_bounds.0, spec_matrix.nrows()).unwrap();
    let mut domains = vec![MultiObjInputDomain {
        input_bounds: Arc::new(child_input),
        obj_bounds: parent_obj_bounds.clone(),
        linear_bounds: None,
        depth: 1,
        priority: 1.0,
        needs_bounding: true,
        node_bounds_override: Some(node_bounds_override),
        inherited_alpha_state: None,
    }];

    assert!(
        unguarded_obj_bounds
            .iter()
            .zip(parent_obj_bounds.iter())
            .all(|((new_l, _), (parent_l, _))| new_l < parent_l),
        "fixture should exercise the override monotonicity guard: direct child bounds={unguarded_obj_bounds:?}, parent={parent_obj_bounds:?}"
    );

    bound_deferred_multi_obj_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        None,
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
        None,
        0,
    )
    .expect("override-backed deferred multi-objective batch should apply the parent floor");

    let expected_obj_bounds: Vec<(f32, f32)> = unguarded_obj_bounds
        .iter()
        .zip(parent_obj_bounds.iter())
        .map(|((_, new_u), (parent_l, _))| (*parent_l, *new_u))
        .collect();
    assert_obj_bounds_close(
        "override monotonicity-guarded child",
        &domains[0].obj_bounds,
        &expected_obj_bounds,
    );
    assert_eq!(
        domains[0].priority,
        multi_obj_domain_priority(&domains[0].obj_bounds, &thresholds),
        "override-backed rebound should recompute priority from the clamped per-spec bounds"
    );
    assert!(!domains[0].needs_bounding);
    assert!(domains[0].node_bounds_override.is_none());
}

fn build_multi_objective_status_parity_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear")),
    ));
    graph.set_output("out");
    graph
}

#[test]
fn test_multi_objective_reorder_bab_preserves_verification_status_4116() {
    let graph = build_multi_objective_status_parity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];

    let eager_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: false,
        ..Default::default()
    });
    let reorder_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        ..eager_verifier.config.clone()
    });

    let eager_result = eager_verifier
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("eager multi-objective input split should not error");
    let reorder_result = reorder_verifier
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("reordered multi-objective input split should not error");

    assert_eq!(
        std::mem::discriminant(&eager_result.result),
        std::mem::discriminant(&reorder_result.result),
        "reordered multi-objective input split should preserve verification status: eager={:?}, reorder={:?}",
        eager_result.result,
        reorder_result.result,
    );
    assert!(
        !matches!(reorder_result.result, BabVerificationStatus::Verified),
        "the parity harness should exercise the split path, not verify at the root"
    );
    assert!(
        reorder_result.domains_explored >= 2,
        "reordered multi-objective path must still split at least once, got {}",
        reorder_result.domains_explored,
    );
}
