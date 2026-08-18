// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::super::batching::bound_deferred_disjunctive_domains_batch;
use super::super::grouped_semantics::disjunctive_domain_priority;
use super::super::root_bounds::collect_input_split_root_node_bounds;
use super::super::shared::extract_obj_bounds;
use super::*;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::result::BabVerificationStatus;
use crate::bounds::AlphaCrownConfig;
use crate::BranchingHeuristic;
use ny_test_utils::CountingGemmEngine;

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

fn direct_disjunctive_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    node_bounds_override: Option<&HashMap<String, BoundedTensor>>,
) -> (BoundedTensor, Option<LinearBounds>) {
    direct_disjunctive_bounds_with_options(
        graph,
        input,
        spec_matrix,
        root_node_bounds,
        node_bounds_override,
        None,
        None,
    )
}

fn direct_disjunctive_bounds_with_options(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    node_bounds_override: Option<&HashMap<String, BoundedTensor>>,
    engine: Option<&dyn ny_core::GemmEngine>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> (BoundedTensor, Option<LinearBounds>) {
    compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input,
        spec_matrix,
        engine,
        root_node_bounds,
        node_bounds_override,
        None,
        mul_binary_alphas,
        None,
        None,
        false,
    )
    .expect("independent direct grouped call should succeed")
}

fn direct_disjunctive_mul_binary_baselines_4284(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    mul_binary_alphas: &HashMap<String, Array2<f32>>,
    child_a: &BoundedTensor,
    child_b: &BoundedTensor,
) -> (
    (BoundedTensor, Option<LinearBounds>),
    (BoundedTensor, Option<LinearBounds>),
    usize,
) {
    let baseline_engine = CountingGemmEngine::new();
    let expected_a = direct_disjunctive_bounds_with_options(
        graph,
        child_a,
        spec_matrix,
        None,
        None,
        Some(&baseline_engine),
        Some(mul_binary_alphas),
    );
    let expected_b = direct_disjunctive_bounds_with_options(
        graph,
        child_b,
        spec_matrix,
        None,
        None,
        Some(&baseline_engine),
        Some(mul_binary_alphas),
    );
    (expected_a, expected_b, baseline_engine.gemm_calls())
}

fn deferred_disjunctive_domain(
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

fn assert_mul_binary_disjunctive_domains_4284(
    domains: &[MultiObjInputDomain],
    thresholds: &[f32],
    clause_sizes: &[usize],
    expected_a: &(BoundedTensor, Option<LinearBounds>),
    expected_b: &(BoundedTensor, Option<LinearBounds>),
) {
    assert_deferred_domain_matches_direct(
        "mulbinary child_a",
        &domains[0],
        &expected_a.0,
        expected_a.1.clone(),
    );
    assert_deferred_domain_matches_direct(
        "mulbinary child_b",
        &domains[1],
        &expected_b.0,
        expected_b.1.clone(),
    );

    for (idx, domain) in domains.iter().enumerate() {
        assert_eq!(
            domain.priority,
            disjunctive_domain_priority(&domain.obj_bounds, thresholds, clause_sizes),
            "mulbinary deferred grouped domain {idx} should recompute disjunctive priority after batched rebound"
        );
        assert!(
            !domain.needs_bounding,
            "mulbinary deferred grouped domain {idx} should be marked bounded after batched rebound"
        );
        assert!(
            domain.node_bounds_override.is_none(),
            "mulbinary deferred grouped domain {idx} should not retain node-bounds overrides"
        );
    }
}

fn build_disjunctive_deferred_rebound_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "hidden",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("hidden linear")),
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
        vec!["hidden".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn build_disjunctive_override_bounds() -> Arc<HashMap<String, BoundedTensor>> {
    let hidden_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("hidden override bounds");
    let out_bounds = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("out override bounds");
    Arc::new(HashMap::from([
        ("hidden".to_string(), hidden_bounds),
        ("out".to_string(), out_bounds),
    ]))
}

#[test]
fn test_bound_deferred_disjunctive_domains_batch_matches_independent_calls_4267() {
    let graph = build_disjunctive_deferred_rebound_graph();
    let spec_matrix = arr2(&[[1.0_f32], [0.5_f32]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize];
    let node_bounds_override = build_disjunctive_override_bounds();

    let child_a = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[0.6_f32]).into_dyn())
        .expect("valid child_a");
    let child_b = BoundedTensor::new(arr1(&[-0.4_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid child_b");
    let child_override =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid child_override");

    let expected_a = direct_disjunctive_bounds(&graph, &child_a, &spec_matrix, None, None);
    let expected_b = direct_disjunctive_bounds(&graph, &child_b, &spec_matrix, None, None);
    let expected_override = direct_disjunctive_bounds(
        &graph,
        &child_override,
        &spec_matrix,
        None,
        Some(node_bounds_override.as_ref()),
    );

    let mut domains = vec![
        deferred_disjunctive_domain(child_a, None),
        deferred_disjunctive_domain(child_b, None),
        deferred_disjunctive_domain(child_override, Some(node_bounds_override)),
    ];

    bound_deferred_disjunctive_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
        None,
        0,
    )
    .expect("deferred grouped batch should match independent calls");

    assert_deferred_domain_matches_direct("child_a", &domains[0], &expected_a.0, expected_a.1);
    assert_deferred_domain_matches_direct("child_b", &domains[1], &expected_b.0, expected_b.1);
    assert_deferred_domain_matches_direct(
        "override child",
        &domains[2],
        &expected_override.0,
        expected_override.1,
    );

    for (idx, domain) in domains.iter().enumerate() {
        assert_eq!(
            domain.priority,
            disjunctive_domain_priority(&domain.obj_bounds, &thresholds, &clause_sizes),
            "deferred grouped domain {idx} should recompute disjunctive priority after rebound"
        );
        assert!(
            !domain.needs_bounding,
            "deferred grouped domain {idx} should be marked bounded after the batch rebound pass"
        );
        assert!(
            domain.node_bounds_override.is_none(),
            "deferred grouped domain {idx} should consume any queued node-bounds override"
        );
    }
}

#[test]
fn test_bound_deferred_disjunctive_domains_batch_uses_batched_kernel_with_mul_binary_alphas_4284() {
    let (graph, spec_matrix, mul_binary_alphas, child_a, child_b) =
        build_mul_binary_dense_spec_batch_fixture_4284();
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize];

    let (expected_a, expected_b, baseline_gemm_calls) =
        direct_disjunctive_mul_binary_baselines_4284(
            &graph,
            &spec_matrix,
            &mul_binary_alphas,
            &child_a,
            &child_b,
        );

    let mut domains = vec![
        deferred_disjunctive_domain(child_a, None),
        deferred_disjunctive_domain(child_b, None),
    ];

    let batch_engine = CountingGemmEngine::new();
    bound_deferred_disjunctive_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        Some(&batch_engine),
        None,
        None,
        Some(&mul_binary_alphas),
        None,
        None,
        &BetaCrownConfig::default(),
        None,
        0,
    )
    .expect("deferred disjunctive mulbinary rebound should use the batched kernel");

    assert!(
        batch_engine.gemm_calls() > 0,
        "mulbinary batched rebound should invoke the GEMM engine"
    );
    assert!(
        batch_engine.gemm_calls() < baseline_gemm_calls,
        "mulbinary batched rebound should use fewer GEMM calls than two scalar direct calls: batched={} baseline={}",
        batch_engine.gemm_calls(),
        baseline_gemm_calls
    );

    assert_mul_binary_disjunctive_domains_4284(
        &domains,
        &thresholds,
        &clause_sizes,
        &expected_a,
        &expected_b,
    );
}

#[test]
fn test_bound_deferred_disjunctive_domains_batch_override_path_applies_parent_floor_4354() {
    let graph = build_disjunctive_deferred_rebound_graph();
    let spec_matrix = arr2(&[[1.0_f32], [1.0_f32]]);
    let thresholds = [0.0_f32, 0.0_f32];
    let clause_sizes = [1usize, 1usize];
    let node_bounds_override = build_disjunctive_override_bounds();
    let child_input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid child input");
    let parent_obj_bounds = vec![(0.1_f32, 1.0_f32), (0.15_f32, 1.0_f32)];
    let direct_bounds = direct_disjunctive_bounds(
        &graph,
        &child_input,
        &spec_matrix,
        None,
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

    bound_deferred_disjunctive_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        &thresholds,
        &clause_sizes,
        None,
        None,
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
        None,
        0,
    )
    .expect("override-backed deferred disjunctive batch should apply the parent floor");

    let expected_obj_bounds: Vec<(f32, f32)> = unguarded_obj_bounds
        .iter()
        .zip(parent_obj_bounds.iter())
        .map(|((_, new_u), (parent_l, _))| (*parent_l, *new_u))
        .collect();
    assert_obj_bounds_close(
        "override monotonicity-guarded grouped child",
        &domains[0].obj_bounds,
        &expected_obj_bounds,
    );
    assert_eq!(
        domains[0].priority,
        disjunctive_domain_priority(&domains[0].obj_bounds, &thresholds, &clause_sizes),
        "override-backed grouped rebound should recompute priority from the clamped per-spec bounds"
    );
    assert!(!domains[0].needs_bounding);
    assert!(domains[0].node_bounds_override.is_none());
}

fn build_disjunctive_status_parity_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear")),
    ));
    graph.set_output("out");
    graph
}

fn build_disjunctive_status_config_4354() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: false,
        ..Default::default()
    }
}

/// Extend the shared branched dense-Concat fixture into a three-activation DAG,
/// the minimum depth that selects LinearizeNN's dense-DAG CROWN-IBP reference
/// collector:
///
/// ```text
/// input(2) -> Linear/ReLU --┐
///       \-> Linear/ReLU ----+-> Concat -> Linear/ReLU -> Linear(2)
/// ```
///
/// Reusing `build_dag_concat_graph_4384` keeps this composition test focused;
/// the larger exact `Slice -> Linear -> Concat` AllInOne-shaped fixture is
/// already pinned by the graph-alpha collector regression.
fn build_linearizenn_production_composition_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = build_dag_concat_graph_4384();
    graph.add_node(GraphNode::new(
        "relu_out",
        Layer::ReLU(ReLULayer),
        vec!["linear_out".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32], [-0.7]]), Some(arr1(&[0.02, -0.03])))
                .expect("valid final linear"),
        ),
        vec!["relu_out".to_string()],
    ));
    graph.set_output("output");
    (graph, dag_concat_input_4384())
}

fn linearizenn_child_boxes(root: &BoundedTensor) -> [BoundedTensor; 2] {
    let mut left_upper = root.upper().clone();
    left_upper[[0]] = 0.0;
    let mut right_lower = root.lower().clone();
    right_lower[[0]] = 0.0;
    [
        BoundedTensor::new(root.lower().clone(), left_upper).expect("valid left child"),
        BoundedTensor::new(right_lower, root.upper().clone()).expect("valid right child"),
    ]
}

fn linearizenn_production_composition_config() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: true,
        alpha_config: AlphaCrownConfig {
            iterations: 0,
            ..AlphaCrownConfig::default()
        },
        input_split_alpha_iteration: 0,
        input_split_ibp_enhancement: true,
        input_split_stacked_rebound: true,
        reorder_bab: true,
        enable_cuts: false,
        enable_relaxed_clip: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        timeout: Duration::from_secs(10),
        ..BetaCrownConfig::default()
    }
}

fn total_spec_width(bounds: &[BoundedTensor]) -> f32 {
    bounds
        .iter()
        .map(|bounds| {
            bounds
                .upper()
                .iter()
                .zip(bounds.lower())
                .map(|(upper, lower)| upper - lower)
                .sum::<f32>()
        })
        .sum()
}

/// Compose the LinearizeNN root collector, reordered child rebound, and grouped
/// verifier. The smaller tests pin these pieces separately; this test asserts
/// that the exact production knobs meet at the same seams.
#[test]
fn linearizenn_grouped_reorder_root_crown_child_stacked_ibp_matches_conservative_verdict() {
    let (graph, input) = build_linearizenn_production_composition_graph();
    let config = linearizenn_production_composition_config();
    assert_eq!(config.alpha_config.iterations, 0);
    assert_eq!(config.input_split_alpha_iteration, 0);
    assert!(config.input_split_ibp_enhancement);
    assert!(config.input_split_stacked_rebound);
    assert!(config.reorder_bab);

    let exec_order = graph.exec_order().expect("LinearizeNN DAG should sort");
    assert!(
        !graph.is_sequential_graph(exec_order),
        "fixture must retain the LinearizeNN input-skip DAG"
    );
    let (reference, source) = graph
        .collect_alpha_reference_bounds_with_engine_and_source(
            &input,
            &config.alpha_config,
            None,
            exec_order,
        )
        .expect("root alpha reference should collect");
    assert!(
        source.is_crown_ibp(),
        "the production root must select the dense-DAG CROWN-IBP collector"
    );

    let (root_node_bounds, root_alpha_state) = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        None,
        None,
        "LinearizeNN production-composition test",
        None,
    )
    .expect("zero-iteration root alpha bootstrap should succeed");
    let root_node_bounds =
        root_node_bounds.expect("alpha-CROWN root should return its reusable reference map");
    let root_alpha_state =
        root_alpha_state.expect("zero root iterations must still initialize reusable alpha");
    assert!(
        root_alpha_state.num_unstable() > 0,
        "zero iterations should initialize, not omit, unstable ReLU alpha state"
    );
    let expected = reference
        .get("linear_out")
        .expect("reference map missing linear_out");
    let actual = root_node_bounds
        .get("linear_out")
        .expect("root bootstrap map missing linear_out");
    assert_eq!(
        (actual.lower(), actual.upper()),
        (expected.lower(), expected.upper()),
        "zero-iteration bootstrap must reuse the CROWN-IBP reference map"
    );

    let children = linearizenn_child_boxes(&input);
    let child_refs: Vec<&BoundedTensor> = children.iter().collect();
    let spec_matrix = arr2(&[[1.0_f32, -0.5], [-0.75, 1.0]]);
    let rebound = |ibp_enhancement, stacked_rebound| {
        compute_crown_or_ibp_bounds_batched_specs(
            &graph,
            &child_refs,
            &spec_matrix,
            None,
            Some(&root_node_bounds),
            Some(&root_alpha_state),
            None,
            None,
            None,
            ibp_enhancement,
            stacked_rebound,
        )
        .expect("dense-spec child rebound should succeed")
    };
    let shared_only = rebound(false, false);
    let refreshed = rebound(true, true);
    assert_eq!(
        shared_only.rebound_timing.mode,
        crate::beta_crown::DenseSpecReboundMode::BatchedFastPath
    );
    assert_eq!(
        refreshed.rebound_timing.mode,
        crate::beta_crown::DenseSpecReboundMode::BatchedFastPath,
        "production child rebound must stay on the dense-spec batched path"
    );

    // Fresh child IBP must contribute to the intersection, and that production
    // adapter path must materially tighten the downstream spec enclosure.
    for (child_idx, child) in children.iter().enumerate() {
        let child_ibp = graph
            .collect_node_bounds(child)
            .expect("fresh child IBP should collect");
        let has_strict_intersection = child_ibp.iter().any(|(node_name, child_bounds)| {
            root_node_bounds.get(node_name).is_some_and(|root_bounds| {
                root_bounds.shape() == child_bounds.shape()
                    && (root_bounds
                        .lower()
                        .iter()
                        .zip(child_bounds.lower())
                        .any(|(root, child)| child > &(root + 1e-6))
                        || root_bounds
                            .upper()
                            .iter()
                            .zip(child_bounds.upper())
                            .any(|(root, child)| child < &(root - 1e-6)))
            })
        });
        assert!(
            has_strict_intersection,
            "child {child_idx} must contribute at least one strict fresh-IBP intersection"
        );
    }
    let shared_width = total_spec_width(&shared_only.bounds);
    let refreshed_width = total_spec_width(&refreshed.bounds);
    assert!(
        refreshed_width < shared_width,
        "fresh child IBP must materially re-anchor the production rebound: refreshed={refreshed_width}, shared={shared_width}"
    );

    let objectives = vec![vec![1.0_f32, -0.5], vec![-0.75, 1.0]];
    // Deliberately unreachable thresholds keep every sound implementation in
    // the split/unknown lane, so an accidental false proof cannot hide behind
    // a root-verifiable fixture.
    let thresholds = vec![1_000.0_f32, 1_000.0_f32];
    let clause_sizes = vec![1usize, 1usize];
    let verify = |config| {
        BetaCrownVerifier::new(config)
            .verify_graph_input_split_multi_clause_disjunctive(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clause_sizes,
                None,
                None,
            )
            .expect("grouped verifier should complete")
    };
    let production = verify(config.clone());
    let conservative = verify(BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        input_split_ibp_enhancement: false,
        input_split_stacked_rebound: false,
        reorder_bab: false,
        ..config
    });

    assert_eq!(
        std::mem::discriminant(&production.result),
        std::mem::discriminant(&conservative.result),
        "LinearizeNN production composition changed the conservative verifier verdict: production={:?}, conservative={:?}",
        production.result,
        conservative.result
    );
    assert!(
        !matches!(production.result, BabVerificationStatus::Verified),
        "unreachable-threshold harness must not produce a false proof"
    );
    assert!(
        production.domains_explored >= 2,
        "production reordered verifier must exercise deferred child rebound, got {} explored",
        production.domains_explored
    );
}

#[test]
fn test_disjunctive_reorder_bab_preserves_verification_status_4267() {
    let graph = build_disjunctive_status_parity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];
    let clause_sizes = vec![1usize, 1usize];

    let eager_verifier = BetaCrownVerifier::new(build_disjunctive_status_config_4354());
    let reorder_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        reorder_bab: true,
        ..eager_verifier.config.clone()
    });

    let eager_result = eager_verifier
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            None,
        )
        .expect("eager grouped input split should not error");
    let reorder_result = reorder_verifier
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            None,
        )
        .expect("reordered grouped input split should not error");

    assert_eq!(
        std::mem::discriminant(&eager_result.result),
        std::mem::discriminant(&reorder_result.result),
        "reordered grouped input split should preserve verification status: eager={:?}, reorder={:?}",
        eager_result.result,
        reorder_result.result,
    );
    assert!(
        !matches!(reorder_result.result, BabVerificationStatus::Verified),
        "the grouped parity harness should exercise the split path, not verify at the root"
    );
    assert!(
        reorder_result.domains_explored >= 2,
        "reordered grouped path must still split at least once, got {}",
        reorder_result.domains_explored,
    );
}

#[test]
fn test_disjunctive_build_batch_size_preserves_status_4354() {
    let graph = build_disjunctive_status_parity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];
    let clause_sizes = vec![1usize, 1usize];

    let baseline_verifier = BetaCrownVerifier::new(build_disjunctive_status_config_4354());
    let chunked_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        build_batch_size: Some(1),
        ..baseline_verifier.config.clone()
    });

    let baseline_result = baseline_verifier
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            None,
        )
        .expect("baseline disjunctive input split should not error");
    let chunked_result = chunked_verifier
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            None,
        )
        .expect("chunked disjunctive input split should not error");

    assert_eq!(
        std::mem::discriminant(&baseline_result.result),
        std::mem::discriminant(&chunked_result.result),
        "build_batch_size should preserve the disjunctive verifier status: baseline={:?}, chunked={:?}",
        baseline_result.result,
        chunked_result.result,
    );
    assert_eq!(
        baseline_result.domains_explored, chunked_result.domains_explored,
        "build_batch_size should not perturb the explored-domain count on the grouped parity harness"
    );
    assert_eq!(
        baseline_result.domains_verified, chunked_result.domains_verified,
        "build_batch_size should preserve verified-domain accounting"
    );
}
