// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::bounds::GraphAlphaState;
use ny_test_utils::CountingGemmEngine;

fn build_single_objective_batching_graph_4210() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.1_f32, -0.4_f32], [0.3_f32, 0.8_f32]]), None)
                .expect("valid linear1"),
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
            LinearLayer::new(arr2(&[[0.7_f32, -0.2_f32]]), Some(arr1(&[0.05_f32])))
                .expect("valid linear2"),
        ),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn build_batched_kernel_fixture_4210() -> (
    GraphNetwork,
    HashMap<String, BoundedTensor>,
    Array2<f32>,
    BoundedTensor,
    BoundedTensor,
) {
    let graph = build_single_objective_batching_graph_4210();
    let root_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid root input");
    let root_node_bounds = graph
        .collect_node_bounds_with_engine(&root_input, None)
        .expect("root node bounds");
    let spec_matrix = arr2(&[[1.0_f32]]);
    let child_a = BoundedTensor::new(
        arr1(&[-0.8_f32, -0.5_f32]).into_dyn(),
        arr1(&[0.4_f32, 0.9_f32]).into_dyn(),
    )
    .expect("valid child a");
    let child_b = BoundedTensor::new(
        arr1(&[-0.2_f32, -0.7_f32]).into_dyn(),
        arr1(&[0.9_f32, 0.3_f32]).into_dyn(),
    )
    .expect("valid child b");
    (graph, root_node_bounds, spec_matrix, child_a, child_b)
}

fn baseline_single_objective_bounds_4210(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    child_a: &BoundedTensor,
    child_b: &BoundedTensor,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
) -> (
    CountingGemmEngine,
    (BoundedTensor, Option<LinearBounds>),
    (BoundedTensor, Option<LinearBounds>),
) {
    baseline_bounds_with_alpha_4210(graph, spec_matrix, child_a, child_b, root_node_bounds, None)
}

fn baseline_bounds_with_alpha_4210(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    child_a: &BoundedTensor,
    child_b: &BoundedTensor,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
) -> (
    CountingGemmEngine,
    (BoundedTensor, Option<LinearBounds>),
    (BoundedTensor, Option<LinearBounds>),
) {
    let baseline_engine = CountingGemmEngine::new();
    let baseline_a = compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        child_a,
        spec_matrix,
        Some(&baseline_engine),
        root_node_bounds,
        None,
        alpha_state,
        None,
        None,
        None,
        false,
    )
    .expect("baseline child_a");
    let baseline_b = compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        child_b,
        spec_matrix,
        Some(&baseline_engine),
        root_node_bounds,
        None,
        alpha_state,
        None,
        None,
        None,
        false,
    )
    .expect("baseline child_b");
    (baseline_engine, baseline_a, baseline_b)
}

fn assert_batched_bounds_match_baseline_4210(
    domains: &[GraphInputDomain],
    baseline_a: &(BoundedTensor, Option<LinearBounds>),
    baseline_b: &(BoundedTensor, Option<LinearBounds>),
) {
    assert!(!domains[0].needs_bounding && !domains[1].needs_bounding);
    assert!(
        (domains[0].lower_bound - baseline_a.0.lower_scalar()).abs() <= 1e-6,
        "child_a lower bound {} diverged from baseline {}",
        domains[0].lower_bound,
        baseline_a.0.lower_scalar()
    );
    assert!(
        (domains[0].upper_bound - baseline_a.0.upper_scalar()).abs() <= 1e-6,
        "child_a upper bound {} diverged from baseline {}",
        domains[0].upper_bound,
        baseline_a.0.upper_scalar()
    );
    assert!(
        (domains[1].lower_bound - baseline_b.0.lower_scalar()).abs() <= 1e-6,
        "child_b lower bound {} diverged from baseline {}",
        domains[1].lower_bound,
        baseline_b.0.lower_scalar()
    );
    assert!(
        (domains[1].upper_bound - baseline_b.0.upper_scalar()).abs() <= 1e-6,
        "child_b upper bound {} diverged from baseline {}",
        domains[1].upper_bound,
        baseline_b.0.upper_scalar()
    );
}

/// Assert the batched path produces bounds at least as tight as the per-domain baseline.
///
/// The batched backward path inherently tightens intermediate bounds via
/// `compute_constrained_forward_bounds` (re-propagates IBP through the sub-domain
/// input region), while the per-domain baseline (Mode A, ibp_enhancement=false) uses
/// root-level alpha_node_bounds directly without re-propagation. So the batched path
/// produces TIGHTER final bounds (higher lower, lower upper).
///
/// Tolerance of 1e-6 in the "wrong" direction guards against f32 rounding.
fn assert_batched_bounds_at_least_as_tight_4210(
    domains: &[GraphInputDomain],
    baseline_a: &(BoundedTensor, Option<LinearBounds>),
    baseline_b: &(BoundedTensor, Option<LinearBounds>),
) {
    let tol = 1e-6;
    assert!(!domains[0].needs_bounding && !domains[1].needs_bounding);

    // Batched lower bounds >= baseline lower bounds (tighter from below).
    assert!(
        domains[0].lower_bound >= baseline_a.0.lower_scalar() - tol,
        "child_a batched lower {} should be >= baseline lower {} (tighter)",
        domains[0].lower_bound,
        baseline_a.0.lower_scalar()
    );
    assert!(
        domains[1].lower_bound >= baseline_b.0.lower_scalar() - tol,
        "child_b batched lower {} should be >= baseline lower {} (tighter)",
        domains[1].lower_bound,
        baseline_b.0.lower_scalar()
    );

    // Batched upper bounds <= baseline upper bounds (tighter from above).
    assert!(
        domains[0].upper_bound <= baseline_a.0.upper_scalar() + tol,
        "child_a batched upper {} should be <= baseline upper {} (tighter)",
        domains[0].upper_bound,
        baseline_a.0.upper_scalar()
    );
    assert!(
        domains[1].upper_bound <= baseline_b.0.upper_scalar() + tol,
        "child_b batched upper {} should be <= baseline upper {} (tighter)",
        domains[1].upper_bound,
        baseline_b.0.upper_scalar()
    );
}

fn assert_linear_bounds_finite_and_shape_match_4210(
    actual: &LinearBounds,
    baseline: &LinearBounds,
    label: &str,
) {
    fn assert_finite<'a>(values: impl Iterator<Item = &'a f32>, label: &str, field: &str) {
        assert!(
            values.copied().all(f32::is_finite),
            "{label} {field} should stay finite on the batched alpha path"
        );
    }

    assert_eq!(
        actual.lower_a().dim(),
        baseline.lower_a().dim(),
        "{label} lower_a shape diverged from baseline"
    );
    assert_eq!(
        actual.upper_a().dim(),
        baseline.upper_a().dim(),
        "{label} upper_a shape diverged from baseline"
    );
    assert_eq!(
        actual.lower_b().len(),
        baseline.lower_b().len(),
        "{label} lower_b shape diverged from baseline"
    );
    assert_eq!(
        actual.upper_b().len(),
        baseline.upper_b().len(),
        "{label} upper_b shape diverged from baseline"
    );

    assert_finite(actual.lower_a().iter(), label, "lower_a");
    assert_finite(actual.upper_a().iter(), label, "upper_a");
    assert_finite(actual.lower_b().iter(), label, "lower_b");
    assert_finite(actual.upper_b().iter(), label, "upper_b");
}

fn deferred_domains_4210(child_a: BoundedTensor, child_b: BoundedTensor) -> Vec<GraphInputDomain> {
    vec![
        GraphInputDomain {
            input_bounds: Arc::new(child_a),
            lower_bound: f32::NEG_INFINITY,
            upper_bound: f32::INFINITY,
            depth: 1,
            priority: 0.0,
            linear_bounds: None,
            needs_bounding: true,
            node_bounds_override: None,
            inherited_alpha_state: None,
        },
        GraphInputDomain {
            input_bounds: Arc::new(child_b),
            lower_bound: f32::NEG_INFINITY,
            upper_bound: f32::INFINITY,
            depth: 1,
            priority: 0.0,
            linear_bounds: None,
            needs_bounding: true,
            node_bounds_override: None,
            inherited_alpha_state: None,
        },
    ]
}

#[test]
fn test_bound_deferred_domains_batch_uses_batched_backward_kernel_4210() {
    let (graph, _root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();
    let (baseline_engine, baseline_a, baseline_b) =
        baseline_single_objective_bounds_4210(&graph, &spec_matrix, &child_a, &child_b, None);
    let mut domains = deferred_domains_4210(child_a, child_b);

    let batch_engine = CountingGemmEngine::new();
    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        Some(&batch_engine),
        None,
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
    )
    .expect("batched deferred bounds");

    assert!(batch_engine.gemm_calls() > 0);
    assert!(batch_engine.gemm_calls() < baseline_engine.gemm_calls());
    assert_batched_bounds_match_baseline_4210(&domains, &baseline_a, &baseline_b);

    match (&domains[0].linear_bounds, &baseline_a.1) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "child_a linear bounds diverged: batched={} baseline={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
    match (&domains[1].linear_bounds, &baseline_b.1) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "child_b linear bounds diverged: batched={} baseline={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

/// The batched backward path re-propagates IBP through the sub-domain input region
/// during `compute_constrained_forward_bounds`, producing tighter intermediate bounds
/// than the per-domain baseline (Mode A, ibp_enhancement=false) which uses root-level
/// alpha_node_bounds directly. Verify the batched path produces bounds at least as
/// tight as the per-domain baseline.
#[test]
fn test_bound_deferred_domains_batch_tightens_with_root_bounds_4210() {
    let (graph, root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();
    let (_baseline_engine, baseline_a, baseline_b) = baseline_single_objective_bounds_4210(
        &graph,
        &spec_matrix,
        &child_a,
        &child_b,
        Some(&root_node_bounds),
    );
    let mut domains = deferred_domains_4210(child_a, child_b);

    let batch_engine = CountingGemmEngine::new();
    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        Some(&batch_engine),
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
    )
    .expect("deferred bounds with root bounds");

    assert!(batch_engine.gemm_calls() > 0);
    assert_batched_bounds_at_least_as_tight_4210(&domains, &baseline_a, &baseline_b);
}

/// Verify the batched kernel fires with alpha state. Part of #4210.
/// Before this fix, `alpha_state.is_some()` forced rayon par_iter fallback.
#[test]
fn test_bound_deferred_domains_batch_uses_batched_kernel_with_alpha_state_4210() {
    let (graph, root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();

    // Build GraphAlphaState simulating warmup α-CROWN pass.
    let pre_relu = root_node_bounds.get("linear1").expect("pre-activation");
    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .add_relu_node("relu1", pre_relu, false)
        .expect("alpha");

    let (baseline_engine, baseline_a, baseline_b) = baseline_bounds_with_alpha_4210(
        &graph,
        &spec_matrix,
        &child_a,
        &child_b,
        Some(&root_node_bounds),
        Some(&alpha_state),
    );

    let mut domains = deferred_domains_4210(child_a, child_b);
    let batch_engine = CountingGemmEngine::new();
    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        Some(&batch_engine),
        Some(&root_node_bounds),
        Some(&alpha_state),
        None,
        None,
        None,
        &BetaCrownConfig::default(),
    )
    .expect("batched deferred bounds with alpha state");

    assert!(batch_engine.gemm_calls() > 0, "batched kernel should fire");
    assert!(
        batch_engine.gemm_calls() < baseline_engine.gemm_calls(),
        "batched ({}) should use fewer GEMM calls than 2x scalar ({})",
        batch_engine.gemm_calls(),
        baseline_engine.gemm_calls()
    );
    assert_batched_bounds_at_least_as_tight_4210(&domains, &baseline_a, &baseline_b);

    match (&domains[0].linear_bounds, &baseline_a.1) {
        (Some(actual), Some(expected)) => {
            assert_linear_bounds_finite_and_shape_match_4210(actual, expected, "child_a");
        }
        (actual, expected) => panic!(
            "child_a linear bounds diverged: batched={} baseline={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
    match (&domains[1].linear_bounds, &baseline_b.1) {
        (Some(actual), Some(expected)) => {
            assert_linear_bounds_finite_and_shape_match_4210(actual, expected, "child_b");
        }
        (actual, expected) => panic!(
            "child_b linear bounds diverged: batched={} baseline={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

/// Verify the batched kernel fires when ibp_enhancement=true. Part of #4210.
/// Before this fix, `ibp_enhancement` forced rayon par_iter fallback.
#[test]
fn test_bound_deferred_domains_batch_fires_with_ibp_enhancement_4210() {
    let (graph, root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();

    // Non-ibp_enhancement baseline (the batched path uses warmup base_bounds
    // without fresh IBP forward, matching this baseline's mechanism).
    let (_baseline_engine, baseline_a, baseline_b) = baseline_single_objective_bounds_4210(
        &graph,
        &spec_matrix,
        &child_a,
        &child_b,
        Some(&root_node_bounds),
    );

    let mut domains = deferred_domains_4210(child_a, child_b);
    let batch_engine = CountingGemmEngine::new();
    let config = BetaCrownConfig {
        input_split_ibp_enhancement: true,
        ..BetaCrownConfig::default()
    };
    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        Some(&batch_engine),
        Some(&root_node_bounds),
        None,
        None,
        None,
        None,
        &config,
    )
    .expect("batched deferred bounds with ibp_enhancement");

    assert!(
        batch_engine.gemm_calls() > 0,
        "batched kernel should fire with ibp_enhancement=true"
    );
    assert!(!domains[0].needs_bounding && !domains[1].needs_bounding);
    // The batched path with ibp_enhancement=true still uses warmup base_bounds
    // (not fresh subdomain IBP), so bounds match the non-ibp_enhancement
    // baseline with root_node_bounds (at-least-as-tight due to constrained
    // forward bounds).
    assert_batched_bounds_at_least_as_tight_4210(&domains, &baseline_a, &baseline_b);
}
