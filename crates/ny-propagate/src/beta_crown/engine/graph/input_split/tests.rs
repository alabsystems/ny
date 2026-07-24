// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::batching::bound_deferred_domains_batch;
use super::grouped_semantics::{disjunctive_domain_priority, disjunctive_domain_verified};
use super::mul_binary::maybe_optimize_mul_binary_alphas;
use super::shared::{
    compute_crown_or_ibp_bounds_batched, compute_crown_or_ibp_bounds_with_node_bounds,
    graph_spec_crown_with_mul_binary_and_truncation, multi_obj_domain_verified, GraphInputDomain,
    MultiObjInputDomain,
};
use super::shared_specs::{compute_crown_or_ibp_bounds_batched_specs, BatchedSpecBounds};
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::layers::ConcatLayer;
use crate::layers::ReLULayer;
use crate::{GraphNetwork, GraphNode, Layer, LinearBounds, LinearLayer, MulBinaryLayer};
use ndarray::{arr1, arr2, Array1, Array2};
use ny_tensor::BoundedTensor;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod adv_check_dag_engine;
mod batching_rebound;
mod disjunctive_domain_verified;
mod disjunctive_reorder_batching;
mod metrics_emission;
mod multi_objective_parity;
mod multi_objective_reorder_batching;
mod single_objective_batched_kernel;
mod stacked_rebound;
mod warm_alpha_rebound;
mod warmup_deadline;

fn test_domain(priority: f32) -> GraphInputDomain {
    let input_bounds = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid test tensor");
    GraphInputDomain {
        input_bounds: Arc::new(input_bounds),
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 0,
        priority,
        linear_bounds: None,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn test_multi_obj_domain(priority: f32) -> MultiObjInputDomain {
    let input_bounds = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid test tensor");
    MultiObjInputDomain {
        input_bounds: Arc::new(input_bounds),
        obj_bounds: vec![(-1.0, 1.0)],
        linear_bounds: None,
        depth: 0,
        priority,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn build_mul_binary_deadline_test_case() -> (GraphNetwork, BoundedTensor, Array2<f32>) {
    let mut graph = GraphNetwork::new();
    let hidden = 2;

    let up_linear = LinearLayer::new(
        Array2::from_diag(&Array1::from_elem(hidden, 1.0_f32)),
        Some(Array1::zeros(hidden)),
    )
    .expect("valid up linear");
    graph.add_node(GraphNode::new(
        "up",
        Layer::Linear(up_linear),
        vec!["_input".to_string()],
    ));

    let gate_linear = LinearLayer::new(
        Array2::from_diag(&Array1::from_vec(vec![0.5_f32, 0.75_f32])),
        Some(Array1::zeros(hidden)),
    )
    .expect("valid gate linear");
    graph.add_node(GraphNode::new(
        "gate",
        Layer::Linear(gate_linear),
        vec!["_input".to_string()],
    ));

    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "up",
        "gate",
    ));
    graph.set_output("mul");

    let input = BoundedTensor::new(
        Array2::from_elem((1, hidden), -0.5_f32).into_dyn(),
        Array2::from_elem((1, hidden), 0.5_f32).into_dyn(),
    )
    .expect("valid bounded input");

    let mut spec_matrix = Array2::zeros((1, hidden));
    spec_matrix[[0, 0]] = 1.0;

    (graph, input, spec_matrix)
}

fn build_mul_binary_dense_spec_batch_fixture_4284() -> (
    GraphNetwork,
    Array2<f32>,
    HashMap<String, Array2<f32>>,
    BoundedTensor,
    BoundedTensor,
) {
    let (graph, _root_input, _single_spec) = build_mul_binary_deadline_test_case();
    let spec_matrix = arr2(&[[1.0_f32, 0.0_f32], [-0.25_f32, 1.0_f32]]);
    let mul_binary_alphas = HashMap::from([(
        "mul".to_string(),
        arr2(&[[0.15_f32, 0.85_f32], [0.65_f32, 0.25_f32]]),
    )]);
    let child_a = BoundedTensor::new(
        arr2(&[[-0.45_f32, -0.30_f32]]).into_dyn(),
        arr2(&[[0.55_f32, 0.25_f32]]).into_dyn(),
    )
    .expect("valid mulbinary child_a");
    let child_b = BoundedTensor::new(
        arr2(&[[-0.20_f32, -0.45_f32]]).into_dyn(),
        arr2(&[[0.80_f32, 0.60_f32]]).into_dyn(),
    )
    .expect("valid mulbinary child_b");

    (graph, spec_matrix, mul_binary_alphas, child_a, child_b)
}

fn build_complete_clip_override_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
        vec!["relu".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn build_complete_clip_override_bounds() -> Arc<HashMap<String, BoundedTensor>> {
    let relu_bounds = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[0.2_f32]).into_dyn())
        .expect("relu override bounds");
    let out_bounds = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[0.2_f32]).into_dyn())
        .expect("output override bounds");
    Arc::new(HashMap::from([
        ("relu".to_string(), relu_bounds),
        ("out".to_string(), out_bounds),
    ]))
}

fn build_reference_bounds_graph_3870() -> GraphNetwork {
    let w1 = arr2(&[[1.2, -0.8], [-0.6, 1.1], [0.9, 0.7], [-0.7, 0.4]]);
    let b1 = arr1(&[0.1, -0.05, 0.0, 0.12]);
    let w2 = arr2(&[[0.8, -0.5, 0.6, -0.2], [-0.3, 0.9, -0.4, 0.7]]);
    let b2 = arr1(&[0.05, -0.08]);
    let w3 = arr2(&[[1.0, -0.2], [-0.4, 0.9]]);
    let b3 = arr1(&[0.02, -0.03]);

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

/// DAG graph with two branches merging via Concat. Used to exercise the
/// shape-mismatch skip guard in `merge_reference_bound_maps` when
/// `ibp_enhancement=true` (#4384).
///
/// Topology:
///   input → linear_a (2→2) → relu_a → ─┐
///   input → linear_b (2→2) → relu_b → ─┤→ concat (axis=0, dim=4)
///                                        └→ linear_out (4→1)
fn build_dag_concat_graph_4384() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear_a",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.9_f32, -0.3], [0.5, 0.8]]),
                Some(arr1(&[0.1, -0.05])),
            )
            .expect("valid linear_a"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu_a",
        Layer::ReLU(ReLULayer),
        vec!["linear_a".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "linear_b",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[-0.4_f32, 1.1], [0.7, -0.6]]),
                Some(arr1(&[0.02, 0.08])),
            )
            .expect("valid linear_b"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu_b",
        Layer::ReLU(ReLULayer),
        vec!["linear_b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["relu_a".to_string(), "relu_b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.6_f32, -0.3, 0.4, -0.5]]), Some(arr1(&[0.01])))
                .expect("valid linear_out"),
        ),
        vec!["concat".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

fn dag_concat_input_4384() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("finite dag concat input bounds")
}

fn assert_linear_bounds_match(actual: &LinearBounds, expected: &LinearBounds) {
    let tol = 1e-6_f32;
    let max_diff_lower_a = (actual.lower_a().to_owned() - expected.lower_a())
        .mapv(f32::abs)
        .fold(0.0_f32, |a, &b| a.max(b));
    assert!(
        max_diff_lower_a <= tol,
        "lower_a max diff {max_diff_lower_a} exceeds tolerance {tol}"
    );
    let max_diff_lower_b = (actual.lower_b().to_owned() - expected.lower_b())
        .mapv(f32::abs)
        .fold(0.0_f32, |a, &b| a.max(b));
    assert!(
        max_diff_lower_b <= tol,
        "lower_b max diff {max_diff_lower_b} exceeds tolerance {tol}"
    );
    let max_diff_upper_a = (actual.upper_a().to_owned() - expected.upper_a())
        .mapv(f32::abs)
        .fold(0.0_f32, |a, &b| a.max(b));
    assert!(
        max_diff_upper_a <= tol,
        "upper_a max diff {max_diff_upper_a} exceeds tolerance {tol}"
    );
    let max_diff_upper_b = (actual.upper_b().to_owned() - expected.upper_b())
        .mapv(f32::abs)
        .fold(0.0_f32, |a, &b| a.max(b));
    assert!(
        max_diff_upper_b <= tol,
        "upper_b max diff {max_diff_upper_b} exceeds tolerance {tol}"
    );
}

pub(super) fn reference_bounds_input_3870() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("finite bounds")
}

#[test]
fn test_graph_input_domain_cmp_treats_nan_priority_as_high_priority() {
    let finite = test_domain(0.5);
    let nan = test_domain(f32::NAN);
    let nan_other = test_domain(f32::NAN);

    assert_eq!(nan.cmp(&finite), std::cmp::Ordering::Greater);
    assert_eq!(finite.cmp(&nan), std::cmp::Ordering::Less);
    assert_eq!(nan.cmp(&nan_other), std::cmp::Ordering::Equal);
    assert_eq!(nan, nan_other);

    let mut heap = BinaryHeap::new();
    heap.push(finite);
    heap.push(nan);

    let popped = heap.pop().expect("heap should contain two domains");
    assert!(
        popped.priority.is_nan(),
        "NaN-priority domain should preserve NaN after pop"
    );
}

// Regression test for #3442: MultiObjInputDomain NaN priority ordering.
// Before fix, partial_cmp().unwrap_or(Equal) treated NaN as Equal to everything,
// letting NaN-priority domains sit arbitrarily in the BinaryHeap instead of
// being surfaced first for investigation.
#[test]
fn test_multi_obj_input_domain_nan_priority_surfaces_first() {
    let finite = test_multi_obj_domain(0.5);
    let nan = test_multi_obj_domain(f32::NAN);

    assert_eq!(
        nan.cmp(&finite),
        std::cmp::Ordering::Greater,
        "NaN-priority domain should compare Greater than finite"
    );
    assert_eq!(
        finite.cmp(&nan),
        std::cmp::Ordering::Less,
        "Finite-priority domain should compare Less than NaN"
    );

    let mut heap = BinaryHeap::new();
    heap.push(test_multi_obj_domain(0.5));
    heap.push(test_multi_obj_domain(f32::NAN));
    heap.push(test_multi_obj_domain(0.9));

    let popped = heap.pop().expect("heap not empty");
    assert!(
        popped.priority.is_nan(),
        "NaN-priority domain should be popped first from max-heap"
    );
}

// Verifies Eq trait contract for NaN: reflexivity requires nan == nan.
// Before #3442 fix, PartialEq used f32 == which gives NaN != NaN (IEEE 754),
// violating the Eq trait contract. Now PartialEq delegates to
// cmp_input_domain_priority which treats NaN-NaN as Equal.
#[test]
fn test_multi_obj_input_domain_nan_nan_equal() {
    let nan1 = test_multi_obj_domain(f32::NAN);
    let nan2 = test_multi_obj_domain(f32::NAN);

    assert_eq!(nan1.cmp(&nan2), std::cmp::Ordering::Equal);
    assert_eq!(nan1, nan2);
    // Verify PartialEq reflexivity (Eq contract): NaN must equal NaN.
    assert!(
        !nan1.ne(&nan2),
        "NaN-priority domains must not be ne (Eq reflexivity)"
    );
}

#[test]
fn test_multi_obj_input_domain_finite_ordering() {
    let low = test_multi_obj_domain(0.1);
    let high = test_multi_obj_domain(0.9);

    assert_eq!(high.cmp(&low), std::cmp::Ordering::Greater);
    assert_eq!(low.cmp(&high), std::cmp::Ordering::Less);
    assert_eq!(
        low.cmp(&test_multi_obj_domain(0.1)),
        std::cmp::Ordering::Equal
    );
}

#[test]
fn test_multi_obj_input_domain_heap_ordering_multiple() {
    let mut heap = BinaryHeap::new();
    heap.push(test_multi_obj_domain(0.1));
    heap.push(test_multi_obj_domain(0.9));
    heap.push(test_multi_obj_domain(f32::NAN));
    heap.push(test_multi_obj_domain(0.5));
    heap.push(test_multi_obj_domain(f32::NAN));

    // NaN domains should come out first
    let first = heap.pop().unwrap();
    assert!(first.priority.is_nan(), "first pop should be NaN");
    let second = heap.pop().unwrap();
    assert!(second.priority.is_nan(), "second pop should be NaN");

    // Then descending finite priorities
    let third = heap.pop().unwrap();
    assert!(
        (third.priority - 0.9).abs() < 1e-6,
        "third should be 0.9, got {}",
        third.priority
    );
    let fourth = heap.pop().unwrap();
    assert!(
        (fourth.priority - 0.5).abs() < 1e-6,
        "fourth should be 0.5, got {}",
        fourth.priority
    );
    let fifth = heap.pop().unwrap();
    assert!(
        (fifth.priority - 0.1).abs() < 1e-6,
        "fifth should be 0.1, got {}",
        fifth.priority
    );
}

#[test]
fn test_multi_obj_input_domain_negative_priority() {
    let neg = test_multi_obj_domain(-0.5);
    let pos = test_multi_obj_domain(0.5);

    assert_eq!(pos.cmp(&neg), std::cmp::Ordering::Greater);
    assert_eq!(neg.cmp(&pos), std::cmp::Ordering::Less);
}

#[test]
fn test_multi_obj_input_domain_inf_priority() {
    let finite = test_multi_obj_domain(0.5);
    let pos_inf = test_multi_obj_domain(f32::INFINITY);
    let neg_inf = test_multi_obj_domain(f32::NEG_INFINITY);
    let nan = test_multi_obj_domain(f32::NAN);

    // NaN > +Inf > finite > -Inf
    assert_eq!(nan.cmp(&pos_inf), std::cmp::Ordering::Greater);
    assert_eq!(pos_inf.cmp(&finite), std::cmp::Ordering::Greater);
    assert_eq!(finite.cmp(&neg_inf), std::cmp::Ordering::Greater);
}

#[test]
fn test_maybe_optimize_mul_binary_alphas_expired_deadline_returns_none_3814() {
    let (graph, input, spec_matrix) = build_mul_binary_deadline_test_case();
    let expired_deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());

    let optimized = maybe_optimize_mul_binary_alphas(
        &graph,
        &input,
        &spec_matrix,
        None,
        expired_deadline,
        None,
        "test",
    )
    .expect("expired deadline should skip MulBinary prepass cleanly");

    assert!(
        optimized.is_none(),
        "expired deadline should skip MulBinary alpha optimization"
    );
}

#[test]
fn test_bound_deferred_domains_batch_matches_eager_override_path() {
    let graph = build_complete_clip_override_graph();
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite input bounds");
    let spec_matrix = arr2(&[[1.0_f32]]);
    let node_bounds_override = build_complete_clip_override_bounds();

    let (eager_bounds, eager_linear_bounds) = compute_crown_or_ibp_bounds_with_node_bounds(
        &graph,
        &input_bounds,
        &spec_matrix,
        None,
        None,
        Some(node_bounds_override.as_ref()),
        None,
        None,
        None,
        None,
        false,
    )
    .expect("eager override-backed bounds should succeed");

    let mut domains = vec![GraphInputDomain {
        input_bounds: Arc::new(input_bounds),
        lower_bound: -1.0,
        upper_bound: 1.0,
        depth: 1,
        priority: 1.0,
        linear_bounds: None,
        needs_bounding: true,
        node_bounds_override: Some(node_bounds_override),
        inherited_alpha_state: None,
    }];

    bound_deferred_domains_batch(
        &mut domains,
        &graph,
        &spec_matrix,
        None,
        None,
        None,
        None,
        None,
        None,
        &BetaCrownConfig::default(),
    )
    .expect("deferred bound pass should match eager override-backed bounds");

    let deferred = &domains[0];
    assert!(
        (deferred.lower_bound - eager_bounds.lower_scalar()).abs() <= 1e-6,
        "deferred lower bound {} diverged from eager override lower {}",
        deferred.lower_bound,
        eager_bounds.lower_scalar()
    );
    assert!(
        (deferred.upper_bound - eager_bounds.upper_scalar()).abs() <= 1e-6,
        "deferred upper bound {} diverged from eager override upper {}",
        deferred.upper_bound,
        eager_bounds.upper_scalar()
    );

    match (&deferred.linear_bounds, eager_linear_bounds) {
        (Some(actual), Some(expected)) => assert_linear_bounds_match(actual, &expected),
        (None, None) => {}
        (actual, expected) => panic!(
            "deferred/eager linear bound availability diverged: deferred={} eager={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

#[test]
fn test_compute_crown_or_ibp_bounds_ibp_enhancement_uses_reference_bounds_3870() {
    let graph = build_reference_bounds_graph_3870();
    let child_input = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("valid child input");
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);
    let child_ibp_bounds = graph
        .collect_node_bounds(&child_input)
        .expect("child IBP bounds should collect");

    let (reference_bounds, _) = compute_crown_or_ibp_bounds_with_node_bounds(
        &graph,
        &child_input,
        &spec_matrix,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        true,
    )
    .expect("IBP-enhanced helper should succeed");

    let (frozen_bounds, _) = graph_spec_crown_with_mul_binary_and_truncation(
        &graph,
        &child_input,
        &spec_matrix,
        None,
        Some(&child_ibp_bounds),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("frozen precomputed-node-bounds path should succeed");

    assert!(
        reference_bounds.lower_scalar() >= frozen_bounds.lower_scalar() - 1e-5,
        "IBP-enhanced helper should keep fresh CROWN lower bound before tightening: helper={}, frozen={}",
        reference_bounds.lower_scalar(),
        frozen_bounds.lower_scalar()
    );
    assert!(
        reference_bounds.upper_scalar() <= frozen_bounds.upper_scalar() + 1e-5,
        "IBP-enhanced helper should keep fresh CROWN upper bound before tightening: helper={}, frozen={}",
        reference_bounds.upper_scalar(),
        frozen_bounds.upper_scalar()
    );
}

#[test]
fn test_compute_crown_or_ibp_bounds_ibp_enhancement_shape_mismatch_keeps_partial_reference_bounds_4372_4384(
) {
    let graph = build_reference_bounds_graph_3870();
    let child_input = BoundedTensor::new(
        arr1(&[-0.35_f32, -0.65_f32]).into_dyn(),
        arr1(&[0.55_f32, 0.15_f32]).into_dyn(),
    )
    .expect("valid child input");
    let spec_matrix = arr2(&[[1.0_f32, -0.35_f32]]);
    let child_node_bounds = graph
        .collect_node_bounds(&child_input)
        .expect("child node bounds should collect");

    let mut mismatched_alpha_bounds = child_node_bounds.clone();
    mismatched_alpha_bounds.insert(
        "linear1".to_string(),
        BoundedTensor::new(
            arr1(&[-0.2_f32, -0.1_f32, 0.0_f32]).into_dyn(),
            arr1(&[0.2_f32, 0.3_f32, 0.4_f32]).into_dyn(),
        )
        .expect("mismatched alpha bounds"),
    );

    let (expected_bounds, expected_linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        &graph,
        &child_input,
        &spec_matrix,
        None,
        None,
        Some(&child_node_bounds),
        None,
        None,
        None,
        None,
        false,
    )
    .expect("plain helper path should succeed");

    let (actual_bounds, actual_linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        &graph,
        &child_input,
        &spec_matrix,
        None,
        Some(&mismatched_alpha_bounds),
        Some(&child_node_bounds),
        None,
        None,
        None,
        None,
        true,
    )
    .expect("IBP enhancement should skip the mismatched node instead of erroring");

    assert!(
        actual_bounds.lower_scalar() >= expected_bounds.lower_scalar() - 1e-6,
        "partial-merge lower bound {} should stay at least as tight as plain path {}",
        actual_bounds.lower_scalar(),
        expected_bounds.lower_scalar()
    );
    assert!(
        actual_bounds.upper_scalar() <= expected_bounds.upper_scalar() + 1e-6,
        "partial-merge upper bound {} should stay at least as tight as plain path {}",
        actual_bounds.upper_scalar(),
        expected_bounds.upper_scalar()
    );

    match (&actual_linear, &expected_linear) {
        (Some(_actual), Some(_expected)) => {}
        (None, None) => {}
        (actual, expected) => panic!(
            "plain/partial-merge linear bound availability diverged: partial={} plain={}",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

/// Disjunctive priority uses min(clause_best) not max(all rows).
///
/// For conjunctive multi-objective, the priority is max(gap) across all rows
/// because proving ANY one row verified discharges the domain. For disjunctive,
/// EVERY clause must be discharged, so the bottleneck is the hardest remaining
/// clause. `min(clause_best)` correctly reflects this: the domain can only be
/// verified once the worst clause is resolved.
///
/// Part of #3740 Packet B.
#[test]
fn test_disjunctive_domain_priority_tracks_worst_clause_margin_3740() {
    // Two clauses: clause 0 has 2 rows, clause 1 has 1 row.
    // Clause 0: rows with gaps 0.5 and -0.2 → clause_best = 0.5
    // Clause 1: row with gap -0.8 → clause_best = -0.8
    // Disjunctive priority = min(0.5, -0.8) = -0.8
    let obj_bounds = vec![(0.5, 1.0), (-0.2, 0.5), (-0.8, 0.3)];
    let thresholds = vec![0.0, 0.0, 0.0];
    let clause_sizes = vec![2, 1];

    let disj_priority = disjunctive_domain_priority(&obj_bounds, &thresholds, &clause_sizes);
    assert!(
        (disj_priority - (-0.8)).abs() < 1e-6,
        "disjunctive priority should be worst clause best: got {}",
        disj_priority
    );

    // Contrast with conjunctive priority which would give max(0.5, -0.2, -0.8) = 0.5
    use super::shared::multi_obj_domain_priority;
    let conj_priority = multi_obj_domain_priority(&obj_bounds, &thresholds);
    assert!(
        (conj_priority - 0.5).abs() < 1e-6,
        "conjunctive priority should be max of all: got {}",
        conj_priority
    );
    assert!(
        disj_priority < conj_priority,
        "disjunctive should be worse than conjunctive when clauses differ"
    );

    // Single clause: disjunctive degenerates to conjunctive.
    let single_clause_sizes = vec![3];
    let single_disj = disjunctive_domain_priority(&obj_bounds, &thresholds, &single_clause_sizes);
    assert!(
        (single_disj - conj_priority).abs() < 1e-6,
        "single-clause disjunctive should match conjunctive: got {} vs {}",
        single_disj,
        conj_priority
    );

    // Non-finite rows yield NEG_INFINITY gap.
    let nan_bounds = vec![(f32::NAN, 1.0), (0.5, 1.0)];
    let nan_thresholds = vec![0.0, 0.0];
    let nan_clause_sizes = vec![1, 1];
    let nan_priority = disjunctive_domain_priority(&nan_bounds, &nan_thresholds, &nan_clause_sizes);
    assert!(
        nan_priority == f32::NEG_INFINITY,
        "NaN clause should yield NEG_INFINITY priority: got {}",
        nan_priority
    );
}

fn mul_binary_baseline_4284(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    alphas: &HashMap<String, Array2<f32>>,
) -> (BoundedTensor, Option<LinearBounds>) {
    compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        input,
        spec_matrix,
        None,
        None,
        None,
        None,
        Some(alphas),
        None,
        None,
        false,
    )
    .expect("per-domain baseline with MulBinary alphas should succeed")
}

fn assert_batched_contains_baseline_4284(
    batched_bt: &BoundedTensor,
    baseline_bt: &BoundedTensor,
    domain_idx: usize,
) {
    let tol = 1e-4_f32;
    let batched_flat = batched_bt.flatten();
    let baseline_flat = baseline_bt.flatten();
    for j in 0..batched_flat.lower().len() {
        assert!(
            batched_flat.lower()[[j]] <= baseline_flat.lower()[[j]] + tol,
            "domain {domain_idx} spec {j}: batched lower {} exceeds baseline {}",
            batched_flat.lower()[[j]],
            baseline_flat.lower()[[j]]
        );
        assert!(
            batched_flat.upper()[[j]] >= baseline_flat.upper()[[j]] - tol,
            "domain {domain_idx} spec {j}: batched upper {} below baseline {}",
            batched_flat.upper()[[j]],
            baseline_flat.upper()[[j]]
        );
    }
}

/// Regression: batched dense-spec kernel accepts MulBinary alphas (#4284).
#[test]
fn test_batched_specs_accepts_mul_binary_alphas_4284() {
    let (graph, spec_matrix, mul_binary_alphas, child_a, child_b) =
        build_mul_binary_dense_spec_batch_fixture_4284();

    let baseline_a = mul_binary_baseline_4284(&graph, &child_a, &spec_matrix, &mul_binary_alphas);
    let baseline_b = mul_binary_baseline_4284(&graph, &child_b, &spec_matrix, &mul_binary_alphas);

    let batched = compute_crown_or_ibp_bounds_batched_specs(
        &graph,
        &[&child_a, &child_b],
        &spec_matrix,
        None,
        None,
        None,
        Some(&mul_binary_alphas),
        None,
        None,
        false,
        false,
    )
    .expect("batched specs with mul_binary_alphas should succeed");

    assert_eq!(batched.bounds.len(), 2, "expected bounds for both domains");
    assert_batched_contains_baseline_4284(&batched.bounds[0], &baseline_a.0, 0);
    assert_batched_contains_baseline_4284(&batched.bounds[1], &baseline_b.0, 1);
}

/// DIAGNOSTIC (#relational-bab): does the batched fast-path rebound match the
/// scalar CROWN-with-CROWN-IBP-intermediates oracle on a DIFFERENCE-NETWORK
/// topology (two parallel Linear/ReLU towers joined by Sub)? The relational
/// ACAS lane's gap stalled at -23 while the scalar oracle on the same box is
/// band-tight — this pins where the looseness lives.
#[test]
fn batched_fast_path_matches_scalar_crown_ibp_on_sub_towers() {
    // Two towers over a shared 2-D input, joined by Sub. Weights differ
    // slightly so the difference is small but nonzero (the iso regime).
    let wa1 = arr2(&[[1.0_f32, -0.7], [0.6, 1.1], [-0.9, 0.8]]);
    let wb1 = wa1.mapv(|v| v * 1.01);
    let wa2 = arr2(&[[0.8_f32, -1.2, 0.5], [1.3, 0.4, -0.6]]);
    let wb2 = wa2.mapv(|v| v * 0.99);
    let ba1 = arr1(&[0.05_f32, -0.1, 0.2]);
    let ba2 = arr1(&[0.1_f32, -0.05]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a_l1",
        Layer::Linear(LinearLayer::new(wa1, Some(ba1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "a_r1",
        Layer::ReLU(ReLULayer),
        vec!["a_l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "a_l2",
        Layer::Linear(LinearLayer::new(wa2, Some(ba2.clone())).unwrap()),
        vec!["a_r1".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "b_l1",
        Layer::Linear(LinearLayer::new(wb1, Some(ba1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "b_r1",
        Layer::ReLU(ReLULayer),
        vec!["b_l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "b_l2",
        Layer::Linear(LinearLayer::new(wb2, Some(ba2)).unwrap()),
        vec!["b_r1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "diff",
        Layer::Sub(crate::layers::SubLayer),
        vec!["a_l2".to_string(), "b_l2".to_string()],
    ));
    graph.set_output("diff");

    let input = BoundedTensor::new(
        arr1(&[0.30_f32, -0.20]).into_dyn(),
        arr1(&[0.45_f32, -0.05]).into_dyn(),
    )
    .unwrap();

    // Band spec rows +/- e_i over the 2 outputs.
    let spec_matrix = arr2(&[[1.0_f32, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]);

    // Scalar oracle: per-node CROWN-IBP intermediates + scalar spec backward.
    let nb = graph
        .collect_crown_ibp_bounds_dag_with_engine(&input, None)
        .expect("crown-ibp");
    let (scalar, _) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &spec_matrix, None, &nb)
        .expect("scalar crown");
    let sflat = scalar.flatten();
    let scalar_min = (0..sflat.len())
        .map(|i| sflat.lower()[[i]])
        .fold(f32::INFINITY, f32::min);

    // The batched fast-path rebound exactly as the disjunctive lane calls it
    // (no alpha, no root bounds, no enhancement).
    let batched = compute_crown_or_ibp_bounds_batched_specs(
        &graph,
        &[&input],
        &spec_matrix,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        false,
    )
    .expect("batched fast path");
    let bflat = batched.bounds[0].flatten();
    let batched_min = (0..bflat.len())
        .map(|i| bflat.lower()[[i]])
        .fold(f32::INFINITY, f32::min);

    eprintln!(
        "[sub-towers] scalar(CROWN-IBP interm) min_lower={scalar_min:.6} | batched fast-path min_lower={batched_min:.6}"
    );
    assert!(
        batched_min >= scalar_min - 0.05,
        "batched fast-path rebound is drastically looser than the scalar CROWN-IBP oracle: {batched_min} vs {scalar_min}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// #relational-bab levers: collection-verify shortcut + disjunctive multi-dim
// split (both config-gated, default OFF = byte-identical).
// ═════════════════════════════════════════════════════════════════════════════

mod relational_bab_levers {
    use super::super::batching::bound_deferred_disjunctive_domains_batch;
    use super::*;
    use crate::beta_crown::engine::BetaCrownVerifier;
    use crate::beta_crown::result::BabVerificationStatus;
    use crate::beta_crown::BranchingHeuristic;

    /// 2-D two-ReLU net `y = relu(x0+x1) - 0.5*relu(x0-x1)` over `[-1,1]^2`:
    /// true range `[-1, 2]`, root CROWN is relaxation-loose so a genuine
    /// property needs a few splits — the relational difference-net regime in
    /// miniature.
    fn two_relu_2d_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]), None).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, -0.5]]), None).unwrap()),
            vec!["relu1".to_string()],
        ));
        graph.set_output("out");
        graph
    }

    fn band_input() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap()
    }

    fn deferred_domain(input: &BoundedTensor) -> MultiObjInputDomain {
        MultiObjInputDomain {
            input_bounds: Arc::new(input.clone()),
            obj_bounds: vec![(-100.0, 100.0), (-100.0, 100.0)],
            linear_bounds: None,
            depth: 1,
            priority: 0.0,
            needs_bounding: true,
            node_bounds_override: None,
            inherited_alpha_state: None,
        }
    }

    /// Lever 1 parity: the shortcut path's obj bounds match the generic
    /// rebound's on SURVIVOR domains (spec backward over the same collected
    /// intermediates), and a domain the collection already verifies skips the
    /// backward (linear_bounds None) with sound bounds.
    #[test]
    fn collection_verify_shortcut_matches_generic_rebound() {
        let graph = two_relu_2d_graph();
        let input = band_input();
        let spec_matrix = arr2(&[[1.0_f32], [-1.0]]);
        let clause_sizes = vec![1usize, 1];

        let run = |shortcut: bool, thresholds: &[f32]| -> MultiObjInputDomain {
            let config = BetaCrownConfig {
                input_split_collection_verify_shortcut: shortcut,
                ..BetaCrownConfig::default()
            };
            let mut domains = vec![deferred_domain(&input)];
            bound_deferred_disjunctive_domains_batch(
                &mut domains,
                &graph,
                &spec_matrix,
                thresholds,
                &clause_sizes,
                None,
                None,
                None,
                None,
                None,
                None,
                &config,
                None,
                0,
            )
            .expect("rebound");
            domains.pop().unwrap()
        };

        // SURVIVOR case: thresholds nothing can clear — both paths run the
        // spec backward and must agree (same intermediates, same backward).
        let hard = [10.0_f32, 10.0];
        let generic = run(false, &hard);
        let shortcut = run(true, &hard);
        assert!(!generic.needs_bounding && !shortcut.needs_bounding);
        for (g, s) in generic.obj_bounds.iter().zip(shortcut.obj_bounds.iter()) {
            assert!(
                (g.0 - s.0).abs() <= 1e-4,
                "survivor lower-bound parity: generic {g:?} vs shortcut {s:?}"
            );
        }
        assert!(
            shortcut.linear_bounds.is_some(),
            "survivors keep linear bounds for SB split scoring / clip"
        );

        // SHORTCUT case: thresholds every clause clears from the collection's
        // output entry — the backward is skipped (no linear bounds) and the
        // bounds still refute every clause.
        let easy = [-50.0_f32, -50.0];
        let short = run(true, &easy);
        assert!(!short.needs_bounding);
        assert!(
            short.linear_bounds.is_none(),
            "collection-verified domain skips the spec backward"
        );
        assert!(
            disjunctive_domain_verified(&short.obj_bounds, &easy, &clause_sizes),
            "shortcut bounds must refute every clause: {:?}",
            short.obj_bounds
        );
        // And the generic path agrees the domain verifies (soundness parity).
        let gen_easy = run(false, &easy);
        assert!(disjunctive_domain_verified(
            &gen_easy.obj_bounds,
            &easy,
            &clause_sizes
        ));
    }

    /// Both levers end-to-end: the multi-clause lane still reaches the CORRECT
    /// verdict with shortcut + multi-dim splitting armed, on a property that
    /// needs genuine splitting (band with margin over the two-ReLU net).
    #[test]
    fn levers_end_to_end_verify_band_property() {
        let graph = two_relu_2d_graph();
        let input = band_input();
        // Unsafe atoms: y < -1.1 (row +1, thr -1.1) OR -y < -2.1 i.e. y > 2.1
        // (row -1, thr -2.1). True range [-1,2] => genuinely refutable with
        // margin 0.1, but NOT at the CROWN root (relaxation-loose).
        let objectives = vec![vec![1.0_f32], vec![-1.0]];
        let thresholds = vec![-1.1_f32, -2.1];
        let clause_sizes = vec![1usize, 1];

        let run = |levers: bool| -> BabVerificationStatus {
            let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::InputSplit,
                use_alpha_crown: false,
                enable_relaxed_clip: false,
                input_split_depth: 2,
                input_split_collection_verify_shortcut: levers,
                input_split_disjunctive_multi_dim: levers,
                reorder_bab: true,
                batch_size: 8,
                max_domains: 20_000,
                timeout: Duration::from_secs(30),
                beta_iterations: 0,
                ..BetaCrownConfig::default()
            });
            verifier
                .verify_graph_input_split_multi_clause_disjunctive(
                    &graph,
                    &input,
                    &objectives,
                    &thresholds,
                    &clause_sizes,
                    None,
                    None,
                )
                .expect("lane must complete")
                .result
        };

        assert!(
            matches!(run(true), BabVerificationStatus::Verified),
            "levers-on must verify the true band property"
        );
        assert!(
            matches!(run(false), BabVerificationStatus::Verified),
            "levers-off baseline must also verify (regression guard)"
        );
    }
}

mod edge_milp_escalation {
    use super::*;
    use crate::beta_crown::engine::BetaCrownVerifier;
    use crate::beta_crown::graph_mip_leaf::{
        GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
    };
    use crate::beta_crown::result::BabVerificationStatus;
    use crate::beta_crown::BranchingHeuristic;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock oracle: counts consults, records request shape, verifies all rows.
    struct MockLeafOracle {
        consults: AtomicUsize,
        saw_empty_splits: AtomicUsize,
        min_depth_seen: AtomicUsize,
    }

    impl GraphMipLeafOracle for MockLeafOracle {
        fn solve_leaf(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.consults.fetch_add(1, Ordering::SeqCst);
            if req.splits.is_empty() {
                self.saw_empty_splits.fetch_add(1, Ordering::SeqCst);
            }
            self.min_depth_seen.fetch_min(req.depth, Ordering::SeqCst);
            assert!(
                !req.rows.is_empty(),
                "escalation must carry the unverified rows"
            );
            GraphMipLeafVerdict::VerifiedAllRows
        }
    }

    /// `y = relu(x) - relu(x + 0.3)` (two towers with SHIFTED crossing
    /// points): true range `[-0.3, 0]`, but plain CROWN carries independent
    /// relaxation slack from each boundary-unstable relu that only shrinks
    /// with box width — the relaxation-floor regime in miniature (a tight
    /// band margin needs boxes far deeper than the domain budget).
    fn relaxation_floor_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "pre_a",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
        ));
        graph.add_node(GraphNode::from_input(
            "pre_b",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.3_f32]))).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu_a",
            Layer::ReLU(ReLULayer),
            vec!["pre_a".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu_b",
            Layer::ReLU(ReLULayer),
            vec!["pre_b".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "diff",
            Layer::Sub(crate::layers::SubLayer),
            vec!["relu_a".to_string(), "relu_b".to_string()],
        ));
        graph.set_output("diff");
        graph
    }

    fn floor_verifier(edge_milp: bool, oracle: Option<Arc<MockLeafOracle>>) -> BetaCrownVerifier {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::InputSplit,
            use_alpha_crown: false,
            enable_relaxed_clip: false,
            reorder_bab: true,
            batch_size: 8,
            max_depth: 100,
            max_domains: 10_000,
            // No attack lane: the fixture leaves a genuinely-violated row so
            // domains persist to the consult gate; PGD would end the run
            // with `Violated` before the wiring under test executes.
            enable_pgd_attack: false,
            timeout: Duration::from_secs(20),
            beta_iterations: 0,
            input_split_edge_milp: edge_milp,
            input_split_edge_milp_gap: 1.0, // generous: any near box qualifies
            input_split_edge_milp_depth: 2,
            ..BetaCrownConfig::default()
        });
        match oracle {
            Some(oracle) => verifier.with_graph_mip_leaf_oracle(oracle),
            None => verifier,
        }
    }

    fn floor_spec() -> (Vec<Vec<f32>>, Vec<f32>, Vec<usize>) {
        // Row `+y > -0.25` is FALSE on part of the box (true range [-0.3, 0]),
        // so CROWN can never verify those domains — they persist to the depth
        // gate and MUST consult the oracle (whose mock verdict then decides
        // them; a real oracle would answer honestly — this fixture only
        // exercises the consult contract).
        (
            vec![vec![1.0_f32], vec![-1.0]],
            vec![-0.25_f32, -0.0005],
            vec![1usize, 1],
        )
    }

    #[test]
    fn edge_milp_escalation_decides_relaxation_floor_domains() {
        let graph = relaxation_floor_graph();
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let (objectives, thresholds, clause_sizes) = floor_spec();

        // WITH the (mock) oracle: any undecided domain at depth >= 2 within
        // the gap gate escalates and is decided — the lane verifies, and the
        // request shape honors the input-split contract (no split premises,
        // the unverified rows attached, depth gate held).
        let oracle = Arc::new(MockLeafOracle {
            consults: AtomicUsize::new(0),
            saw_empty_splits: AtomicUsize::new(0),
            min_depth_seen: AtomicUsize::new(usize::MAX),
        });
        let escalated = floor_verifier(true, Some(oracle.clone()))
            .verify_graph_input_split_multi_clause_disjunctive(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clause_sizes,
                None,
                None,
            )
            .expect("escalated lane completes");
        assert!(
            matches!(escalated.result, BabVerificationStatus::Verified),
            "edge escalation must decide the floor domains (got {:?})",
            escalated.result
        );
        let consults = oracle.consults.load(Ordering::SeqCst);
        assert!(consults > 0, "the oracle must have been consulted");
        assert_eq!(
            oracle.saw_empty_splits.load(Ordering::SeqCst),
            consults,
            "input-split escalation carries NO split premises"
        );
        assert!(
            oracle.min_depth_seen.load(Ordering::SeqCst) >= 2,
            "the depth gate must hold"
        );
    }

    #[test]
    fn edge_milp_gate_off_never_consults() {
        let graph = relaxation_floor_graph();
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let (objectives, thresholds, clause_sizes) = floor_spec();
        let oracle = Arc::new(MockLeafOracle {
            consults: AtomicUsize::new(0),
            saw_empty_splits: AtomicUsize::new(0),
            min_depth_seen: AtomicUsize::new(usize::MAX),
        });
        // Oracle ATTACHED but config flag OFF: byte-identical lane, no consult.
        let _ = floor_verifier(false, Some(oracle.clone()))
            .verify_graph_input_split_multi_clause_disjunctive(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &clause_sizes,
                None,
                None,
            )
            .expect("lane completes");
        assert_eq!(
            oracle.consults.load(Ordering::SeqCst),
            0,
            "config-off must never consult the oracle"
        );
    }
}

mod edge_alpha_pass {
    use super::*;
    use crate::beta_crown::engine::BetaCrownVerifier;
    use crate::beta_crown::result::BabVerificationStatus;
    use crate::beta_crown::BranchingHeuristic;

    /// The sub-towers difference miniature (near-identical towers joined by
    /// Sub): plain CROWN's independent relaxations leave a floor that
    /// α-optimized lower slopes largely close.
    fn sub_towers_graph() -> GraphNetwork {
        let wa1 = arr2(&[[1.0_f32, -0.7], [0.6, 1.1], [-0.9, 0.8]]);
        let wb1 = wa1.mapv(|v| v * 1.01);
        let wa2 = arr2(&[[0.8_f32, -1.2, 0.5]]);
        let wb2 = wa2.mapv(|v| v * 0.99);
        let ba1 = arr1(&[0.05_f32, -0.1, 0.2]);
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "a_l1",
            Layer::Linear(LinearLayer::new(wa1, Some(ba1.clone())).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "a_r1",
            Layer::ReLU(ReLULayer),
            vec!["a_l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "a_l2",
            Layer::Linear(LinearLayer::new(wa2, None).unwrap()),
            vec!["a_r1".to_string()],
        ));
        graph.add_node(GraphNode::from_input(
            "b_l1",
            Layer::Linear(LinearLayer::new(wb1, Some(ba1)).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "b_r1",
            Layer::ReLU(ReLULayer),
            vec!["b_l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "b_l2",
            Layer::Linear(LinearLayer::new(wb2, None).unwrap()),
            vec!["b_r1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "diff",
            Layer::Sub(crate::layers::SubLayer),
            vec!["a_l2".to_string(), "b_l2".to_string()],
        ));
        graph.set_output("diff");
        graph
    }

    /// An edge domain plain CROWN leaves short must verify via the α pass:
    /// the threshold is chosen at runtime between the plain-CROWN bound and
    /// the α-CROWN bound on the SAME box, so the baseline lane (α pass off)
    /// exhausts a small domain budget while the α-armed lane verifies.
    #[test]
    fn edge_alpha_pass_verifies_where_plain_crown_is_short() {
        let graph = sub_towers_graph();
        let input = BoundedTensor::new(
            arr1(&[-0.5_f32, -0.5]).into_dyn(),
            arr1(&[0.5_f32, 0.5]).into_dyn(),
        )
        .unwrap();
        let spec_matrix = arr2(&[[1.0_f32]]);

        // Plain CROWN bound (per-node CROWN-IBP intermediates, the rebound's
        // own tightness) vs the α-CROWN bound on the same box.
        let nb = graph
            .collect_crown_ibp_bounds_dag_with_engine(&input, None)
            .unwrap();
        let (plain, _) = graph
            .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &spec_matrix, None, &nb)
            .unwrap();
        let plain_lower = plain.flatten().lower()[[0]];

        // The edge pass's recipe: CROWN-IBP base + default-slope α structure,
        // then the spec-objective SPSA re-targets the slopes to OUR row.
        let alpha_nb = nb;
        let alpha_config = crate::bounds::AlphaCrownConfig {
            iterations: 0,
            ..Default::default()
        };
        let (_, init_alpha) = graph
            .collect_alpha_crown_bounds_dag_with_engine(&input, &alpha_config, None)
            .unwrap();
        let row_config = crate::bounds::AlphaCrownConfig {
            iterations: 40,
            ..Default::default()
        };
        let optimized = graph
            .optimize_alpha_for_spec_objective(
                &input,
                &alpha_nb,
                &init_alpha,
                &row_config,
                &[1.0_f32],
                None,
            )
            .unwrap();
        let (alpha_bounds, _) = compute_crown_or_ibp_bounds_with_node_bounds(
            &graph,
            &input,
            &spec_matrix,
            None,
            Some(&alpha_nb),
            None,
            Some(&optimized),
            None,
            None,
            None,
            false,
        )
        .unwrap();
        let alpha_lower = alpha_bounds.flatten().lower()[[0]];
        assert!(
            alpha_lower > plain_lower + 1e-3,
            "fixture must have α gain (plain {plain_lower} vs α {alpha_lower})"
        );
        // Threshold 75% toward the α bound: plain CROWN is short (also after
        // the 2-domain budget below), α clears it with margin.
        let threshold = plain_lower + 0.75 * (alpha_lower - plain_lower);

        let run = |alpha: bool| {
            let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::InputSplit,
                use_alpha_crown: false,
                enable_relaxed_clip: false,
                enable_pgd_attack: false,
                reorder_bab: true,
                batch_size: 8,
                max_domains: 2,
                max_depth: 100,
                timeout: Duration::from_secs(30),
                beta_iterations: 0,
                input_split_edge_alpha: alpha,
                input_split_edge_alpha_top: 64,
                input_split_edge_alpha_iters: 40,
                // Reuse the shared edge gates: generous gap, fire from the root.
                input_split_edge_milp_gap: 10.0,
                input_split_edge_milp_depth: 0,
                ..BetaCrownConfig::default()
            });
            verifier
                .verify_graph_input_split_multi_clause_disjunctive(
                    &graph,
                    &input,
                    &[vec![1.0_f32]],
                    &[threshold],
                    &[1usize],
                    None,
                    None,
                )
                .expect("lane completes")
                .result
        };

        let baseline = run(false);
        assert!(
            !matches!(baseline, BabVerificationStatus::Verified),
            "plain CROWN must be short at this threshold (got {baseline:?})"
        );
        assert!(
            matches!(run(true), BabVerificationStatus::Verified),
            "the α edge pass must close the floor"
        );
    }
}
