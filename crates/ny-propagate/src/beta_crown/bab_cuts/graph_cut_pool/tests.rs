// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::beta_crown::bab_cuts::{CutKind, CutMetadata, GraphCutTerm, GraphCuttingPlane};
use crate::network::GraphNode;
use crate::{BoundedTensor, GraphNetwork};
use ndarray::{arr1, arr2, Array1};
use std::collections::HashMap;
use std::sync::Arc;

fn make_cut(terms: Vec<(&str, usize, f32)>, bias: f32) -> GraphCuttingPlane {
    GraphCuttingPlane {
        terms: terms
            .into_iter()
            .map(|(node, neuron, coeff)| GraphCutTerm {
                node_name: node.to_string(),
                neuron_idx: neuron,
                coefficient: coeff,
            })
            .collect(),
        bias,
        lambda: 0.01,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 2,
        metadata: CutMetadata::new(0, CutKind::Verified),
    }
}

fn build_two_relu_graph_with_bounds() -> (GraphNetwork, HashMap<String, Arc<BoundedTensor>>) {
    use crate::layers::{Layer, LinearLayer, ReLULayer};

    let mut graph = GraphNetwork::new();
    let linear1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
        Some(arr1(&[0.1_f32, -0.1])),
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.3], [0.3, 1.0]]),
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");

    let bounds_linear1 = BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -0.5]).into_dyn(),
        Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
    )
    .unwrap();
    let bounds_linear2 = BoundedTensor::new(
        Array1::from_vec(vec![-0.8, -0.3]).into_dyn(),
        Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
    )
    .unwrap();

    let mut node_bounds = HashMap::new();
    node_bounds.insert("linear1".to_string(), Arc::new(bounds_linear1));
    node_bounds.insert("linear2".to_string(), Arc::new(bounds_linear2));
    (graph, node_bounds)
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_sibling_cuts_basic() {
    let cut_a = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, -1.0)], 0.0);
    let mut pool = GraphCutPool::new(10);
    pool.add_cut(cut_a);
    pool.add_cut(cut_b);

    let count = pool.merge_cuts();

    assert_eq!(count, 1, "Expected 1 cut after merge, got {}", count);
    assert_eq!(pool.cuts[0].terms.len(), 1);
    assert_eq!(pool.cuts[0].bias, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_generate_proactive_cuts_returns_result_2998() {
    let (graph, node_bounds) = build_two_relu_graph_with_bounds();
    let mut pool = GraphCutPool::new(100);

    let generated = pool
        .generate_proactive_cuts(&graph, &node_bounds, 50)
        .expect("valid graph proactive cut generation should return Ok(count)");

    assert!(
        generated > 0,
        "expected proactive cuts for unstable ReLU nodes"
    );
    assert!(
        pool.cuts.iter().all(|cut| cut.source_depth == 0),
        "graph proactive cuts must be tagged with source_depth=0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_no_siblings() {
    let cut_a = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![("node_a", 0, -1.0), ("node_a", 1, -1.0)], -1.0);
    let mut pool = GraphCutPool::new(10);
    pool.add_cut(cut_a);
    pool.add_cut(cut_b);

    let count = pool.merge_cuts();

    assert_eq!(count, 2, "Expected 2 cuts (no merge), got {}", count);
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_handles_duplicate_term_ids_with_mixed_signs() {
    // Regression for sibling_signature matching only (node, neuron). If a cut
    // contains both signs for the same term id, the positive term must still be
    // the one that flips during sibling lookup.
    let cut_a = make_cut(vec![("node_a", 0, -1.0), ("node_a", 0, 1.0)], 0.0);
    let cut_b = make_cut(vec![("node_a", 0, -1.0), ("node_a", 0, -1.0)], -1.0);
    let mut pool = GraphCutPool::new(10);
    pool.add_cut(cut_a);
    pool.add_cut(cut_b);

    let count = pool.merge_cuts();

    assert_eq!(count, 1, "Expected merge despite duplicate term ids");
    assert_eq!(pool.cuts[0].terms.len(), 1, "Expected single-term parent");
    assert!(
        pool.cuts[0].terms[0].coefficient < 0.0,
        "Expected remaining parent term to keep the negative sign"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_exhausts_duplicate_single_term_siblings() {
    // Regression: the sibling index must keep all duplicate candidates. With
    // single-term sibling cuts, every pair should disappear because the parent
    // would be empty and therefore dropped.
    let cut_a1 = make_cut(vec![("node_a", 0, 1.0)], 0.0);
    let cut_a2 = make_cut(vec![("node_a", 0, 1.0)], 0.0);
    let cut_b1 = make_cut(vec![("node_a", 0, -1.0)], -1.0);
    let cut_b2 = make_cut(vec![("node_a", 0, -1.0)], -1.0);
    let mut pool = GraphCutPool::new(10);
    pool.add_cut(cut_a1);
    pool.add_cut(cut_a2);
    pool.add_cut(cut_b1);
    pool.add_cut(cut_b2);

    let count = pool.merge_cuts();

    assert_eq!(
        count, 0,
        "Expected all duplicate sibling pairs to merge away"
    );
    assert!(
        pool.cuts.is_empty(),
        "Expected no residual cuts after merging"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_prune_redundant_after_merge() {
    let cut_a = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, -1.0)], 0.0);
    let cut_c = make_cut(
        vec![("node_a", 0, 1.0), ("node_a", 1, 1.0), ("node_b", 0, 1.0)],
        2.0,
    );
    let mut pool = GraphCutPool::new(10);
    pool.add_cut(cut_a);
    pool.add_cut(cut_b);
    pool.add_cut(cut_c);

    let count = pool.merge_cuts();

    assert_eq!(count, 1, "Expected exactly 1 cut after merge and prune");
    assert_eq!(pool.cuts[0].terms.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_iterative_merge_dedup_identical_parents() {
    let cut_a = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![("node_a", 0, 1.0), ("node_a", 1, -1.0)], 0.0);
    let cut_c = make_cut(vec![("node_a", 0, 1.0), ("node_a", 2, 1.0)], 1.0);
    let cut_d = make_cut(vec![("node_a", 0, 1.0), ("node_a", 2, -1.0)], 0.0);
    let mut pool = GraphCutPool::new(10);
    pool.add_cut(cut_a);
    pool.add_cut(cut_b);
    pool.add_cut(cut_c);
    pool.add_cut(cut_d);

    let count = pool.merge_cuts();

    assert_eq!(count, 1, "Expected exactly 1 cut after merge and dedup");
    assert_eq!(pool.cuts[0].terms.len(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_nan_lambda_cut_evicted_as_hard_stale_2598() {
    // A cut with NaN lambda must be eligible for hard-stale eviction.
    // Before the fix, `NaN.abs() < cut_lambda_min` returned false (IEEE 754),
    // making NaN-lambda cuts immortal in the pool.
    let mut pool = GraphCutPool::new(1);
    pool.cut_hard_stale_iters = 5;

    // Insert a NaN-lambda cut at iter 0.
    let mut nan_cut = make_cut(vec![("node_a", 0, 1.0)], 0.5);
    nan_cut.lambda = f32::NAN;
    nan_cut.metadata = CutMetadata::new(0, CutKind::NearMiss);
    pool.cuts.push(nan_cut);

    // Advance iter well past hard_stale threshold.
    pool.iter_counter.store(100, Ordering::Relaxed);

    // Adding a new cut should evict the NaN cut.
    let healthy_cut = make_cut(vec![("node_b", 1, 2.0)], 1.0);
    let added = pool.add_cut(healthy_cut);
    assert!(
        added,
        "healthy cut should be accepted after evicting NaN cut"
    );
    assert_eq!(
        pool.cuts.len(),
        1,
        "pool should have exactly 1 cut after eviction"
    );
    assert!(
        !pool.cuts[0].lambda.is_nan(),
        "remaining cut should be the healthy one (lambda={}, not NaN)",
        pool.cuts[0].lambda
    );
    assert!(
        pool.cuts_evicted_stale > 0,
        "eviction should be counted as stale eviction"
    );
}

/// Regression test for #3148: the create_parent_cut guard must reject
/// non-finite parent biases. We inject -Inf bias via direct struct push
/// (bypassing GraphCuttingPlane::new validation) to verify the guard fires.
#[ntest::timeout(10000)]
#[test]
fn test_merge_f32_min_bias_no_overflow_3148() {
    // Use -Inf bias to actually trigger the non-finite guard.
    // Note: f32::MIN - 1.0 == f32::MIN (1.0 is below ULP), so f32::MIN
    // would NOT trigger the guard. -Inf - 1.0 == -Inf which IS non-finite.
    let cut_a = make_cut(
        vec![("node_a", 0, 1.0), ("node_a", 1, 1.0)],
        f32::NEG_INFINITY,
    );
    let cut_b = make_cut(
        vec![("node_a", 0, 1.0), ("node_a", 1, -1.0)],
        f32::NEG_INFINITY,
    );

    let mut pool = GraphCutPool::new(10);
    // Bypass add_cut validation by pushing directly
    pool.cuts.push(cut_a);
    pool.cuts.push(cut_b);

    let count = pool.merge_cuts();

    // The guard in create_parent_cut returns None for non-finite bias,
    // so no parent is added.
    for cut in &pool.cuts {
        assert!(
            cut.bias.is_finite(),
            "No cut should have non-finite bias after merge, got {}",
            cut.bias
        );
    }
    assert!(count <= 2, "Expected at most 2 cuts, got {}", count);
}
