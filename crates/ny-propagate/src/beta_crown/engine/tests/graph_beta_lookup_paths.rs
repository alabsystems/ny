// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph β lookup path regressions.
//!
//! Part of #2936: prove the public constrained-CROWN graph path accumulates
//! duplicate same-node β entries correctly on both fresh and stale indexes.

use super::gpu_bab::simple_graph_network;
use super::prelude::*;
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::state::GraphBetaEntry;
use std::collections::HashMap;

fn setup_graph_with_bounds() -> (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, Arc<BoundedTensor>>,
) {
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .unwrap()
        .into_iter()
        .map(|(name, bounds)| (name, Arc::new(bounds)))
        .collect();
    (graph, input, node_bounds)
}

fn beta_entry(
    node_name: &str,
    neuron_idx: usize,
    split_point: f32,
    value: f32,
    sign: f32,
) -> GraphBetaEntry {
    GraphBetaEntry::new(node_name.to_string(), neuron_idx, split_point, value, sign)
        .expect("test beta entry should be valid")
}

fn assert_close(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "{label}: expected {expected}, got {actual}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_constrained_crown_matches_aggregated_duplicate_beta_entries_2936() {
    let (graph, input, node_bounds) = setup_graph_with_bounds();
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let objective = [1.0_f32];

    let aggregated_beta =
        GraphBetaState::from_entries(vec![beta_entry("relu1", 0, 0.0, 0.75, 1.0)]);
    let indexed_multi_beta = GraphBetaState::from_entries(vec![
        beta_entry("relu1", 0, 0.0, 1.0, 1.0),
        beta_entry("relu1", 0, 0.5, 0.25, -1.0),
    ]);
    let mut stale_multi_beta =
        GraphBetaState::from_entries(vec![beta_entry("relu1", 0, 0.0, 1.0, 1.0)]);
    stale_multi_beta
        .entries
        .push(beta_entry("relu1", 0, 0.5, 0.25, -1.0));

    let (baseline_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, Some(&objective))
        .expect("baseline constrained CROWN should succeed");
    let (aggregated_output, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input,
            &context,
            Some(&aggregated_beta),
            Some(&objective),
        )
        .expect("aggregated beta constrained CROWN should succeed");
    let (indexed_multi_output, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input,
            &context,
            Some(&indexed_multi_beta),
            Some(&objective),
        )
        .expect("fresh indexed duplicate beta constrained CROWN should succeed");
    let (stale_multi_output, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input,
            &context,
            Some(&stale_multi_beta),
            Some(&objective),
        )
        .expect("stale duplicate beta constrained CROWN should succeed");

    assert_close(
        indexed_multi_output.lower_scalar(),
        aggregated_output.lower_scalar(),
        "fresh duplicate lower bound",
    );
    assert_close(
        indexed_multi_output.upper_scalar(),
        aggregated_output.upper_scalar(),
        "fresh duplicate upper bound",
    );
    assert_close(
        stale_multi_output.lower_scalar(),
        aggregated_output.lower_scalar(),
        "stale duplicate lower bound",
    );
    assert_close(
        stale_multi_output.upper_scalar(),
        aggregated_output.upper_scalar(),
        "stale duplicate upper bound",
    );

    let total_shift = (aggregated_output.lower_scalar() - baseline_output.lower_scalar()).abs()
        + (aggregated_output.upper_scalar() - baseline_output.upper_scalar()).abs();
    assert!(
        total_shift > 1e-4,
        "beta contributions should measurably change the constrained-CROWN bounds; total shift={total_shift}"
    );
}
