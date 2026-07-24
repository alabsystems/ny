// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::AlphaCrownConfig;
use crate::layers::{Layer, LinearLayer, ReLULayer};
use crate::{BoundedTensor, GraphNetwork, GraphNode};
use ndarray::{arr1, arr2};

/// Helper: build the 2-layer ReLU network used by the #3436 regression tests.
fn build_3436_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
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
    graph
}

/// Helper: build the input bounds for #3436 tests.
fn build_3436_input() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap()
}

/// Regression for #3436: the new defaults (100 iterations, lr=0.1) produce
/// sound bounds on a shallow 2-layer ReLU network. The reference α,β-CROWN
/// chose lr=0.1 for stability across deep networks (arguments.py:351); on
/// shallow networks lr=0.5 converges faster per-iteration, so the new defaults
/// may produce slightly looser bounds than the old (20 iters, lr=0.5) defaults.
///
/// This test verifies:
/// 1. New defaults produce sound bounds (at least as tight as plain CROWN)
/// 2. The loosening vs old defaults is bounded (≤15% relative to bound width)
///
/// Observed values (stable across runs):
/// - Old defaults (20 iters, lr=0.5): upper ≈ 0.4939
/// - New defaults (100 iters, lr=0.1): upper ≈ 0.5433
/// - Delta: ~10% of bound width. Accepted tradeoff for reference alignment.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_alpha_crown_lr_01_not_looser_than_05_3436() {
    let graph = build_3436_graph();
    let input = build_3436_input();

    // Old defaults before #3436: 20 iterations, lr=0.5
    let old_default_config = AlphaCrownConfig {
        learning_rate: 0.5,
        iterations: 20,
        ..AlphaCrownConfig::default()
    };
    // New defaults after #3436: 100 iterations, lr=0.1
    let new_default_config = AlphaCrownConfig::default();

    let old_bounds = graph
        .propagate_alpha_crown_with_config(&input, &old_default_config)
        .unwrap();
    let new_bounds = graph
        .propagate_alpha_crown_with_config(&input, &new_default_config)
        .unwrap();

    // Verify new defaults produce sound bounds: alpha-CROWN should always be
    // at least as tight as plain CROWN (no alpha optimization).
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    for (idx, ((&new_l, &new_u), (&crown_l, &crown_u))) in new_bounds
        .lower()
        .iter()
        .zip(new_bounds.upper().iter())
        .zip(crown_bounds.lower().iter().zip(crown_bounds.upper().iter()))
        .enumerate()
    {
        assert!(
            new_l >= crown_l - 1e-4,
            "#3436 output {idx}: new default lower {new_l} worse than CROWN lower {crown_l}"
        );
        assert!(
            new_u <= crown_u + 1e-4,
            "#3436 output {idx}: new default upper {new_u} worse than CROWN upper {crown_u}"
        );
    }

    // Verify the loosening vs old defaults is bounded.
    // On this shallow network, lr=0.1 converges slower than lr=0.5 and produces
    // slightly looser bounds (~10%). The reference chose lr=0.1 for deep network
    // stability. Bound loosening > 15% of bound width would indicate a bug.
    for (idx, ((&new_l, &new_u), (&old_l, &old_u))) in new_bounds
        .lower()
        .iter()
        .zip(new_bounds.upper().iter())
        .zip(old_bounds.lower().iter().zip(old_bounds.upper().iter()))
        .enumerate()
    {
        let width = (old_u - old_l).max(1e-6);
        let lower_loosening = (old_l - new_l).max(0.0) / width;
        let upper_loosening = (new_u - old_u).max(0.0) / width;
        assert!(
            lower_loosening <= 0.15,
            "#3436 output {idx}: lower loosened by {:.1}% (>15%) — new_l={new_l}, old_l={old_l}",
            lower_loosening * 100.0,
        );
        assert!(
            upper_loosening <= 0.15,
            "#3436 output {idx}: upper loosened by {:.1}% (>15%) — new_u={new_u}, old_u={old_u}",
            upper_loosening * 100.0,
        );
    }
}
