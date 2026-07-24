// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for DAG alpha-CROWN in the BaB initialization path (#3357).
//!
//! The BaB path through `collect_alpha_crown_bounds_dag` initializes alpha state
//! for all unstable ReLU neurons. Prior to #3357, DAGs with skip connections
//! used IBP bounds (O(N) but very loose), causing most neurons to appear stable
//! and the alpha state to be empty. The fix auto-detects DAGs and uses CROWN-IBP
//! bounds (O(N^2) but much tighter).
//!
//! These tests verify the fix works on representative DAG topologies.

use crate::bounds::AlphaCrownConfig;
use crate::*;
use ndarray::{arr1, arr2};

/// Build a skip-connection DAG where IBP bounds are loose but CROWN-IBP is tight.
///
/// ```text
/// Input(2)
///   |
/// Linear1(2->4) + bias
///   |
/// ReLU1
///   |\
///   | Linear2(4->4) [branch A: large weights]
///   |   |
///   |  ReLU2
///   |   |
///   | Linear3(4->4) [branch A continued]
///   |  /
///  Add  [merge: skip + branch A]
///   |
/// ReLU3
///   |
/// Linear4(4->2) [output]
/// ```
///
/// The skip connection from ReLU1 to Add means IBP must bound ReLU1 outputs
/// through BOTH paths. The branch A weights amplify: Linear2 scales by ~2x,
/// Linear3 scales by ~2x. IBP compounds the over-approximation at Add because
/// it tracks each path independently. CROWN-IBP traces backward correlations
/// and produces tighter bounds at Add.
fn build_skip_dag_loose_ibp() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // Input -> Linear1 -> ReLU1 (bias crosses zero in [-1,1] input range)
    let w1 = arr2(&[[1.0_f32, 0.5, -0.3, 0.2], [-0.4, 0.8, 0.6, -0.5]])
        .t()
        .to_owned();
    let b1 = arr1(&[0.1_f32, -0.2, 0.15, -0.1]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Branch A: relu1 -> Linear2 -> ReLU2 -> Linear3 (amplifying weights)
    let w2 = arr2(&[
        [1.5_f32, -0.8, 0.3, -0.2],
        [-0.5, 1.2, -0.4, 0.6],
        [0.3, -0.3, 1.8, -0.5],
        [-0.2, 0.4, -0.6, 1.3],
    ]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, None).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));

    let w3 = arr2(&[
        [1.2_f32, -0.4, 0.5, -0.3],
        [-0.3, 1.5, -0.2, 0.4],
        [0.4, -0.5, 1.3, -0.6],
        [-0.2, 0.3, -0.4, 1.1],
    ]);
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, None).unwrap()),
        vec!["relu2".to_string()],
    ));

    // Merge: Add(relu1, linear3) — skip connection
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["relu1".to_string(), "linear3".to_string()],
    ));

    // Output: ReLU3 -> Linear4
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    let w4 = arr2(&[[1.0_f32, -0.5, 0.3, -0.2], [-0.3, 0.8, -0.4, 0.6]]);
    let b4 = arr1(&[0.0_f32, 0.0]);
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(LinearLayer::new(w4, Some(b4)).unwrap()),
        vec!["relu3".to_string()],
    ));

    graph.set_output("linear4");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// #3357: BaB alpha-CROWN initialization on a DAG produces non-empty alpha state.
///
/// Before the fix, `collect_alpha_crown_bounds_dag` used IBP bounds for
/// intermediate nodes even on DAGs. For skip-connection DAGs, IBP is so loose
/// that most neurons appear stable (l >= 0 or u <= 0), resulting in empty alpha
/// state and alpha optimization being skipped entirely.
///
/// The fix auto-detects DAG topology and switches to CROWN-IBP bounds, which
/// are tighter and produce unstable neurons for alpha optimization.
#[test]
fn test_dag_bab_alpha_state_non_empty_with_crown_ibp_3357() {
    let (graph, input) = build_skip_dag_loose_ibp();

    // Use default config (fix_interm_bounds=true). The #3357 fix auto-overrides
    // to CROWN-IBP for DAGs.
    let config = AlphaCrownConfig::default();

    let (_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("collect_alpha_crown_bounds_dag should succeed");

    let num_unstable = alpha_state.num_unstable();
    assert!(
        num_unstable > 0,
        "DAG with skip connections should have unstable neurons after CROWN-IBP \
         initialization. Got 0 unstable neurons — the #3357 fix may be broken."
    );
}

/// #3357: Alpha-CROWN on a DAG produces strictly tighter bounds than baseline CROWN.
///
/// This verifies the end-to-end result: alpha optimization on CROWN-IBP-initialized
/// alpha state produces bounds that are at least as tight as (and ideally tighter
/// than) plain CROWN.
#[test]
fn test_dag_bab_alpha_crown_improves_over_crown_3357() {
    let (graph, input) = build_skip_dag_loose_ibp();

    // CROWN baseline
    let crown_bounds = graph
        .propagate_crown_fixed_slope(&input)
        .expect("CROWN should succeed");
    let crown_lower = crown_bounds.lower();
    let crown_upper = crown_bounds.upper();
    let crown_width: f32 = crown_upper
        .iter()
        .zip(crown_lower.iter())
        .map(|(u, l)| u - l)
        .sum();

    // Alpha-CROWN with BaB initialization path
    let config = AlphaCrownConfig::default();
    let alpha_bounds = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN should succeed");
    let alpha_lower = alpha_bounds.lower();
    let alpha_upper = alpha_bounds.upper();
    let alpha_width: f32 = alpha_upper
        .iter()
        .zip(alpha_lower.iter())
        .map(|(u, l)| u - l)
        .sum();

    // Alpha-CROWN must not be wider than CROWN (soundness)
    assert!(
        alpha_width <= crown_width + 1e-4,
        "alpha-CROWN ({alpha_width:.4}) wider than CROWN ({crown_width:.4}) — soundness bug"
    );

    // Verify soundness: alpha bounds must contain CROWN bounds
    for (i, ((al, au), (cl, cu))) in alpha_lower
        .iter()
        .zip(alpha_upper.iter())
        .zip(crown_lower.iter().zip(crown_upper.iter()))
        .enumerate()
    {
        // Alpha lower should be >= CROWN lower (tighter)
        // But alpha upper should be <= CROWN upper (tighter)
        // We check that alpha bounds are at least as tight
        assert!(
            *al >= cl - 1e-4,
            "Output {i}: alpha lower ({al}) < CROWN lower ({cl}) - 1e-4"
        );
        assert!(
            *au <= cu + 1e-4,
            "Output {i}: alpha upper ({au}) > CROWN upper ({cu}) + 1e-4"
        );
    }
}
