// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Two-ReLU-layer regressions (#1926).

use ndarray::{arr1, arr2};

use crate::beta_crown::{GraphCrownContext, GraphNeuronConstraint, GraphSplitHistory};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer,
};

use super::support::scalar_interval;
use super::two_neuron::{build_two_neuron_input_bounds, build_two_neuron_relu_graph};
use super::TOL;

use ny_test_utils::assert_bounded_tensor_close;
// ====================================================================
// Two-layer ReLU graph: tests backward through multiple ReLU+linear stages.
// ====================================================================

/// Build a deeper graph: linear1(2→2) → relu1 → linear2(2→2) → relu2 → linear3(2→1)
///
/// All linears are identity (or sum at output). Tests backward traversal through
/// multiple ReLU layers with constraints at different depths.
fn build_two_relu_layer_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[0.0, 0.0])))
        .expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[0.0, 0.0])))
        .expect("valid linear2");
    let linear3 = LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).expect("valid linear3");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
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
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");
    graph
}

#[test]
fn test_two_relu_layers_constraints_at_different_depths() {
    // Constrain relu1[0] inactive and relu2[1] active.
    // This tests backward.rs traversing two ReLU layers with constraints at each.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_relu_layer_graph();
    let input = build_two_neuron_input_bounds();

    let history = GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: true,
            score: 0.0,
        });

    let unconstrained_history = GraphSplitHistory::new();
    let unconstrained_ctx = GraphCrownContext::for_history(&unconstrained_history);
    let constrained_ctx = GraphCrownContext::for_history(&history);

    let (unconstrained_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &unconstrained_ctx, None, None)
        .expect("unconstrained should succeed");
    let (constrained_output, cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &constrained_ctx, None, None)
        .expect("multi-layer constrained should succeed");

    let (u_lower, u_upper) = scalar_interval(&unconstrained_output);
    let (c_lower, c_upper) = scalar_interval(&constrained_output);

    // Constrained should be at least as tight.
    assert!(
        c_lower >= u_lower - TOL,
        "multi-layer constrained lower must not be looser: unconstrained={}, constrained={}",
        u_lower,
        c_lower
    );
    assert!(
        c_upper <= u_upper + TOL,
        "multi-layer constrained upper must not be looser: unconstrained={}, constrained={}",
        u_upper,
        c_upper
    );

    // Verify constraint effects in cache:
    // relu1[0] inactive → must be 0.
    let relu1 = cache.get("relu1").expect("relu1 must be in cache");
    let relu1_flat = relu1.flatten();
    assert!(
        relu1_flat.lower()[[0]].abs() < TOL && relu1_flat.upper()[[0]].abs() < TOL,
        "relu1[0] should be dead: [{}, {}]",
        relu1_flat.lower()[[0]],
        relu1_flat.upper()[[0]]
    );
    // relu2[1] active → lower must be ≥ 0.
    let relu2 = cache.get("relu2").expect("relu2 must be in cache");
    let relu2_flat = relu2.flatten();
    assert!(
        relu2_flat.lower()[[1]] >= -TOL,
        "relu2[1] active: lower should be ≥ 0, got {}",
        relu2_flat.lower()[[1]]
    );
}

#[test]
fn test_two_relu_layers_storing_intermediates_captures_both_relus() {
    // Constrain both relu1[0] and relu2[1]. The StoringIntermediates mode should
    // capture A matrices and pre-ReLU bounds at both constrained ReLU layers.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_relu_layer_graph();
    let input = build_two_neuron_input_bounds();

    let history = GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: true,
            score: 0.0,
        });

    let context = GraphCrownContext::for_history(&history);

    let (standard_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("standard should succeed");
    let (intermediate_output, _, intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("intermediate should succeed");

    // Output parity.
    assert_bounded_tensor_close(
        &standard_output,
        &intermediate_output,
        TOL,
        "two-layer output parity",
    );

    // Both constrained ReLUs should have captured A matrices.
    assert!(
        intermediate.a_at_relu.contains_key("relu1"),
        "intermediate must capture A at relu1"
    );
    assert!(
        intermediate.a_at_relu.contains_key("relu2"),
        "intermediate must capture A at relu2"
    );

    // Both constrained ReLUs should have captured pre-ReLU bounds.
    assert!(
        intermediate.pre_relu_bounds.contains_key("relu1"),
        "intermediate must capture pre-ReLU bounds at relu1"
    );
    assert!(
        intermediate.pre_relu_bounds.contains_key("relu2"),
        "intermediate must capture pre-ReLU bounds at relu2"
    );

    // Pre-ReLU bounds dimensions should match neuron count (2).
    for name in ["relu1", "relu2"] {
        let (lo, hi) = &intermediate.pre_relu_bounds[name];
        assert_eq!(lo.len(), 2, "{name} pre-lower should have 2 neurons");
        assert_eq!(hi.len(), 2, "{name} pre-upper should have 2 neurons");
    }
}

/// Shared setup for #1926: constrain both neurons so backward.rs enters
/// the pre-ReLU storage path.
fn build_both_constrained_history_1926() -> GraphSplitHistory {
    GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 1,
            is_active: true,
            score: 0.0,
        })
}

/// Regression test for #1926: pre-ReLU bounds stored during intermediates mode
/// must not be silent zeros from a failed `into_dimensionality::<Ix1>()`.
#[test]
fn test_pre_relu_bounds_not_silent_zeros_1926() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();
    let history = build_both_constrained_history_1926();
    let context = GraphCrownContext::for_history(&history);

    let (_output, _cache, intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("storing intermediates should succeed");

    let (pre_lower, pre_upper) = intermediate
        .pre_relu_bounds
        .get("relu1")
        .expect("relu1 pre-ReLU bounds must be stored");

    let any_nonzero_lower = pre_lower.iter().any(|&v: &f32| v.abs() > 1e-7);
    let any_nonzero_upper = pre_upper.iter().any(|&v: &f32| v.abs() > 1e-7);
    assert!(
        any_nonzero_lower || any_nonzero_upper,
        "pre-ReLU bounds must not be all zeros (silent fallback regression #1926): \
         lower={:?}, upper={:?}",
        pre_lower.as_slice().unwrap(),
        pre_upper.as_slice().unwrap()
    );
}

/// Regression test for #1926: stored pre-ReLU bounds must match forward cache.
#[test]
fn test_pre_relu_bounds_match_forward_cache_1926() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();
    let history = build_both_constrained_history_1926();
    let context = GraphCrownContext::for_history(&history);

    let (forward_cache, _) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("forward bounds should succeed");
    let (_output, _cache, intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("storing intermediates should succeed");

    let (pre_lower, pre_upper) = intermediate
        .pre_relu_bounds
        .get("relu1")
        .expect("relu1 pre-ReLU bounds must be stored");
    let fwd = forward_cache.get("linear1").expect("linear1 in cache");
    let fwd_flat = fwd.flatten();
    let fwd_lo: Vec<f32> = fwd_flat.lower().iter().copied().collect();
    let fwd_hi: Vec<f32> = fwd_flat.upper().iter().copied().collect();

    assert_eq!(pre_lower.len(), fwd_lo.len(), "dimension mismatch");
    for i in 0..pre_lower.len() {
        assert!(
            (pre_lower[i] - fwd_lo[i]).abs() <= TOL,
            "pre-ReLU lower[{i}]: stored={}, forward={}",
            pre_lower[i],
            fwd_lo[i]
        );
        assert!(
            (pre_upper[i] - fwd_hi[i]).abs() <= TOL,
            "pre-ReLU upper[{i}]: stored={}, forward={}",
            pre_upper[i],
            fwd_hi[i]
        );
    }
}

#[test]
fn test_beta_on_two_relu_layers_affects_bounds() {
    // Apply non-zero beta to constraints at both ReLU layers.
    // Verify bounds change relative to zero-beta and remain valid.
    use crate::beta_crown::state::{GraphBetaEntry, GraphBetaState};

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_relu_layer_graph();
    let input = build_two_neuron_input_bounds();

    let history = GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: true,
            score: 0.0,
        });
    let context = GraphCrownContext::for_history(&history);

    // Zero-beta baseline
    let (zero_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("zero-beta should succeed");

    // Non-zero beta
    let beta_state = GraphBetaState::from_entries(vec![
        GraphBetaEntry {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            split_point: 0.0,
            value: 0.3,
            sign: -1.0,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        },
        GraphBetaEntry {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            split_point: 0.0,
            value: 0.2,
            sign: 1.0,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        },
    ]);

    let (beta_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, Some(&beta_state), None)
        .expect("nonzero-beta should succeed");

    let (z_lower, z_upper) = scalar_interval(&zero_output);
    let (b_lower, b_upper) = scalar_interval(&beta_output);

    // Bounds must differ with non-zero beta.
    let changed = (b_lower - z_lower).abs() > 1e-6 || (b_upper - z_upper).abs() > 1e-6;
    assert!(
        changed,
        "non-zero β on two layers must change bounds: zero=({}, {}), beta=({}, {})",
        z_lower, z_upper, b_lower, b_upper
    );

    // Bounds must remain valid.
    assert!(
        b_lower <= b_upper + TOL,
        "beta bounds invalid: lower={} > upper={}",
        b_lower,
        b_upper
    );
}
