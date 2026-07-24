// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph zonotope softmax-cut diagnostics.

use ny_test_utils::assert_bounded_tensor_close;

use crate::*;
use ndarray::{arr1, arr2};

fn assert_bounds_within_ibp_and_tightened(
    actual: &BoundedTensor,
    ibp: &BoundedTensor,
    context: &str,
) {
    assert_eq!(actual.shape(), ibp.shape(), "{context} shape mismatch");
    let mut any_tightened = false;
    for (idx, ((&actual_lo, &actual_hi), (&ibp_lo, &ibp_hi))) in actual
        .lower()
        .iter()
        .zip(actual.upper().iter())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
        .enumerate()
    {
        assert!(
            actual_lo <= actual_hi + 1e-6,
            "{context} inverted at index {idx}: lower={actual_lo}, upper={actual_hi}"
        );
        assert!(
            actual_lo >= ibp_lo - 1e-6,
            "{context} lower widened: actual={actual_lo}, ibp={ibp_lo}"
        );
        assert!(
            actual_hi <= ibp_hi + 1e-6,
            "{context} upper widened: actual={actual_hi}, ibp={ibp_hi}"
        );
        any_tightened |= actual_lo > ibp_lo + 1e-6 || actual_hi < ibp_hi - 1e-6;
    }
    assert!(
        any_tightened,
        "{context} must remain locally re-zonotized and tighten at least one endpoint relative to IBP"
    );
}

/// #318 Packet B: the explicit Softmax interval cut must be operator-local.
///
/// Nodes before Softmax should stay on the ordinary zonotope path, while the
/// Softmax node itself should match the existing IBP fallback behavior exactly.
/// Downstream affine consumers may tighten again once the fallback interval is
/// re-embedded as a zonotope, but they must not widen beyond the IBP result.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_zonotope_softmax_interval_cut_is_operator_local_318() {
    let dim = 3_usize;
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "scores",
        Layer::Linear(LinearLayer::new(ndarray::Array2::<f32>::eye(dim), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.5_f32, -0.3, 0.2], [0.1, 0.4, -0.6]]),
                Some(arr1(&[0.05_f32, -0.1])),
            )
            .unwrap(),
        ),
        vec!["probs".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr2(&[[0.0_f32, 0.5, -0.5], [1.0, -1.0, 0.25]]).into_dyn(),
        arr2(&[[0.2_f32, 0.7, -0.3], [1.2, -0.8, 0.45]]).into_dyn(),
    )
    .unwrap();
    let options =
        ZonotopePropagationOptions::new().with_softmax_mode(ZonotopeSoftmaxMode::IntervalFallback);

    let mut scores_graph = graph.clone();
    scores_graph.set_output("scores");
    let scores_default = scores_graph.propagate_zonotope(&input, 0.1).unwrap();
    let scores_cut = scores_graph
        .propagate_zonotope_with_options(&input, 0.1, options)
        .unwrap();
    assert_bounded_tensor_close(&scores_cut, &scores_default, 1e-6, "pre-softmax zonotope");

    let mut probs_graph = graph.clone();
    probs_graph.set_output("probs");
    let probs_cut = probs_graph
        .propagate_zonotope_with_options(&input, 0.1, options)
        .unwrap();
    let probs_ibp = probs_graph.propagate_ibp(&input).unwrap();
    assert_bounded_tensor_close(&probs_cut, &probs_ibp, 1e-6, "softmax interval cut");

    let out_cut = graph
        .propagate_zonotope_with_options(&input, 0.1, options)
        .unwrap();
    let out_ibp = graph.propagate_ibp(&input).unwrap();
    assert_bounds_within_ibp_and_tightened(&out_cut, &out_ibp, "downstream softmax interval cut");
}
