// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline-threading regressions (#3816).

use ndarray::{arr1, arr2};

use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer,
    ReLULayer,
};
// =========================================================================
// Regression tests for #3816: constrained BaB CROWN deadline threading
// =========================================================================

/// Build a minimal Conv2d graph with proper 3D input shape:
/// _input → conv2d([1,1,3,3], stride=1, pad=0) → flatten → relu → linear(1→1)
///
/// Input shape: [1, 4, 4] (in_channels=1, H=4, W=4)
/// Conv2d output: [1, 2, 2] → flatten → [4] → relu → [4] → linear → [1]
///
/// Conv2d is the layer that checks the deadline via
/// `propagate_linear_with_engine_and_deadline`. Without deadline threading
/// from constrained backward, an expired deadline is silently ignored.
fn build_conv2d_deadline_graph() -> GraphNetwork {
    use crate::{Conv2dLayer, FlattenLayer};
    use ndarray::Array4;

    // 1×1×3×3 kernel (out_channels=1, in_channels=1, kH=3, kW=3)
    let kernel = Array4::ones((1, 1, 3, 3)).into_dyn();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid conv2d");

    let linear_out =
        LinearLayer::new(arr2(&[[1.0; 4]]), Some(arr1(&[0.0]))).expect("valid linear_out");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv2d", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["conv2d".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["flatten".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

/// Regression test #3816: constrained backward CROWN (standard mode) must
/// respect the BaB deadline.
///
/// Before this fix, `propagate_crown_with_graph_constraints` constructed
/// `BackwardParams { deadline: None }`, so an already-expired per-domain
/// budget did not stop constrained Conv2d backward.
///
/// Source: Prover's reproduction on committed f1dc8ee18, constraints/mod.rs:89.
#[test]
fn test_constrained_backward_expired_deadline_returns_error_3816() {
    use std::time::Duration;

    let config = BetaCrownConfig {
        timeout: Duration::from_millis(1),
        ..BetaCrownConfig::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Sleep past the deadline
    std::thread::sleep(Duration::from_millis(10));

    let graph = build_conv2d_deadline_graph();
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_err(),
        "Constrained backward CROWN should return DeadlineExceeded after expired timeout"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("deadline") || err_msg.contains("Deadline"),
        "Error should mention deadline, got: {err_msg}"
    );
}

/// Regression test #3816: constrained backward CROWN (storing intermediates
/// mode) must also respect the BaB deadline.
///
/// Before this fix, `propagate_crown_with_graph_constraints_storing_intermediates`
/// also constructed `BackwardParams { deadline: None }`.
///
/// Source: Prover's reproduction on committed f1dc8ee18, constraints/mod.rs:433.
#[test]
fn test_constrained_backward_storing_intermediates_expired_deadline_returns_error_3816() {
    use std::time::Duration;

    let config = BetaCrownConfig {
        timeout: Duration::from_millis(1),
        ..BetaCrownConfig::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Sleep past the deadline
    std::thread::sleep(Duration::from_millis(10));

    let graph = build_conv2d_deadline_graph();
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result = verifier.propagate_crown_with_graph_constraints_storing_intermediates(
        &graph, &input, &context, None, None,
    );
    assert!(
        result.is_err(),
        "Constrained backward CROWN (storing intermediates) should return DeadlineExceeded after expired timeout"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("deadline") || err_msg.contains("Deadline"),
        "Error should mention deadline, got: {err_msg}"
    );
}
