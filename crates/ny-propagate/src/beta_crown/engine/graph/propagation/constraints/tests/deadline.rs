// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline-threading regressions (#3816).

use ndarray::{arr1, arr2};
use ny_core::NyError;

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

/// Expired authority must be rejected at wrapper entry, before malformed graph
/// preparation can replace the timeout with an unrelated validation error.
#[test]
fn test_constrained_wrappers_preflight_deadline_before_preparation() {
    use std::time::{Duration, Instant};

    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    verifier.config.alpha_config.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before the current instant"),
    );

    // Deliberately has no output node. Reaching constrained forward preparation
    // would therefore return InvalidSpec instead of the authority timeout.
    let graph = GraphNetwork::new();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);
    let spec_matrix = arr2(&[[1.0]]);

    let standard =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(matches!(standard, Err(NyError::DeadlineExceeded(_))));

    let storing = verifier.propagate_crown_with_graph_constraints_storing_intermediates(
        &graph, &input, &context, None, None,
    );
    assert!(matches!(storing, Err(NyError::DeadlineExceeded(_))));

    let spec = verifier.propagate_crown_with_graph_constraints_with_spec_matrix(
        &graph,
        &input,
        &context,
        None,
        &spec_matrix,
        None,
        false,
    );
    assert!(matches!(spec, Err(NyError::DeadlineExceeded(_))));

    let spec_storing = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates_with_spec_matrix(
            &graph,
            &input,
            &context,
            None,
            &spec_matrix,
        );
    assert!(matches!(spec_storing, Err(NyError::DeadlineExceeded(_))));
}

#[test]
fn test_constrained_preflight_none_preserves_unbounded_path() {
    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    verifier.config.alpha_config.deadline = None;

    let graph = build_conv2d_deadline_graph();
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_ok(),
        "deadline=None must preserve constrained propagation: {result:?}"
    );
}

#[test]
fn constrained_forward_inner_captures_effective_deadline() {
    use std::time::{Duration, Instant};

    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    verifier.config.alpha_config.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before the current instant"),
    );
    let graph = build_conv2d_deadline_graph();
    let input = BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 1.0_f32),
    )
    .expect("valid input bounds");
    let history = GraphSplitHistory::new();

    let error = verifier
        .compute_constrained_forward_bounds_inner(&graph, &input, &history, None, None, false)
        .expect_err("the inner constrained forward must retain deadline authority");
    assert!(
        matches!(error, NyError::DeadlineExceeded(_)),
        "unexpected error: {error}"
    );
}

#[test]
fn constrained_forward_conv2d_intersection_keeps_certified_cancellation_widening() {
    let kernel = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 1, 1, 3]),
        vec![16_777_216.0_f32, 1.0, -16_777_216.0],
    )
    .expect("kernel");
    let conv =
        crate::Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 1, 3).expect("conv");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");
    let input = BoundedTensor::concrete(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 3]), vec![1.0_f32; 3])
            .expect("input"),
    )
    .expect("concrete input");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let history = GraphSplitHistory::new();

    let (bounds, _) = verifier
        .compute_constrained_forward_bounds_inner(&graph, &input, &history, None, None, false)
        .expect("constrained forward");
    let conv_bounds = bounds.get("conv").expect("conv bounds");
    assert!(
        conv_bounds.lower()[[0, 0, 0]] <= 1.0 && conv_bounds.upper()[[0, 0, 0]] >= 1.0,
        "the constrained intersection must retain the exact real sum 1.0, got [{}, {}]",
        conv_bounds.lower()[[0, 0, 0]],
        conv_bounds.upper()[[0, 0, 0]],
    );
}
