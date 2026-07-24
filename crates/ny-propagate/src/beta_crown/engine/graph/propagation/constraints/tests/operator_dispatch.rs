// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Operator dispatch regressions: sigmoid, max-binary, conv2d, conv1d,
//! conv-transpose1d (#1934, #1995, #2000).

use ndarray::{arr1, arr2};

use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer,
    ReLULayer,
};

use super::support::{assert_cache_bounds_close, scalar_interval};
use super::TOL;

use ny_test_utils::assert_bounded_tensor_close;
// =========================================================================
// Regression tests for #1934: wildcard identity fallback in constrained
// backward CROWN dispatch replaced with generic BoundPropagation dispatch.
// Same fix class as #1929 (graph_alpha/bounds.rs).
// =========================================================================

/// Build a graph with Sigmoid (nonlinear unary op, not explicitly handled in
/// the constrained backward dispatch table): linear1(1→1) → sigmoid → linear2(1→1).
///
/// Sigmoid was previously handled by the wildcard `_` arm, which passed bounds
/// through unchanged (identity) — unsound for a nonlinear layer. After #1934,
/// the generic dispatch routes through `BoundPropagation::propagate_crown_backward`,
/// which correctly calls `propagate_linear_with_bounds` for nonlinear layers.
pub(super) fn build_sigmoid_graph() -> GraphNetwork {
    use crate::SigmoidLayer;

    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "sigmoid1",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["sigmoid1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Regression test #1934: Sigmoid through constrained backward CROWN must produce
/// sound bounds (not the identity fallback which was unsound for nonlinear ops).
///
/// Without the fix, the wildcard arm treated Sigmoid as identity, so CROWN bounds
/// for y = sigmoid(x) would be the same as y = x — clearly wrong.
///
/// With the fix, generic dispatch uses sigmoid's CROWN relaxation, producing
/// correct linear bounds.
#[test]
fn test_constrained_backward_sigmoid_soundness_1934() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_sigmoid_graph();

    // Input x ∈ [-2, 2]. True output: sigmoid(x) ∈ [sigmoid(-2), sigmoid(2)] ≈ [0.119, 0.881].
    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn())
        .expect("valid input bounds");

    // No constraints — just exercise the backward dispatch through sigmoid.
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_ok(),
        "Sigmoid through constrained CROWN should succeed, got: {:?}",
        result.err()
    );

    let (output, _cache) = result.unwrap();
    let (lower, upper) = scalar_interval(&output);

    // Soundness: bounds must contain the true output range.
    // sigmoid(-2) ≈ 0.1192, sigmoid(2) ≈ 0.8808
    let true_lower = 1.0 / (1.0 + 2.0f32.exp()); // sigmoid(-2)
    let true_upper = 1.0 / (1.0 + (-2.0f32).exp()); // sigmoid(2)

    assert!(
        lower <= true_lower + TOL,
        "CROWN lower bound must be ≤ true min {}: got {}",
        true_lower,
        lower
    );
    assert!(
        upper >= true_upper - TOL,
        "CROWN upper bound must be ≥ true max {}: got {}",
        true_upper,
        upper
    );

    // The identity fallback would give bounds [-2, 2] (pass-through).
    // Correct sigmoid CROWN bounds should be strictly inside [-2, 2].
    assert!(
        lower > -2.0 + 0.01,
        "Sigmoid CROWN lower should be much tighter than identity fallback -2.0: got {}",
        lower
    );
    assert!(
        upper < 2.0 - 0.01,
        "Sigmoid CROWN upper should be much tighter than identity fallback 2.0: got {}",
        upper
    );
}

/// Regression test #1934: Sigmoid through constrained backward CROWN with
/// StoringIntermediates mode must produce same bounds as Standard mode.
#[test]
fn test_constrained_backward_sigmoid_parity_1934() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_sigmoid_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let (standard_output, standard_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("standard sigmoid should succeed");
    let (intermediate_output, intermediate_cache, _intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("intermediate sigmoid should succeed");

    assert_bounded_tensor_close(
        &standard_output,
        &intermediate_output,
        TOL,
        "sigmoid standard vs intermediate output parity",
    );
    assert_cache_bounds_close(
        &standard_cache,
        &intermediate_cache,
        "sigmoid standard vs intermediate cache parity",
    );
}

/// Build a graph with MaxBinary (multi-input op without explicit handler):
/// _input → linear1a(1→1) ↘
///                          max1 → linear2(1→1)
/// _input → linear1b(1→1) ↗
///
/// MaxBinary has no CROWN backward handler and is a multi-input op. After #1934,
/// this should return UnsupportedOp error instead of silently using identity.
fn build_max_binary_graph() -> GraphNetwork {
    use crate::MaxBinaryLayer;

    let linear1a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1a");
    let linear1b = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[1.0]))).expect("valid linear1b");
    let linear2 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1a", Layer::Linear(linear1a)));
    graph.add_node(GraphNode::from_input("linear1b", Layer::Linear(linear1b)));
    graph.add_node(GraphNode::binary(
        "max1",
        Layer::MaxBinary(MaxBinaryLayer),
        "linear1a",
        "linear1b",
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["max1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// #1934 (updated): a binary layer (MaxBinary) through constrained backward
/// CROWN must produce SOUND bounds via the n-ary dispatch (routing coefficients
/// to BOTH inputs) — NOT silently use identity on input[0] (the old unsound
/// path), and NOT error out (the old over-conservative behavior that pinned BaB
/// at depth 0 on residual/binary graphs).
///
/// Graph: linear1a = x, linear1b = x + 1, max1 = max(x, x+1) = x+1, out = x+1.
/// Over x ∈ [-1, 1] the true output is exactly x+1 ∈ [0, 2], so any SOUND bound
/// must satisfy lower ≤ 0 and upper ≥ 2.
#[test]
fn test_constrained_backward_multi_input_is_sound_1934() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_max_binary_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_ok(),
        "MaxBinary through constrained CROWN should now propagate soundly via the \
         n-ary dispatch, got Err: {:?}",
        result.err()
    );
    let (output, _cache) = result.unwrap();
    let lo = output.lower()[[0]];
    let hi = output.upper()[[0]];
    assert!(
        lo.is_finite() && hi.is_finite(),
        "bounds must be finite, got [{lo}, {hi}]"
    );
    // Soundness: the bound must contain the TRUE output range [0, 2]. A silent
    // identity-on-input[0] bug would compute max(x) = x ∈ [-1, 1] and miss the
    // +1 from input_b, so upper < 2 — this assertion catches that unsoundness.
    assert!(
        lo <= 0.0 + TOL,
        "unsound: lower bound {lo} > true min 0.0 (input_b contribution dropped?)"
    );
    assert!(
        hi >= 2.0 - TOL,
        "unsound: upper bound {hi} < true max 2.0 (input_b contribution dropped?)"
    );
}

// =========================================================================
// Regression test for #1995: Conv2d/ConvTranspose2d identity fallback removed
// =========================================================================

/// Build a graph where a Conv2d node receives 1D pre-activation bounds
/// (shape < 3D). Before #1995, the backward pass silently used identity
/// pass-through, skipping the convolution weight matrix — unsound.
/// After #1995, this returns UnsupportedOp error.
///
/// Graph: linear1(1→1) → conv2d(1×1×1×1 kernel) → output
///
/// The Conv2d receives 1D bounds from linear1 output (shape [1]).
/// Conv2d CROWN backward requires >= 3D input shape (H, W, C).
fn build_conv2d_1d_preact_graph() -> GraphNetwork {
    use crate::Conv2dLayer;
    use ndarray::Array4;

    // 1×1×1×1 kernel (out_channels=1, in_channels=1, H=1, W=1)
    let kernel = Array4::from_elem((1, 1, 1, 1), 2.0_f32).into_dyn();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid conv2d");

    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "conv2d",
        Layer::Conv2d(conv),
        vec!["linear1".to_string()],
    ));
    graph.set_output("conv2d");
    graph
}

/// Regression test #1995: Conv2d with 1D input must return error at some point
/// in the pipeline — not silently pass bounds through unchanged (identity fallback).
///
/// The forward pass (IBP) catches shape mismatches for Conv2d with < 3D input.
/// If forward somehow produced 1D bounds (e.g., from domain overrides in BaB),
/// the backward pass now also returns UnsupportedOp instead of identity (#1995).
/// This test verifies the end-to-end error behavior.
#[test]
fn test_conv2d_sub3d_preact_returns_error_not_identity_1995() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_conv2d_1d_preact_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);

    // Must be an error — either from forward (shape mismatch) or backward (UnsupportedOp).
    // Before #1995, the backward path would silently use identity if reached.
    assert!(
        result.is_err(),
        "Conv2d with 1D input shape should return error, not silently use identity."
    );
}

// =========================================================================
// Regression tests for #2000: Conv1d/ConvTranspose1d backward dispatch
// =========================================================================

/// Build a graph with Conv1d receiving proper 2D input:
/// _input → conv1d([2,1,3], stride=1, pad=0) → flatten → relu → linear(12→1)
///
/// Input shape: [1, 8] (in_channels=1, in_length=8)
/// Conv1d output: [2, 6], flatten → [1, 12], relu → [1, 12], linear → [1, 1]
///
/// Before #2000, Conv1d fell through to the generic backward dispatch which
/// called propagate_crown_backward without setting input_length, causing
/// "Conv1d CROWN requires input_length to be set" error.
fn build_conv1d_graph() -> GraphNetwork {
    use crate::{Conv1dLayer, FlattenLayer};
    use ndarray::ArrayD;

    // Kernel: [out_c=2, in_c=1, k=3]
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;
    let conv = Conv1dLayer::new(kernel, None, 1, 0).expect("valid conv1d");

    let linear_out =
        LinearLayer::new(arr2(&[[1.0; 12]]), Some(arr1(&[0.0]))).expect("valid linear_out");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1d", Layer::Conv1d(conv)));
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["conv1d".to_string()],
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

/// Regression test #2000: Conv1d through constrained backward CROWN must
/// succeed and produce sound bounds. Before #2000, Conv1d had no explicit
/// handler in the constrained backward dispatch — it fell through to the
/// generic path which failed because input_length was not set.
#[test]
fn test_constrained_backward_conv1d_dispatch_2000() {
    use ndarray::ArrayD;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_conv1d_graph();

    // Input: [1, 8] with perturbation around 0.5
    let lower = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.4_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.6_f32);
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_ok(),
        "Conv1d through constrained CROWN should succeed (not fail with input_length unset), got: {:?}",
        result.err()
    );

    let (output, _cache) = result.unwrap();
    let (lower_val, upper_val) = scalar_interval(&output);

    // Basic soundness: bounds must be finite and ordered
    assert!(
        lower_val.is_finite() && upper_val.is_finite(),
        "Conv1d CROWN bounds must be finite: [{}, {}]",
        lower_val,
        upper_val
    );
    assert!(
        lower_val <= upper_val + TOL,
        "Conv1d CROWN lower {} must be <= upper {}",
        lower_val,
        upper_val
    );
}

/// Build a graph with ConvTranspose1d:
/// _input → conv_transpose1d([1,2,3], stride=1, pad=0) → flatten → relu → linear(12→1)
///
/// Input shape: [1, 4] (in_channels=1, in_length=4)
/// ConvTranspose1d output: [2, 6], flatten → [1, 12], relu → [1, 12], linear → [1, 1]
fn build_conv_transpose1d_graph() -> GraphNetwork {
    use crate::{ConvTranspose1dLayer, FlattenLayer};
    use ndarray::ArrayD;

    // Kernel: [in_c=1, out_c=2, k=3]
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 0.5;
    kernel[[0, 0, 2]] = 0.25;
    kernel[[0, 1, 0]] = 0.5;
    kernel[[0, 1, 1]] = 1.0;
    kernel[[0, 1, 2]] = 0.5;
    let conv = ConvTranspose1dLayer::new(kernel, None, 1, 0).expect("valid conv_transpose1d");

    // ConvTranspose1d with in_c=1, out_c=2, k=3, stride=1, pad=0:
    // output_length = (input_length - 1) * stride + kernel_size - 2 * padding
    //              = (4 - 1) * 1 + 3 - 0 = 6
    // output_elements = 2 * 6 = 12
    let linear_out =
        LinearLayer::new(arr2(&[[1.0; 12]]), Some(arr1(&[0.0]))).expect("valid linear_out");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "conv_t1d",
        Layer::ConvTranspose1d(conv),
    ));
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["conv_t1d".to_string()],
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

/// Regression test #2000: ConvTranspose1d through constrained backward CROWN.
#[test]
fn test_constrained_backward_conv_transpose1d_dispatch_2000() {
    use ndarray::ArrayD;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_conv_transpose1d_graph();

    // Input: [1, 4] with perturbation around 0.5
    let lower = ArrayD::from_elem(ndarray::IxDyn(&[1, 4]), 0.4_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[1, 4]), 0.6_f32);
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_ok(),
        "ConvTranspose1d through constrained CROWN should succeed, got: {:?}",
        result.err()
    );

    let (output, _cache) = result.unwrap();
    let (lower_val, upper_val) = scalar_interval(&output);

    assert!(
        lower_val.is_finite() && upper_val.is_finite(),
        "ConvTranspose1d CROWN bounds must be finite: [{}, {}]",
        lower_val,
        upper_val
    );
    assert!(
        lower_val <= upper_val + TOL,
        "ConvTranspose1d CROWN lower {} must be <= upper {}",
        lower_val,
        upper_val
    );
}
