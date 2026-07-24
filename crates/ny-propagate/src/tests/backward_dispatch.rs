// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `network::backward_dispatch::dispatch_backward_layer`.
//!
//! Tests all match arms: Linear, Add, Sub, SkipMerge, OpaqueSkip, Conv1d
//! dimension error, Conv2d dimension error, MatMul/BilinearCrown input
//! count error, MulBinary → Binary (#3439), ReLU/Where → Unsupported,
//! Flatten via catch-all.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::layers::{
    AddLayer, Conv1dLayer, Conv2dLayer, FlattenLayer, Layer, LinearLayer, MulBinaryLayer,
    OpaqueSkipLayer, ReLULayer, SkipMergeLayer, SubLayer, WhereLayer,
};
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::MulBinaryRelaxationMode;

/// Helper: create identity LinearBounds of given dimension.
fn identity_lb(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

/// Helper: create a simple BoundedTensor of shape [dim].
fn simple_bounds(dim: usize) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[dim]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[dim]), 1.0_f32);
    BoundedTensor::new(lower, upper).unwrap()
}

/// Helper: build a DispatchContext for a node.
fn make_ctx<'a>(
    layer: &'a Layer,
    pre_act: &'a BoundedTensor,
    net_input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    inputs: &'a [String],
) -> DispatchContext<'a> {
    DispatchContext {
        node_name: "test_node",
        layer,
        inputs,
        pre_activation: pre_act,
        network_input: net_input,
        node_bounds: node_bounds.into(),
        engine: None,
        deadline: None,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    }
}

// ===================================================================
// Linear layer dispatch
// ===================================================================

#[test]
fn dispatch_linear_returns_single() {
    let weight = Array2::eye(3);
    let bias = Array1::zeros(3);
    let layer = Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap());

    let bounds = simple_bounds(3);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(3);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "Linear should return Single"
    );
}

// ===================================================================
// Add layer dispatch
// ===================================================================

#[test]
fn dispatch_add_returns_binary() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::Binary { .. }),
        "Add should return Binary"
    );
}

// ===================================================================
// Sub layer dispatch
// ===================================================================

#[test]
fn dispatch_sub_returns_binary() {
    let layer = Layer::Sub(SubLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::Binary { .. }),
        "Sub should return Binary"
    );
}

// ===================================================================
// SkipMerge dispatch
// ===================================================================

#[test]
fn dispatch_skip_merge_single_input_returns_passthrough() {
    let layer = Layer::SkipMerge(SkipMergeLayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::PassThrough),
        "SkipMerge with 1 input should return PassThrough"
    );
}

#[test]
fn dispatch_skip_merge_multiple_inputs_returns_error() {
    let layer = Layer::SkipMerge(SkipMergeLayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "SkipMerge with 2 inputs should error");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("expects exactly 1 input"),
        "Expected 'expects exactly 1 input', got: {err_msg}"
    );
}

// ===================================================================
// OpaqueSkip dispatch
// ===================================================================

#[test]
fn dispatch_opaque_skip_single_input_returns_single() {
    let layer = Layer::OpaqueSkip(OpaqueSkipLayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "OpaqueSkip with 1 input should return Single"
    );
}

#[test]
fn dispatch_opaque_skip_multi_input_returns_nary() {
    let layer = Layer::OpaqueSkip(OpaqueSkipLayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Nary {
            bounds: entries, ..
        } => {
            assert_eq!(entries.len(), 3, "Expected 3 entries for 3 inputs");
            assert!(
                entries.iter().all(|e| e.is_some()),
                "All entries should be Some"
            );
        }
        _ => panic!("Expected Nary for OpaqueSkip with 3 inputs"),
    }
}

#[test]
fn dispatch_opaque_skip_no_inputs_returns_error() {
    let layer = Layer::OpaqueSkip(OpaqueSkipLayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs: Vec<String> = vec![];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "OpaqueSkip with no inputs should error");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("has no inputs"),
        "Expected 'has no inputs', got: {err_msg}"
    );
}

// ===================================================================
// ReLU, Where return Unsupported; MulBinary returns Binary (#3439)
// ===================================================================

#[test]
fn dispatch_relu_returns_unsupported() {
    let layer = Layer::ReLU(ReLULayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Unsupported(msg) => {
            assert!(msg.contains("ReLU"), "Expected ReLU in message, got: {msg}");
        }
        _ => panic!("Expected Unsupported for ReLU"),
    }
}

/// MulBinary dispatches via McCormick CROWN backward (#3439).
#[test]
fn dispatch_mulbinary_returns_binary_3439() {
    let layer = Layer::MulBinary(MulBinaryLayer);
    let bounds = simple_bounds(2);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("a".to_string(), simple_bounds(2));
    node_bounds.insert("b".to_string(), simple_bounds(2));
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            // #2617: A-matrix bounds must have zero bias
            assert!(bounds_a.lower_b.iter().all(|&v| v == 0.0));
            assert!(bounds_a.upper_b.iter().all(|&v| v == 0.0));
            assert!(bounds_b.lower_b.iter().all(|&v| v == 0.0));
            assert!(bounds_b.upper_b.iter().all(|&v| v == 0.0));
            assert_eq!(bias_lower.len(), 2);
            assert_eq!(bias_upper.len(), 2);
        }
        other => panic!("Expected Binary for MulBinary, got {other:?}"),
    }
}

#[test]
fn dispatch_where_returns_unsupported() {
    let layer = Layer::Where(WhereLayer::new());
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["cond".to_string(), "x".to_string(), "y".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::Unsupported(_)),
        "Where should return Unsupported"
    );
}

// ===================================================================
// Conv1d with <2D input returns error
// ===================================================================

#[test]
fn dispatch_conv1d_1d_input_returns_error() {
    let weight = ArrayD::from_elem(IxDyn(&[4, 3, 3]), 0.0_f32); // out_ch, in_ch, kernel_size
    let conv = Conv1dLayer::new(weight, None, 1, 0).unwrap();
    let layer = Layer::Conv1d(conv);

    // 1D pre-activation bounds — below the 2D requirement
    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[3]), 1.0_f32),
    )
    .unwrap();
    let net_input = simple_bounds(3);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = LinearBounds::new(
        Array2::eye(4),
        Array1::zeros(4),
        Array2::eye(4),
        Array1::zeros(4),
    )
    .unwrap();

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "Conv1d with 1D input should error");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("Conv1d") && err_msg.contains(">= 2D"),
        "Expected Conv1d dimension error, got: {err_msg}"
    );
}

// ===================================================================
// Conv2d with <3D input returns error
// ===================================================================

#[test]
fn dispatch_conv2d_2d_input_returns_error() {
    let weight = ArrayD::from_elem(IxDyn(&[4, 3, 3, 3]), 0.0_f32); // out_ch, in_ch, kH, kW
    let conv = Conv2dLayer::new(weight, None, (1, 1), (0, 0)).unwrap();
    let layer = Layer::Conv2d(conv);

    // 2D pre-activation — below the 3D requirement for Conv2d
    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[3, 3]), 1.0_f32),
    )
    .unwrap();
    let net_input = simple_bounds(3);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = LinearBounds::new(
        Array2::eye(4),
        Array1::zeros(4),
        Array2::eye(4),
        Array1::zeros(4),
    )
    .unwrap();

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "Conv2d with 2D input should error");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("Conv2d") && err_msg.contains(">= 3D"),
        "Expected Conv2d dimension error, got: {err_msg}"
    );
}

// ===================================================================
// MatMul with <2 inputs returns error
// ===================================================================

#[test]
fn dispatch_matmul_insufficient_inputs_returns_error() {
    use crate::layers::MatMulLayer;

    let layer = Layer::MatMul(MatMulLayer::new(false, None));
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()]; // only 1 input, need 2
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "MatMul with 1 input should error");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("MatMul"),
        "Expected MatMul in error, got: {err_msg}"
    );
}

// ===================================================================
// BilinearCrown with <2 inputs returns error
// ===================================================================

#[test]
fn dispatch_bilinear_insufficient_inputs_returns_error() {
    use crate::layers::BilinearCrownLayer;

    let layer = Layer::BilinearCrown(BilinearCrownLayer::new(false, None));
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()]; // only 1 input, need 2
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "BilinearCrown with 1 input should error");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("BilinearCrown"),
        "Expected BilinearCrown in error, got: {err_msg}"
    );
}

// ===================================================================
// Flatten (trait dispatch via catch-all) returns Single
// ===================================================================

#[test]
fn dispatch_flatten_via_trait_returns_single() {
    let layer = Layer::Flatten(FlattenLayer::new(1));
    let bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 3]), 1.0_f32),
    )
    .unwrap();
    let net_input = simple_bounds(6);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &bounds, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(6);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    assert!(
        matches!(result, BackwardDispatchResult::Single(_)),
        "Flatten should return Single via trait dispatch"
    );
}
