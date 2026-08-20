// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for backward CROWN dispatch.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use super::dispatch::{dispatch_backward_layer, dispatch_backward_layer_finite_boundary};
use super::helpers::resolve_input_bounds;
use super::types::{BackwardDispatchResult, DispatchContext};
use crate::bounds::LinearBounds;
use crate::layers::{
    AddLayer, BoundPropagation, ConcatLayer, Conv1dLayer, Conv2dLayer, ConvTranspose2dLayer,
    FlattenLayer, Layer, LayerNormCrownMode, LayerNormLayer, LinearLayer, MulBinaryLayer,
    OpaqueSkipLayer, PadLayer, PadMode, ReLULayer, SkipMergeLayer, SubLayer, WhereLayer,
};
use crate::MulBinaryRelaxationMode;
use ny_tensor::BoundedTensor;

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

fn sample_conv_transpose2d() -> ConvTranspose2dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -0.5, 0.25, 2.0])
        .expect("valid kernel shape");
    let bias = Array1::from_vec(vec![0.75_f32]);
    ConvTranspose2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).expect("valid conv transpose")
}

fn sample_conv_transpose2d_pre_act() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0_f32, 0.0, 0.5, 1.0])
            .expect("valid lower input"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.5_f32, 2.0, 2.5, 3.0])
            .expect("valid upper input"),
    )
    .expect("valid bounded tensor")
}

fn sample_conv_transpose2d_node_lb() -> LinearBounds {
    LinearBounds::new(
        Array2::from_shape_vec(
            (2, 9),
            vec![
                1.0, -0.5, 0.25, 0.0, 1.5, -1.0, 0.75, 0.2, -0.3, -0.1, 0.4, 1.25, -0.75, 0.5, 0.6,
                -1.2, 0.8, 0.9,
            ],
        )
        .expect("valid lower_a"),
        Array1::from_vec(vec![0.1_f32, -0.3]),
        Array2::from_shape_vec(
            (2, 9),
            vec![
                0.9, 0.1, -0.25, 0.7, -1.5, 0.4, 0.0, 1.1, -0.8, 1.2, -0.6, 0.3, 0.5, 0.9, -1.4,
                0.2, 0.75, 0.6,
            ],
        )
        .expect("valid upper_a"),
        Array1::from_vec(vec![0.4_f32, 0.2]),
    )
    .expect("valid linear bounds")
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
        "Linear dispatch should return Single, got {result:?}"
    );
}

// EXPIRY, NOT PRESENCE. These three pinned the strict finite boundary closing
// whenever a deadline was merely PRESENT. That is the defect that was shipped
// for a week: every scored run carries a deadline, so the boundary closed on
// every row, the carrier densified, and the walk paid the dense-path bill. The
// boundary is now decided by EXPIRY, so a LIVE deadline proceeds and an EXPIRED
// one still closes.
//
// Each test now asserts BOTH arms. The invariant they were written to protect —
// that the refusal is atomic and touches nothing before declining — is retained
// and still checked on the arm where the refusal actually happens.
#[test]
fn strict_finite_boundary_linear_closes_only_once_the_deadline_expires() {
    let weight = Array2::eye(3);
    let bias = Array1::zeros(3);
    let layer = Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap());
    let bounds = simple_bounds(3);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let mut ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(3);

    // Live: the cooperative route runs.
    ctx.deadline = Some(Instant::now() + Duration::from_secs(30));
    let live = dispatch_backward_layer_finite_boundary(&ctx, &lb).unwrap();
    assert!(
        matches!(live, BackwardDispatchResult::Single(_)),
        "a live deadline must not close the boundary, got {live:?}"
    );

    // Expired: it still closes, typed.
    ctx.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now"),
    );
    let expired = dispatch_backward_layer_finite_boundary(&ctx, &lb);
    assert!(
        expired.is_err() || matches!(expired, Ok(BackwardDispatchResult::Unsupported(_))),
        "an expired deadline must still close the boundary"
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
        "Add dispatch should return Binary, got {result:?}"
    );
}

#[test]
fn expired_strict_finite_boundary_declines_before_split_and_leaves_source_atomic() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let mut ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    ctx.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now"),
    );
    let lb = LinearBounds {
        lower_a: Array2::eye(2),
        lower_b: Array1::from_vec(vec![1.0, 2.0]),
        upper_a: Array2::eye(2),
        upper_b: Array1::from_vec(vec![3.0, 4.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let before = lb.clone();

    let error = dispatch_backward_layer_finite_boundary(&ctx, &lb).unwrap_err();
    assert!(matches!(error, ny_core::NyError::DeadlineExceeded(_)));
    assert_eq!(lb.lower_a(), before.lower_a());
    assert_eq!(lb.upper_a(), before.upper_a());
    assert_eq!(lb.lower_b(), before.lower_b());
    assert_eq!(lb.upper_b(), before.upper_b());
}

#[test]
fn ordinary_finite_dense_add_preserves_historical_split() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let mut ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    ctx.deadline = Some(Instant::now() + Duration::from_secs(30));
    let lb = LinearBounds {
        lower_a: Array2::eye(2),
        lower_b: Array1::from_vec(vec![1.0, 2.0]),
        upper_a: Array2::eye(2),
        upper_b: Array1::from_vec(vec![3.0, 4.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let before = lb.clone();

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            assert_eq!(bounds_a.lower_a(), &Array2::<f32>::eye(2));
            assert_eq!(bounds_a.upper_a(), &Array2::<f32>::eye(2));
            assert_eq!(bounds_b.lower_a(), &Array2::<f32>::eye(2));
            assert_eq!(bounds_b.upper_a(), &Array2::<f32>::eye(2));
            assert_eq!(bias_lower, Array1::from_vec(vec![1.0, 2.0]));
            assert_eq!(bias_upper, Array1::from_vec(vec![3.0, 4.0]));
        }
        other => panic!("ordinary finite Dense Add should split, got {other:?}"),
    }
    assert_eq!(lb.lower_a(), before.lower_a());
    assert_eq!(lb.upper_a(), before.upper_a());
    assert_eq!(lb.lower_b(), before.lower_b());
    assert_eq!(lb.upper_b(), before.upper_b());
}

#[test]
fn strict_finite_boundary_closes_the_unpollable_split_only_once_expired() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let mut ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    ctx.deadline = Some(Instant::now() + Duration::from_secs(30));
    let lb = LinearBounds {
        lower_a: Array2::eye(2),
        lower_b: Array1::from_vec(vec![1.0, 2.0]),
        upper_a: Array2::eye(2),
        upper_b: Array1::from_vec(vec![3.0, 4.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let before = lb.clone();

    // Live: the split proceeds.
    let live = dispatch_backward_layer_finite_boundary(&ctx, &lb).unwrap();
    assert!(
        matches!(live, BackwardDispatchResult::Binary { .. }),
        "a live deadline must not close the unpollable split, got {live:?}"
    );

    // Expired: closed, and ATOMIC — the carrier is untouched by the refusal.
    ctx.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now"),
    );
    let expired = dispatch_backward_layer_finite_boundary(&ctx, &lb);
    assert!(
        expired.is_err() || matches!(expired, Ok(BackwardDispatchResult::Unsupported(_))),
        "an expired deadline must still close the unpollable split"
    );
    assert_eq!(lb.lower_a(), before.lower_a());
    assert_eq!(lb.upper_a(), before.upper_a());
    assert_eq!(lb.lower_b(), before.lower_b());
    assert_eq!(lb.upper_b(), before.upper_b());
}

#[test]
fn strict_finite_boundary_declines_coeff_err_duplication_only_once_expired() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let mut ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    ctx.deadline = Some(Instant::now() + Duration::from_secs(30));
    let mut lb = identity_lb(2);
    lb.lower_a_err = Some(Array2::from_elem((2, 2), 0.25));
    lb.upper_a_err = Some(Array2::from_elem((2, 2), 0.5));
    let before = lb.clone();

    // Live: the carrier's certified error is composed rather than declined.
    let live = dispatch_backward_layer_finite_boundary(&ctx, &lb).unwrap();
    assert!(
        !matches!(live, BackwardDispatchResult::Unsupported(_)),
        "a live deadline must not decline the certified-error discharge"
    );

    // Expired: the typed refusal returns, and names the discharge it declined.
    ctx.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now"),
    );
    match dispatch_backward_layer_finite_boundary(&ctx, &lb) {
        Err(error) => assert!(error.is_deadline_exceeded(), "unexpected error: {error}"),
        Ok(BackwardDispatchResult::Unsupported(message)) => {
            assert!(message.contains("certified-error discharge"));
        }
        Ok(other) => panic!("expected typed finite refusal once expired, got {other:?}"),
    }
    assert_eq!(lb.lower_a_err, before.lower_a_err);
    assert_eq!(lb.upper_a_err, before.upper_a_err);
    assert_eq!(lb.lower_b(), before.lower_b());
    assert_eq!(lb.upper_b(), before.upper_b());
}

#[test]
fn live_strict_finite_boundary_skip_merge_with_coeff_err_is_copy_free_pass_through() {
    let layer = Layer::SkipMerge(SkipMergeLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()];
    let mut ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    ctx.deadline = Some(Instant::now() + Duration::from_secs(30));
    let mut lb = identity_lb(2);
    lb.lower_a_err = Some(Array2::from_elem((2, 2), 0.25));
    lb.upper_a_err = Some(Array2::from_elem((2, 2), 0.5));

    assert!(matches!(
        dispatch_backward_layer_finite_boundary(&ctx, &lb).unwrap(),
        BackwardDispatchResult::PassThrough
    ));
}

#[test]
fn dispatch_add_returns_binary_with_separate_bias_2617() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    // Non-zero bias in incoming bounds
    let lb = LinearBounds {
        lower_a: Array2::eye(2),
        lower_b: Array1::from_vec(vec![1.0, 2.0]),
        upper_a: Array2::eye(2),
        upper_b: Array1::from_vec(vec![3.0, 4.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            // Bias must be in separate channel, not in bounds
            assert!(
                bounds_a.lower_b.iter().all(|&v| v == 0.0),
                "bounds_a lower_b should be zero"
            );
            assert!(
                bounds_a.upper_b.iter().all(|&v| v == 0.0),
                "bounds_a upper_b should be zero"
            );
            assert!(
                bounds_b.lower_b.iter().all(|&v| v == 0.0),
                "bounds_b lower_b should be zero"
            );
            assert!(
                bounds_b.upper_b.iter().all(|&v| v == 0.0),
                "bounds_b upper_b should be zero"
            );
            // Bias channel must contain incoming bias (tolerance for directed rounding)
            for (got, expected) in bias_lower.iter().zip(&[1.0_f32, 2.0]) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "bias_lower mismatch: got {got}, expected {expected}"
                );
            }
            for (got, expected) in bias_upper.iter().zip(&[3.0_f32, 4.0]) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "bias_upper mismatch: got {got}, expected {expected}"
                );
            }
        }
        other => panic!("Expected Binary, got {other:?}"),
    }
}

#[test]
fn dispatch_add_insufficient_inputs_returns_error() {
    let layer = Layer::Add(AddLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()]; // Only 1 input, need 2
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "Add with 1 input should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("requires exactly 2 inputs"),
        "Expected 'requires exactly 2 inputs', got: {err_msg}"
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
        "Sub dispatch should return Binary, got {result:?}"
    );
}

#[test]
fn dispatch_sub_returns_binary_with_separate_bias_2617() {
    let layer = Layer::Sub(SubLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    // Non-zero incoming bias must stay in the separate channel.
    let lb = LinearBounds {
        lower_a: Array2::eye(2),
        lower_b: Array1::from_vec(vec![1.25, -0.5]),
        upper_a: Array2::eye(2),
        upper_b: Array1::from_vec(vec![2.75, 0.25]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            assert!(
                bounds_a.lower_b.iter().all(|&v| v == 0.0),
                "Sub bounds_a lower_b should be zero (bias in separate channel)"
            );
            assert!(
                bounds_a.upper_b.iter().all(|&v| v == 0.0),
                "Sub bounds_a upper_b should be zero (bias in separate channel)"
            );
            assert!(
                bounds_b.lower_b.iter().all(|&v| v == 0.0),
                "Sub bounds_b lower_b should be zero (bias in separate channel)"
            );
            assert!(
                bounds_b.upper_b.iter().all(|&v| v == 0.0),
                "Sub bounds_b upper_b should be zero (bias in separate channel)"
            );

            for (got, expected) in bias_lower.iter().zip(&[1.25_f32, -0.5]) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "bias_lower mismatch: got {got}, expected {expected}"
                );
            }
            for (got, expected) in bias_upper.iter().zip(&[2.75_f32, 0.25]) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "bias_upper mismatch: got {got}, expected {expected}"
                );
            }
        }
        other => panic!("Expected Binary, got {other:?}"),
    }
}

#[test]
fn dispatch_sub_insufficient_inputs_returns_error() {
    let layer = Layer::Sub(SubLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs: Vec<String> = vec![]; // 0 inputs, need 2
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "Sub with 0 inputs should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("requires exactly 2 inputs"),
        "Expected 'requires exactly 2 inputs', got: {err_msg}"
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
        "SkipMerge single-input should return PassThrough, got {result:?}"
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
    assert!(result.is_err(), "SkipMerge with 2 inputs should fail");
    let err_msg = format!("{}", result.unwrap_err());
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
        "OpaqueSkip single-input should return Single, got {result:?}"
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
            bounds: entries,
            bias_lower,
            bias_upper,
        } => {
            assert_eq!(entries.len(), 3, "Expected 3 entries for 3 inputs");
            assert!(
                entries.iter().all(|e| e.is_some()),
                "All entries should be Some"
            );
            // OpaqueSkip's propagate_linear returns unbounded [-inf, +inf] bias
            // (conservative "I know nothing" bounds). The separate bias channel
            // captures this, which will trigger IBP fallback at the consumer.
            assert!(
                bias_lower.iter().all(|&v| v == f32::NEG_INFINITY),
                "OpaqueSkip bias_lower should be -inf (unbounded)"
            );
            assert!(
                bias_upper.iter().all(|&v| v == f32::INFINITY),
                "OpaqueSkip bias_upper should be +inf (unbounded)"
            );
            // A-matrix bounds should have zero bias
            for entry in entries.iter().flatten() {
                assert!(
                    entry.lower_b.iter().all(|&v| v == 0.0),
                    "OpaqueSkip entry lower_b should be zero (bias in separate channel)"
                );
                assert!(
                    entry.upper_b.iter().all(|&v| v == 0.0),
                    "OpaqueSkip entry upper_b should be zero (bias in separate channel)"
                );
            }
        }
        other => panic!("Expected Nary, got {other:?}"),
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
    assert!(result.is_err(), "OpaqueSkip with 0 inputs should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("has no inputs"),
        "Expected 'has no inputs', got: {err_msg}"
    );
}

// ===================================================================
// ReLU, MulBinary, Where return Unsupported
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
        other => panic!("Expected Unsupported, got {other:?}"),
    }
}

/// MulBinary now dispatches via McCormick CROWN backward (#3439).
/// Both input bounds resolve to `node_bounds` entries "a" and "b".
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
            // #2617/#2530: A-matrix bounds must have zero bias
            assert!(
                bounds_a.lower_b.iter().all(|&v| v == 0.0),
                "bounds_a lower_b should be zero"
            );
            assert!(
                bounds_a.upper_b.iter().all(|&v| v == 0.0),
                "bounds_a upper_b should be zero"
            );
            assert!(
                bounds_b.lower_b.iter().all(|&v| v == 0.0),
                "bounds_b lower_b should be zero"
            );
            assert!(
                bounds_b.upper_b.iter().all(|&v| v == 0.0),
                "bounds_b upper_b should be zero"
            );
            // Bias channel should contain McCormick relaxation bias
            assert_eq!(bias_lower.len(), 2);
            assert_eq!(bias_upper.len(), 2);
        }
        other => panic!("Expected Binary for MulBinary, got {other:?}"),
    }
}

/// MulBinary dispatch with alpha parameters uses interpolated McCormick (#3439 Phase 2).
#[test]
fn dispatch_mulbinary_with_alpha_returns_binary_3439() {
    let layer = Layer::MulBinary(MulBinaryLayer);
    let bounds = simple_bounds(2);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("a".to_string(), simple_bounds(2));
    node_bounds.insert("b".to_string(), simple_bounds(2));
    let inputs = vec!["a".to_string(), "b".to_string()];

    // Alpha parameters: [2, n] with r_l=0.3, r_u=0.7
    let mut mul_alphas = HashMap::new();
    mul_alphas.insert(
        "test_node".to_string(),
        Array2::from_shape_vec((2, 2), vec![0.3, 0.3, 0.7, 0.7]).unwrap(),
    );
    let ctx = DispatchContext {
        node_name: "test_node",
        layer: &layer,
        inputs: &inputs,
        pre_activation: &bounds,
        network_input: &bounds,
        node_bounds: (&node_bounds).into(),
        engine: None,
        deadline: None,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: Some(&mul_alphas),
        norm_inv_rms_override: None,
    };
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            // A-matrix bounds must have zero bias
            assert!(
                bounds_a.lower_b.iter().all(|&v| v == 0.0),
                "MulBinary+alpha bounds_a lower_b should be zero"
            );
            assert!(
                bounds_a.upper_b.iter().all(|&v| v == 0.0),
                "MulBinary+alpha bounds_a upper_b should be zero"
            );
            assert!(
                bounds_b.lower_b.iter().all(|&v| v == 0.0),
                "MulBinary+alpha bounds_b lower_b should be zero"
            );
            assert!(
                bounds_b.upper_b.iter().all(|&v| v == 0.0),
                "MulBinary+alpha bounds_b upper_b should be zero"
            );
            assert_eq!(bias_lower.len(), 2);
            assert_eq!(bias_upper.len(), 2);
        }
        other => panic!("Expected Binary for MulBinary with alpha, got {other:?}"),
    }
}

/// MulBinary dispatch soundness: verify bounds from dispatch path contain true output.
///
/// The existing structural tests (`dispatch_mulbinary_returns_binary_3439`) only check
/// that the correct variant/shape is returned. This test verifies the mathematical
/// property: for all (xa, xb) in input bounds, xa*xb lies within the concretized
/// CROWN bounds produced by the dispatch path.
///
/// Closes the integration gap between layer-level proptests and dispatch-level
/// structural tests. Part of P1 1114 reflection.
#[test]
fn dispatch_mulbinary_soundness_bounds_contain_product() {
    // Use asymmetric bounds to exercise sign-dependent McCormick plane selection.
    // Input A: [-2, 1] per element (zero-crossing).
    // Input B: [0.5, 3] per element (strictly positive).
    let dim = 2;
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-2.0f32, -1.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0f32, 0.5]).unwrap(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.5f32, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![3.0f32, 2.5]).unwrap(),
    )
    .unwrap();

    let layer = Layer::MulBinary(MulBinaryLayer);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("a".to_string(), input_a.clone());
    node_bounds.insert("b".to_string(), input_b.clone());
    let inputs = vec!["a".to_string(), "b".to_string()];
    let ctx = make_ctx(&layer, &input_a, &input_a, &node_bounds, &inputs);
    let lb = identity_lb(dim);

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    let (bounds_a, bounds_b, bias_lower, bias_upper) = match result {
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => (bounds_a, bounds_b, bias_lower, bias_upper),
        other => panic!("Expected Binary for MulBinary, got {other:?}"),
    };

    // Concretize: combined lower = concretize_a.lower + concretize_b.lower + bias_lower
    let concrete_a = bounds_a.concretize(&input_a);
    let concrete_b = bounds_b.concretize(&input_b);

    let crown_lower: Vec<f32> = (0..dim)
        .map(|i| concrete_a.lower()[[i]] + concrete_b.lower()[[i]] + bias_lower[i])
        .collect();
    let crown_upper: Vec<f32> = (0..dim)
        .map(|i| concrete_a.upper()[[i]] + concrete_b.upper()[[i]] + bias_upper[i])
        .collect();

    // Sample grid: 11 points per dimension, 11^4 = 14641 samples total.
    let tolerance = 1e-3;
    let sample_count = 11;
    for ia in 0..sample_count {
        let ta = ia as f32 / (sample_count - 1) as f32;
        for ib in 0..sample_count {
            let tb = ib as f32 / (sample_count - 1) as f32;
            for elem in 0..dim {
                let xa = input_a.lower()[[elem]]
                    + ta * (input_a.upper()[[elem]] - input_a.lower()[[elem]]);
                let xb = input_b.lower()[[elem]]
                    + tb * (input_b.upper()[[elem]] - input_b.lower()[[elem]]);
                let y = xa * xb;
                assert!(
                    y >= crown_lower[elem] - tolerance,
                    "Soundness violation (lower) at elem={elem}, xa={xa}, xb={xb}: \
                     y={y} < lb={}",
                    crown_lower[elem]
                );
                assert!(
                    y <= crown_upper[elem] + tolerance,
                    "Soundness violation (upper) at elem={elem}, xa={xa}, xb={xb}: \
                     y={y} > ub={}",
                    crown_upper[elem]
                );
            }
        }
    }
}

/// MulBinary with insufficient inputs returns an error.
#[test]
fn dispatch_mulbinary_insufficient_inputs_returns_error() {
    let layer = Layer::MulBinary(MulBinaryLayer);
    let bounds = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["a".to_string()]; // Only 1 input, need 2
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(2);

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("MulBinary") && err_msg.contains("requires exactly 2 inputs"),
        "Expected MulBinary input count error, got: {err_msg}"
    );
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
        "Where dispatch should return Unsupported, got {result:?}"
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
    let lb = LinearBounds {
        lower_a: Array2::eye(4),
        lower_b: Array1::zeros(4),
        upper_a: Array2::eye(4),
        upper_b: Array1::zeros(4),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "Conv1d with 1D input should fail");
    let err_msg = format!("{}", result.unwrap_err());
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
    let lb = LinearBounds {
        lower_a: Array2::eye(4),
        lower_b: Array1::zeros(4),
        upper_a: Array2::eye(4),
        upper_b: Array1::zeros(4),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = dispatch_backward_layer(&ctx, &lb);
    assert!(result.is_err(), "Conv2d with 2D input should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Conv2d") && err_msg.contains(">= 3D"),
        "Expected Conv2d dimension error, got: {err_msg}"
    );
}

#[test]
fn ordinary_finite_dense_conv2d_matches_no_deadline_dispatch() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.25_f32]).unwrap();
    let bias = Array1::from_vec(vec![0.2_f32]);
    let layer =
        Layer::Conv2d(Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).expect("valid Conv2d"));
    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32),
    )
    .unwrap();
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let node_lb = identity_lb(4);

    let baseline_ctx = make_ctx(&layer, &pre_act, &pre_act, &node_bounds, &inputs);
    let baseline = match dispatch_backward_layer(&baseline_ctx, &node_lb).unwrap() {
        BackwardDispatchResult::Single(bounds) => *bounds,
        other => panic!("no-deadline Conv2d should return Single, got {other:?}"),
    };
    let mut finite_ctx = make_ctx(&layer, &pre_act, &pre_act, &node_bounds, &inputs);
    finite_ctx.deadline = Some(Instant::now() + Duration::from_secs(30));
    let finite = match dispatch_backward_layer(&finite_ctx, &node_lb).unwrap() {
        BackwardDispatchResult::Single(bounds) => *bounds,
        other => panic!("ordinary finite Dense Conv2d should return Single, got {other:?}"),
    };

    assert_eq!(finite.lower_a, baseline.lower_a);
    assert_eq!(finite.lower_b, baseline.lower_b);
    assert_eq!(finite.upper_a, baseline.upper_a);
    assert_eq!(finite.upper_b, baseline.upper_b);
    assert_eq!(finite.lower_a_err, baseline.lower_a_err);
    assert_eq!(finite.upper_a_err, baseline.upper_a_err);
}

#[test]
fn dispatch_conv_transpose2d_engine_none_matches_trait_path_3697() {
    let conv = sample_conv_transpose2d();
    let layer = Layer::ConvTranspose2d(conv.clone());
    let pre_act = sample_conv_transpose2d_pre_act();
    let net_input = pre_act.clone();
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let node_lb = sample_conv_transpose2d_node_lb();

    let actual = match dispatch_backward_layer(&ctx, &node_lb).expect("dispatch should succeed") {
        BackwardDispatchResult::Single(bounds) => *bounds,
        other => panic!("expected Single result, got {other:?}"),
    };

    let mut expected_layer = conv;
    expected_layer.set_input_shape(2, 2);
    let expected = expected_layer
        .propagate_crown_backward(&node_lb, Some(&pre_act))
        .expect("legacy trait path should succeed");

    assert_eq!(actual.lower_a, expected.lower_a);
    assert_eq!(actual.lower_b, expected.lower_b);
    assert_eq!(actual.upper_a, expected.upper_a);
    assert_eq!(actual.upper_b, expected.upper_b);
}

// ===================================================================
// MatMul with <2 inputs returns error (new guard from W3)
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
    assert!(result.is_err(), "MatMul with 1 input should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("MatMul") && err_msg.contains("requires exactly 2 inputs"),
        "Expected MatMul input count error, got: {err_msg}"
    );
}

// ===================================================================
// BilinearCrown with <2 inputs returns error (new guard from W3)
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
    assert!(result.is_err(), "BilinearCrown with 1 input should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("BilinearCrown") && err_msg.contains("requires exactly 2 inputs"),
        "Expected BilinearCrown input count error, got: {err_msg}"
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
        "Flatten dispatch should return Single, got {result:?}"
    );
}

// ===================================================================
// Concat N-ary: separate bias channel (#2617, supersedes #2529)
// ===================================================================

/// Regression test for #2529/#2617: Concat with constant-input bias goes through
/// the separate bias channel (not split across per-input bounds).
#[test]
fn dispatch_concat_nary_constant_input_bias_in_separate_channel_2617() {
    // 3-input concat: sizes [2, 1, 2], input[1] is constant.
    let constant_tensor = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 42.0_f32),
        ArrayD::from_elem(IxDyn(&[1]), 42.0_f32),
    )
    .unwrap();
    let concat = ConcatLayer::with_constants(
        0,
        vec![vec![2], vec![1], vec![2]],
        vec![None, Some(constant_tensor), None],
    );
    let layer = Layer::Concat(concat);
    let net_input = simple_bounds(2);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("c".to_string(), simple_bounds(2));
    let inputs = vec![
        "_input".to_string(),
        "const_node".to_string(),
        "c".to_string(),
    ];
    let ctx = make_ctx(&layer, &net_input, &net_input, &node_bounds, &inputs);

    let lb = LinearBounds {
        lower_a: Array2::ones((1, 5)),
        lower_b: Array1::from_vec(vec![0.9]),
        upper_a: Array2::ones((1, 5)),
        upper_b: Array1::from_vec(vec![1.5]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Nary {
            bounds: entries,
            bias_lower,
            bias_upper,
        } => {
            assert_eq!(entries.len(), 3);
            assert!(entries[1].is_none(), "Constant input should be None");
            assert!(entries[0].is_some(), "Input 0 should be Some");
            assert!(entries[2].is_some(), "Input 2 should be Some");
            // Bias = incoming bias + W*constant for constant positions (#4112).
            // lower_a slice for the constant (col 2) is ones(1,1), constant=42.0
            // → contribution ≈ 42.0; total ≈ 0.9 + 42.0 = 42.9
            let expected_lower = 0.9_f32 + 42.0_f32;
            let expected_upper = 1.5_f32 + 42.0_f32;
            assert!(
                (bias_lower[0] - expected_lower).abs() < 1e-4,
                "bias_lower mismatch: got {}, expected {}",
                bias_lower[0],
                expected_lower,
            );
            assert!(
                (bias_upper[0] - expected_upper).abs() < 1e-4,
                "bias_upper mismatch: got {}, expected {}",
                bias_upper[0],
                expected_upper,
            );
            // Per-entry bounds must have zero bias
            for entry in entries.iter().flatten() {
                assert!(
                    entry.lower_b.iter().all(|&v| v == 0.0),
                    "entry lower_b should be zero"
                );
                assert!(
                    entry.upper_b.iter().all(|&v| v == 0.0),
                    "entry upper_b should be zero"
                );
            }
        }
        other => panic!("Expected Nary, got {other:?}"),
    }
}

/// Verify that without constant inputs, N-ary dispatch returns bias in separate channel.
#[test]
fn dispatch_concat_nary_no_constants_bias_in_separate_channel_2617() {
    let concat = ConcatLayer::with_input_shapes(0, vec![vec![2], vec![1], vec![2]]);
    let layer = Layer::Concat(concat);
    let net_input = simple_bounds(2);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("b".to_string(), simple_bounds(1));
    node_bounds.insert("c".to_string(), simple_bounds(2));
    let inputs = vec!["_input".to_string(), "b".to_string(), "c".to_string()];
    let ctx = make_ctx(&layer, &net_input, &net_input, &node_bounds, &inputs);

    let lb = LinearBounds {
        lower_a: Array2::ones((1, 5)),
        lower_b: Array1::from_vec(vec![0.9]),
        upper_a: Array2::ones((1, 5)),
        upper_b: Array1::from_vec(vec![1.5]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = dispatch_backward_layer(&ctx, &lb).unwrap();
    match result {
        BackwardDispatchResult::Nary {
            bounds: entries,
            bias_lower,
            bias_upper,
        } => {
            assert_eq!(entries.len(), 3);
            assert!(entries.iter().all(|e| e.is_some()), "All should be Some");
            // Bias in separate channel must equal incoming bias (tolerance for directed rounding)
            assert!(
                (bias_lower[0] - 0.9).abs() < 1e-6,
                "bias_lower mismatch: got {}, expected 0.9",
                bias_lower[0]
            );
            assert!(
                (bias_upper[0] - 1.5).abs() < 1e-6,
                "bias_upper mismatch: got {}, expected 1.5",
                bias_upper[0]
            );
            // Per-entry bounds must have zero bias
            for entry in entries.iter().flatten() {
                assert!(
                    entry.lower_b.iter().all(|&v| v == 0.0),
                    "entry lower_b should be zero"
                );
                assert!(
                    entry.upper_b.iter().all(|&v| v == 0.0),
                    "entry upper_b should be zero"
                );
            }
        }
        other => panic!("Expected Nary, got {other:?}"),
    }
}

// ===================================================================
// resolve_input_bounds tests
// ===================================================================

#[test]
fn resolve_input_bounds_returns_network_input_for_underscore() {
    let net_input = simple_bounds(3);
    let node_bounds: HashMap<String, BoundedTensor> = HashMap::new();
    let result = resolve_input_bounds("_input", &net_input, (&node_bounds).into(), "test", "label");
    let bounds = result.expect("underscore prefix should resolve to network input");
    assert_eq!(bounds.shape(), &[3]);
    assert_eq!(bounds.lower(), net_input.lower());
    assert_eq!(bounds.upper(), net_input.upper());
}

#[test]
fn resolve_input_bounds_returns_cached_node() {
    let net_input = simple_bounds(3);
    let mut node_bounds = HashMap::new();
    let cached = simple_bounds(4);
    node_bounds.insert("node_a".to_string(), cached.clone());

    let result = resolve_input_bounds("node_a", &net_input, (&node_bounds).into(), "test", "label");
    let bounds = result.expect("cached node should resolve");
    assert_eq!(bounds.shape(), &[4]);
    assert_eq!(bounds.lower(), cached.lower());
    assert_eq!(bounds.upper(), cached.upper());
}

#[test]
fn resolve_input_bounds_missing_node_returns_error() {
    let net_input = simple_bounds(3);
    let node_bounds: HashMap<String, BoundedTensor> = HashMap::new();
    let result = resolve_input_bounds(
        "missing",
        &net_input,
        (&node_bounds).into(),
        "test",
        "label",
    );
    assert!(
        result.is_err(),
        "Missing node 'missing' should return error"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("not found"),
        "Expected 'not found', got: {err_msg}"
    );
}

// ===================================================================
// NumericalInstability → Unsupported fallback (#2888)
// ===================================================================

/// Regression test for #2888: NumericalInstability from a layer's CROWN backward
/// (e.g., Exp with non-finite pre-activation bounds) must return Unsupported,
/// not propagate as a hard error. The caller uses this to fall back to IBP.
#[test]
fn dispatch_numerical_instability_returns_unsupported_2888() {
    use crate::layers::activations::ExpLayer;

    let layer = Layer::Exp(ExpLayer::new());
    // Pre-activation bounds containing +Inf trigger non_finite_domain_guard
    // → NumericalInstability from Exp's propagate_crown_backward.
    let pre_act = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY),
    )
    .unwrap();
    let net_input = simple_bounds(2);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(2);

    // Before #2888 fix: this would return Err(NumericalInstability).
    // After fix: returns Ok(Unsupported) so callers can fall back to IBP.
    let result = dispatch_backward_layer(&ctx, &lb)
        .expect("NumericalInstability should be caught as Unsupported, not propagated as error");
    assert!(
        matches!(result, BackwardDispatchResult::Unsupported(_)),
        "Expected Unsupported for NumericalInstability, got {result:?}"
    );
}

#[test]
fn dispatch_pad_shape_mismatch_preserves_structure_3680() {
    let layer = Layer::Pad(PadLayer::new(vec![(1, 1), (0, 0)], PadMode::Constant(0.0)));
    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 3, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0_f32),
    )
    .unwrap();
    let net_input = simple_bounds(pre_act.len());
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &pre_act, &net_input, &node_bounds, &inputs);
    let lb = identity_lb(pre_act.len());

    let err = dispatch_backward_layer(&ctx, &lb)
        .expect_err("Pad shape mismatch should stay structured, not become Unsupported");
    match err {
        ny_core::NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

// ===================================================================
// SoundnessRefusal propagates as hard error (NOT caught as Unsupported)
// ===================================================================

/// Regression guard: SoundnessRefusal from a layer's propagate_crown_backward
/// must propagate as a hard error through the catch-all dispatch arm.
/// If this were swallowed as Unsupported, the caller would silently fall back
/// to IBP, producing unsound bounds without any warning.
///
/// Counterpart to dispatch_numerical_instability_returns_unsupported_2888:
/// NumericalInstability → Unsupported (safe IBP fallback), but
/// SoundnessRefusal → hard Err (must not be silenced).
#[test]
fn dispatch_soundness_refusal_propagates_as_error() {
    // LayerNorm in Sound mode returns SoundnessRefusal from propagate_crown_backward.
    let ln = LayerNormLayer::new_default(3, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    let layer = Layer::LayerNorm(ln);

    let bounds = simple_bounds(3);
    let node_bounds = HashMap::new();
    let inputs = vec!["_input".to_string()];
    let ctx = make_ctx(&layer, &bounds, &bounds, &node_bounds, &inputs);
    let lb = identity_lb(3);

    let err = dispatch_backward_layer(&ctx, &lb)
        .expect_err("SoundnessRefusal must propagate as error, not be caught as Unsupported");
    assert!(
        matches!(err, ny_core::NyError::SoundnessRefusal(_)),
        "Expected SoundnessRefusal, got: {err}"
    );
}
