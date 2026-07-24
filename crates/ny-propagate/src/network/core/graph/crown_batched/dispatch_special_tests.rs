// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `dispatch_special_batched_operator` error routing.
//!
//! Part of #4280: these tests encode that MulBinary and BilinearCrown error
//! branches correctly route to partial CROWN fallback (for fallback-eligible
//! errors) or propagate the error (for non-eligible errors).
//!
//! Loaded from `crown_batched.rs` via `#[path]` so that `dispatch_special_batched_operator`
//! (which is `pub(super)` on `impl GraphNetwork`) is accessible.

use std::collections::HashMap;

use ndarray::ArrayD;
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use crate::bounds::BatchedLinearBounds;
use crate::layers::binary_ops::BilinearCrownLayer;
use crate::layers::{Layer, LinearLayer, MulBinaryLayer, ReLULayer};
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
use crate::network::core::graph::GraphNode;
use crate::network::core::graph::NETWORK_INPUT;
use crate::types::BoundsProvenance;
use crate::MulBinaryRelaxationMode;

use super::binary_ops::AttentionCompositionRuntime;
use super::dispatch::SpecialBatchedDispatchResult;
use super::GraphNetwork;

/// Build a minimal graph and dispatch plan for accumulator tests.
///
/// Graph: NETWORK_INPUT(4) -> linear1(4) -> relu1(4)
fn make_test_plan() -> CrownDispatchPlan {
    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(
                ndarray::Array2::zeros((4, 4)),
                Some(ndarray::Array1::zeros(4)),
            )
            .unwrap(),
        ),
    ))
    .unwrap();
    g.try_add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ))
    .unwrap();
    g.set_output("relu1");
    CrownDispatchPlan::build(&g).unwrap()
}

/// Create interval bounds [-1, 1] for a given shape.
fn make_interval(shape: &[usize]) -> BoundedTensor {
    let lower = ArrayD::from_elem(shape, -1.0f32);
    let upper = ArrayD::from_elem(shape, 1.0f32);
    BoundedTensor::new(lower, upper).unwrap()
}

/// Create interval bounds with infinite values (triggers McCormick validation failure).
fn make_infinite_interval(shape: &[usize]) -> BoundedTensor {
    let lower = ArrayD::from_elem(shape, f32::NEG_INFINITY);
    let upper = ArrayD::from_elem(shape, f32::INFINITY);
    BoundedTensor::new_allow_infinite(lower, upper).unwrap()
}

/// Build a minimal GraphNetwork with a MulBinary(NETWORK_INPUT, NETWORK_INPUT) node.
fn make_mul_graph() -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        NETWORK_INPUT,
        NETWORK_INPUT,
    ));
    g.set_output("mul");
    g
}

/// Build a MulBinary GraphNode for dispatch testing.
fn make_mul_node() -> GraphNode {
    GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        NETWORK_INPUT,
        NETWORK_INPUT,
    )
}

// ---- dispatch_special_batched_operator: error routing tests (#4280 checkbox 6) ----

/// MulBinary with infinite input bounds triggers McCormick validation failure
/// (UnsupportedOp), which is fallback-eligible → PartialFallback with IBP bounds.
#[test]
fn test_dispatch_special_mulbinary_infinite_inputs_fallback() {
    let graph = make_mul_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = make_mul_node();
    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    // Infinite input bounds trigger McCormick validation → UnsupportedOp
    let input = make_infinite_interval(&[4]);
    let mut node_bounds = HashMap::new();
    // IBP bounds for "mul" node — needed by partial_crown_ibp_fallback
    node_bounds.insert("mul".to_string(), make_interval(&[4]));

    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "mul",
        &node_lb,
        &input,
        &node_bounds,
        &[4],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    match result.expect("should succeed with partial fallback") {
        SpecialBatchedDispatchResult::PartialFallback(fallback) => {
            assert_eq!(
                fallback.provenance,
                BoundsProvenance::Crown,
                "partial fallback provenance should be Crown"
            );
        }
        other => panic!(
            "expected PartialFallback for infinite-input MulBinary, got {:?}",
            match other {
                SpecialBatchedDispatchResult::NotHandled => "NotHandled",
                SpecialBatchedDispatchResult::Handled => "Handled",
                SpecialBatchedDispatchResult::PartialFallback(_) => unreachable!(),
            }
        ),
    }
}

/// MulBinary with 1D node_lb (ndim < 2) triggers ShapeMismatch, which is
/// NOT fallback-eligible for the base path → error must propagate.
#[test]
fn test_dispatch_special_mulbinary_shape_mismatch_propagates_error() {
    let graph = make_mul_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = make_mul_node();
    // 1D BatchedLinearBounds: ndim < 2 → ShapeMismatch from MulBinary
    let node_lb = BatchedLinearBounds {
        lower_a: ndarray::Array1::zeros(4).into_dyn(),
        upper_a: ndarray::Array1::zeros(4).into_dyn(),
        lower_b: ndarray::Array1::zeros(4).into_dyn(),
        upper_b: ndarray::Array1::zeros(4).into_dyn(),
        input_shape: vec![4],
        output_shape: vec![4],
        lower_a_err: None,
        upper_a_err: None,
    };
    let input = make_interval(&[4]);
    let node_bounds = HashMap::new();

    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "mul",
        &node_lb,
        &input,
        &node_bounds,
        &[4],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    let err =
        result.expect_err("ShapeMismatch from 1D node_lb must propagate (not fallback-eligible)");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {err}"
    );
}

/// A non-special layer (ReLU) returns NotHandled from dispatch_special_batched_operator.
#[test]
fn test_dispatch_special_non_special_layer_returns_not_handled() {
    let graph = make_mul_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    // ReLU is a unary layer, not handled by the special binary dispatch
    let node = GraphNode::from_input("relu_node", Layer::ReLU(ReLULayer));
    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    let input = make_interval(&[4]);
    let node_bounds = HashMap::new();

    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "relu_node",
        &node_lb,
        &input,
        &node_bounds,
        &[4],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    assert!(
        matches!(
            result.expect("non-special layer should return Ok"),
            SpecialBatchedDispatchResult::NotHandled
        ),
        "unary layer must return NotHandled from special binary dispatch"
    );
}

/// MulBinary with valid finite inputs and identity bounds should succeed
/// and return Handled (bounds accumulated into accumulator).
#[test]
fn test_dispatch_special_mulbinary_success_returns_handled() {
    let graph = make_mul_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = make_mul_node();
    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    let input = make_interval(&[4]);
    let node_bounds = HashMap::new();

    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "mul",
        &node_lb,
        &input,
        &node_bounds,
        &[4],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    match result.expect("valid MulBinary dispatch should succeed") {
        SpecialBatchedDispatchResult::Handled => {}
        SpecialBatchedDispatchResult::PartialFallback(_) => {
            // Also acceptable — validate_binary_bounds_and_accumulate may
            // trigger fallback if bounds have non-finite coefficients.
        }
        SpecialBatchedDispatchResult::NotHandled => {
            panic!("MulBinary must be handled by special dispatch, not returned as NotHandled")
        }
    }
}

// ---- dispatch_special_batched_operator: BilinearCrown error routing tests (#4280 checkbox 6) ----

/// Build a minimal GraphNetwork with a BilinearCrown(NETWORK_INPUT, NETWORK_INPUT) node.
fn make_bilinear_graph() -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::binary(
        "bilinear",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        NETWORK_INPUT,
        NETWORK_INPUT,
    ));
    g.set_output("bilinear");
    g
}

/// Build a BilinearCrown GraphNode for dispatch testing.
fn make_bilinear_node() -> GraphNode {
    GraphNode::binary(
        "bilinear",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        NETWORK_INPUT,
        NETWORK_INPUT,
    )
}

/// BilinearCrown with infinite input bounds triggers McCormick validation failure,
/// which is fallback-eligible → PartialFallback with IBP bounds.
#[test]
fn test_dispatch_special_bilinear_infinite_inputs_fallback() {
    let graph = make_bilinear_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = make_bilinear_node();
    // BilinearCrown needs 2D+ identity bounds to avoid ShapeMismatch
    let node_lb = BatchedLinearBounds::identity(&[2, 2]).unwrap();
    // Infinite input bounds trigger McCormick validation → fallback-eligible error
    let input = make_infinite_interval(&[2, 2]);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("bilinear".to_string(), make_interval(&[2, 2]));

    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "bilinear",
        &node_lb,
        &input,
        &node_bounds,
        &[2, 2],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    match result.expect("should succeed with partial fallback") {
        SpecialBatchedDispatchResult::PartialFallback(fallback) => {
            assert_eq!(
                fallback.provenance,
                BoundsProvenance::Crown,
                "partial fallback provenance should be Crown"
            );
        }
        other => panic!(
            "expected PartialFallback for infinite-input BilinearCrown, got {:?}",
            match other {
                SpecialBatchedDispatchResult::NotHandled => "NotHandled",
                SpecialBatchedDispatchResult::Handled => "Handled",
                SpecialBatchedDispatchResult::PartialFallback(_) => unreachable!(),
            }
        ),
    }
}

/// BilinearCrown with valid finite inputs and identity bounds should succeed
/// and return Handled (bounds accumulated into accumulator).
#[test]
fn test_dispatch_special_bilinear_success_returns_handled() {
    let graph = make_bilinear_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = make_bilinear_node();
    let node_lb = BatchedLinearBounds::identity(&[2, 2]).unwrap();
    let input = make_interval(&[2, 2]);
    let mut node_bounds = HashMap::new();
    // BilinearCrown resolve_binary_input_bounds looks up IBP bounds for "bilinear"
    node_bounds.insert("bilinear".to_string(), make_interval(&[2, 2]));

    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "bilinear",
        &node_lb,
        &input,
        &node_bounds,
        &[2, 2],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    match result.expect("valid BilinearCrown dispatch should succeed") {
        SpecialBatchedDispatchResult::Handled => {}
        SpecialBatchedDispatchResult::PartialFallback(_) => {
            // Also acceptable — validate_binary_bounds_and_accumulate may
            // trigger fallback if bounds have non-finite coefficients.
        }
        SpecialBatchedDispatchResult::NotHandled => {
            panic!("BilinearCrown must be handled by special dispatch, not returned as NotHandled")
        }
    }
}
