// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `dispatch_batched_unary` has_other_pending soundness guard.
//!
//! Part of #4280: these tests encode the invariant that partial CROWN fallback
//! is only safe when no other paths have accumulated bounds. With pending
//! paths, partial fallback would capture only one path's contribution —
//! producing unsound bounds (#2072).
//!
//! Loaded from `crown_batched.rs` via `#[path]` so that `dispatch_batched_unary`
//! (which is `pub(super)` on `impl GraphNetwork`) is accessible.

use std::collections::HashMap;

use ny_core::NyError;
use ny_tensor::BoundedTensor;

use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;
use crate::layers::{Layer, LinearLayer, NonZeroLayer, ReLULayer};
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::core::graph::dispatch_plan::CrownDispatchPlan;
use crate::network::core::graph::GraphNode;
use crate::GraphNetwork;

/// Build a minimal graph and dispatch plan for accumulator tests.
///
/// Graph: NETWORK_INPUT(3) -> linear1(4) -> relu1(4)
fn make_test_plan() -> CrownDispatchPlan {
    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(
                ndarray::Array2::zeros((4, 3)),
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

/// Create interval bounds [-1, 1] for a given flat size.
fn make_interval(size: usize) -> BoundedTensor {
    let lower = ndarray::Array1::from_elem(size, -1.0f32).into_dyn();
    let upper = ndarray::Array1::from_elem(size, 1.0f32).into_dyn();
    BoundedTensor::new(lower, upper).unwrap()
}

// ---- dispatch_batched_unary: has_other_pending guard (Part of #2072) ----

/// When accumulator is empty and a layer returns UnsupportedOp,
/// dispatch_batched_unary should return Ok(Some(fallback_bounds))
/// (partial CROWN fallback).
#[test]
fn test_dispatch_unary_empty_accumulator_unsupported_triggers_fallback() {
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);
    assert!(acc.is_empty(), "precondition: accumulator must be empty");

    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    let pre_activation = make_interval(4);

    // NonZero returns UnsupportedOp for batched CROWN backward.
    let layer = Layer::NonZero(NonZeroLayer);

    // IBP bounds for the node — needed by partial_crown_fallback.
    let mut node_bounds = HashMap::new();
    node_bounds.insert("nonzero_node".to_string(), make_interval(4));

    let result = GraphNetwork::dispatch_batched_unary(
        &layer,
        "nonzero_node",
        &node_lb,
        &pre_activation,
        None,
        "linear1",
        &node_bounds,
        &[4],
        &mut acc,
    );

    let fallback = result.expect("should succeed with partial fallback");
    assert!(
        fallback.is_some(),
        "empty accumulator + UnsupportedOp must trigger partial CROWN fallback, not error"
    );
}

/// When accumulator has pending paths from a binary op split and a layer
/// returns UnsupportedOp, the error MUST be propagated (not fallback).
/// Partial CROWN with pending paths would produce unsound bounds (#2072).
#[test]
fn test_dispatch_unary_nonempty_accumulator_unsupported_propagates_error() {
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    // Pre-populate: simulate a prior path having accumulated bounds.
    let dummy = BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&[4]).unwrap());
    acc.insert("relu1", dummy);
    assert!(
        !acc.is_empty(),
        "precondition: accumulator must be non-empty"
    );

    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    let pre_activation = make_interval(4);

    let layer = Layer::NonZero(NonZeroLayer);
    let node_bounds = HashMap::new(); // Not needed — error should fire first.

    let result = GraphNetwork::dispatch_batched_unary(
        &layer,
        "nonzero_node",
        &node_lb,
        &pre_activation,
        None,
        "linear1",
        &node_bounds,
        &[4],
        &mut acc,
    );

    let err = result
        .expect_err("non-empty accumulator + UnsupportedOp must propagate error for soundness");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "expected UnsupportedOp, got: {err}"
    );
}

/// When accumulator has pending paths and any fallback-eligible error occurs,
/// the error must be propagated — not just UnsupportedOp but also
/// NumericalInstability and ShapeMismatch (#4146 defense-in-depth).
#[test]
fn test_dispatch_unary_nonempty_accumulator_any_eligible_error_propagates() {
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);
    let dummy = BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&[4]).unwrap());
    acc.insert("relu1", dummy);

    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    let pre_activation = make_interval(4);
    // NonZero returns UnsupportedOp, which is in the fallback-eligible set.
    let layer = Layer::NonZero(NonZeroLayer);

    let result = GraphNetwork::dispatch_batched_unary(
        &layer,
        "test",
        &node_lb,
        &pre_activation,
        None,
        "linear1",
        &HashMap::new(),
        &[4],
        &mut acc,
    );
    assert!(
        result.is_err(),
        "non-empty + any fallback-eligible error must propagate"
    );
}

/// When the layer succeeds, dispatch_batched_unary returns Ok(None) and
/// accumulates bounds into the accumulator for the first_input node.
#[test]
fn test_dispatch_unary_success_accumulates_bounds() {
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    // ReLU supports batched CROWN backward.
    let layer = Layer::ReLU(ReLULayer);
    let node_lb = BatchedLinearBounds::identity(&[4]).unwrap();
    // ReLU needs pre-activation with lower < 0 < upper for non-trivial bounds.
    let pre_activation = make_interval(4);

    let result = GraphNetwork::dispatch_batched_unary(
        &layer,
        "relu1",
        &node_lb,
        &pre_activation,
        None,
        "linear1", // first_input: where bounds get accumulated
        &HashMap::new(),
        &[4],
        &mut acc,
    );

    assert!(
        result.unwrap().is_none(),
        "successful dispatch must return Ok(None)"
    );
    assert!(
        !acc.is_empty(),
        "after successful dispatch, accumulator must contain bounds for first_input"
    );
    assert!(
        acc.contains_key("linear1"),
        "bounds must be accumulated under the first_input key"
    );
}
