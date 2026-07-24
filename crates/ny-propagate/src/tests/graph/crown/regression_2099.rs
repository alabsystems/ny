// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #2099: empty-input and malformed-arity nodes must return
//! `InvalidSpec` errors instead of index-out-of-bounds panics.
//!
//! Updated for #2991: `GraphNode::new()` now asserts arity at construction time
//! (#2481, #2686). These tests use `GraphNode::try_new()` to verify the
//! construction-time validation returns `InvalidSpec` without panicking.

use crate::*;

/// Regression test #2099/#2991: empty-input ReLU is rejected at construction time
/// with InvalidSpec, not a panic.
#[test]
fn test_graph_crown_empty_inputs_relu_returns_invalid_spec_2099() {
    let err = GraphNode::try_new("relu", Layer::ReLU(ReLULayer), vec![])
        .expect_err("empty-input ReLU should return InvalidSpec at construction");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("relu") && msg.contains("1 input"),
        "expected node name and arity in message, got: {msg}"
    );
}

/// Regression test #2099/#2991: spec-guided CROWN path also catches empty-input
/// ReLU — validated at construction before propagation is reached.
#[test]
fn test_spec_guided_crown_empty_inputs_relu_returns_invalid_spec_2099() {
    let err = GraphNode::try_new("relu", Layer::ReLU(ReLULayer), vec![])
        .expect_err("empty-input ReLU should return InvalidSpec at construction");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

/// Regression test #2099: Where with wrong arity (1 input instead of 3)
/// is caught at propagation time by `require_ternary_inputs()`.
///
/// Note: Where's `min_inputs()` returns 1 (embedded-constants variant), so
/// construction succeeds. The 3-input validation happens at propagation time.
#[test]
fn test_graph_crown_where_wrong_arity_returns_invalid_spec_2099() {
    let mut graph = GraphNetwork::new();

    let weight = ndarray::arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "where",
        Layer::Where(WhereLayer::new()),
        vec!["linear".to_string()],
    ));
    graph.set_output("where");

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0_f32, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let err = graph
        .propagate_crown(&input)
        .expect_err("malformed Where arity should return error");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("where") && msg.contains("3 inputs"),
        "expected node name and ternary arity diagnostic, got: {msg}"
    );
}

/// Regression test #2099/#2991: empty-input Add is rejected at construction.
#[test]
fn test_batched_crown_empty_inputs_add_returns_invalid_spec_2099() {
    use crate::layers::AddLayer;
    let err = GraphNode::try_new("add", Layer::Add(AddLayer), vec![])
        .expect_err("empty-input Add should return InvalidSpec at construction");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

/// Regression test #2099/#2991: single-input Add (binary op needing 2) is
/// rejected at construction with InvalidSpec containing diagnostic info.
#[test]
fn test_batched_crown_single_input_add_returns_invalid_spec_2099() {
    use crate::layers::AddLayer;
    let err = GraphNode::try_new("add", Layer::Add(AddLayer), vec!["linear".to_string()])
        .expect_err("single-input Add should return InvalidSpec at construction");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("add") && msg.contains("2 input"),
        "expected node name and binary arity diagnostic, got: {msg}"
    );
}
