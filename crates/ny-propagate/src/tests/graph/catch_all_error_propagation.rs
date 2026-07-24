// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression test: errors through the catch-all `_` arm in graph alpha-CROWN
//! backward pass must propagate, not be silently swallowed.
//!
//! Part of #1972: The catch-all `_` arm in `backward.rs` distinguishes
//! `NyError::UnsupportedOp` (legitimate fallback) from other errors like
//! `ShapeMismatch` (real bugs). This test exercises a `ShapeMismatch` through
//! the catch-all to ensure it propagates as an error rather than being silently
//! absorbed by a CROWN/IBP fallback.
//!
//! If the catch-all is ever simplified to `Err(_) => { ... fallback ... }`,
//! this test will fail.

use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

/// Build a graph: Linear(2->3) -> AddConstant(wrong-shape) -> output
///
/// The AddConstant layer has a constant of length 5, but the LinearBounds
/// coming from the 3-output linear layer have `num_inputs = 3`. Since
/// `3 % 5 != 0`, `AddConstantLayer::propagate_linear` returns `ShapeMismatch`.
///
/// In the alpha-CROWN backward pass, AddConstant hits the catch-all `_` arm
/// (it has no explicit match arm). The error must propagate as `InvalidSpec`
/// (wrapping the `ShapeMismatch`), not be silently swallowed.
fn setup_graph_with_mismatched_add_constant() -> (GraphNetwork, BoundedTensor) {
    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear = LinearLayer::new(w, None).unwrap();

    // Constant of length 5 is incompatible with linear output of length 3
    // (3 % 5 != 0, so propagate_linear returns ShapeMismatch)
    let bad_constant = ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0; 5]).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "add_bad",
        Layer::AddConstant(AddConstantLayer::new(bad_constant)),
        vec!["linear1".to_string()],
    ));
    graph.set_output("add_bad");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Verify that a ShapeMismatch error through the alpha-CROWN catch-all arm
/// propagates as an error, not a silent fallback.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_catch_all_propagates_shape_mismatch_1972() {
    let (graph, input) = setup_graph_with_mismatched_add_constant();

    let result = graph.propagate_alpha_crown(&input);

    assert!(
        result.is_err(),
        "ShapeMismatch through alpha-CROWN catch-all should be an error, not silent fallback. \
         If this test fails, check that backward.rs catch-all distinguishes UnsupportedOp \
         from other errors (commit e464c8a6)."
    );

    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Shape mismatch") || err_msg.contains("add_bad"),
        "Error should mention shape mismatch or the failing node, got: {}",
        err_msg
    );
}

/// Build a graph: Linear(2->3) -> OpaqueSkip -> output
///
/// OpaqueSkipLayer returns unbounded linear bounds from `propagate_linear`,
/// which `propagate_crown_backward` delegates to. OpaqueSkip hits the catch-all
/// `_` arm in the alpha-CROWN backward pass (not Linear, ReLU, or Transpose).
///
/// This tests that the catch-all correctly dispatches through
/// `propagate_crown_backward` for layers that DO support CROWN backward.
fn setup_graph_with_catch_all_layer() -> (GraphNetwork, BoundedTensor) {
    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear = LinearLayer::new(w, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "opaque",
        Layer::OpaqueSkip(OpaqueSkipLayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.set_output("opaque");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Verify that a layer in the catch-all `_` arm of alpha-CROWN backward
/// is dispatched correctly through `propagate_crown_backward`.
///
/// OpaqueSkipLayer hits the catch-all (not Linear/ReLU/Transpose) and returns
/// unbounded linear bounds. The alpha-CROWN loop should succeed and produce
/// valid (though conservative) bounds.
///
/// Part of #2114: both the main loop and single-pass catch-all paths now
/// consistently handle both successful dispatch and UnsupportedOp errors.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_catch_all_dispatches_opaque_skip_2114() {
    let (graph, input) = setup_graph_with_catch_all_layer();

    let result = graph.propagate_alpha_crown(&input);
    assert!(
        result.is_ok(),
        "OpaqueSkipLayer in alpha-CROWN catch-all should be dispatched successfully. Got: {:?}",
        result.err()
    );

    let bounds = result.unwrap();
    assert_eq!(bounds.shape(), &[3]);
    // OpaqueSkip returns [-inf, +inf] — bounds should be valid (lower <= upper).
    for i in 0..3 {
        assert!(
            bounds.lower()[[i]] <= bounds.upper()[[i]],
            "Bounds must be valid at index {}: lower={} > upper={}",
            i,
            bounds.lower()[[i]],
            bounds.upper()[[i]]
        );
    }
}

/// Verify that the single-pass catch-all wraps non-UnsupportedOp errors with
/// node context (part of #2114 fix).
///
/// Uses the same mismatched AddConstant graph as the main loop test, but
/// through `propagate_alpha_crown` which exercises both paths. The error
/// message should include the node name for debugging.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_single_pass_catch_all_wraps_errors_2114() {
    let (graph, input) = setup_graph_with_mismatched_add_constant();

    let result = graph.propagate_alpha_crown(&input);
    assert!(
        result.is_err(),
        "ShapeMismatch through catch-all should still propagate as error"
    );

    let err_msg = format!("{}", result.unwrap_err());
    // The error should contain node context from the catch-all wrapping.
    assert!(
        err_msg.contains("Shape mismatch") || err_msg.contains("add_bad"),
        "Error should mention shape mismatch or the failing node, got: {}",
        err_msg
    );
}

/// Build a graph: Linear(2->4) -> ReLU -> Linear(4->3) -> Sigmoid -> output
///
/// Sigmoid hits the catch-all `_` arm in the backward dispatcher (not
/// explicitly matched like Linear/ReLU/Transpose). The ReLU ensures
/// alpha-CROWN runs its full backward pass (not just plain CROWN).
///
/// With 2 nonlinear layers, CROWN exploits cross-layer correlations that
/// IBP loses, producing strictly tighter bounds after intersection.
fn setup_graph_with_sigmoid_multi_layer() -> (GraphNetwork, BoundedTensor) {
    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3], [0.8, -0.2]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[
        [0.5_f32, -0.3, 0.7, 0.1],
        [0.2, 0.8, -0.4, 0.5],
        [-0.6, 0.1, 0.3, -0.9],
    ]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

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
        "sigmoid1",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear2".to_string()],
    ));
    graph.set_output("sigmoid1");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Verify that the catch-all `_` arm dispatches through CROWN backward by
/// checking that alpha-CROWN produces tighter bounds than IBP (#2142).
///
/// OpaqueSkipLayer returns [-inf, +inf] through both CROWN and IBP paths,
/// making the dispatch path indistinguishable. This test uses Sigmoid (a
/// nonlinear layer with real CROWN backward) in a multi-layer network where
/// CROWN backward through the catch-all produces strictly tighter bounds
/// than IBP. If the catch-all fell back to IBP, bounds would be identical.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_catch_all_dispatches_crown_not_ibp_2142() {
    let (graph, input) = setup_graph_with_sigmoid_multi_layer();

    let crown_bounds = graph
        .propagate_alpha_crown(&input)
        .expect("Sigmoid alpha-CROWN catch-all should dispatch successfully");
    let ibp_bounds = graph
        .propagate_ibp(&input)
        .expect("Sigmoid IBP should succeed");

    assert_eq!(crown_bounds.shape(), ibp_bounds.shape());
    assert_eq!(crown_bounds.shape(), &[3]);

    // CROWN should be at least as tight as IBP (after CROWN∩IBP intersection)
    for i in 0..3 {
        assert!(crown_bounds.lower()[[i]] <= crown_bounds.upper()[[i]]);
        assert!(ibp_bounds.lower()[[i]] <= ibp_bounds.upper()[[i]]);
        assert!(
            crown_bounds.lower()[[i]] >= ibp_bounds.lower()[[i]] - 1e-6,
            "CROWN lower should be >= IBP lower at {}: {} < {}",
            i,
            crown_bounds.lower()[[i]],
            ibp_bounds.lower()[[i]]
        );
        assert!(
            crown_bounds.upper()[[i]] <= ibp_bounds.upper()[[i]] + 1e-6,
            "CROWN upper should be <= IBP upper at {}: {} > {}",
            i,
            crown_bounds.upper()[[i]],
            ibp_bounds.upper()[[i]]
        );
    }

    // Key: bounds must DIFFER — proves CROWN backward was dispatched.
    let bounds_differ = (0..3).any(|i| {
        (crown_bounds.lower()[[i]] - ibp_bounds.lower()[[i]]).abs() > 1e-6
            || (crown_bounds.upper()[[i]] - ibp_bounds.upper()[[i]]).abs() > 1e-6
    });
    assert!(
        bounds_differ,
        "alpha-CROWN and IBP bounds are identical — catch-all may not dispatch \
         CROWN backward. CROWN: lower={:?}, upper={:?}. IBP: lower={:?}, upper={:?}",
        crown_bounds.lower(),
        crown_bounds.upper(),
        ibp_bounds.lower(),
        ibp_bounds.upper()
    );
}

/// Verify CROWN provenance is not ForwardFallback for Sigmoid catch-all.
///
/// If the catch-all returned UnsupportedOp for Sigmoid (instead of
/// dispatching through propagate_crown_backward), the entire CROWN
/// backward pass would fall back to IBP with ForwardFallback provenance.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_catch_all_provenance_is_crown_2142() {
    let (graph, input) = setup_graph_with_sigmoid_multi_layer();

    let result = graph
        .propagate_crown_with_provenance(&input)
        .expect("Sigmoid CROWN should succeed");

    assert!(
        !result.is_fallback(),
        "CROWN provenance should be Crown, not ForwardFallback. \
         Sigmoid catch-all may have returned UnsupportedOp. Got: {:?}",
        result.provenance
    );
}

/// Same test for plain CROWN (non-alpha) backward pass.
#[ntest::timeout(10000)]
#[test]
fn test_crown_catch_all_propagates_shape_mismatch_1972() {
    let (graph, input) = setup_graph_with_mismatched_add_constant();

    let result = graph.propagate_crown(&input);

    // CROWN may use a different backward path (IBP-based or spec-based).
    // If the layer error propagates, great. If CROWN falls back to IBP for
    // unsupported ops, it should still fail because AddConstant IBP also
    // fails on shape mismatch for incompatible broadcasting.
    // Either way, we should NOT get valid-looking bounds from a mismatched network.
    match result {
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains("Shape mismatch")
                    || msg.contains("add_bad")
                    || msg.contains("broadcast"),
                "Error should relate to shape mismatch, got: {}",
                msg
            );
        }
        Ok(bounds) => {
            // If CROWN somehow produces bounds, they should at least be
            // non-finite or clearly wrong (but ideally we get an error).
            // This is a weaker assertion for the case where the CROWN path
            // falls back to IBP and IBP somehow handles the mismatch.
            panic!(
                "Expected error from mismatched AddConstant, but got bounds: lower={:?}, upper={:?}",
                bounds.lower(),
                bounds.upper()
            );
        }
    }
}
