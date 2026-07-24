// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Storing-intermediates result guard and constructor arity regressions (#2014, #2099).

use crate::beta_crown::GraphCrownContext;
use crate::{BetaCrownConfig, BetaCrownVerifier, GraphNode, Layer, ReLULayer};

use super::support::{active_relu_history, build_input_bounds, build_single_relu_graph};
// =========================================================================
// Regression test for #2014: expect → Result for missing intermediates
// =========================================================================

/// Regression test #2014: `propagate_crown_with_graph_constraints_storing_intermediates`
/// must return `Result` (not panic) when the backward pass fails to produce intermediates.
///
/// The invariant (StoringIntermediates always produces Some) is maintained by
/// backward.rs. This test verifies the happy path exercises the new `ok_or_else`
/// code path (replacing the old `.expect()` that would panic on invariant violation)
/// and that intermediates are correctly returned.
#[test]
fn test_storing_intermediates_returns_result_not_panic_2014() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = active_relu_history();
    let context = GraphCrownContext::for_history(&history);

    // This call goes through the `ok_or_else` path that replaced the old `expect`.
    // If the backward pass correctly populates intermediates, we get Ok.
    // If it returned None, the old code would panic; the new code returns Err.
    let result = verifier.propagate_crown_with_graph_constraints_storing_intermediates(
        &graph, &input, &context, None, None,
    );
    assert!(
        result.is_ok(),
        "StoringIntermediates should succeed (intermediates populated by backward pass), \
         got Err: {:?}",
        result.err()
    );

    let (_output, _cache, intermediate) = result.unwrap();

    // The constrained ReLU should have captured A matrix and pre-ReLU bounds.
    assert!(
        intermediate.a_at_relu.contains_key("relu1"),
        "intermediate must capture A matrix at constrained relu1"
    );
    assert!(
        intermediate.pre_relu_bounds.contains_key("relu1"),
        "intermediate must capture pre-ReLU bounds at constrained relu1"
    );
}

/// Regression test #2099/#2991: empty-input ReLU is now caught at construction
/// time by GraphNode::try_new() arity validation (#2481, #2686).
#[ntest::timeout(5000)]
#[test]
fn test_constrained_backward_empty_inputs_relu_returns_invalid_spec_2099() {
    let err = GraphNode::try_new("relu1", Layer::ReLU(ReLULayer), vec![])
        .expect_err("empty-input ReLU should return InvalidSpec at construction");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("relu1") && msg.contains("1 input"),
        "expected node name and arity diagnostic, got: {msg}"
    );
}
