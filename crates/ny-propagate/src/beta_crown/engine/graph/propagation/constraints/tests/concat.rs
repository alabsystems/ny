// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concat metadata and n-ary routing regressions (#1994, #2398).

use ndarray::{arr1, arr2};

use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, ConcatLayer, GraphNetwork, GraphNode, Layer,
    LinearLayer, ReLULayer,
};

use super::super::patches::ConstrainedPatchesPolicy;
use super::support::{
    active_relu_history, build_input_bounds, build_single_relu_graph, scalar_interval,
};
use super::TOL;

/// Build a graph with Concat and no explicit input_shapes metadata:
/// _input → linear_a(1→1) ↘
///                         concat(axis=0) → linear_out(2→1)
/// _input → linear_b(1→1) ↗
///
/// This shape is intentionally simple so the pre-fix fallback (`vec![pre_activation.len()]`)
/// produced a superficially valid split and masked missing metadata.
fn build_concat_missing_shape_graph() -> GraphNetwork {
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.1]))).expect("valid linear_a");
    let linear_b = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[-0.2]))).expect("valid linear_b");
    let linear_out =
        LinearLayer::new(arr2(&[[0.7, -0.3]]), Some(arr1(&[0.0]))).expect("valid linear_out");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["linear_a".to_string(), "linear_b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["concat".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

/// Regression test #1994: constrained backward Concat must fail fast when shape
/// metadata is unavailable, not synthesize a fake one-dimensional shape.
#[test]
fn test_constrained_backward_concat_missing_shape_metadata_errors_1994() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_concat_missing_shape_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let (mut bounds_cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("forward constrained bounds should succeed");
    let exec_order = graph.exec_order().expect("valid DAG");

    // Force the backward Concat path to miss both shape sources:
    // - `ConcatLayer::new(0)` has no explicit input_shapes metadata.
    // - remove one branch from cache so `bounds_cache.get(inp_name)` fails.
    let removed = bounds_cache.remove("linear_b");
    assert!(
        removed.is_some(),
        "test setup failed: expected linear_b bounds in cache"
    );

    let params = super::super::backward::BackwardParams {
        graph: &graph,
        constrained_input: &constrained_input,
        exec_order,
        context: &context,
        beta_state: None,
        objective: None,
        spec_matrix: None,
        seed_cache: None,
        capture_linear_bounds: false,
        deadline: None,
        patches_policy: ConstrainedPatchesPolicy::selective_matrix_reentry(),
    };
    let err = match verifier.backward_crown_constrained(
        &params,
        &mut bounds_cache,
        super::super::backward::BackwardMode::Standard,
    ) {
        Ok(_) => panic!("missing Concat shape metadata should produce error"),
        Err(e) => e,
    };
    let err_msg = err.to_string();
    assert!(
        (err_msg.contains("missing shape") && err_msg.contains("Concat"))
            || (err_msg.contains("Concat") && err_msg.contains("input")),
        "expected Concat-related error (missing shape or input arity), got: {}",
        err_msg
    );
}

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

// =========================================================================
// Regression test for #2398: 3-input Concat routes through n-ary handler
// =========================================================================

/// Build a graph with three parallel Linear branches feeding into a single Concat.
///
/// ```text
///   _input ──┬── linear_a ──┐
///            ├── linear_b ──┤── concat ── linear_out
///            └── linear_c ──┘
/// ```
///
/// Each linear produces a 1-element output; concat on axis 0 produces a 3-element vector.
fn build_three_input_concat_graph() -> GraphNetwork {
    let linear_a = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.1]))).expect("valid linear_a");
    let linear_b = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[-0.2]))).expect("valid linear_b");
    let linear_c = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.3]))).expect("valid linear_c");
    let linear_out =
        LinearLayer::new(arr2(&[[0.5, -0.3, 0.2]]), Some(arr1(&[0.0]))).expect("valid linear_out");

    let concat = ConcatLayer::with_input_shapes(0, vec![vec![1], vec![1], vec![1]]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("linear_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::from_input("linear_c", Layer::Linear(linear_c)));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(concat),
        vec![
            "linear_a".to_string(),
            "linear_b".to_string(),
            "linear_c".to_string(),
        ],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["concat".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

/// Regression test #2398: constraint forward pass must route 3-input Concat through
/// the n-ary handler (propagate_ibp_nary), not the binary handler which only uses
/// the first 2 inputs. Before the fix, the Concat check at line 250 was dead code
/// because is_binary() (which includes Concat) matched first at line 229.
#[test]
fn test_constrained_forward_three_input_concat_uses_nary_path_2398() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_three_input_concat_graph();
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds");

    // No ReLU constraints — just verify Concat forward propagation is correct.
    let history = GraphSplitHistory::new();

    let (cache, _constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("forward constrained bounds should succeed for 3-input concat");

    // Verify concat node produced bounds with all 3 inputs.
    // Input [-1, 1]: linear_a = [-0.9, 1.1], linear_b = [-1.2, 0.8], linear_c = [-0.7, 1.3]
    // concat = [-0.9, -1.2, -0.7] to [1.1, 0.8, 1.3]  (3-element vector)
    let concat_bounds = cache
        .get("concat")
        .expect("concat bounds should exist in forward cache");
    let concat_shape = concat_bounds.shape();
    assert_eq!(
        concat_shape,
        &[3],
        "3-input concat on axis 0 should produce shape [3], got {:?}. \
         If shape is [2], the binary path was taken instead of n-ary (#2398).",
        concat_shape,
    );

    // Verify individual values from all 3 branches.
    let lower = concat_bounds.lower();
    let upper = concat_bounds.upper();
    // linear_a: 1.0 * x + 0.1, x in [-1, 1] => [-0.9, 1.1]
    assert!(
        (lower[[0]] - (-0.9)).abs() < TOL,
        "concat lower[0] = linear_a lower"
    );
    assert!(
        (upper[[0]] - 1.1).abs() < TOL,
        "concat upper[0] = linear_a upper"
    );
    // linear_b: 1.0 * x + (-0.2), x in [-1, 1] => [-1.2, 0.8]
    assert!(
        (lower[[1]] - (-1.2)).abs() < TOL,
        "concat lower[1] = linear_b lower"
    );
    assert!(
        (upper[[1]] - 0.8).abs() < TOL,
        "concat upper[1] = linear_b upper"
    );
    // linear_c: 1.0 * x + 0.3, x in [-1, 1] => [-0.7, 1.3]
    assert!(
        (lower[[2]] - (-0.7)).abs() < TOL,
        "concat lower[2] = linear_c lower"
    );
    assert!(
        (upper[[2]] - 1.3).abs() < TOL,
        "concat upper[2] = linear_c upper"
    );

    // Verify output node bounds are correct.
    // linear_out: [0.5, -0.3, 0.2] . concat + 0.0
    // lower = min(0.5*[-0.9,1.1]) + min(-0.3*[-1.2,0.8]) + min(0.2*[-0.7,1.3])
    //       = 0.5*(-0.9) + (-0.3)*0.8 + 0.2*(-0.7) = -0.45 + -0.24 + -0.14 = -0.83
    // upper = 0.5*1.1 + (-0.3)*(-1.2) + 0.2*1.3 = 0.55 + 0.36 + 0.26 = 1.17
    let output_bounds = cache
        .get("linear_out")
        .expect("linear_out bounds should exist in forward cache");
    let (out_lo, out_hi) = scalar_interval(output_bounds);
    assert!(
        (out_lo - (-0.83)).abs() < TOL,
        "linear_out lower should be -0.83, got {}",
        out_lo
    );
    assert!(
        (out_hi - 1.17).abs() < TOL,
        "linear_out upper should be 1.17, got {}",
        out_hi
    );
}

// =========================================================================
// Regression test for #ml4acopf-genbab: n-ary bias wrapper width
// =========================================================================

/// The constrained n-ary dispatch accumulates a zero-A bias wrapper under
/// NETWORK_INPUT. Its column count must be the network-input width — the old
/// code used the FIRST CHILD bound's ncols, which is only right when the
/// n-ary node's first input IS the network input. On ml4acopf every
/// constrained pass hit the output `Concat` (first child 38 cols vs input 22),
/// `CrownMergeAccumulator` widened the input accumulator to infinities, and
/// every BaB domain bound came back [-inf, +inf].
///
/// Miniature of the ml4acopf topology: Concat whose first input is a WIDER
/// intermediate node (3 cols) than the network input (2 cols). The network is
/// affine, so constrained CROWN must be exact — and above all FINITE.
#[test]
fn test_constrained_backward_concat_bias_wrapper_uses_input_width_ml4acopf() {
    let linear_a = LinearLayer::new(
        arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        Some(arr1(&[0.5, -0.5, 1.0])),
    )
    .expect("valid linear_a");
    let linear_b = LinearLayer::new(arr2(&[[1.0, -1.0]]), Some(arr1(&[0.25]))).expect("valid b");
    let linear_out =
        LinearLayer::new(arr2(&[[1.0, 2.0, -1.0, 3.0]]), Some(arr1(&[0.0]))).expect("valid out");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("wide_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("narrow_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["wide_a".to_string(), "narrow_b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["concat".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
        .expect("valid input bounds");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);
    let (output, _cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("constrained propagation must succeed");

    let (lo, hi) = scalar_interval(&output);
    assert!(
        lo.is_finite() && hi.is_finite(),
        "bias wrapper width mismatch must not widen the input accumulator \
         to infinities: got [{lo}, {hi}]"
    );
    // y = (x0+0.5) + 2(x1-0.5) - (x0+x1+1.0) + 3(x0-x1+0.25) = 3*x0 - 2*x1 - 0.75.
    // Over x in [-1,1]^2: [-5.75, 4.25]. Affine network => CROWN is exact.
    assert!(
        (lo - (-5.75)).abs() < 1e-4,
        "expected exact affine lower -5.75, got {lo}"
    );
    assert!(
        (hi - 4.25).abs() < 1e-4,
        "expected exact affine upper 4.25, got {hi}"
    );
}
