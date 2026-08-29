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

use ndarray::{Array1, Array2, ArrayD};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use crate::bounds::BatchedLinearBounds;
use crate::layers::binary_ops::BilinearCrownLayer;
use crate::layers::{
    CompareLayer, CompareOp, ConcatLayer, Layer, LinearLayer, MulBinaryLayer, ReLULayer, SubLayer,
    WhereLayer,
};
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
        Layer::Linear(LinearLayer::new(Array2::zeros((4, 4)), Some(Array1::zeros(4))).unwrap()),
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
        lower_a: Array1::zeros(4).into_dyn(),
        upper_a: Array1::zeros(4).into_dyn(),
        lower_b: Array1::zeros(4).into_dyn(),
        upper_b: Array1::zeros(4).into_dyn(),
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

// ============================================================================
// New batched dispatcher arms: Sub, Concat, Where tiers.
//
// Every test below asserts the two properties an added relaxation arm must
// have, on the SAME graph:
//   (a) ENCLOSURE — the batched bound contains the concrete evaluation at
//       sampled points of the input box (soundness);
//   (b) NO LOOSER THAN DENSE — the batched bound is at least as tight as the
//       dense fixed-slope CROWN path, which is the reference implementation
//       these arms mirror.
// Several also pin the tightness the arm buys, by comparing against plain IBP
// (which is what the previous generic fallback concretized through).
// ============================================================================

/// Interval bounds `[lo, hi]` repeated over `shape`.
fn uniform_interval(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(ArrayD::from_elem(shape, lo), ArrayD::from_elem(shape, hi))
        .expect("uniform interval should construct")
}

/// A `LinearLayer` from a row-major weight matrix and bias.
fn linear(rows: usize, cols: usize, weights: &[f32], bias: &[f32]) -> Layer {
    Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((rows, cols), weights.to_vec()).expect("weight shape"),
            Some(Array1::from_vec(bias.to_vec())),
        )
        .expect("linear layer should construct"),
    )
}

/// Add a `cond` node whose IBP bounds are an EXACTLY degenerate 0/1 mask
/// `[0, 1]` over a `[2]` input box of `[-1, 1]`.
///
/// A zero-weight `Linear` is NOT usable here: `LinearLayer`'s sound IBP widens
/// by `in_features + 2` ULPs, so its "constant" output is `[1-1ulp, 1+1ulp]` —
/// non-degenerate, and both the dense and the batched constant-mask tiers
/// (which key on `lower == upper`) correctly refuse it. `Compare` instead emits
/// exactly `(0.0, 0.0)` / `(1.0, 1.0)` when the input interval is entirely on
/// one side of the threshold, which is what a real lowered mask looks like.
fn add_exact_mask_condition(graph: &mut GraphNetwork) {
    // shift = x + [-100, +100] ⇒ position 0 ∈ [-101, -99], position 1 ∈ [99, 101].
    graph.add_node(GraphNode::from_input(
        "mask_shift",
        linear(2, 2, &[1.0, 0.0, 0.0, 1.0], &[-100.0, 100.0]),
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::Compare(CompareLayer::new(0.0, CompareOp::Gt)),
        vec!["mask_shift".to_string()],
    ));
}

/// Assert (a) enclosure at sampled corners/interior of `[lo, hi]^n` and
/// (b) that `batched` is no looser than `dense`.
///
/// `eval` maps a concrete input vector to the concrete graph output.
fn assert_encloses_and_no_looser(
    batched: &BoundedTensor,
    dense: &BoundedTensor,
    lo: f32,
    hi: f32,
    dims: usize,
    eval: impl Fn(&[f32]) -> Vec<f32>,
    label: &str,
) {
    const TOL: f32 = 1e-4;

    // (b) no looser than the dense reference on the same graph.
    for (index, (&b_lo, &d_lo)) in batched.lower().iter().zip(dense.lower().iter()).enumerate() {
        assert!(
            b_lo >= d_lo - TOL,
            "{label}: batched lower[{index}] = {b_lo} is LOOSER than dense {d_lo}"
        );
    }
    for (index, (&b_hi, &d_hi)) in batched.upper().iter().zip(dense.upper().iter()).enumerate() {
        assert!(
            b_hi <= d_hi + TOL,
            "{label}: batched upper[{index}] = {b_hi} is LOOSER than dense {d_hi}"
        );
    }

    // (a) enclosure of sampled concrete evaluations over a grid of the box.
    let steps = 5usize;
    let total = steps.pow(dims as u32);
    for sample in 0..total {
        let mut point = Vec::with_capacity(dims);
        let mut remaining = sample;
        for _ in 0..dims {
            let step = remaining % steps;
            remaining /= steps;
            point.push(lo + (hi - lo) * (step as f32) / ((steps - 1) as f32));
        }
        let concrete = eval(&point);
        for (index, &value) in concrete.iter().enumerate() {
            let b_lo = batched.lower().iter().nth(index).copied().expect("lower");
            let b_hi = batched.upper().iter().nth(index).copied().expect("upper");
            assert!(
                value >= b_lo - TOL && value <= b_hi + TOL,
                "{label}: concrete f({point:?})[{index}] = {value} escapes batched \
                 bound [{b_lo}, {b_hi}]"
            );
        }
    }
}

/// Sub arm: `diff = (2x + 1) - (x - 1) = x + 2`.
///
/// Without a batched `Sub` arm the node fell through to the generic unary
/// dispatch, whose partial-CROWN fallback concretizes `diff` through its IBP
/// forward — the interval difference `[a_lo - b_hi, a_hi - b_lo]`, which throws
/// away the fact that both operands are the SAME `x`. On this graph that is
/// `[-1, 5]` instead of the exact `[1, 3]`.
#[test]
fn sub_batched_arm_is_exact_encloses_samples_and_matches_dense() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear(2, 2, &[2.0, 0.0, 0.0, 2.0], &[1.0, 1.0]),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear(2, 2, &[1.0, 0.0, 0.0, 1.0], &[-1.0, -1.0]),
    ));
    graph.add_node(GraphNode::binary("diff", Layer::Sub(SubLayer), "a", "b"));
    graph.set_output("diff");

    let input = uniform_interval(&[2], -1.0, 1.0);
    let batched = graph
        .propagate_crown_batched(&input)
        .expect("batched CROWN should succeed");
    let dense = graph
        .propagate_crown_fixed_slope(&input)
        .expect("dense CROWN should succeed");
    let ibp = graph.propagate_ibp(&input).expect("IBP should succeed");

    assert_encloses_and_no_looser(
        &batched,
        &dense,
        -1.0,
        1.0,
        2,
        |x| vec![x[0] + 2.0, x[1] + 2.0],
        "Sub",
    );

    // The arm actually fired: the exact difference is [1, 3], while the
    // interval difference the old fallback concretized through is [-1, 5].
    assert!(
        (batched.lower()[[0]] - 1.0).abs() < 1e-4 && (batched.upper()[[0]] - 3.0).abs() < 1e-4,
        "Sub batched bound should be the exact [1, 3], got [{}, {}]",
        batched.lower()[[0]],
        batched.upper()[[0]]
    );
    assert!(
        batched.lower()[[0]] > ibp.lower()[[0]] + 1e-4,
        "Sub batched lower {} should beat the IBP difference {}",
        batched.lower()[[0]],
        ibp.lower()[[0]]
    );
}

/// Concat arm: `out = a0 + a1 + b0` where `a = x` and `b = -(x0 + x1)`, so the
/// exact output is identically `0`.
///
/// The implemented batched Concat backward had ZERO production callers, so this
/// join concretized through IBP (`a ∈ [-1,1]^2`, `b ∈ [-2,2]` → `out ∈ [-4,4]`)
/// and destroyed the cancellation.
#[test]
fn concat_batched_arm_preserves_cancellation_and_matches_dense() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear(2, 2, &[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0]),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear(1, 2, &[-1.0, -1.0], &[0.0]),
    ));
    graph.add_node(GraphNode::binary(
        "cat",
        Layer::Concat(ConcatLayer::new(0)),
        "a",
        "b",
    ));
    graph.add_node(GraphNode::new(
        "out",
        linear(1, 3, &[1.0, 1.0, 1.0], &[0.0]),
        vec!["cat".to_string()],
    ));
    graph.set_output("out");

    let input = uniform_interval(&[2], -1.0, 1.0);
    let batched = graph
        .propagate_crown_batched(&input)
        .expect("batched CROWN should succeed");
    let dense = graph
        .propagate_crown_fixed_slope(&input)
        .expect("dense CROWN should succeed");

    assert_encloses_and_no_looser(&batched, &dense, -1.0, 1.0, 2, |_x| vec![0.0], "Concat");

    assert!(
        batched.lower()[[0]] > -1e-3 && batched.upper()[[0]] < 1e-3,
        "Concat batched bound should recover the exact 0, got [{}, {}]",
        batched.lower()[[0]],
        batched.upper()[[0]]
    );
}

/// A Concat join that the batched column split cannot express must stay
/// refused (`NotHandled`), keeping the sound IBP fallback.
///
/// `[2, 2] ++ [2, 2]` along axis 0 puts the two operands in different BATCH
/// positions of the batched carrier, not in different columns; and
/// `product([2,2]) = 4 != 2 = last dim`, so the layer's product-based sizing
/// would cut the coefficient matrix in the wrong place.
#[test]
fn concat_batched_arm_refuses_a_join_it_cannot_express() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "cat",
        Layer::Concat(ConcatLayer::new(0)),
        NETWORK_INPUT,
        NETWORK_INPUT,
    ));
    graph.set_output("cat");
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = GraphNode::binary(
        "cat",
        Layer::Concat(ConcatLayer::new(0)),
        NETWORK_INPUT,
        NETWORK_INPUT,
    );
    let node_lb = BatchedLinearBounds::identity(&[2, 4]).expect("identity bounds");
    let input = make_interval(&[2, 2]);
    let node_bounds = HashMap::new();
    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "cat",
        &node_lb,
        &input,
        &node_bounds,
        &[2, 4],
        MulBinaryRelaxationMode::default(),
        None,
        &mut runtime,
        &mut acc,
    );

    assert!(
        matches!(
            result.expect("guard should refuse, not error"),
            SpecialBatchedDispatchResult::NotHandled
        ),
        "a non-last-axis / product-mismatched Concat join must stay refused"
    );
    assert!(
        acc.is_empty(),
        "a refused Concat must not have accumulated a partially-split carrier"
    );
}

/// Where tier 4 (constant mask): `cond = [1, 0]` is bound-independent with BOTH
/// branches live, so neither `cond_all_true` nor `cond_all_false` fires. The
/// exact per-column select gives `[x0, x1 + 10]`; the condition-blind union the
/// old fallback concretized through gives `[-1, 11]` at BOTH positions.
///
/// A causal / padding attention mask is exactly this shape.
#[test]
fn where_constant_mask_batched_splits_columns_exactly() {
    let mut graph = GraphNetwork::new();
    add_exact_mask_condition(&mut graph);
    graph.add_node(GraphNode::from_input(
        "t",
        linear(2, 2, &[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0]),
    ));
    graph.add_node(GraphNode::from_input(
        "f",
        linear(2, 2, &[1.0, 0.0, 0.0, 1.0], &[10.0, 10.0]),
    ));
    graph.add_node(GraphNode::new(
        "w",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "t".to_string(), "f".to_string()],
    ));
    graph.set_output("w");

    let input = uniform_interval(&[2], -1.0, 1.0);
    let batched = graph
        .propagate_crown_batched(&input)
        .expect("batched CROWN should succeed");
    let dense = graph
        .propagate_crown_fixed_slope(&input)
        .expect("dense CROWN should succeed");

    // mask = [false, true] ⇒ position 0 takes `f` (= x + 10), position 1 takes
    // `t` (= x). The condition-blind union would give [-1, 11] at BOTH.
    assert_encloses_and_no_looser(
        &batched,
        &dense,
        -1.0,
        1.0,
        2,
        |x| vec![x[0] + 10.0, x[1]],
        "Where/constant-mask",
    );

    assert!(
        (batched.lower()[[0]] - 9.0).abs() < 1e-3 && (batched.upper()[[0]] - 11.0).abs() < 1e-3,
        "masked position 0 must select the false branch exactly, got [{}, {}]",
        batched.lower()[[0]],
        batched.upper()[[0]]
    );
    assert!(
        (batched.lower()[[1]] - (-1.0)).abs() < 1e-3 && (batched.upper()[[1]] - 1.0).abs() < 1e-3,
        "masked position 1 must select the true branch exactly, got [{}, {}]",
        batched.lower()[[1]],
        batched.upper()[[1]]
    );
}

/// Where tier 1 (embedded constants): both branches are frozen constants, so
/// the output is the exact per-element select `[5, -5]`.
///
/// The removed bail claimed "the generic fallback concretizes through its
/// forward bound, which already performs the exact constant select". It does
/// not: `WhereLayer::propagate_ibp` → `propagate_ibp_with_condition` →
/// `propagate_ibp_ternary` is the condition-BLIND union `[-5, 5]`. This test
/// pins the exact select the dense arm has always produced.
#[test]
fn where_embedded_constant_batched_folds_the_exact_select_not_the_union() {
    let true_const =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![5.0_f32, 7.0]).expect("true constant");
    let false_const =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-5.0_f32, -3.0]).expect("false constant");

    let mut graph = GraphNetwork::new();
    add_exact_mask_condition(&mut graph);
    graph.add_node(GraphNode::new(
        "w",
        Layer::Where(WhereLayer::with_constants(
            Some(true_const),
            Some(false_const),
        )),
        vec!["cond".to_string()],
    ));
    graph.set_output("w");

    let input = uniform_interval(&[2], -1.0, 1.0);
    let batched = graph
        .propagate_crown_batched(&input)
        .expect("batched CROWN should succeed");
    let dense = graph
        .propagate_crown_fixed_slope(&input)
        .expect("dense CROWN should succeed");
    let ibp = graph.propagate_ibp(&input).expect("IBP should succeed");

    // mask = [false, true] ⇒ the exact select is [-5, 7]; the condition-blind
    // union is [-5, 5] at position 0 and [-3, 7] at position 1.
    assert_encloses_and_no_looser(
        &batched,
        &dense,
        -1.0,
        1.0,
        2,
        |_x| vec![-5.0, 7.0],
        "Where/embedded-constant",
    );

    assert!(
        (batched.lower()[[0]] - (-5.0)).abs() < 1e-4
            && (batched.upper()[[0]] - (-5.0)).abs() < 1e-4,
        "position 0 must fold to the exact constant -5, got [{}, {}]",
        batched.lower()[[0]],
        batched.upper()[[0]]
    );
    assert!(
        (batched.lower()[[1]] - 7.0).abs() < 1e-4 && (batched.upper()[[1]] - 7.0).abs() < 1e-4,
        "position 1 must fold to the exact constant 7, got [{}, {}]",
        batched.lower()[[1]],
        batched.upper()[[1]]
    );
    assert!(
        batched.upper()[[0]] < ibp.upper()[[0]] - 1e-4,
        "the exact select must beat the condition-blind IBP union: batched upper {} \
         vs IBP {}",
        batched.upper()[[0]],
        ibp.upper()[[0]]
    );
}

/// A data-dependent `Where` condition has no exact linear form, so the batched
/// arm must keep refusing it — the sound IBP union fallback stands.
#[test]
fn where_data_dependent_condition_stays_refused() {
    let graph = make_mul_graph();
    let plan = make_test_plan();
    let mut acc = BatchedCrownAccumulator::new(&plan);

    let node = GraphNode::new(
        "w",
        Layer::Where(WhereLayer::new()),
        vec![
            NETWORK_INPUT.to_string(),
            NETWORK_INPUT.to_string(),
            NETWORK_INPUT.to_string(),
        ],
    );
    let node_lb = BatchedLinearBounds::identity(&[4]).expect("identity bounds");
    // Condition straddles the 0.5 decision boundary ⇒ not provable, not constant.
    let input = make_interval(&[4]);
    let node_bounds = HashMap::new();
    let mut runtime = AttentionCompositionRuntime::production();

    let result = graph.dispatch_special_batched_operator(
        &node,
        "w",
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
            result.expect("data-dependent Where should refuse, not error"),
            SpecialBatchedDispatchResult::NotHandled
        ),
        "a data-dependent Where condition must keep the sound union fallback"
    );
    assert!(acc.is_empty(), "a refused Where must accumulate nothing");
}

/// The constant-mask column split is a partition: the two branch carriers
/// re-sum to the incoming carrier exactly (coefficients AND bias), which is the
/// algebraic identity the arm's soundness rests on.
#[test]
fn where_constant_mask_columns_partition_the_carrier() {
    use super::dispatch::mask_batched_bounds_columns_for_test;

    let node_lb = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("lower_a"),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.5, -0.5]).expect("lower_b"),
        ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .expect("upper_a"),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.5, -1.5]).expect("upper_b"),
        vec![3],
        vec![2],
    );
    let mask = [true, false, true];

    let true_lb =
        mask_batched_bounds_columns_for_test(&node_lb, &mask, true).expect("true branch split");
    let false_lb =
        mask_batched_bounds_columns_for_test(&node_lb, &mask, false).expect("false branch split");

    for row in 0..2 {
        for column in 0..3 {
            let sum = true_lb.lower_a()[[row, column]] + false_lb.lower_a()[[row, column]];
            assert_eq!(
                sum,
                node_lb.lower_a()[[row, column]],
                "lower_a[{row},{column}] must be partitioned, not duplicated or dropped"
            );
            let sum = true_lb.upper_a()[[row, column]] + false_lb.upper_a()[[row, column]];
            assert_eq!(
                sum,
                node_lb.upper_a()[[row, column]],
                "upper_a[{row},{column}] must be partitioned"
            );
            // Exactly one branch owns each column.
            assert!(
                true_lb.lower_a()[[row, column]] == 0.0 || false_lb.lower_a()[[row, column]] == 0.0,
                "column {column} must belong to exactly one branch"
            );
        }
        // The bias rides the TRUE branch only — never halved, never doubled.
        assert_eq!(true_lb.lower_b()[[row]], node_lb.lower_b()[[row]]);
        assert_eq!(true_lb.upper_b()[[row]], node_lb.upper_b()[[row]]);
        assert_eq!(false_lb.lower_b()[[row]], 0.0);
        assert_eq!(false_lb.upper_b()[[row]], 0.0);
    }
}

/// The same `Sub` arm with a DOWNSTREAM `Linear`, so the carrier arriving at the
/// `Sub` node carries a certified `γ_n·S` coefficient error.
///
/// This is the shape of every real graph: the batched `Linear` backward attaches
/// a fresh per-coefficient error, so a split that ignored it would publish a
/// bound tighter than the arithmetic justifies. The arm runs the split on the
/// error carrier as well and re-attaches it, which is why the exact
/// cancellation still comes out (`out = 2x0 - 2x1 ∈ [-4, 4]`) instead of the
/// interval-difference fallback's `[-8, 8]`.
#[test]
fn sub_batched_arm_carries_the_certified_coefficient_error() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear(2, 2, &[3.0, 0.0, 0.0, 3.0], &[1.0, 1.0]),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear(2, 2, &[1.0, 0.0, 0.0, 1.0], &[-1.0, -1.0]),
    ));
    graph.add_node(GraphNode::binary("diff", Layer::Sub(SubLayer), "a", "b"));
    graph.add_node(GraphNode::new(
        "out",
        linear(1, 2, &[1.0, -1.0], &[0.0]),
        vec!["diff".to_string()],
    ));
    graph.set_output("out");

    let input = uniform_interval(&[2], -1.0, 1.0);
    let batched = graph
        .propagate_crown_batched(&input)
        .expect("batched CROWN should succeed");
    let dense = graph
        .propagate_crown_fixed_slope(&input)
        .expect("dense CROWN should succeed");
    let ibp = graph.propagate_ibp(&input).expect("IBP should succeed");

    assert_encloses_and_no_looser(
        &batched,
        &dense,
        -1.0,
        1.0,
        2,
        |x| vec![2.0 * x[0] - 2.0 * x[1]],
        "Sub/with-coeff-err",
    );

    assert!(
        (batched.lower()[[0]] - (-4.0)).abs() < 1e-3 && (batched.upper()[[0]] - 4.0).abs() < 1e-3,
        "Sub batched bound should recover the exact [-4, 4], got [{}, {}]",
        batched.lower()[[0]],
        batched.upper()[[0]]
    );
    assert!(
        batched.upper()[[0]] < ibp.upper()[[0]] - 1e-3,
        "the exact difference must beat the IBP interval difference: batched {} vs IBP {}",
        batched.upper()[[0]],
        ibp.upper()[[0]]
    );
    // The certified error survives as an OUTWARD widening, never inward.
    assert!(
        batched.lower()[[0]] <= -4.0 && batched.upper()[[0]] >= 4.0,
        "the coefficient-error penalty must widen outward, got [{}, {}]",
        batched.lower()[[0]],
        batched.upper()[[0]]
    );
}
