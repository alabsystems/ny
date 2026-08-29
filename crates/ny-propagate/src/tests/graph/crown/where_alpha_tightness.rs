// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG α-CROWN Where tightness regressions: on a sign-provable Where condition,
//! α-CROWN must be at least as tight as fixed-slope graph CROWN, which routes
//! the whole output to the proven branch instead of concretizing the union.
//!
//! Regression shape: `Where(CompareTensor(x, 0), x, MulBinary(x, s))` with no
//! optimizable activations. The alpha entry early-returns into the BATCHED
//! CROWN backward; `MulBinary` blocklists per-node CROWN-IBP intermediates, so
//! the Where node's forward bound is the condition-blind IBP branch UNION, and
//! the batched backward's generic partial-CROWN fallback concretized through
//! it — ~1e6x looser than fixed-slope CROWN on the identical graph. The fix
//! routes the same `cond_all_true`/`cond_all_false` decision the DAG-CROWN
//! Where arm uses into the batched dispatch; a NOT-provable condition must
//! keep the sound branch union.

use crate::layers::{CompareLayer, CompareOp, CompareTensorLayer, MulBinaryLayer};
use crate::*;
use ndarray::arr1;

/// "if x >= 0 then x else s*x" with the condition and scaled branch built the
/// way an ONNX import produces them: cond = CompareTensor(x, zero-node) and
/// s*x = MulBinary(x, s_const-node). The MulBinary node is what routes the
/// alpha path into the batched CROWN backward (no per-node CROWN-IBP).
fn build_sign_select_graph_mul_binary(s: f32) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "zero",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "s_const",
        Layer::AddConstant(AddConstantLayer::new(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[]),
            s,
        ))),
        vec!["zero".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::CompareTensor(CompareTensorLayer::new(CompareOp::Ge)),
        vec!["x".to_string(), "zero".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sx",
        Layer::MulBinary(MulBinaryLayer),
        vec!["x".to_string(), "s_const".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "sx".to_string()],
    ));
    graph.set_output("out");
    graph
}

/// Same select with a unary threshold Compare and a MulConstant branch — this
/// variant keeps per-node CROWN-IBP intermediates eligible, guarding the other
/// alpha route to the same contract.
fn build_sign_select_graph_unary(s: f32) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::Compare(CompareLayer::new(0.0, CompareOp::Ge)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sx",
        Layer::MulConstant(MulConstantLayer::scalar(s)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "sx".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn max_width(bounds: &BoundedTensor) -> f32 {
    let flat = bounds.flatten();
    flat.upper()
        .iter()
        .zip(flat.lower().iter())
        .map(|(u, l)| u - l)
        .fold(0.0_f32, f32::max)
}

/// The α-CROWN ≥-as-tight-as-CROWN contract, plus containment of the true
/// output value (soundness witness at the given point).
fn assert_alpha_no_looser_than_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    true_value: f32,
    label: &str,
) {
    let crown = graph.propagate_crown_fixed_slope(input).unwrap();
    let alpha = graph
        .propagate_alpha_crown_with_config(input, &AlphaCrownConfig::default())
        .unwrap();

    eprintln!(
        "{label}: CROWN width={:.3e}  alpha width={:.3e}",
        max_width(&crown),
        max_width(&alpha)
    );
    let crown_flat = crown.flatten();
    let alpha_flat = alpha.flatten();
    assert_eq!(
        crown_flat.lower().len(),
        alpha_flat.lower().len(),
        "{label}"
    );
    for dim in 0..crown_flat.lower().len() {
        assert!(
            alpha_flat.lower()[[dim]] >= crown_flat.lower()[[dim]] - 1e-5,
            "{label}: alpha lower[{dim}]={} looser than CROWN lower {}",
            alpha_flat.lower()[[dim]],
            crown_flat.lower()[[dim]]
        );
        assert!(
            alpha_flat.upper()[[dim]] <= crown_flat.upper()[[dim]] + 1e-5,
            "{label}: alpha upper[{dim}]={} looser than CROWN upper {}",
            alpha_flat.upper()[[dim]],
            crown_flat.upper()[[dim]]
        );
    }
    assert!(
        true_value >= alpha_flat.lower()[[0]] - 1e-5
            && true_value <= alpha_flat.upper()[[0]] + 1e-5,
        "{label}: true output {} outside alpha bounds [{}, {}]",
        true_value,
        alpha_flat.lower()[[0]],
        alpha_flat.upper()[[0]]
    );
    // The routed bound must actually be tight (the proven branch, not the
    // ~0.8-wide branch union the regression returned).
    assert!(
        max_width(&alpha) < 1e-3,
        "{label}: alpha width {} is not the routed single-branch bound",
        max_width(&alpha)
    );
}

/// Condition provably TRUE over the box: alpha must match the true branch.
#[ntest::timeout(30000)]
#[test]
fn test_dag_alpha_where_sign_provable_true_no_looser_than_crown() {
    let input =
        BoundedTensor::new(arr1(&[0.9_f32]).into_dyn(), arr1(&[0.9_f32]).into_dyn()).unwrap();
    assert_alpha_no_looser_than_crown(
        &build_sign_select_graph_mul_binary(0.1),
        &input,
        0.9,
        "Where cond all-true (batched alpha route)",
    );
    assert_alpha_no_looser_than_crown(
        &build_sign_select_graph_unary(0.1),
        &input,
        0.9,
        "Where cond all-true (per-node CROWN-IBP route)",
    );
}

/// Condition provably FALSE over the box: alpha must match the false branch.
#[ntest::timeout(30000)]
#[test]
fn test_dag_alpha_where_sign_provable_false_no_looser_than_crown() {
    let input =
        BoundedTensor::new(arr1(&[-0.9_f32]).into_dyn(), arr1(&[-0.9_f32]).into_dyn()).unwrap();
    assert_alpha_no_looser_than_crown(
        &build_sign_select_graph_mul_binary(0.1),
        &input,
        -0.09,
        "Where cond all-false (batched alpha route)",
    );
    assert_alpha_no_looser_than_crown(
        &build_sign_select_graph_unary(0.1),
        &input,
        -0.09,
        "Where cond all-false (per-node CROWN-IBP route)",
    );
}

/// A NOT-provable condition must keep the sound branch union: both branch
/// values over the whole box stay enclosed (no unsound tightening).
#[ntest::timeout(30000)]
#[test]
fn test_dag_alpha_where_sign_not_provable_keeps_branch_union() {
    let graph = build_sign_select_graph_mul_binary(0.1);
    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let alpha = graph
        .propagate_alpha_crown_with_config(&input, &AlphaCrownConfig::default())
        .unwrap();
    let alpha_flat = alpha.flatten();

    // True output at sampled points of the box: x for x >= 0, 0.1*x otherwise.
    for x in [-1.0_f32, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0] {
        let y = if x >= 0.0 { x } else { 0.1 * x };
        assert!(
            y >= alpha_flat.lower()[[0]] - 1e-5 && y <= alpha_flat.upper()[[0]] + 1e-5,
            "mixed cond: true output {} at x={} outside alpha bounds [{}, {}]",
            y,
            x,
            alpha_flat.lower()[[0]],
            alpha_flat.upper()[[0]]
        );
    }
    // The union must genuinely remain: both extremes are reachable
    // (x=1 -> 1, x=-1 -> -0.1), so the bound cannot be tighter than that hull.
    assert!(
        alpha_flat.lower()[[0]] <= -0.1 + 1e-5 && alpha_flat.upper()[[0]] >= 1.0 - 1e-5,
        "mixed cond must keep the reachable hull, got [{}, {}]",
        alpha_flat.lower()[[0]],
        alpha_flat.upper()[[0]]
    );
}
