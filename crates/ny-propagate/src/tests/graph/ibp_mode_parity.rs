// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #2706 IBP mode parity.

use crate::domain_clip::DomainClipper;
use crate::*;
use ndarray::{arr1, Array2};
use ny_core::NyError;

fn assert_close(actual: f32, expected: f32, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= 1e-6,
        "{label}: expected {expected:.6e}, got {actual:.6e} (diff {diff:.6e})"
    );
}

fn assert_bounds_equal(actual: &BoundedTensor, expected: &BoundedTensor, label: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape mismatch actual={:?} expected={:?}",
        actual.shape(),
        expected.shape()
    );
    assert_eq!(actual.lower(), expected.lower(), "{label}: lower mismatch");
    assert_eq!(actual.upper(), expected.upper(), "{label}: upper mismatch");
}

fn bounds_min_max(bounds: &BoundedTensor) -> (f32, f32) {
    bounds.lower().iter().zip(bounds.upper().iter()).fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_bound, max_bound), (&l, &u)| (min_bound.min(l), max_bound.max(u)),
    )
}

fn assert_output_info_matches_bounds(
    node_name: &str,
    info: &NodeBoundsInfo,
    bounds: &BoundedTensor,
    expected_layer_type: &str,
    label: &str,
) {
    let (min_bound, max_bound) = bounds_min_max(bounds);
    assert_eq!(info.name, node_name, "{label}: unexpected output node");
    assert_eq!(
        info.layer_type, expected_layer_type,
        "{label}: unexpected output layer type"
    );
    assert_eq!(
        info.output_shape,
        bounds.shape().to_vec(),
        "{label}: output shape mismatch"
    );
    assert!(!info.has_nan, "{label}: output should stay NaN-free");
    assert!(
        !info.has_infinite,
        "{label}: output should stay finite, got {:?}",
        info
    );
    assert!(!info.saturated, "{label}: output should not saturate");
    assert_close(
        info.output_width,
        bounds.max_width(),
        &format!("{label} output_width"),
    );
    assert_close(info.min_bound, min_bound, &format!("{label} min_bound"));
    assert_close(info.max_bound, max_bound, &format!("{label} max_bound"));
}

fn build_single_block_swiglu_graph_2706() -> (GraphNetwork, BoundedTensor, f32) {
    let hidden = 3;
    let epsilon = 0.05_f32;
    let mut graph = GraphNetwork::new();

    let up_weights = Array2::<f32>::from_shape_fn((hidden, hidden), |(i, j)| {
        0.2 + ((i * 3 + j * 5) as f32) * 0.05
    });
    let up_bias = arr1(&[0.05_f32, -0.02, 0.03]);
    graph.add_node(GraphNode::from_input(
        "layer0_ffn_up",
        Layer::Linear(
            LinearLayer::new(up_weights, Some(up_bias))
                .expect("single-block SwiGLU up projection should be valid"),
        ),
    ));

    let gate_weights = Array2::<f32>::from_shape_fn((hidden, hidden), |(i, j)| {
        -0.15 + ((i * 7 + j * 2) as f32) * 0.04
    });
    let gate_bias = arr1(&[-0.01_f32, 0.04, -0.03]);
    graph.add_node(GraphNode::from_input(
        "layer0_ffn_gate",
        Layer::Linear(
            LinearLayer::new(gate_weights, Some(gate_bias))
                .expect("single-block SwiGLU gate projection should be valid"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "layer0_silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["layer0_ffn_gate".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "layer0_swiglu",
        Layer::MulBinary(MulBinaryLayer),
        "layer0_ffn_up",
        "layer0_silu",
    ));
    graph.set_output("layer0_swiglu");

    let input = BoundedTensor::from_epsilon(Array2::<f32>::zeros((1, hidden)).into_dyn(), epsilon)
        .expect("single-block SwiGLU input should be valid");

    (graph, input, epsilon)
}

fn assert_block_wise_matches_bounds(
    result: &BlockWiseResult,
    expected: &BoundedTensor,
    label: &str,
) {
    assert_eq!(result.total_blocks, 1, "{label}: expected one block");
    assert_eq!(
        result.degraded_blocks, 0,
        "{label}: no degraded blocks expected"
    );
    let block = result
        .blocks
        .first()
        .unwrap_or_else(|| panic!("{label}: block-wise result should contain one block"));
    assert_eq!(block.block_name, "layer0", "{label}: unexpected block name");
    assert_close(
        block.output_width,
        expected.max_width(),
        &format!("{label} output_width"),
    );
    let swiglu_width = block
        .swiglu_width
        .unwrap_or_else(|| panic!("{label}: block-wise SwiGLU path should record zonotope width"));
    assert_close(
        swiglu_width,
        expected.max_width(),
        &format!("{label} swiglu_width"),
    );
    let block_output = block
        .nodes
        .last()
        .unwrap_or_else(|| panic!("{label}: block-wise result should include the output node"));
    assert_output_info_matches_bounds("layer0_swiglu", block_output, expected, "MulBinary", label);
}

fn assert_numerical_instability(err: NyError, label: &str) {
    match err {
        NyError::NumericalInstability(message) => {
            assert!(
                message.contains("NaN"),
                "{label}: expected NaN diagnostic, got: {message}"
            );
        }
        other => panic!("{label}: expected NumericalInstability, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_modes_match_on_single_block_swiglu_graph_2706() {
    let (graph, input, epsilon) = build_single_block_swiglu_graph_2706();

    let ibp = graph
        .propagate_ibp(&input)
        .expect("#2706 baseline IBP should succeed on single-block SwiGLU");

    let detailed = graph
        .propagate_ibp_detailed(&input, epsilon)
        .expect("#2706 detailed IBP should match baseline on single-block SwiGLU");
    assert_eq!(detailed.total_nodes, 4, "#2706 expected 4 graph nodes");
    let detailed_output = detailed
        .nodes
        .last()
        .expect("#2706 detailed result should include the output node");
    assert_output_info_matches_bounds(
        "layer0_swiglu",
        detailed_output,
        &ibp,
        "MulBinary",
        "#2706 detailed",
    );

    let mut clipper = DomainClipper::default();
    let clipped = graph
        .propagate_ibp_with_clipper(&input, &mut clipper)
        .expect("#2706 clipper IBP should stay sound on the single-block SwiGLU graph");
    let raw_up = graph
        .nodes
        .get("layer0_ffn_up")
        .expect("#2706 graph must include layer0_ffn_up")
        .layer
        .propagate_ibp(&input)
        .expect("#2706 raw up projection should succeed");
    let raw_gate = graph
        .nodes
        .get("layer0_ffn_gate")
        .expect("#2706 graph must include layer0_ffn_gate")
        .layer
        .propagate_ibp(&input)
        .expect("#2706 raw gate projection should succeed");
    let raw_silu = graph
        .nodes
        .get("layer0_silu")
        .expect("#2706 graph must include layer0_silu")
        .layer
        .propagate_ibp(&raw_gate)
        .expect("#2706 raw SiLU propagation should succeed");
    let raw_mul = graph
        .nodes
        .get("layer0_swiglu")
        .expect("#2706 graph must include layer0_swiglu")
        .layer
        .propagate_ibp_binary(&raw_up, &raw_silu)
        .expect("#2706 raw MulBinary propagation should succeed");
    assert_bounds_equal(&clipped, &raw_mul, "#2706 clipper raw interval");
    let differs_from_tightened = raw_mul
        .lower()
        .iter()
        .zip(raw_mul.upper().iter())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
        .any(|((&raw_lower, &raw_upper), (&tight_lower, &tight_upper))| {
            (raw_lower - tight_lower).abs() > 1e-6 || (raw_upper - tight_upper).abs() > 1e-6
        });
    assert!(
        differs_from_tightened,
        "#2706 clipper regression graph should exercise a real difference from tightened baseline"
    );

    let block_wise = graph
        .propagate_ibp_block_wise(&input, epsilon)
        .expect("#2706 block-wise IBP should match baseline on a single block");
    assert_block_wise_matches_bounds(&block_wise, &ibp, "#2706 block-wise");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_modes_fail_fast_on_nan_input_2706() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let nan_input = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, -1.0_f32]).into_dyn(),
        arr1(&[f32::NAN, 1.0_f32]).into_dyn(),
    )
    .expect("#2706 test input should allow NaN bounds");

    let baseline_err = graph
        .propagate_ibp(&nan_input)
        .expect_err("#2706 baseline IBP must fail fast on NaN input");
    assert_numerical_instability(baseline_err, "#2706 baseline");

    let detailed_err = graph
        .propagate_ibp_detailed(&nan_input, 0.1)
        .expect_err("#2706 detailed IBP must fail fast on NaN input");
    assert_numerical_instability(detailed_err, "#2706 detailed");

    let mut clipper = DomainClipper::default();
    let clipper_err = graph
        .propagate_ibp_with_clipper(&nan_input, &mut clipper)
        .expect_err("#2706 clipper IBP must fail fast on NaN input");
    assert_numerical_instability(clipper_err, "#2706 clipper");

    let block_wise_err = graph
        .propagate_ibp_block_wise(&nan_input, 0.1)
        .expect_err("#2706 block-wise IBP fallback must fail fast on NaN input");
    assert_numerical_instability(block_wise_err, "#2706 block-wise");
}

/// Regression test for #2585: the clipper path must reject a node whose IBP
/// output contains NaN before `clip_bounds()` or `bounds_cache` can observe it.
///
/// ReLU IBP preserves NaN via `nan_propagating_max_zero`, and the first
/// interception layer is `BoundedTensor::new_allow_infinite` inside the ReLU
/// layer itself.  The graph-level `check_nan_firewall()` is a second defence
/// that would trigger if a layer ever bypassed the tensor constructor check.
/// Either interception satisfies the #2585 contract: NaN never reaches the
/// bounds cache or the clipper.
#[ntest::timeout(10000)]
#[test]
fn test_graph_network_ibp_with_clipper_rejects_nan_node_output_2585() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let nan_input = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, -1.0_f32]).into_dyn(),
        arr1(&[f32::NAN, 1.0_f32]).into_dyn(),
    )
    .expect("#2585 test input should allow NaN bounds");

    let mut clipper = DomainClipper::default();
    let err = graph
        .propagate_ibp_with_clipper(&nan_input, &mut clipper)
        .expect_err("#2585 clipper IBP must fail before clipping poisoned node output");

    match err {
        NyError::NumericalInstability(message) => {
            assert!(
                message.contains("NaN"),
                "#2585 expected NaN diagnostic, got: {message}"
            );
        }
        other => panic!("#2585 expected NumericalInstability, got {other:?}"),
    }

    assert_eq!(
        clipper.clip_count, 0,
        "#2585 NaN firewall must reject before DomainClipper::clip_bounds runs"
    );
}

/// Regression test for #3768: `collect_node_bounds` is the 5th IBP path and was
/// missing the NaN firewall that the other 4 paths share.
#[ntest::timeout(10000)]
#[test]
fn test_collect_node_bounds_nan_input_returns_numerical_instability_3768() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let nan_input = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, -1.0_f32]).into_dyn(),
        arr1(&[f32::NAN, 1.0_f32]).into_dyn(),
    )
    .expect("#3768 test input should allow NaN bounds");

    let collect_err = graph
        .collect_node_bounds(&nan_input)
        .expect_err("#3768 collect_node_bounds must fail fast on NaN input");
    assert_numerical_instability(collect_err, "#3768 collect_node_bounds");
}
