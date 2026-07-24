// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #3720: graph CROWN fallback provenance contract.
//!
//! These tests verify that the shared `graph_crown_dispatch_fallback_reason`
//! helper is wired into DAG-CROWN and that recoverable dispatch errors produce
//! correct `BoundsProvenance::ForwardFallback(reason)` provenance.
//!
//! The shared helper mapping is unit-tested in
//! `network/core/graph/fallback_reason.rs`. This module adds public
//! `propagate_crown_with_provenance(...)` regressions for unsupported fallback
//! behavior and for rank-2 LayerNorm staying on the real CROWN path after the
//! multi-dimensional scalar uplift in #4148.

use crate::layers::{
    Conv1dLayer, Layer, LayerNormLayer, LinearLayer, NonZeroLayer, TransposeLayer,
};
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn assert_arrays_close_4148(actual: &ArrayD<f32>, expected: &ArrayD<f32>, tol: f32, context: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: shape mismatch"
    );
    for (idx, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "{context}: idx {idx} actual={actual_value} expected={expected_value} tol={tol}"
        );
    }
}

/// Regression for #3720: DAG-CROWN classifies dispatch errors through
/// the shared `graph_crown_dispatch_fallback_reason` helper and records
/// structured fallback provenance.
///
/// Uses NonZero (data-dependent output) which returns Unsupported from
/// backward dispatch. DAG-CROWN catches this at the Unsupported arm and
/// records `ForwardFallback(CrownPropagationError)`.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_fallback_provenance_unsupported_3720() {
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[0.0_f32, 0.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("nonzero");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let ibp = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed for NonZero graph");
    let crown = graph
        .propagate_crown_with_provenance(&input)
        .expect("DAG-CROWN should fall back for NonZero, not hard-fail");

    assert_eq!(
        crown.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError),
        "NonZero CROWN backward returns Unsupported → CrownPropagationError provenance"
    );
    assert_eq!(
        crown.bounds.lower(),
        ibp.lower(),
        "fallback lower bounds should match IBP"
    );
    assert_eq!(
        crown.bounds.upper(),
        ibp.upper(),
        "fallback upper bounds should match IBP"
    );
}

/// Regression for #4148: rank-2 LayerNorm should stay on the real DAG-CROWN
/// path instead of tripping the old scalar `ShapeMismatch` fallback.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_rank2_layernorm_routes_to_multidim_crown_4148() {
    let mut graph = GraphNetwork::new();

    let ny = arr1(&[1.0_f32, 0.8, 1.2]);
    let beta = arr1(&[0.0_f32, 0.1, -0.1]);
    let layernorm = LayerNormLayer::new(ny, beta, 1e-5).unwrap();
    graph.add_node(GraphNode::from_input(
        "layernorm",
        Layer::LayerNorm(layernorm.clone()),
    ));
    graph.set_output("layernorm");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0_f32, -0.5, 0.25, -0.25, 0.5, 1.0])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0_f32, 0.25, 0.75, 0.5, 1.0, 1.5]).unwrap(),
    )
    .unwrap();

    let crown = graph
        .propagate_crown_with_provenance(&input)
        .expect("DAG-CROWN should use multi-dim LayerNorm CROWN, not fall back");
    let direct = layernorm
        .propagate_linear_batched_with_bounds(
            &BatchedLinearBounds::identity(&[2, 3]).unwrap(),
            &input,
        )
        .expect("direct batched LayerNorm helper should succeed")
        .concretize_sound(&input)
        .expect("direct batched LayerNorm concretization should succeed");

    assert_eq!(
        crown.provenance,
        BoundsProvenance::Crown,
        "rank-2 LayerNorm should no longer report ShapeMismatch fallback"
    );
    // Both paths concretize through the batched path, which now accumulates the
    // dot product in f64 (BLAS DGEMV) and rounds only the single f64->f32 cast.
    // With the conservative `gamma_{2n+2}*S` envelope gone, the two paths agree
    // to ordinary FP slack again — restored to the former 1e-6 tolerance.
    assert_arrays_close_4148(
        crown.bounds.lower(),
        direct.lower(),
        1e-6,
        "graph DAG-CROWN lower bounds should match direct batched LayerNorm helper",
    );
    assert_arrays_close_4148(
        crown.bounds.upper(),
        direct.upper(),
        1e-6,
        "graph DAG-CROWN upper bounds should match direct batched LayerNorm helper",
    );
}

fn build_conv1d_transpose_layernorm_graph_4148() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 3]),
        vec![0.45_f32, -0.2, 0.35, -0.15, 0.4, 0.25],
    )
    .unwrap();
    let conv = Conv1dLayer::with_input_length(kernel, Some(arr1(&[0.05_f32, -0.03])), 1, 1, 4)
        .expect("valid Conv1d params");
    graph.add_node(GraphNode::from_input("conv", Layer::Conv1d(conv)));
    graph.add_node(GraphNode::new(
        "transpose",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        vec!["conv".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(
            LayerNormLayer::new(arr1(&[1.0_f32, 0.8]), arr1(&[0.0_f32, -0.1]), 1e-5)
                .expect("valid LayerNorm"),
        ),
        vec!["transpose".to_string()],
    ));
    graph.set_output("layernorm");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-0.8_f32, -0.3, 0.2, 0.7]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.1_f32, 0.6, 1.0, 1.4]).unwrap(),
    )
    .unwrap();

    (graph, input)
}

/// Regression for #4148 on the exact Kokoro-shaped route:
/// Conv1d -> Transpose -> LayerNorm should keep the LayerNorm node and graph
/// output on the real CROWN path.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_conv1d_transpose_layernorm_node_stays_on_crown_4148() {
    let (graph, input) = build_conv1d_transpose_layernorm_graph_4148();

    let ibp_nodes = graph
        .collect_node_bounds(&input)
        .expect("IBP node collection should succeed");
    let node_result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_nodes.clone(), None)
        .expect("CROWN-IBP node collection should succeed");
    let output = graph
        .propagate_crown_with_provenance(&input)
        .expect("graph DAG-CROWN should succeed");
    let ibp_output = graph.propagate_ibp(&input).expect("IBP should succeed");

    assert_eq!(
        node_result.provenance_for_node("layernorm"),
        Some(BoundsProvenance::Crown),
        "Conv1d -> Transpose -> LayerNorm should keep the LayerNorm node on CROWN"
    );
    assert_eq!(
        output.provenance,
        BoundsProvenance::Crown,
        "the graph output should stay on the CROWN path for this LayerNorm route"
    );

    let node_crown = node_result
        .bounds
        .get("layernorm")
        .expect("layernorm node bounds missing");
    let node_ibp = ibp_nodes
        .get("layernorm")
        .expect("layernorm IBP bounds missing");
    assert_eq!(
        node_crown.shape(),
        node_ibp.shape(),
        "LayerNorm node CROWN and IBP bounds should agree on shape"
    );
    assert!(
        node_crown
            .lower()
            .iter()
            .chain(node_crown.upper().iter())
            .all(|value| value.is_finite()),
        "LayerNorm node CROWN bounds should stay finite on the multi-dim route"
    );
    assert_eq!(
        output.bounds.shape(),
        ibp_output.shape(),
        "graph output CROWN and IBP bounds should agree on shape"
    );
    assert!(
        output
            .bounds
            .lower()
            .iter()
            .chain(output.bounds.upper().iter())
            .all(|value| value.is_finite()),
        "graph output CROWN bounds should stay finite on the multi-dim route"
    );
}

/// Packet C (#4171): The transpose node itself must stay on exact CROWN and
/// produce finite, correctly-shaped bounds. Before #4171 Packet A, grouped
/// multi-dimensional bounds triggered UnsupportedConfiguration fallback.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_transpose_provenance_and_finiteness_4171() {
    let (graph, input) = build_conv1d_transpose_layernorm_graph_4148();

    let ibp_nodes = graph
        .collect_node_bounds(&input)
        .expect("IBP node collection should succeed");
    let node_result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_nodes.clone(), None)
        .expect("CROWN-IBP node collection should succeed");

    assert_eq!(
        node_result.provenance_for_node("transpose"),
        Some(BoundsProvenance::Crown),
        "Conv1d -> Transpose: the transpose node itself should stay on exact CROWN (#4171)"
    );

    let transpose_crown = node_result
        .bounds
        .get("transpose")
        .expect("transpose node bounds missing from CROWN-IBP collection");
    let transpose_ibp = ibp_nodes
        .get("transpose")
        .expect("transpose node bounds missing from IBP collection");
    assert_eq!(
        transpose_crown.shape(),
        transpose_ibp.shape(),
        "transpose node CROWN and IBP bounds should agree on shape"
    );
    assert!(
        transpose_crown
            .lower()
            .iter()
            .chain(transpose_crown.upper().iter())
            .all(|v| v.is_finite()),
        "transpose node CROWN bounds should be finite after exact flat-grouped backward (#4171)"
    );
}
