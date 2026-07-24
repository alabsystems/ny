// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #4146 batched unary fallback behavior.

use crate::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;

fn assert_bounds_contain_ibp_reference(
    crown_bounds: &BoundedTensor,
    ibp_bounds: &BoundedTensor,
    context: &str,
) {
    assert_eq!(
        crown_bounds.shape(),
        ibp_bounds.shape(),
        "{context}: batched CROWN shape must match IBP"
    );

    for (idx, ((&crown_l, &crown_u), (&ibp_l, &ibp_u))) in crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
        .enumerate()
    {
        assert!(
            crown_l <= ibp_l,
            "{context}: lower[{idx}] {crown_l} must be <= IBP lower {ibp_l}"
        );
        assert!(
            crown_u >= ibp_u,
            "{context}: upper[{idx}] {crown_u} must be >= IBP upper {ibp_u}"
        );
    }
}

fn resize_shape_mismatch_input_4146() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-0.5_f32, -0.25, 0.0, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.75_f32, 0.5, 0.4, 0.9]).unwrap(),
    )
    .unwrap()
}

fn assert_resize_direct_shape_mismatch_4146(resize: &ResizeLayer, input: &BoundedTensor) {
    let resize_bounds = BatchedLinearBounds::identity(&[1, 4, 4]).unwrap();
    let err = resize
        .propagate_linear_batched(&resize_bounds, input)
        .expect_err("resize should still emit ShapeMismatch for this reduced-column setup");
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(
                expected,
                vec![16],
                "resize output should flatten to 16 columns"
            );
            assert_eq!(
                got,
                vec![4],
                "layernorm backward should keep only the last axis"
            );
        }
        other => {
            panic!("expected ShapeMismatch from direct resize batched CROWN setup, got {other}")
        }
    }
}

fn build_resize_layernorm_graph_4146(resize: ResizeLayer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("resize", Layer::Resize(resize)));
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(LayerNormLayer::new_default(4, 1e-5).unwrap()),
        vec!["resize".into()],
    ));
    graph.set_output("layernorm");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_transpose_layernorm_succeeds_4146() {
    tests::with_crown_dense_budget_mb("2048", || {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "transpose",
            Layer::Transpose(TransposeLayer::batched_transpose()),
        ));
        graph.add_node(GraphNode::new(
            "layernorm",
            Layer::LayerNorm(LayerNormLayer::new_default(8, 1e-5).unwrap()),
            vec!["transpose".into()],
        ));
        graph.set_output("layernorm");

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[1, 8, 3]),
                (0..24).map(|idx| -0.4_f32 + idx as f32 * 0.05).collect(),
            )
            .unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[1, 8, 3]),
                (0..24).map(|idx| 0.35_f32 + idx as f32 * 0.05).collect(),
            )
            .unwrap(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        // After #4148 (multi-dimensional LayerNorm routing), this graph-level
        // batched CROWN call must succeed end-to-end instead of surfacing the
        // old Transpose/LayerNorm routing failure. The stronger Kokoro-shaped
        // Conv1d -> Transpose -> LayerNorm DAG-CROWN provenance contract is
        // covered separately in fallback_reason.rs; this regression only
        // requires batched graph-level success.
        let crown_bounds = graph
            .propagate_crown_batched(&input)
            .expect("#4146 transpose -> layernorm batched CROWN should now succeed");
        assert_bounds_contain_ibp_reference(
            &crown_bounds,
            &ibp_bounds,
            "#4146 transpose -> layernorm graph success",
        );
        // Guard against degenerate NaN/Inf bounds that technically "succeed"
        assert!(
            crown_bounds
                .lower()
                .iter()
                .chain(crown_bounds.upper().iter())
                .all(|v| v.is_finite()),
            "#4146 transpose -> layernorm graph-success bounds must be finite"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_resize_shape_mismatch_falls_back_4146() {
    tests::with_crown_dense_budget_mb("2048", || {
        let resize = ResizeLayer::new(2, 2);
        let input = resize_shape_mismatch_input_4146();
        // Verify Resize still emits ShapeMismatch at the layer level
        // (Fix 1 only changed Transpose; Resize retains ShapeMismatch).
        assert_resize_direct_shape_mismatch_4146(&resize, &input);

        let graph = build_resize_layernorm_graph_4146(resize);

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        // After #4148 (multi-dimensional LayerNorm routing), the graph-level
        // dispatch must catch Resize's ShapeMismatch and return sound fallback
        // bounds end-to-end. Failure here is a real regression.
        let crown_bounds = graph
            .propagate_crown_batched(&input)
            .expect("#4146 resize -> layernorm batched CROWN fallback should now succeed");
        assert_bounds_contain_ibp_reference(
            &crown_bounds,
            &ibp_bounds,
            "#4146 resize -> layernorm ShapeMismatch fallback",
        );
        // Guard against degenerate NaN/Inf bounds that technically "succeed"
        assert!(
            crown_bounds
                .lower()
                .iter()
                .chain(crown_bounds.upper().iter())
                .all(|v| v.is_finite()),
            "#4146 resize -> layernorm CROWN fallback bounds must be finite"
        );
    });
}
