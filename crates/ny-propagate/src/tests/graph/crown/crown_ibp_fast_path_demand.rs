// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for #3775 demand-policy parity in the sequential
//! graph CROWN-IBP fast path.

use ndarray::{arr1, arr2};
use ny_core::NaiveCpuGemmEngine;
use ny_test_utils::assert_bounded_tensor_close;

use crate::layers::LayerNormCrownMode;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::{
    BoundedTensor, GELULayer, GeluApproximation, GraphNetwork, GraphNode, Layer, LayerNormLayer,
    LinearLayer, NonZeroLayer,
};

fn build_unary_transformer_sequential_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear0",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.2], [0.4, 0.7], [-0.5, 0.3]]),
                Some(arr1(&[0.05_f32, -0.03, 0.02])),
            )
            .unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(LayerNormLayer::new_default(3, 1e-5).unwrap()),
        vec!["linear0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.8_f32, -0.3, 0.2], [0.1, 0.6, -0.4]]), None).unwrap(),
        ),
        vec!["layernorm".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::new(GeluApproximation::Tanh)),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[0.7_f32, -0.2], [0.3, 0.5]]), None).unwrap()),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    (
        graph,
        BoundedTensor::new(
            arr1(&[-0.2_f32, -0.1]).into_dyn(),
            arr1(&[0.3_f32, 0.25]).into_dyn(),
        )
        .unwrap(),
    )
}

fn build_linear_nonzero_linear_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear0",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.5], [0.4, 0.8], [-0.2, 0.6]]),
                Some(arr1(&[0.1_f32, -0.05, 0.02])),
            )
            .unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["linear0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[0.6_f32, -0.2, 0.1]]), None).unwrap()),
        vec!["nonzero".to_string()],
    ));
    graph.set_output("linear1");

    (
        graph,
        BoundedTensor::new(
            arr1(&[-0.4_f32, -0.3]).into_dyn(),
            arr1(&[0.6_f32, 0.5]).into_dyn(),
        )
        .unwrap(),
    )
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_sequential_fast_path_keeps_skipped_bounds_on_ibp_3775() {
    let (mut graph, input) = build_unary_transformer_sequential_graph();
    assert_eq!(
        graph.set_layernorm_crown_mode(LayerNormCrownMode::Sampling),
        1
    );

    let baseline = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    let with_engine = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&NaiveCpuGemmEngine))
        .unwrap();

    for name in ["layernorm", "gelu"] {
        assert_eq!(
            baseline.provenance_for_node(name),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DemandDrivenSkip
            )),
            "{name} should be skipped by #3775 demand selection"
        );
        assert_eq!(
            with_engine.provenance_for_node(name),
            baseline.provenance_for_node(name),
            "engine fast path changed skipped-node provenance at '{name}'"
        );
        assert_bounded_tensor_close(
            with_engine.bounds.get(name).unwrap(),
            baseline.bounds.get(name).unwrap(),
            1e-6,
            "engine fast path skipped-node bound parity",
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_sequential_fast_path_drops_skipped_fallback_events_3775() {
    let (graph, input) = build_linear_nonzero_linear_graph();

    let with_engine = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&NaiveCpuGemmEngine))
        .unwrap();

    assert_eq!(
        with_engine.provenance_for_node("nonzero"),
        Some(BoundsProvenance::ForwardFallback(
            CrownIbpFallbackReason::DemandDrivenSkip
        )),
        "skipped unsupported node must remain a demand-driven skip"
    );
    assert_eq!(
        with_engine
            .fallback_events
            .iter()
            .filter(|event| event.details.contains("node 'nonzero'"))
            .count(),
        0,
        "engine fast path should not report fallback telemetry for skipped nodes"
    );
}
