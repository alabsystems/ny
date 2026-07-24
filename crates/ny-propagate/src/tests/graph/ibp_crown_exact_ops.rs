// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regressions for exact shape/affine operators in the #3775 demand set.

use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn linear4(weights: [[f32; 4]; 4], bias: [f32; 4]) -> LinearLayer {
    LinearLayer::new(arr2(&weights), Some(arr1(&bias)))
        .expect("invariant: valid 4x4 linear parameters")
}

fn build_encoder_prefix_3775(graph: &mut GraphNetwork) {
    let linear0 = linear4(
        [
            [1.0_f32, -0.2, 0.1, 0.4],
            [0.3, 0.8, -0.5, 0.2],
            [-0.6, 0.4, 0.9, -0.1],
            [0.2, -0.7, 0.3, 1.1],
        ],
        [0.05_f32, -0.03, 0.02, 0.1],
    );
    graph.add_node(GraphNode::from_input("linear0", Layer::Linear(linear0)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear0".to_string()],
    ));
    let linear_pre_ln = linear4(
        [
            [0.6_f32, -0.1, 0.2, 0.3],
            [-0.3, 0.5, 0.4, -0.2],
            [0.1, 0.3, -0.5, 0.6],
            [0.4, -0.4, 0.1, 0.7],
        ],
        [0.02_f32, -0.01, 0.03, -0.02],
    );
    graph.add_node(GraphNode::new(
        "linear_pre_ln",
        Layer::Linear(linear_pre_ln),
        vec!["relu".to_string()],
    ));
}

/// Graph: linear0 -> slice -> batchnorm -> relu -> linear_out.
///
/// `Slice` and `BatchNorm` use pre-activation bounds for shape/channel metadata,
/// but they are still exact linear operators and must not pull upstream demand.
fn build_exact_shape_affine_relu_graph_3775() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    build_encoder_prefix_3775(&mut graph);
    graph.add_node(GraphNode::new(
        "slice",
        Layer::Slice(SliceLayer::new(0, 0, 2)),
        vec!["linear_pre_ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "batchnorm",
        Layer::BatchNorm(
            BatchNormLayer::from_scale_bias(
                arr1(&[1.25_f32, -0.75]).into_dyn(),
                arr1(&[0.10_f32, -0.05]).into_dyn(),
            )
            .expect("invariant: valid BatchNorm scale/bias"),
        ),
        vec!["slice".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu_bn",
        Layer::ReLU(ReLULayer),
        vec!["batchnorm".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.9_f32, -0.4], [0.3, 0.8]]),
                Some(arr1(&[0.01, -0.02])),
            )
            .expect("invariant: valid exact-affine regression projection"),
        ),
        vec!["relu_bn".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), 0.2_f32)
        .expect("invariant: exact-shape affine test interval should construct");
    (graph, input)
}

/// Graph: linear0 -> reduce_mean -> relu -> linear_out.
///
/// `ReduceMean` is an exact linear reduction and should not demand-tighten its
/// producer just because it needs pre-activation shape for backward.
fn build_exact_reduction_relu_graph_3775() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    build_encoder_prefix_3775(&mut graph);
    graph.add_node(GraphNode::new(
        "reduce_mean",
        Layer::ReduceMean(ReduceMeanLayer::new(vec![0], true)),
        vec!["linear_pre_ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu_mean",
        Layer::ReLU(ReLULayer),
        vec!["reduce_mean".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.1_f32], [-0.6]]), Some(arr1(&[0.0, 0.04])))
                .expect("invariant: valid reduction regression projection"),
        ),
        vec!["relu_mean".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), 0.2_f32)
        .expect("invariant: exact-reduction test interval should construct");
    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_exact_shape_and_affine_ops_do_not_pull_upstream_demand_3775() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_exact_shape_affine_relu_graph_3775();

        let with_status = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("#3775 exact shape/affine graph should collect CROWN-IBP bounds");

        for skipped in ["linear_pre_ln", "slice"] {
            assert_eq!(
                with_status.provenance_for_node(skipped),
                Some(BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::DemandDrivenSkip
                )),
                "#3775 exact operator `{skipped}` should not request upstream tightening"
            );
        }
        assert_eq!(
            with_status.provenance_for_node("batchnorm"),
            Some(BoundsProvenance::Crown),
            "#3775 BatchNorm output should still tighten for the downstream ReLU"
        );
        assert!(
            with_status
                .fallback_events
                .iter()
                .all(|event| event.reason != CrownIbpFallbackReason::DemandDrivenSkip),
            "#3775 demand policy skips must stay out of fallback events"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_exact_reduction_does_not_pull_upstream_demand_3775() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_exact_reduction_relu_graph_3775();

        let with_status = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .expect("#3775 exact reduction graph should collect CROWN-IBP bounds");

        assert_eq!(
            with_status.provenance_for_node("linear_pre_ln"),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::DemandDrivenSkip
            )),
            "#3775 ReduceMean should not demand-tighten its producer"
        );
        assert_eq!(
            with_status.provenance_for_node("reduce_mean"),
            Some(BoundsProvenance::Crown),
            "#3775 ReduceMean output should still tighten for the downstream ReLU"
        );
        assert!(
            with_status
                .fallback_events
                .iter()
                .all(|event| event.reason != CrownIbpFallbackReason::DemandDrivenSkip),
            "#3775 demand policy skips must stay out of fallback events"
        );
    });
}
