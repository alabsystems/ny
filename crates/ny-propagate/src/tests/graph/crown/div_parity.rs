// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Full-graph DAG CROWN parity regressions for Div backward (#3676).

use crate::types::BoundsProvenance;
use crate::*;
use ndarray::{array, ArrayD, IxDyn};

fn build_rowwise_broadcast_div_graph_3676() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "reduce",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
    ));
    graph.add_node(GraphNode::new(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[1]),
            4.0_f32,
        ))),
        vec!["reduce".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "div",
        Layer::Div(DivLayer),
        vec![NETWORK_INPUT.to_string(), "shift".to_string()],
    ));
    graph.set_output("div");
    graph
}

fn eval_rowwise_broadcast_div_graph_3676(x: &[f32; 4]) -> [f32; 4] {
    let denom0 = x[0] + x[1] + 4.0;
    let denom1 = x[2] + x[3] + 4.0;
    [x[0] / denom0, x[1] / denom0, x[2] / denom1, x[3] / denom1]
}

fn assert_div_sample_contained(bounds: &BoundedTensor, x: [f32; 4], label: &str) {
    let bounds = bounds.flatten();
    let output = eval_rowwise_broadcast_div_graph_3676(&x);
    for (dim, &value) in output.iter().enumerate() {
        assert!(
            value >= bounds.lower()[[dim]] - 1e-4 && value <= bounds.upper()[[dim]] + 1e-4,
            "{label}: output {} not in [{}, {}] at [{}, {}, {}, {}] dim {}",
            value,
            bounds.lower()[[dim]],
            bounds.upper()[[dim]],
            x[0],
            x[1],
            x[2],
            x[3],
            dim
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_div_full_backward_keeps_crown_provenance_3676() {
    let graph = build_rowwise_broadcast_div_graph_3676();
    let input = BoundedTensor::new(
        array![[-1.5_f32, -1.0_f32], [0.2_f32, 0.3_f32]].into_dyn(),
        array![[0.5_f32, 1.0_f32], [1.0_f32, 1.2_f32]].into_dyn(),
    )
    .expect("valid bounded input");

    let dag_result = graph
        .propagate_crown_with_provenance(&input)
        .expect("full DAG-CROWN should succeed on Div graph");
    assert_eq!(
        dag_result.provenance,
        BoundsProvenance::Crown,
        "#3676 Div graph should stay on the DAG-CROWN path"
    );

    let lower = input.lower().iter().copied().collect::<Vec<_>>();
    let upper = input.upper().iter().copied().collect::<Vec<_>>();
    for &x0 in &[lower[0], f32::midpoint(lower[0], upper[0]), upper[0]] {
        for &x1 in &[lower[1], f32::midpoint(lower[1], upper[1]), upper[1]] {
            for &x2 in &[lower[2], f32::midpoint(lower[2], upper[2]), upper[2]] {
                for &x3 in &[lower[3], f32::midpoint(lower[3], upper[3]), upper[3]] {
                    let point = [x0, x1, x2, x3];
                    assert_div_sample_contained(&dag_result.bounds, point, "DAG-CROWN soundness");
                }
            }
        }
    }
}
