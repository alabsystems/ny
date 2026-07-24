// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #2817: zero-row linear layer (`out_features == 0`) must
//! not panic via `% 0` in graph CROWN dimension checks.

use crate::*;
use ndarray::{arr1, arr2, Array2};

/// Regression test #2817: DAG-CROWN with zero-row linear layer falls back to
/// IBP instead of panicking on `% 0`.
#[test]
fn test_graph_crown_zero_row_linear_falls_back_to_ibp_2817() {
    let mut graph = GraphNetwork::new();

    // Linear layer with zero output features (0-row weight matrix).
    // This is a degenerate model that should trigger a graceful fallback,
    // not a `% 0` panic in the dimension check.
    let weight = Array2::<f32>::zeros((0, 2));
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    // Must not panic. The exact result (error or fallback bounds) depends on
    // how the IBP path handles zero-row linear, but no `% 0` panic.
    let _result = graph.propagate_crown(&input);
}

/// Regression test #2817: spec-guided CROWN with zero-row linear layer falls
/// back to IBP instead of panicking on `% 0`.
#[test]
fn test_spec_guided_crown_zero_row_linear_falls_back_to_ibp_2817() {
    let mut graph = GraphNetwork::new();

    let weight = Array2::<f32>::zeros((0, 2));
    let linear = LinearLayer::new(weight, None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    // Spec matrix with 1 row (single classification property).
    // The spec references a single output, but the zero-row linear produces
    // zero outputs — the dimension check should catch this before `% 0`.
    let spec_matrix = arr2(&[[1.0_f32]]);

    // Must not panic.
    let _result = graph.propagate_crown_with_specs_and_engine(&input, &spec_matrix, None);
}
