// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::GraphNetwork;
use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;
use ndarray::{Array, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32};
use std::collections::HashMap;

fn test_graph() -> GraphNetwork {
    GraphNetwork {
        output_node: "output".to_string(),
        ..GraphNetwork::new()
    }
}

fn scalar_batched_linear_bounds(value: f32) -> BatchedLinearBounds {
    BatchedLinearBounds::from_parts_unchecked(
        Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![value]).expect("shape should be valid"),
        Array::from_shape_vec(IxDyn(&[1, 1]), vec![value]).expect("shape should be valid"),
        Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![value]).expect("shape should be valid"),
        Array::from_shape_vec(IxDyn(&[1, 1]), vec![value]).expect("shape should be valid"),
        vec![1, 1],
        vec![1, 1],
    )
}

fn zero_batched_linear_bounds(
    input_shape: &[usize],
    output_shape: &[usize],
) -> BatchedLinearBounds {
    let in_dim = *input_shape.last().expect("input shape should be non-empty");
    let out_dim = *output_shape
        .last()
        .expect("output shape should be non-empty");
    let mut a_shape = output_shape[..output_shape.len() - 1].to_vec();
    a_shape.push(out_dim);
    a_shape.push(in_dim);

    BatchedLinearBounds::from_parts_unchecked(
        Array::zeros(IxDyn(&a_shape)),
        Array::zeros(IxDyn(output_shape)),
        Array::zeros(IxDyn(&a_shape)),
        Array::zeros(IxDyn(output_shape)),
        input_shape.to_vec(),
        output_shape.to_vec(),
    )
}

fn filled_batched_linear_bounds(
    input_shape: &[usize],
    output_shape: &[usize],
    start: f32,
) -> BatchedLinearBounds {
    let in_dim = *input_shape.last().expect("input shape should be non-empty");
    let out_dim = *output_shape
        .last()
        .expect("output shape should be non-empty");
    let mut a_shape = output_shape[..output_shape.len() - 1].to_vec();
    a_shape.push(out_dim);
    a_shape.push(in_dim);
    let a_len: usize = a_shape.iter().product();
    let b_len: usize = output_shape.iter().product();

    BatchedLinearBounds::from_parts_unchecked(
        Array::from_shape_vec(
            IxDyn(&a_shape),
            (0..a_len).map(|idx| start + idx as f32).collect(),
        )
        .expect("lower_a shape should be valid"),
        Array::from_shape_vec(
            IxDyn(output_shape),
            (0..b_len).map(|idx| start + 1000.0 + idx as f32).collect(),
        )
        .expect("lower_b shape should be valid"),
        Array::from_shape_vec(
            IxDyn(&a_shape),
            (0..a_len).map(|idx| start + 2000.0 + idx as f32).collect(),
        )
        .expect("upper_a shape should be valid"),
        Array::from_shape_vec(
            IxDyn(output_shape),
            (0..b_len).map(|idx| start + 3000.0 + idx as f32).collect(),
        )
        .expect("upper_b shape should be valid"),
        input_shape.to_vec(),
        output_shape.to_vec(),
    )
}

fn assert_reshape_compatible_merge_matches(
    merged: &BatchedLinearBounds,
    expected: &BatchedLinearBounds,
) {
    assert!(
        merged
            .lower_a()
            .iter()
            .chain(merged.lower_b().iter())
            .chain(merged.upper_a().iter())
            .chain(merged.upper_b().iter())
            .all(|value| value.is_finite()),
        "reshape-compatible batched merge should stay finite"
    );

    let expected_lower_a = expected
        .lower_a()
        .view()
        .into_shape_with_order(IxDyn(merged.lower_a().shape()))
        .expect("lower_a reshape should succeed");
    let expected_lower_b = expected
        .lower_b()
        .view()
        .into_shape_with_order(IxDyn(merged.lower_b().shape()))
        .expect("lower_b reshape should succeed");
    let expected_upper_a = expected
        .upper_a()
        .view()
        .into_shape_with_order(IxDyn(merged.upper_a().shape()))
        .expect("upper_a reshape should succeed");
    let expected_upper_b = expected
        .upper_b()
        .view()
        .into_shape_with_order(IxDyn(merged.upper_b().shape()))
        .expect("upper_b reshape should succeed");

    for (actual, expected) in merged.lower_a().iter().zip(expected_lower_a.iter()) {
        assert_eq!(*actual, next_down_f32(*expected));
    }
    for (actual, expected) in merged.lower_b().iter().zip(expected_lower_b.iter()) {
        assert_eq!(*actual, next_down_f32(*expected));
    }
    for (actual, expected) in merged.upper_a().iter().zip(expected_upper_a.iter()) {
        assert_eq!(*actual, next_up_f32(*expected));
    }
    for (actual, expected) in merged.upper_b().iter().zip(expected_upper_b.iter()) {
        assert_eq!(*actual, next_up_f32(*expected));
    }
}

#[test]
fn test_accumulate_batched_crown_bounds_preserves_three_term_cancellation_3904() {
    let graph = test_graph();
    let mut node_bounds = HashMap::new();
    let mut input_accumulated = false;

    let contributions = [1_099_511_627_776.0_f32, 1.0_f32, -1_099_511_627_776.0_f32];
    for contribution in contributions {
        graph
            .accumulate_batched_crown_bounds_to_input(
                "residual",
                BatchedCrownBounds::Dense(scalar_batched_linear_bounds(contribution)),
                &mut node_bounds,
                &mut input_accumulated,
            )
            .expect("batched accumulation should succeed");
    }

    let merged = node_bounds
        .remove("residual")
        .expect("residual entry should exist")
        .into_batched_dense_checked(
            "test_accumulate_batched_crown_bounds_preserves_three_term_cancellation_3904",
        )
        .expect("merged residual entry should materialize");

    let mut naive_a = Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![contributions[0]])
        .expect("shape should be valid");
    let mut naive_b = Array::from_shape_vec(IxDyn(&[1, 1]), vec![contributions[0]])
        .expect("shape should be valid");
    for contribution in contributions.iter().skip(1) {
        let add_a = Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![*contribution])
            .expect("shape should be valid");
        let add_b = Array::from_shape_vec(IxDyn(&[1, 1]), vec![*contribution])
            .expect("shape should be valid");
        naive_a = GraphNetwork::safe_add(&naive_a, &add_a, true);
        naive_b = GraphNetwork::safe_add(&naive_b, &add_b, true);
    }

    assert_eq!(
        naive_a[[0, 0, 0]],
        0.0,
        "serial f32 batched merge should lose the low-order term"
    );
    assert_eq!(
        naive_b[[0, 0]],
        0.0,
        "serial f32 batched bias merge should lose the low-order term"
    );
    assert_eq!(merged.lower_a()[[0, 0, 0]], next_down_f32(1.0));
    assert_eq!(merged.lower_b()[[0, 0]], next_down_f32(1.0));
    assert_eq!(merged.upper_a()[[0, 0, 0]], next_up_f32(1.0));
    assert_eq!(merged.upper_b()[[0, 0]], next_up_f32(1.0));
}

#[test]
fn test_accumulate_batched_crown_bounds_reshape_compatible_merge_4243() {
    let graph = test_graph();
    let mut node_bounds = HashMap::new();
    let mut input_accumulated = false;
    let input_shape = [1, 2, 4, 8];
    let flattened_output_shape = [1, 4, 16];
    let per_head_output_shape = [1, 2, 4, 8];

    let existing = zero_batched_linear_bounds(&input_shape, &flattened_output_shape);
    let new_bounds = filled_batched_linear_bounds(&input_shape, &per_head_output_shape, 10.0);

    graph
        .accumulate_batched_crown_bounds_to_input(
            "residual",
            BatchedCrownBounds::Dense(existing),
            &mut node_bounds,
            &mut input_accumulated,
        )
        .expect("initial flattened entry should accumulate");
    graph
        .accumulate_batched_crown_bounds_to_input(
            "residual",
            BatchedCrownBounds::Dense(new_bounds.clone()),
            &mut node_bounds,
            &mut input_accumulated,
        )
        .expect("reshape-compatible contribution should merge");

    let merged = node_bounds
        .remove("residual")
        .expect("residual entry should exist")
        .into_batched_dense_checked(
            "test_accumulate_batched_crown_bounds_reshape_compatible_merge_4243",
        )
        .expect("merged residual entry should materialize");

    assert_eq!(merged.output_shape(), &flattened_output_shape);
    assert_reshape_compatible_merge_matches(&merged, &new_bounds);
}
