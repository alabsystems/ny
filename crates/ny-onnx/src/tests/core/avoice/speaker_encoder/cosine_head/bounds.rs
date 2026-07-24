// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::shared::SPEAKER_COMPONENT_SPEC_DEADLINE_SECS;
use ny_propagate::{BoundPropagation, GraphNetwork};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Build complete node bounds for a cosine component graph by reusing
/// pre-computed encoder-prefix IBP bounds and only propagating the 2-3
/// head nodes that differ between the dot and norm-squared graphs.
///
/// This avoids running redundant full-encoder IBP forward passes - on
/// the deep ECAPA-TDNN encoder each pass costs ~70s, and sharing saves
/// enough budget to keep the cosine distance test within the 600s cargo
/// wrapper timeout under CPU contention from concurrent workers.
pub(in super::super) fn build_component_node_bounds(
    graph: &GraphNetwork,
    encoder_node_bounds: &HashMap<String, BoundedTensor>,
    encoder_output_name: &str,
) -> HashMap<String, BoundedTensor> {
    let mut bounds = encoder_node_bounds.clone();
    assert!(
        bounds.contains_key(encoder_output_name),
        "encoder output '{}' missing from node bounds",
        encoder_output_name
    );

    // Incrementally propagate each head node that is NOT in the encoder prefix.
    for name in graph.node_names() {
        if bounds.contains_key(name) {
            continue;
        }
        let node = graph
            .node(name)
            .unwrap_or_else(|| panic!("node '{}' missing from graph", name));
        // All cosine head nodes are unary: MulConstant, PowConstant, ReduceSum.
        assert_eq!(
            node.inputs().len(),
            1,
            "cosine head node '{}' should be unary, has {} inputs",
            name,
            node.inputs().len()
        );
        let input_name = &node.inputs()[0];
        let input_bounds = bounds.get(input_name).unwrap_or_else(|| {
            panic!(
                "input '{}' for node '{}' not yet computed",
                input_name, name
            )
        });
        let output_bounds = node
            .layer()
            .propagate_ibp(input_bounds)
            .unwrap_or_else(|e| panic!("IBP for cosine head node '{}' failed: {}", name, e));
        bounds.insert(name.clone(), output_bounds);
    }
    bounds
}

pub(in super::super) fn scalar_spec_bounds_with_node_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    label: &str,
) -> (f32, f32) {
    let spec = ndarray::arr2(&[[1.0_f32]]);
    let deadline = Instant::now() + Duration::from_secs(SPEAKER_COMPONENT_SPEC_DEADLINE_SECS);
    let crown = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
            input,
            &spec,
            None,
            node_bounds,
            Some(deadline),
        )
        .unwrap_or_else(|e| panic!("{label} spec-guided CROWN should succeed: {e}"));
    let flat = crown.flatten();
    assert_eq!(
        flat.lower().len(),
        1,
        "{label} spec-guided CROWN should stay scalar, got shape {:?}",
        flat.lower().shape()
    );
    let lower = flat.lower()[0];
    let upper = flat.upper()[0];
    assert!(
        lower.is_finite() && upper.is_finite(),
        "{label} spec-guided CROWN bounds should be finite: [{lower}, {upper}]"
    );
    (lower, upper)
}

pub(in super::super) fn scalar_width(lower: f32, upper: f32) -> f32 {
    upper - lower
}
