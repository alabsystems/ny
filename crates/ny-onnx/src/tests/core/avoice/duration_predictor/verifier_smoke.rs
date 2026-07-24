// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Duration predictor verifier-smoke helpers.
//!
//! Graph construction and duration-range assertions for the duration
//! predictor verifier smokes. Test functions live in the root
//! `avoice/verifier_smoke.rs` entrypoint (#3950).

use super::super::common;
use super::proof_head::KOKORO_DURATION_BUCKETS;
use super::surrogate::load_surrogate_graph;
use super::*;
use ny_core::Bound;
use ny_propagate::{GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// Build the surrogate graph with sigmoid+sum proof head appended,
/// and return the graph, bounded input, output size, and max duration.
///
/// Architecture:
///   encoded_features -> surrogate (MatMul+Add) -> duration_logits [T, 50]
///   -> Sigmoid -> sigmoid_probs [T, 50]
///   -> ReduceSum(axis=-1) -> expected_durations [T]
pub(crate) fn duration_expected_duration_verifier_setup(
) -> (GraphNetwork, BoundedTensor, usize, f32) {
    let (model, graph) = load_surrogate_graph();
    let output_name = graph.output_name().to_string();

    let mut dur_graph = graph;
    dur_graph.add_node(GraphNode::new(
        "sigmoid_probs",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![output_name],
    ));
    dur_graph.add_node(GraphNode::new(
        "expected_durations",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], false)),
        vec!["sigmoid_probs".to_string()],
    ));
    dur_graph.set_output("expected_durations");

    let input_spec = common::input_spec_by_name(&model, "encoded_features");
    let shape = common::unbatched_shape_from_input_spec(input_spec, 4, "encoded_features");
    let center = ArrayD::zeros(IxDyn(&shape));
    let input = BoundedTensor::from_epsilon(center, 1e-3).expect("valid epsilon ball");

    let ibp_output = dur_graph
        .propagate_ibp(&input)
        .expect("duration expected-duration graph IBP should succeed");
    let output_size = ibp_output.lower().len();
    let max_duration = KOKORO_DURATION_BUCKETS as f32;

    (dur_graph, input, output_size, max_duration)
}

pub(crate) fn assert_bounds_in_duration_range(
    output_bounds: &[Bound],
    expected_size: usize,
    max_duration: f32,
    label: &str,
) {
    assert_eq!(
        output_bounds.len(),
        expected_size,
        "{label}: output count should match ({expected_size}), got {}",
        output_bounds.len()
    );
    for (idx, bound) in output_bounds.iter().enumerate() {
        assert!(
            bound.lower() >= -1e-6,
            "{label}: duration[{idx}] lower should be >= 0, got {}",
            bound.lower()
        );
        assert!(
            bound.upper() <= max_duration + 1e-4,
            "{label}: duration[{idx}] upper should be <= {max_duration}, got {}",
            bound.upper()
        );
    }
}
