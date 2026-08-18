// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::shared::{
    assert_node_bounds_maps_match, avoice_speaker_encoder, avoice_speaker_encoder_graph,
    bounded_speaker_encoder_cosine_input, SPEAKER_ENCODER_EPSILON, SPEAKER_ENCODER_SEQUENCE_LEN,
};
use super::{
    build_component_node_bounds, build_speaker_cosine_component_graphs,
    scalar_spec_bounds_with_node_bounds,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use std::time::Instant;

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_cosine_component_incremental_node_bounds_match_full_ibp_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let (dot_graph, norm_sq_graph, _reference_embedding) = build_speaker_cosine_component_graphs();
    let model = avoice_speaker_encoder();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let base_graph = avoice_speaker_encoder_graph();
    let encoder_node_bounds = base_graph
        .collect_node_bounds(&input)
        .expect("base encoder IBP node-bound collection should succeed");
    let encoder_output_name = base_graph.output_name().to_string();

    for (label, graph) in [("dot", &dot_graph), ("norm squared", &norm_sq_graph)] {
        let incremental =
            build_component_node_bounds(graph, &encoder_node_bounds, &encoder_output_name);
        let full = graph.collect_node_bounds(&input).unwrap_or_else(|e| {
            panic!("{label}: full component IBP node-bound collection failed: {e}")
        });

        assert_node_bounds_maps_match(
            &incremental,
            &full,
            graph,
            &format!("{label}: incremental vs full"),
        );
    }
}

fn run_speaker_cosine_distance_crown_bounds(
    dot_graph: &GraphNetwork,
    norm_sq_graph: &GraphNetwork,
    input: &BoundedTensor,
    start: &Instant,
) -> ((f32, f32), (f32, f32)) {
    let base_graph = avoice_speaker_encoder_graph();
    let encoder_node_bounds = base_graph
        .collect_node_bounds(input)
        .expect("base encoder IBP node-bound collection should succeed");
    let encoder_output_name = base_graph.output_name().to_string();
    eprintln!(
        "cosine distance: encoder IBP ({} nodes) in {:.1}s total",
        encoder_node_bounds.len(),
        start.elapsed().as_secs_f64()
    );

    let dot_node_bounds =
        build_component_node_bounds(dot_graph, &encoder_node_bounds, &encoder_output_name);
    let norm_sq_node_bounds =
        build_component_node_bounds(norm_sq_graph, &encoder_node_bounds, &encoder_output_name);
    eprintln!(
        "cosine distance: component node bounds (dot={}, norm_sq={}) in {:.1}s total",
        dot_node_bounds.len(),
        norm_sq_node_bounds.len(),
        start.elapsed().as_secs_f64()
    );

    let dot_bounds = scalar_spec_bounds_with_node_bounds(dot_graph, input, &dot_node_bounds, "dot");
    eprintln!(
        "cosine distance: dot CROWN in {:.1}s total",
        start.elapsed().as_secs_f64()
    );
    let norm_sq_bounds = scalar_spec_bounds_with_node_bounds(
        norm_sq_graph,
        input,
        &norm_sq_node_bounds,
        "norm squared",
    );
    eprintln!(
        "cosine distance: norm_sq CROWN in {:.1}s total",
        start.elapsed().as_secs_f64()
    );

    (dot_bounds, norm_sq_bounds)
}

/// Verify that the real ECAPA-TDNN speaker encoder + cosine-distance surface
/// stays numerically well-defined and reports whether the current bound meets
/// the `#3499` acceptance target or is still on the explicit vacuous fallback.
///
/// The bound is computed analytically from separate CROWN passes on the
/// dot-product and norm-squared graphs:
///
///   distance_upper = 1 - dot_lower / sqrt(norm_sq_upper)
///
/// Performance note: The encoder-prefix IBP bounds are pre-computed once and
/// shared between the dot and norm-squared component graphs. On the deep
/// ECAPA-TDNN DAG each full IBP forward pass costs ~70s, so sharing saves
/// ~70s per component graph - enough to keep the total under the 600s cargo
/// wrapper timeout under CPU contention from concurrent workers.
///
/// `#3596` separately measures the real speaker scalar heads under explicit
/// IBP-intermediate CROWN and currently proves that path runs without
/// loosening the pure-IBP scalar bounds on this deep InstanceNorm-heavy DAG.
/// This end-to-end progress test keeps using the spec-guided CROWN path with
/// pre-computed IBP intermediates, so if `dot_lower` stays non-positive here
/// the test should report the explicit vacuous sentinel `2.0` instead of
/// pretending the `< 0.1` acceptance target is already proven.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_crown_avoice_speaker_cosine_distance_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let t_start = Instant::now();
    let (dot_graph, norm_sq_graph, _reference_embedding) = build_speaker_cosine_component_graphs();
    let model = avoice_speaker_encoder();
    let input = bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    eprintln!(
        "cosine distance: graphs + input built in {:.1}s",
        t_start.elapsed().as_secs_f64()
    );

    let ((dot_crown_lower, dot_crown_upper), (nsq_crown_lower, nsq_crown_upper)) =
        run_speaker_cosine_distance_crown_bounds(&dot_graph, &norm_sq_graph, &input, &t_start);

    assert!(
        nsq_crown_upper > 0.0,
        "CROWN norm_sq upper should be positive, got {nsq_crown_upper}"
    );

    super::distance::assert_speaker_cosine_distance_bound(
        dot_crown_lower,
        dot_crown_upper,
        nsq_crown_lower,
        nsq_crown_upper,
    );
}
