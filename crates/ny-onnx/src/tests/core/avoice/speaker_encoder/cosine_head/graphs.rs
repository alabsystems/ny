// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::shared::{
    avoice_speaker_encoder, bounded_speaker_encoder_cosine_input, SPEAKER_ENCODER_FILE,
    SPEAKER_ENCODER_SEQUENCE_LEN,
};
use super::super::{load_onnx, require_test_model_with_hint, OnnxModel, AVOICE_TEST_MODEL_HINT};
use ndarray::ArrayD;
use ny_propagate::layers::{MulConstantLayer, PowConstantLayer, ReduceSumLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

// ---------------------------------------------------------------------------
// Real-weight cosine distance head (#3499 progress surface)
// ---------------------------------------------------------------------------
//
// The cosine distance verification is split into two CROWN-friendly graphs
// that avoid the Reciprocal node entirely:
//
//   1. **Dot graph**: encoder -> scale(1/||ref||) -> dot(normalized_ref) -> scalar
//   2. **Norm^2 graph**: encoder -> scale(1/||ref||) -> pow(2) -> sum -> scalar
//
// IBP through the deep ECAPA-TDNN widens intervals so much that the embedding
// norm's IBP lower bound reaches 0, triggering the Reciprocal's zero-crossing
// fallback ([-FALLBACK_BOUND, +FALLBACK_BOUND]). The two-graph approach
// sidesteps this. CROWN bounds on each component are combined analytically:
//
//   distance_upper = 1 - dot_lower / sqrt(norm_sq_upper)

/// Compute reference embedding norm in f64 to avoid f32 overflow.
fn ref_norm_f64(reference: &ArrayD<f32>) -> f64 {
    let norm = (reference
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>())
    .sqrt();
    assert!(
        norm > 0.0,
        "reference embedding must have positive norm, got {norm}"
    );
    norm
}

/// Append a dot-product head: sum(scaled_emb x normalized_ref).
fn add_cosine_dot_head(
    graph: &mut GraphNetwork,
    encoder_output: &str,
    reference_embedding: &ArrayD<f32>,
) {
    let norm = ref_norm_f64(reference_embedding);
    let inv_norm = (1.0 / norm) as f32;
    let normalized_ref = reference_embedding.mapv(|v| (v as f64 / norm) as f32);
    let axes: Vec<i64> = (0..reference_embedding.ndim() as i64).collect();

    graph.add_node(GraphNode::new(
        "cosine_embed_scaled",
        Layer::MulConstant(MulConstantLayer::new(ArrayD::from_elem(
            reference_embedding.raw_dim(),
            inv_norm,
        ))),
        vec![encoder_output.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cosine_dot_terms",
        Layer::MulConstant(MulConstantLayer::new(normalized_ref)),
        vec!["cosine_embed_scaled".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cosine_dot",
        Layer::ReduceSum(ReduceSumLayer::new(axes, false)),
        vec!["cosine_dot_terms".to_string()],
    ));
    graph.set_output("cosine_dot");
}

/// Append a norm-squared head: sum(scaled_emb^2).
fn add_cosine_norm_sq_head(
    graph: &mut GraphNetwork,
    encoder_output: &str,
    reference_embedding: &ArrayD<f32>,
) {
    let norm = ref_norm_f64(reference_embedding);
    let inv_norm = (1.0 / norm) as f32;
    let axes: Vec<i64> = (0..reference_embedding.ndim() as i64).collect();

    graph.add_node(GraphNode::new(
        "cosine_embed_scaled",
        Layer::MulConstant(MulConstantLayer::new(ArrayD::from_elem(
            reference_embedding.raw_dim(),
            inv_norm,
        ))),
        vec![encoder_output.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cosine_embedding_sq",
        Layer::PowConstant(PowConstantLayer::new(2.0)),
        vec!["cosine_embed_scaled".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cosine_embedding_sq_sum",
        Layer::ReduceSum(ReduceSumLayer::new(axes, false)),
        vec!["cosine_embedding_sq".to_string()],
    ));
    graph.set_output("cosine_embedding_sq_sum");
}

/// Build both cosine-component graphs from an arbitrary base encoder graph.
///
/// The `model` reference is used only for computing the reference embedding
/// via a zero-epsilon center input. The `base_graph` is the encoder graph
/// to which the dot and norm-squared heads are appended.
///
/// Returns `(dot_graph, norm_sq_graph, reference_embedding)`.
fn build_speaker_cosine_component_graphs_from_base_graph(
    model: &OnnxModel,
    base_graph: GraphNetwork,
) -> (GraphNetwork, GraphNetwork, ArrayD<f32>) {
    let encoder_output = base_graph.output_name().to_string();

    let center_input =
        bounded_speaker_encoder_cosine_input(model, SPEAKER_ENCODER_SEQUENCE_LEN, 0.0);
    let ref_output = base_graph
        .propagate_ibp(&center_input)
        .expect("reference embedding evaluation should succeed");
    let reference_embedding = ref_output.lower().clone();

    let mut dot_graph = base_graph.clone();
    add_cosine_dot_head(&mut dot_graph, &encoder_output, &reference_embedding);

    let mut norm_sq_graph = base_graph;
    add_cosine_norm_sq_head(&mut norm_sq_graph, &encoder_output, &reference_embedding);

    (dot_graph, norm_sq_graph, reference_embedding)
}

/// Build both cosine-component graphs from the real speaker encoder.
///
/// Returns `(dot_graph, norm_sq_graph, reference_embedding)`.
pub(in super::super) fn build_speaker_cosine_component_graphs(
) -> (GraphNetwork, GraphNetwork, ArrayD<f32>) {
    let model = avoice_speaker_encoder();
    let base_graph = model
        .to_graph_network()
        .expect("speaker encoder graph conversion for cosine components");
    build_speaker_cosine_component_graphs_from_base_graph(model, base_graph)
}

/// Build both cosine-component graphs from a GraphModel round-trip encoder.
///
/// Loads a fresh `OnnxModel`, serializes to `GraphModel`, rebuilds to
/// `GraphNetwork`, then attaches the same dot and norm-squared heads.
/// This exercises the `OnnxModel -> GraphModel -> GraphNetwork` path that
/// builder-style downstream consumers use.
pub(in super::super) fn build_speaker_cosine_component_graphs_round_trip(
) -> (GraphNetwork, GraphNetwork, ArrayD<f32>) {
    let model = avoice_speaker_encoder();
    let path = require_test_model_with_hint(SPEAKER_ENCODER_FILE, AVOICE_TEST_MODEL_HINT);
    let round_trip_graph = load_onnx(&path)
        .expect("speaker encoder round-trip load should succeed")
        .to_graph_model()
        .build_graph_network(crate::GraphNetworkOptions::default())
        .expect("speaker encoder GraphModel round-trip build should succeed");
    build_speaker_cosine_component_graphs_from_base_graph(model, round_trip_graph)
}
