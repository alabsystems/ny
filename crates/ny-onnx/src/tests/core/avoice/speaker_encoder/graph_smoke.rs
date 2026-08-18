// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common::{
    assert_finite_and_ordered, assert_node_bounds_finite_and_ordered, node_name_hits,
    node_names_by_layer_type, node_names_by_layer_types,
};
use super::*;

fn assert_speaker_encoder_io_shapes(model: &OnnxModel) {
    assert_eq!(
        model.network.inputs.len(),
        1,
        "avoice speaker encoder should expose one activation input"
    );
    assert_eq!(
        model.network.outputs.len(),
        1,
        "avoice speaker encoder should expose one embedding output"
    );

    let input_spec = &model.network.inputs[0];
    let output_spec = &model.network.outputs[0];
    assert_eq!(
        input_spec.shape.last(),
        Some(&128),
        "speaker encoder input should end in 128 mel bins, got {:?}",
        input_spec.shape
    );
    assert_eq!(
        output_spec.shape.len(),
        2,
        "speaker encoder output should remain rank-2, got {:?}",
        output_spec.shape
    );
    assert!(
        matches!(output_spec.shape.last(), Some(1024 | -1)),
        "speaker encoder output should end in a static 1024 dim or a dynamic placeholder, got {:?}",
        output_spec.shape
    );
}

fn assert_speaker_encoder_layer_inventory(model: &OnnxModel) {
    let layer_types: Vec<LayerType> = model
        .network
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect();
    assert!(
        layer_types.contains(&LayerType::Softmax),
        "expected Softmax in speaker_encoder.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types.contains(&LayerType::Sqrt),
        "expected Sqrt in speaker_encoder.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types.contains(&LayerType::Concat),
        "expected Concat in speaker_encoder.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types.contains(&LayerType::Expand),
        "expected activation-path Expand in speaker_encoder.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types.iter().any(|layer_type| {
            matches!(
                layer_type,
                LayerType::Linear | LayerType::MatMul | LayerType::Conv1d | LayerType::Conv2d
            )
        }),
        "expected at least one affine or convolutional stage in speaker_encoder.onnx, got {:?}",
        layer_types
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_load_avoice_speaker_encoder_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();

    assert_speaker_encoder_io_shapes(model);
    assert_speaker_encoder_layer_inventory(model);
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_ibp_avoice_speaker_encoder_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = shared::bounded_speaker_encoder_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );
    let output = graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("speaker encoder graph IBP should succeed: {e}"));

    assert_eq!(
        output.lower().shape().last().copied(),
        Some(1024),
        "speaker encoder graph IBP output should end in 1024 dims, got {:?}",
        output.lower().shape()
    );
    assert_finite_and_ordered(&output, "speaker encoder graph IBP output");
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_ibp_avoice_speaker_encoder_rejects_t4_3873() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = shared::bounded_speaker_encoder_input(model, 4, shared::SPEAKER_ENCODER_EPSILON);

    let err = graph
        .propagate_ibp(&input)
        .expect_err("speaker encoder graph IBP should reject T=4");
    let err_text = err.to_string();
    assert!(
        err_text.contains("pad < dim") || err_text.contains("reflect"),
        "speaker encoder T=4 failure should mention reflect-pad floor, got {err_text}"
    );
}

/// Prove the speaker encoder graph handles multiple sequence lengths via the
/// Expand lowering path. The exported speaker_encoder.onnx has dynamic T on the
/// time axis; this test runs IBP at T=5 and T=16 to confirm the
/// ExpandLikeLastAxisLayer correctly broadcasts `[C,1] -> [C,T]` for different T.
/// Part of #3600.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_ibp_avoice_speaker_encoder_multi_seq_len_3600() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();
    for &dynamic_t in &[SPEAKER_ENCODER_SEQUENCE_LEN, 16] {
        let graph = model
            .to_graph_network()
            .unwrap_or_else(|e| panic!("graph conversion at T={dynamic_t} should succeed: {e}"));
        let input = shared::bounded_speaker_encoder_input(
            model,
            dynamic_t,
            shared::SPEAKER_ENCODER_EPSILON,
        );
        let output = graph.propagate_ibp(&input).unwrap_or_else(|e| {
            panic!("speaker encoder graph IBP at T={dynamic_t} should succeed: {e}")
        });

        assert_eq!(
            output.lower().shape().last().copied(),
            Some(1024),
            "speaker encoder graph IBP output at T={dynamic_t} should end in 1024 dims, got {:?}",
            output.lower().shape()
        );
        assert_finite_and_ordered(
            &output,
            &format!("speaker encoder graph IBP output at T={dynamic_t}"),
        );
    }
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_crown_avoice_speaker_encoder_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = shared::bounded_speaker_encoder_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("speaker encoder graph IBP should succeed");
    let crown = graph
        .propagate_crown_batched_with_provenance(&input)
        .expect("speaker encoder graph batched CROWN should succeed");
    let crown_output = crown.bounds;
    assert!(
        crown.provenance == BoundsProvenance::Crown,
        "speaker encoder graph batched CROWN should use backward CROWN, got {:?}",
        crown.provenance
    );

    assert_eq!(
        crown_output.lower().shape().last().copied(),
        Some(1024),
        "speaker encoder graph batched CROWN output should end in 1024 dims, got {:?}",
        crown_output.lower().shape()
    );
    assert_finite_and_ordered(&crown_output, "speaker encoder graph batched CROWN output");
    shared::assert_crown_tighter_than_ibp(&crown_output, &ibp_output, "speaker encoder CROWN");
}

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_collect_node_bounds_avoice_speaker_encoder_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = shared::avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input = shared::bounded_speaker_encoder_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );

    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("speaker encoder node-bound collection should succeed");

    let output_bounds = node_bounds.get(graph.output_name()).unwrap_or_else(|| {
        panic!(
            "speaker encoder node-bound collection missing output node {}",
            graph.output_name()
        )
    });
    assert_finite_and_ordered(output_bounds, "speaker encoder output node bounds");

    let softmax_nodes = node_names_by_layer_type(graph, "Softmax");
    assert!(
        !softmax_nodes.is_empty(),
        "expected at least one Softmax node in speaker encoder graph; name hits={:?}",
        node_name_hits(graph, "softmax")
    );
    assert_node_bounds_finite_and_ordered(&node_bounds, &softmax_nodes, "speaker encoder softmax");

    let sqrt_nodes = node_names_by_layer_type(graph, "Sqrt");
    assert!(
        !sqrt_nodes.is_empty(),
        "expected at least one Sqrt node in speaker encoder graph; name hits={:?}",
        node_name_hits(graph, "sqrt")
    );
    assert_node_bounds_finite_and_ordered(&node_bounds, &sqrt_nodes, "speaker encoder sqrt");

    let pooling_reduce_nodes = node_names_by_layer_types(graph, &["ReduceSum", "ReduceMean"]);
    assert!(
        !pooling_reduce_nodes.is_empty(),
        "expected at least one attentive-pooling reduction node in speaker encoder graph; reduce_sum hits={:?}; reduce_mean hits={:?}",
        node_name_hits(graph, "reducesum"),
        node_name_hits(graph, "reducemean")
    );
    assert_node_bounds_finite_and_ordered(
        &node_bounds,
        &pooling_reduce_nodes,
        "speaker encoder pooling reduction",
    );
}
