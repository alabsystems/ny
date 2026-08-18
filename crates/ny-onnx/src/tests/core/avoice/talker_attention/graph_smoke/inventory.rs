// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn assert_talker_attention_io_shapes(model: &OnnxModel) {
    let input_names: Vec<&str> = model
        .network
        .inputs
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    for expected in &["hidden_states", "cos", "sin", "mask"] {
        assert!(
            input_names.contains(expected),
            "expected {expected} in input inventory, got {:?}",
            input_names
        );
    }
    assert_eq!(
        model.network.outputs.len(),
        1,
        "talker attention should expose one output (attn_output)"
    );
    assert_eq!(
        model.network.outputs[0].shape.last(),
        Some(&(TALKER_ATTENTION_HIDDEN_DIM as i64)),
        "talker attention output should end in {TALKER_ATTENTION_HIDDEN_DIM}, got {:?}",
        model.network.outputs[0].shape
    );
}

fn assert_talker_attention_layer_inventory(model: &OnnxModel) {
    let layer_types: Vec<LayerType> = model
        .network
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect();
    assert!(
        layer_types.contains(&LayerType::Softmax)
            || layer_types.contains(&LayerType::CausalSoftmax),
        "expected Softmax or CausalSoftmax in talker_attention_layer0.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types.contains(&LayerType::MatMul),
        "expected MatMul in talker_attention_layer0.onnx, got {:?}",
        layer_types
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_load_avoice_talker_attention_3497() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let model = avoice_talker_attention_raw();
    assert_talker_attention_io_shapes(model);
    assert_talker_attention_layer_inventory(model);
}
