// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::LayerType;

use super::lookup::{layer_supports_missing_output_shape_fallback, ShapeLookup};
use super::tests::{
    load_duration_predictor_timing_model, load_kokoro_vocoder_timing_model,
    load_speaker_encoder_timing_model, load_talker_attention_timing_model,
};
use crate::OnnxModel;

#[derive(Debug)]
struct ShapeFallbackUse {
    layer_name: String,
    layer_type: LayerType,
    output_name: String,
    inferred_shape: Vec<usize>,
    runtime_input_shapes: Vec<Vec<usize>>,
}

fn collect_shape_fallback_uses(model: &OnnxModel) -> Vec<ShapeFallbackUse> {
    let mut lookup = ShapeLookup::new(model);
    let mut fallback_uses = Vec::new();

    for layer in &model.network.layers {
        let mut output_shapes = Vec::new();
        for output_name in layer
            .outputs
            .iter()
            .filter(|name| !model.constant_tensors().contains(*name))
        {
            match lookup.tensor_shape(output_name) {
                Ok(shape) => output_shapes.push(shape),
                Err(_) => {
                    let runtime_input_shapes =
                        super::layer_metadata::activation_input_names(model, layer)
                            .into_iter()
                            .filter_map(|name| lookup.tensor_shape(name).ok())
                            .collect::<Vec<_>>();
                    let inferred_shape = lookup.infer_output_shape(layer).unwrap_or_else(|e| {
                        panic!(
                            "shape fallback should infer '{}' in layer '{}' (type {}): {e}",
                            output_name, layer.name, layer.layer_type
                        )
                    });
                    fallback_uses.push(ShapeFallbackUse {
                        layer_name: layer.name.clone(),
                        layer_type: layer.layer_type.clone(),
                        output_name: output_name.clone(),
                        inferred_shape: inferred_shape.clone(),
                        runtime_input_shapes,
                    });
                    output_shapes.push(inferred_shape);
                }
            }
        }

        for (name, shape) in layer
            .outputs
            .iter()
            .filter(|name| !model.constant_tensors().contains(*name))
            .zip(output_shapes.iter())
        {
            lookup.register_shape(name.clone(), shape.clone());
        }
    }

    fallback_uses
}

fn is_shape_changing_audited_fallback_layer(layer_type: &LayerType) -> bool {
    matches!(
        layer_type,
        LayerType::ReduceMean
            | LayerType::ReduceSum
            | LayerType::Concat
            | LayerType::MatMul
            | LayerType::Reshape
            | LayerType::Slice
            | LayerType::Transpose
            | LayerType::Unsqueeze
            | LayerType::Conv1d
            | LayerType::Conv2d
            | LayerType::ConvTranspose1d
            | LayerType::ConvTranspose2d
            | LayerType::Linear
            | LayerType::Add
            | LayerType::Sub
            | LayerType::Mul
            | LayerType::Div
            | LayerType::Pow
            | LayerType::Min
            | LayerType::Max
            | LayerType::Pad
    )
}

#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_timing_shape_fallback_stays_on_audited_layers_3498() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    crate::test_fixtures::assert_test_model_available!("kokoro_duration_predictor.onnx");
    let models = [
        ("speaker encoder", load_speaker_encoder_timing_model()),
        ("talker attention", load_talker_attention_timing_model()),
        ("kokoro vocoder", load_kokoro_vocoder_timing_model()),
        ("duration predictor", load_duration_predictor_timing_model()),
    ];
    let mut total_fallbacks = 0usize;

    for (label, model) in models {
        let fallback_uses = collect_shape_fallback_uses(&model);
        total_fallbacks += fallback_uses.len();

        for fallback in fallback_uses {
            assert!(
                !fallback.runtime_input_shapes.is_empty(),
                "{label}: fallback layer '{}' (type {}) had no known runtime input shapes",
                fallback.layer_name,
                fallback.layer_type
            );
            assert!(
                layer_supports_missing_output_shape_fallback(
                    model
                        .network
                        .layers
                        .iter()
                        .find(|layer| layer.name == fallback.layer_name)
                        .expect("fallback layer should exist in the model")
                ),
                "{label}: fallback output '{}' used unaudited layer '{}' ({}) with runtime input shapes {:?} and inferred shape {:?}",
                fallback.output_name,
                fallback.layer_name,
                fallback.layer_type,
                fallback.runtime_input_shapes,
                fallback.inferred_shape
            );
            if !fallback
                .runtime_input_shapes
                .iter()
                .all(|shape| *shape == fallback.inferred_shape)
            {
                assert!(
                    is_shape_changing_audited_fallback_layer(&fallback.layer_type),
                    "{label}: fallback output '{}' in layer '{}' ({}) inferred {:?} from runtime input shapes {:?}",
                    fallback.output_name,
                    fallback.layer_name,
                    fallback.layer_type,
                    fallback.inferred_shape,
                    fallback.runtime_input_shapes
                );
            }
        }
    }

    assert!(
        total_fallbacks > 0,
        "avoice timing smoke should exercise at least one missing-shape fallback path"
    );
}
