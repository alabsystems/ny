// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "propagate")]

//! Pronunciation-style traced-producer handoff through the curated `GraphModel` API.
//!
//! The flow routes through a native verifier packet where ny materializes the
//! graph and a downstream external consumer interprets a packed
//! `[f0..., energy...]` output vector as a phoneme certificate. This test keeps
//! the ny-side scope narrow: it proves the current
//! `GraphModelBuilder -> build_graph_network(...)` handoff supports a packed
//! four-lane phoneme-feature head with structural text/style slicing.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_api::model::{
    AttributeValue, DataType, GraphModel, GraphModelBuilder, GraphNetworkOptions, LayerSpec,
    LayerType,
};
use ny_api::verify::{PropagationConfig, PropagationMethod, Verifier};
use ny_api::{Bound, VerificationResult, VerificationSpec};
use ny_tensor::{next_down_f32, next_up_f32};

fn layer_spec(
    name: &str,
    layer_type: LayerType,
    inputs: &[&str],
    outputs: &[&str],
    attributes: HashMap<String, AttributeValue>,
) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: inputs.iter().map(|value| value.to_string()).collect(),
        outputs: outputs.iter().map(|value| value.to_string()).collect(),
        weights: None,
        attributes,
    }
}

fn slice_layer(name: &str, input: &str, output: &str, start: i64, end: i64) -> LayerSpec {
    layer_spec(
        name,
        LayerType::Slice,
        &[input],
        &[output],
        HashMap::from([
            // This packet's graph inputs are rank-1 ([4]), so the model is
            // globally unbatched and axes describe the tensors VERBATIM (no
            // batch axis was ever stripped): slicing the flat feature vector
            // is axis 0. (The legacy convention wrote batched ONNX axis=1
            // here and relied on a blanket batch-squeeze decrement.)
            ("axis".to_string(), AttributeValue::Int(0)),
            ("start".to_string(), AttributeValue::Int(start)),
            ("end".to_string(), AttributeValue::Int(end)),
        ]),
    )
}

fn add_layer(name: &str, left: &str, right: &str, output: &str) -> LayerSpec {
    layer_spec(
        name,
        LayerType::Add,
        &[left, right],
        &[output],
        HashMap::new(),
    )
}

fn relu_layer(name: &str, input: &str, output: &str) -> LayerSpec {
    layer_spec(name, LayerType::ReLU, &[input], &[output], HashMap::new())
}

fn linear_layer(name: &str, input: &str, weight: &str, bias: &str, output: &str) -> LayerSpec {
    layer_spec(
        name,
        LayerType::Linear,
        &[input, weight, bias],
        &[output],
        HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
    )
}

fn phoneme_feature_graph_model() -> GraphModel {
    GraphModelBuilder::new("phoneme-feature-certificate")
        .input("flat_input", &[4], DataType::Float32)
        .output("phoneme_features", &[4], DataType::Float32)
        .weight(
            "phoneme_projection_weight",
            ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![
                    1.0, 0.0, //
                    0.0, 1.0, //
                    0.5, 0.0, //
                    0.0, 0.5,
                ],
            )
            .expect("valid phoneme projection weight"),
        )
        .weight(
            "phoneme_projection_bias",
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, 0.5, 2.0, 2.0])
                .expect("valid phoneme projection bias"),
        )
        .layer(slice_layer("text_slice", "flat_input", "text_flat", 0, 2))
        .layer(slice_layer("style_slice", "flat_input", "style_flat", 2, 4))
        .layer(add_layer(
            "style_condition",
            "text_flat",
            "style_flat",
            "conditioned_features",
        ))
        .layer(relu_layer(
            "stability_relu",
            "conditioned_features",
            "activated_features",
        ))
        .layer(linear_layer(
            "phoneme_projection",
            "activated_features",
            "phoneme_projection_weight",
            "phoneme_projection_bias",
            "phoneme_features",
        ))
        .tensor_shape("flat_input", &[4])
        .tensor_shape("text_flat", &[2])
        .tensor_shape("style_flat", &[2])
        .tensor_shape("conditioned_features", &[2])
        .tensor_shape("activated_features", &[2])
        .tensor_shape("phoneme_features", &[4])
        .build()
}

fn input_bounds() -> Vec<Bound> {
    vec![
        Bound::new(0.0, 0.5),
        Bound::new(0.5, 1.0),
        Bound::new(0.25, 0.5),
        Bound::new(0.75, 1.0),
    ]
}

fn expected_phoneme_feature_bounds() -> Vec<Bound> {
    vec![
        Bound::new(next_down_f32(0.75), next_up_f32(1.5)),
        Bound::new(next_down_f32(1.75), next_up_f32(2.5)),
        Bound::new(next_down_f32(2.125), next_up_f32(2.5)),
        Bound::new(next_down_f32(2.625), next_up_f32(3.0)),
    ]
}

#[test]
fn phoneme_feature_graph_model_builds_through_curated_api() {
    let graph = phoneme_feature_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("phoneme feature GraphModel should build through ny_api::model");

    let style_condition = graph
        .node("style_condition")
        .expect("style-conditioned merge should exist");
    assert_eq!(
        style_condition.inputs(),
        &["text_slice".to_string(), "style_slice".to_string()],
        "the native packet should keep the split text/style paths separate until style conditioning"
    );

    let projection = graph
        .node("phoneme_projection")
        .expect("packed phoneme feature projection should exist");
    assert_eq!(
        projection.inputs(),
        &["stability_relu".to_string()],
        "the packed phoneme feature head should stay downstream of the conditioned activation path"
    );

    assert_eq!(
        graph.output_name(),
        "phoneme_projection",
        "the final graph output should be the packed phoneme feature projection"
    );
}

#[test]
fn phoneme_feature_graph_model_verifies_packed_f0_energy_bounds() {
    let graph = phoneme_feature_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("phoneme feature GraphModel should build");
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        input_bounds(),
        expected_phoneme_feature_bounds(),
        Some(5_000),
        Some(vec![4]),
    )
    .expect("valid phoneme feature verification spec");

    let result = verifier.verify_graph(&graph, &spec).expect(
        "CROWN should verify the packed phoneme feature graph through the curated GraphModel path",
    );

    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                actual_method.as_deref(),
                Some("Crown"),
                "the phoneme feature packet should stay on CROWN rather than falling back"
            );
            assert_eq!(
                output_bounds,
                expected_phoneme_feature_bounds(),
                "packed [f0..., energy...] bounds should remain finite and ordered for downstream certificate interpretation"
            );
            assert!(
                output_bounds
                    .iter()
                    .all(|bound| bound.lower().is_finite() && bound.upper().is_finite()),
                "all phoneme feature bounds must remain finite"
            );
        }
        other => panic!("expected phoneme feature graph to verify, got {other:?}"),
    }
}
