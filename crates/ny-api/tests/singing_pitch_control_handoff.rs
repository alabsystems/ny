// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "propagate")]

//! Singing-style traced-producer handoff through the curated `GraphModel` API.
//!
//! The flow now routes through a native verifier builder packet, but the
//! ny-owned external contract remains `GraphModel ->
//! build_graph_network(...)`. This test proves committed `HEAD` already
//! supports a score-indexed pitch-control head through that stable boundary.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_api::model::{
    AttributeValue, DataType, GraphModel, GraphNetworkOptions, LayerSpec, LayerType, NetworkSpec,
    TensorSpec, WeightStore,
};
use ny_api::verify::{PropagationConfig, PropagationMethod, Verifier};
use ny_api::{Bound, VerificationResult, VerificationSpec};

fn tensor_spec(name: &str, shape: &[i64]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: DataType::Float32,
    }
}

fn linear_layer(name: &str, input: &str, weight: &str, bias: &str, output: &str) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Linear,
        inputs: vec![input.to_string(), weight.to_string(), bias.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
    }
}

fn relu_layer(name: &str, input: &str, output: &str) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes: HashMap::new(),
    }
}

fn singing_pitch_control_graph_model() -> GraphModel {
    let network_spec = NetworkSpec {
        name: "singing-pitch-control".to_string(),
        inputs: vec![tensor_spec("score_notes", &[1, 4])],
        outputs: vec![tensor_spec("pitch_notes", &[1, 4])],
        layers: vec![
            linear_layer(
                "pitch_encoder",
                "score_notes",
                "encoder_weight",
                "encoder_bias",
                "encoded_notes",
            ),
            relu_layer("pitch_relu", "encoded_notes", "encoded_relu"),
            linear_layer(
                "pitch_projection",
                "encoded_relu",
                "projection_weight",
                "projection_bias",
                "pitch_notes",
            ),
        ],
        param_count: 0,
    };

    let mut weights = WeightStore::new();
    weights.insert(
        "encoder_weight".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[4, 4]),
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        .expect("valid encoder weights"),
    );
    weights.insert(
        "encoder_bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 1.0, 1.0]).expect("valid encoder bias"),
    );
    weights.insert(
        "projection_weight".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[4, 4]),
            vec![
                0.5, 0.0, 0.0, 0.0, //
                0.0, 0.5, 0.0, 0.0, //
                0.0, 0.0, 0.5, 0.0, //
                0.0, 0.0, 0.0, 0.5,
            ],
        )
        .expect("valid projection weights"),
    );
    weights.insert(
        "projection_bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.125, 0.25, 0.375])
            .expect("valid projection bias"),
    );

    GraphModel::new(network_spec, weights).with_tensor_shapes(HashMap::from([
        ("score_notes".to_string(), vec![1, 4]),
        ("encoded_notes".to_string(), vec![1, 4]),
        ("encoded_relu".to_string(), vec![1, 4]),
        ("pitch_notes".to_string(), vec![1, 4]),
    ]))
}

fn note_bounds() -> Vec<Bound> {
    vec![
        Bound::new(0.0, 0.25),
        Bound::new(0.25, 0.5),
        Bound::new(0.5, 0.75),
        Bound::new(0.75, 1.0),
    ]
}

fn expected_pitch_bounds() -> Vec<Bound> {
    vec![
        Bound::new(0.5, 0.625),
        Bound::new(0.75, 0.875),
        Bound::new(1.0, 1.125),
        Bound::new(1.25, 1.375),
    ]
}

/// Verification target for the spec: the nominal pitch box widened by a small
/// margin. CROWN computes these bounds exactly in infinite precision, but the
/// f32 GEMM reduction order is not bit-stable across linear-algebra backend
/// versions (e.g. faer), so the realised bounds sit a few ULPs (~3e-7) outside
/// the knife-edge nominal box. The margin keeps the test checking the real
/// behaviour (CROWN stays sound and tight) instead of a backend's exact
/// rounding; a genuine looseness regression would exceed it by far.
fn spec_pitch_target() -> Vec<Bound> {
    const MARGIN: f32 = 1e-4;
    expected_pitch_bounds()
        .into_iter()
        .map(|b| Bound::new(b.lower() - MARGIN, b.upper() + MARGIN))
        .collect()
}

#[test]
fn singing_pitch_control_graph_model_builds_through_curated_api() {
    let graph = singing_pitch_control_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("singing pitch control GraphModel should build through ny_api::model");

    let projection = graph
        .node("pitch_projection")
        .expect("pitch projection node should exist");
    assert_eq!(
        projection.inputs(),
        &["pitch_relu".to_string()],
        "score-indexed control graphs should preserve the encoder -> relu -> projection chain"
    );
}

#[test]
fn singing_pitch_control_graph_model_verifies_per_note_crown_bounds() {
    let graph = singing_pitch_control_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("singing pitch control GraphModel should build");
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        note_bounds(),
        spec_pitch_target(),
        Some(5_000),
        Some(vec![1, 4]),
    )
    .expect("valid singing pitch control verification spec");

    let result = verifier.verify_graph(&graph, &spec).expect(
        "CROWN should verify the singing pitch control graph through the curated GraphModel path",
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
                "the singing control proof surface should stay on CROWN rather than falling back"
            );
            // The realised CROWN bounds equal the nominal pitch box up to f32
            // reduction-order rounding (~ a few ULPs); compare with a tolerance
            // rather than bit-for-bit so the test is stable across backend
            // versions while still pinning the per-note values.
            let nominal = expected_pitch_bounds();
            assert_eq!(output_bounds.len(), nominal.len(), "per-note bound count");
            for (i, (got, want)) in output_bounds.iter().zip(nominal.iter()).enumerate() {
                assert!(
                    (got.lower() - want.lower()).abs() <= 1e-4
                        && (got.upper() - want.upper()).abs() <= 1e-4,
                    "note {i} pitch bound off nominal beyond f32 rounding: \
                     got [{}, {}] want [{}, {}]",
                    got.lower(),
                    got.upper(),
                    want.lower(),
                    want.upper()
                );
            }
            assert!(
                output_bounds
                    .iter()
                    .all(|bound| bound.lower().is_finite() && bound.upper().is_finite()),
                "all per-note pitch bounds must remain finite"
            );
            assert!(
                output_bounds
                    .windows(2)
                    .all(|pair| pair[0].upper() < pair[1].lower()),
                "successive note bounds should remain strictly ordered in this synthetic control head"
            );
        }
        other => panic!("expected singing pitch control graph to verify, got {other:?}"),
    }
}
