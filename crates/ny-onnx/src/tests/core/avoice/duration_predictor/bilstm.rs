// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::proof_head::{avg_bound_width, KOKORO_DURATION_BUCKETS};
use super::*;
use crate::onnx_proto::{
    self, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::types::BoundsProvenance;
use prost::Message;

// ---------------------------------------------------------------------------
// BiLSTM full-sequence admission regression.
//
// This models the path the real Kokoro duration predictor wants to use:
// - ONNX LSTM node with num_directions=2 (BiLSTM)
// - Y output (full sequence, all timesteps)
// - Linear projection to 50-bucket logits
//
// Raw ONNX Y is four-dimensional. The old lowering substituted a three-
// dimensional convenience layout under the same name; these tests require
// model admission to stop until the exact representation is implemented.
//
// Sources:
// - ./avoice/scripts/export_kokoro_onnx.py (DurationWrapper)
// - crates/ny-onnx/src/loader/lstm_unroll/tests_bilstm.rs (protobuf helpers)
// ---------------------------------------------------------------------------

fn bilstm_tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn bilstm_float_tensor(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name: name.to_string(),
        float_data: data,
        ..Default::default()
    }
}

/// Build an ONNX BiLSTM node with Y output (full sequence), layout=1.
fn make_bilstm_onnx_node(hidden_size: i64) -> NodeProto {
    NodeProto {
        op_type: "LSTM".to_string(),
        input: vec![
            "encoded_features".to_string(),
            "W".to_string(),
            "R".to_string(),
            "B".to_string(),
        ],
        output: vec!["Y".to_string(), String::new(), String::new()],
        name: "bilstm".to_string(),
        attribute: vec![
            AttributeProto {
                name: "hidden_size".to_string(),
                r#type: onnx_proto::attribute_proto::AttributeType::Int as i32,
                i: Some(hidden_size),
                ..Default::default()
            },
            AttributeProto {
                name: "direction".to_string(),
                r#type: onnx_proto::attribute_proto::AttributeType::String as i32,
                s: Some(b"bidirectional".to_vec()),
                ..Default::default()
            },
            AttributeProto {
                name: "layout".to_string(),
                r#type: onnx_proto::attribute_proto::AttributeType::Int as i32,
                i: Some(1),
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

/// Deterministic pseudo-random weight data for test reproducibility.
fn test_weights(count: i64, modulus: i64, offset: f32, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i % modulus) as f32 + offset) * scale)
        .collect()
}

/// Build the legacy BiLSTM logits fixture. Its projection assumes the old 3-D
/// substitution for Y, so it is useful only as a fail-closed admission test.
fn build_bilstm_logits_model_bytes(seq: i64, input_size: i64, hidden_size: i64) -> Vec<u8> {
    let num_dir = 2_i64;
    let proj_dim = num_dir * hidden_size;
    let num_buckets = KOKORO_DURATION_BUCKETS as i64;

    let nodes = vec![
        make_bilstm_onnx_node(hidden_size),
        NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["Y".to_string(), "W_proj".to_string()],
            output: vec!["duration_logits".to_string()],
            name: "proj_matmul".to_string(),
            ..Default::default()
        },
    ];

    let graph = GraphProto {
        name: "bilstm_duration_logits".to_string(),
        input: vec![bilstm_tensor_value_info(
            "encoded_features",
            &[1, seq, input_size],
        )],
        output: vec![bilstm_tensor_value_info(
            "duration_logits",
            &[1, seq, num_buckets],
        )],
        node: nodes,
        initializer: vec![
            bilstm_float_tensor(
                "W",
                &[num_dir, 4 * hidden_size, input_size],
                test_weights(num_dir * 4 * hidden_size * input_size, 7, -3.0, 0.1),
            ),
            bilstm_float_tensor(
                "R",
                &[num_dir, 4 * hidden_size, hidden_size],
                test_weights(num_dir * 4 * hidden_size * hidden_size, 5, -2.0, 0.05),
            ),
            bilstm_float_tensor(
                "B",
                &[num_dir, 8 * hidden_size],
                vec![0.0; (num_dir * 8 * hidden_size) as usize],
            ),
            bilstm_float_tensor(
                "W_proj",
                &[proj_dim, num_buckets],
                test_weights(proj_dim * num_buckets, 11, -5.0, 0.02),
            ),
        ],
        ..Default::default()
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-bilstm-logits-integration-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    }
    .encode_to_vec()
}

fn assert_bilstm_sequence_output_rejected(name: &str, bytes: &[u8]) {
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    let error = crate::load_onnx_bytes_with_config(name, bytes, &config)
        .expect_err("bidirectional sequence output must fail closed");
    assert!(error.to_string().contains("LSTM"), "{error}");
}

/// The former end-to-end proof used a three-dimensional substitute for ONNX's
/// four-dimensional bidirectional Y. It must stop at model admission until the
/// exact layout is implemented.
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_bilstm_duration_predictor_sequence_output_fails_closed_3497() {
    let bytes = build_bilstm_logits_model_bytes(4, 8, 4);
    assert_bilstm_sequence_output_rejected("bilstm_dur_logits", &bytes);
}

/// Log and assert CROWN-vs-IBP comparison: CROWN no looser per dimension.
pub(super) fn assert_crown_no_looser_than_ibp_durations(
    crown_durations: &BoundedTensor,
    ibp_durations: &BoundedTensor,
    provenance: &BoundsProvenance,
    label: &str,
) {
    let ibp_width = avg_bound_width(ibp_durations);
    let crown_width = avg_bound_width(crown_durations);

    eprintln!("--- {label} CROWN duration results ---");
    eprintln!("  provenance: {provenance:?}");
    eprintln!("  IBP avg duration width:   {ibp_width:.6}");
    eprintln!("  CROWN avg duration width: {crown_width:.6}");
    if ibp_width > 0.0 {
        eprintln!(
            "  tightening: {:.2}%",
            (1.0 - crown_width / ibp_width) * 100.0
        );
    }
    for (idx, ((&ibp_lo, &ibp_hi), (&c_lo, &c_hi))) in ibp_durations
        .lower()
        .iter()
        .zip(ibp_durations.upper().iter())
        .zip(
            crown_durations
                .lower()
                .iter()
                .zip(crown_durations.upper().iter()),
        )
        .enumerate()
    {
        eprintln!(
            "  dim {idx}: IBP [{ibp_lo:.4}, {ibp_hi:.4}] w={:.4}  CROWN [{c_lo:.4}, {c_hi:.4}] w={:.4}",
            ibp_hi - ibp_lo,
            c_hi - c_lo
        );
        assert!(
            c_lo >= ibp_lo - 1e-5,
            "{label} CROWN lower >= IBP lower at dim {idx}: crown={c_lo}, ibp={ibp_lo}"
        );
        assert!(
            c_hi <= ibp_hi + 1e-5,
            "{label} CROWN upper <= IBP upper at dim {idx}: crown={c_hi}, ibp={ibp_hi}"
        );
    }
}

/// The unauthenticated layout must not reach CROWN either.
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_bilstm_duration_predictor_never_reaches_crown_3497() {
    let bytes = build_bilstm_logits_model_bytes(4, 8, 4);
    assert_bilstm_sequence_output_rejected("bilstm_dur_crown", &bytes);
}

/// Refusal does not depend on a verification epsilon.
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_bilstm_duration_predictor_refusal_is_epsilon_independent_3497() {
    let bytes = build_bilstm_logits_model_bytes(4, 8, 4);
    assert_bilstm_sequence_output_rejected("bilstm_dur_eps", &bytes);
}
