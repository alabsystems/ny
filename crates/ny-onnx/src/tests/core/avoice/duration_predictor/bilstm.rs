// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::proof_head::{
    assert_positive_expected_durations, assert_production_duration_counts, avg_bound_width,
    kokoro_duration_count_bounds_from_logits, kokoro_expected_duration_bounds_from_logits,
    KOKORO_DEFAULT_SPEED, KOKORO_DURATION_BUCKETS,
};
use super::*;
use crate::onnx_proto::{
    self, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ndarray::{ArrayD, IxDyn};
use ny_propagate::types::BoundsProvenance;
use prost::Message;

// ---------------------------------------------------------------------------
// BiLSTM integration pipeline (ONNX BiLSTM → unrolling → graph → IBP →
// external proof head → positive expected durations / production duration counts)
//
// This tests the exact pipeline the real Kokoro duration predictor will use:
// - ONNX LSTM node with num_directions=2 (BiLSTM)
// - Y output (full sequence, all timesteps)
// - Linear projection to 50-bucket logits
// - External sigmoid+sum proof head (NOT in the ONNX model)
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
                i: hidden_size,
                ..Default::default()
            },
            AttributeProto {
                name: "direction".to_string(),
                r#type: onnx_proto::attribute_proto::AttributeType::String as i32,
                s: b"bidirectional".to_vec(),
                ..Default::default()
            },
            AttributeProto {
                name: "layout".to_string(),
                r#type: onnx_proto::attribute_proto::AttributeType::Int as i32,
                i: 1,
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

/// Build a BiLSTM model that outputs LOGITS (not expected durations).
///
/// Architecture matches the real Kokoro DurationWrapper:
///   encoded_features [1, T, input_size] → BiLSTM → Y [1, T, 2*H]
///   → MatMul(W_proj [2*H, 50]) → duration_logits [1, T, 50]
///
/// The post-logit interval head is applied externally so tests can check both
/// the raw expected-duration property and the production count surface.
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

fn log_bilstm_duration_predictor_results(
    seq: i64,
    input_size: i64,
    hidden_size: i64,
    logits: &BoundedTensor,
    expected_durations: &BoundedTensor,
    duration_counts: &BoundedTensor,
) {
    eprintln!("--- BiLSTM duration predictor integration results ---");
    eprintln!("  seq_len: {seq}, input_size: {input_size}, hidden_size: {hidden_size}");
    eprintln!("  logit shape: {:?}", logits.lower().shape());
    eprintln!(
        "  expected-duration shape: {:?}",
        expected_durations.lower().shape()
    );
    eprintln!(
        "  avg expected-duration width: {:.6}",
        avg_bound_width(expected_durations)
    );
    eprintln!(
        "  avg production-count width: {:.6}",
        avg_bound_width(duration_counts)
    );
    for (t, (&lo, &hi)) in expected_durations
        .lower()
        .iter()
        .zip(expected_durations.upper().iter())
        .enumerate()
    {
        eprintln!(
            "  expected duration timestep {t}: [{lo:.4}, {hi:.4}] (width: {:.4})",
            hi - lo
        );
    }
    for (t, (&lo, &hi)) in duration_counts
        .lower()
        .iter()
        .zip(duration_counts.upper().iter())
        .enumerate()
    {
        eprintln!(
            "  production count timestep {t}: [{lo:.4}, {hi:.4}] (width: {:.4})",
            hi - lo
        );
    }
}

/// End-to-end: BiLSTM ONNX → LSTM unrolling → graph → IBP → external proof head
/// → positive expected durations / production duration counts at every timestep.
///
/// This is the integration test for the exact pipeline the real Kokoro duration
/// predictor will use. The model outputs logits [1, T, 50], and the proof head
/// is applied externally:
/// - `kokoro_expected_duration_bounds_from_logits` for the non-vacuous research
///   property
/// - `kokoro_duration_count_bounds_from_logits(..., 1.0)` for the production
///   `duration_to_counts` surface
///
/// Sources:
/// - ./avoice/scripts/export_kokoro_onnx.py (DurationWrapper arch)
/// - designs/2026-03-11-avoice-phase1-onnx-execution.md (section 4)
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_bilstm_duration_predictor_logits_to_positive_expected_duration_3497() {
    let seq = 4_i64;
    let input_size = 8_i64;
    let hidden_size = 4_i64;

    let bytes = build_bilstm_logits_model_bytes(seq, input_size, hidden_size);
    let model = crate::loader::load_onnx_bytes("bilstm_dur_logits", &bytes)
        .expect("BiLSTM duration logits model should load after LSTM unrolling");

    // Verify the ONNX loader correctly unrolled the BiLSTM
    assert_eq!(model.network.inputs.len(), 1, "single activation input");
    let input_spec = common::input_spec_by_name(&model, "encoded_features");
    assert_eq!(
        input_spec.shape.len(),
        3,
        "encoded_features should be 3D [B, T, D]"
    );

    let graph = model
        .to_graph_network()
        .expect("BiLSTM duration predictor should convert to graph");

    // Run IBP to get logit bounds
    let center = ArrayD::zeros(IxDyn(&[seq as usize, input_size as usize]));
    let input = BoundedTensor::from_epsilon(center, 1e-3).expect("valid epsilon ball");
    let logits = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on BiLSTM graph");

    common::assert_finite_and_ordered(&logits, "BiLSTM duration predictor logit bounds");
    assert_eq!(
        logits.lower().shape().last().copied(),
        Some(KOKORO_DURATION_BUCKETS),
        "logit output should have {KOKORO_DURATION_BUCKETS} buckets, got shape {:?}",
        logits.lower().shape()
    );

    // Check both the non-vacuous Bernoulli-sum property and the production
    // avoice count surface.
    let expected_durations = kokoro_expected_duration_bounds_from_logits(&logits);
    assert_positive_expected_durations(&expected_durations);
    let duration_counts = kokoro_duration_count_bounds_from_logits(&logits, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&duration_counts);

    // Every timestep must have a positive lower bound
    assert_eq!(
        expected_durations.lower().len(),
        seq as usize,
        "expected one duration per timestep"
    );

    log_bilstm_duration_predictor_results(
        seq,
        input_size,
        hidden_size,
        &logits,
        &expected_durations,
        &duration_counts,
    );
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

/// CROWN backward through the BiLSTM duration predictor → sigmoid+sum proof
/// head → positive expected durations at every timestep.
///
/// The BiLSTM has nonlinear LSTM gates (sigmoid/tanh) where CROWN linear
/// relaxation should provide tighter bounds than naive IBP.
///
/// Sources:
/// - designs/2026-03-11-avoice-phase1-onnx-execution.md (section 4)
/// - alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/abcrown.py (CROWN backward)
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_bilstm_duration_predictor_crown_positive_expected_duration_3497() {
    let seq = 4_i64;
    let input_size = 8_i64;
    let hidden_size = 4_i64;

    let bytes = build_bilstm_logits_model_bytes(seq, input_size, hidden_size);
    let model = crate::loader::load_onnx_bytes("bilstm_dur_crown", &bytes)
        .expect("BiLSTM duration predictor should load");
    let graph = model
        .to_graph_network()
        .expect("BiLSTM duration predictor should convert to graph");

    let center = ArrayD::zeros(IxDyn(&[seq as usize, input_size as usize]));
    let input = BoundedTensor::from_epsilon(center, 1e-3).expect("valid epsilon ball");

    let crown_result = graph
        .propagate_crown_with_provenance(&input)
        .expect("CROWN backward should succeed on BiLSTM graph");

    common::assert_finite_and_ordered(&crown_result.bounds, "BiLSTM CROWN logit bounds");
    assert_eq!(
        crown_result.bounds.lower().shape().last().copied(),
        Some(KOKORO_DURATION_BUCKETS),
        "CROWN logit output should have {KOKORO_DURATION_BUCKETS} buckets"
    );

    let crown_durations = kokoro_expected_duration_bounds_from_logits(&crown_result.bounds);
    assert_positive_expected_durations(&crown_durations);
    let crown_counts =
        kokoro_duration_count_bounds_from_logits(&crown_result.bounds, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&crown_counts);

    let ibp_logits = graph.propagate_ibp(&input).expect("IBP");
    let ibp_durations = kokoro_expected_duration_bounds_from_logits(&ibp_logits);
    assert_crown_no_looser_than_ibp_durations(
        &crown_durations,
        &ibp_durations,
        &crown_result.provenance,
        "BiLSTM",
    );
}

/// BiLSTM duration predictor tighter bounds at smaller epsilon.
///
/// Validates that the BiLSTM pipeline preserves the fundamental property that
/// tighter input regions yield tighter output bounds, same as the surrogate.
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_bilstm_duration_predictor_tighter_bounds_at_smaller_epsilon_3497() {
    let seq = 4_i64;
    let input_size = 8_i64;
    let hidden_size = 4_i64;

    let bytes = build_bilstm_logits_model_bytes(seq, input_size, hidden_size);
    let model = crate::loader::load_onnx_bytes("bilstm_dur_eps", &bytes)
        .expect("BiLSTM duration predictor should load");
    let graph = model
        .to_graph_network()
        .expect("BiLSTM duration predictor should convert to graph");

    let widths: Vec<f32> = [1e-2, 1e-4]
        .iter()
        .map(|&eps| {
            let center = ArrayD::zeros(IxDyn(&[seq as usize, input_size as usize]));
            let input = BoundedTensor::from_epsilon(center, eps).expect("valid epsilon");
            let logits = graph.propagate_ibp(&input).expect("IBP");
            let durations = kokoro_expected_duration_bounds_from_logits(&logits);
            avg_bound_width(&durations)
        })
        .collect();

    assert!(
        widths[1] < widths[0],
        "BiLSTM: smaller epsilon should yield tighter duration bounds: \
         eps=1e-2 width={:.6}, eps=1e-4 width={:.6}",
        widths[0],
        widths[1]
    );
}
