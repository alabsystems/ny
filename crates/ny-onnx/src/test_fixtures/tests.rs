// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[cfg(feature = "onnx-value-info")]
use crate::onnx_proto::{self, GraphProto, ModelProto, NodeProto, ValueInfoProto};
#[cfg(feature = "onnx-value-info")]
use onnx_proto::tensor_shape_proto::{dimension::Value, Dimension};

const TALKER_CONTRACT_JSON: &str = r#"{
  "version": 1,
  "model": "talker_attention_layer0.onnx",
  "activation_input": "hidden_states",
  "aux_inputs": ["cos", "sin", "mask"],
  "canonical_seq_len": 16,
  "dynamic_axes": {
    "hidden_states": {"1": "T"},
    "cos": {"2": "T"},
    "sin": {"2": "T"},
    "mask": {"2": "T", "3": "T"},
    "attn_output": {"1": "T"}
  },
  "constraints": {
    "hidden_dim": 2048,
    "rope_dim": 64,
    "rope_base": 1000000.0,
    "mask_kind": "causal_upper_neg_inf"
  }
}"#;

fn expected_talker_contract() -> AvoiceFixtureContract {
    let mut dynamic_axes = BTreeMap::new();
    dynamic_axes.insert(
        "hidden_states".to_string(),
        BTreeMap::from([("1".to_string(), "T".to_string())]),
    );
    dynamic_axes.insert(
        "cos".to_string(),
        BTreeMap::from([("2".to_string(), "T".to_string())]),
    );
    dynamic_axes.insert(
        "sin".to_string(),
        BTreeMap::from([("2".to_string(), "T".to_string())]),
    );
    dynamic_axes.insert(
        "mask".to_string(),
        BTreeMap::from([
            ("2".to_string(), "T".to_string()),
            ("3".to_string(), "T".to_string()),
        ]),
    );
    dynamic_axes.insert(
        "attn_output".to_string(),
        BTreeMap::from([("1".to_string(), "T".to_string())]),
    );

    AvoiceFixtureContract {
        version: 1,
        model: "talker_attention_layer0.onnx".to_string(),
        activation_input: "hidden_states".to_string(),
        aux_inputs: vec!["cos".to_string(), "sin".to_string(), "mask".to_string()],
        canonical_seq_len: Some(16),
        dynamic_axes,
        constraints: AvoiceFixtureConstraints {
            hidden_dim: Some(2048),
            rope_dim: Some(64),
            rope_base: Some(1_000_000.0),
            mask_kind: Some("causal_upper_neg_inf".to_string()),
            ..AvoiceFixtureConstraints::default()
        },
    }
}

#[cfg(feature = "onnx-value-info")]
fn dim_param(symbol: &str) -> Dimension {
    Dimension {
        value: Some(Value::DimParam(symbol.to_string())),
    }
}

#[cfg(feature = "onnx-value-info")]
fn dim_value(value: i64) -> Dimension {
    Dimension {
        value: Some(Value::DimValue(value)),
    }
}

#[cfg(feature = "onnx-value-info")]
fn tensor_value_info(name: &str, dims: &[Dimension]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: 1,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims.to_vec() }),
            }),
        }),
    }
}

#[cfg(feature = "onnx-value-info")]
fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|value| (*value).to_string()).collect(),
        output: outputs.iter().map(|value| (*value).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

#[cfg(feature = "onnx-value-info")]
fn kokoro_duration_proto(existing_transpose_value_info: bool) -> ModelProto {
    let mut value_info = vec![tensor_value_info(
        "logit_states",
        &[dim_value(1), dim_param("T"), dim_value(50)],
    )];
    if existing_transpose_value_info {
        value_info.push(tensor_value_info(
            "/lstm/Transpose_output_0",
            &[dim_param("T"), dim_value(1), dim_value(640)],
        ));
    }

    ModelProto {
        graph: Some(GraphProto {
            input: vec![tensor_value_info(
                "encoded_features",
                &[dim_value(1), dim_param("T"), dim_value(640)],
            )],
            output: vec![tensor_value_info(
                "duration_logits",
                &[dim_value(1), dim_param("T"), dim_value(50)],
            )],
            value_info,
            node: vec![
                node(
                    "/lstm/LSTM",
                    "LSTM",
                    &[
                        "/lstm/Transpose_output_0",
                        "W",
                        "R",
                        "B",
                        "sequence_lens",
                        "initial_h",
                        "initial_c",
                    ],
                    &["/lstm/LSTM_output_0"],
                ),
                node(
                    "/duration_proj/linear_layer/MatMul",
                    "MatMul",
                    &["/lstm/Transpose_2_output_0", "duration_weight"],
                    &["duration_proj"],
                ),
                node(
                    "/lstm/Transpose_1",
                    "Transpose",
                    &["/lstm/LSTM_output_0"],
                    &["/lstm/Transpose_1_output_0"],
                ),
                node(
                    "/lstm/Constant_3",
                    "Constant",
                    &[],
                    &["/lstm/Constant_3_output_0"],
                ),
                node(
                    "/lstm/Reshape",
                    "Reshape",
                    &["/lstm/Transpose_1_output_0", "/lstm/Constant_3_output_0"],
                    &["/lstm/Reshape_output_0"],
                ),
                node(
                    "/lstm/Transpose_2",
                    "Transpose",
                    &["/lstm/Reshape_output_0"],
                    &["/lstm/Transpose_2_output_0"],
                ),
                node(
                    "live_relu",
                    "Relu",
                    &["duration_proj"],
                    &["duration_logits"],
                ),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(feature = "onnx-value-info")]
fn dims(info: &ValueInfoProto) -> Vec<i64> {
    info.r#type
        .as_ref()
        .and_then(|ty| ty.tensor_type.as_ref())
        .and_then(|tensor| tensor.shape.as_ref())
        .expect("tensor shape")
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref() {
            Some(Value::DimValue(value)) => *value,
            other => panic!("expected concrete dimension, got {other:?}"),
        })
        .collect()
}

#[test]
fn test_load_avoice_contract_missing_returns_none_3595() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_path = temp.path().join("talker_attention_layer0.onnx");

    assert_eq!(
        avoice_contract_path(&model_path),
        temp.path().join("talker_attention_layer0.contract.json")
    );
    assert_eq!(
        load_avoice_contract(&model_path).expect("missing sidecar should not error"),
        None
    );
}

#[test]
fn test_load_avoice_contract_present_parses_3595() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_path = temp.path().join("talker_attention_layer0.onnx");
    let contract_path = avoice_contract_path(&model_path);
    std::fs::write(&contract_path, TALKER_CONTRACT_JSON).expect("write contract");

    assert_eq!(
        load_avoice_contract(&model_path).expect("valid sidecar should parse"),
        Some(expected_talker_contract())
    );
}

#[test]
fn test_load_avoice_contract_invalid_json_errors_3595() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_path = temp.path().join("kokoro_vocoder.onnx");
    let contract_path = avoice_contract_path(&model_path);
    std::fs::write(&contract_path, "{ invalid json").expect("write invalid contract");

    let error = load_avoice_contract(&model_path).expect_err("invalid JSON should error");
    match error {
        AvoiceContractLoadError::Parse { path, .. } => assert_eq!(path, contract_path),
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn test_load_avoice_contract_directory_errors_as_read_3595() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_path = temp.path().join("speaker_encoder.onnx");
    let contract_path = avoice_contract_path(&model_path);
    std::fs::create_dir(&contract_path).expect("create unreadable contract directory");

    let error = load_avoice_contract(&model_path).expect_err("directory sidecar should error");
    match error {
        AvoiceContractLoadError::Read { path, .. } => assert_eq!(path, contract_path),
        other => panic!("expected read error, got {other:?}"),
    }
}

#[test]
fn missing_required_fixture_is_an_actionable_failure() {
    // Use a deliberately untracked name in the normal fixture directory.
    // Mutating NY_TEST_MODELS_DIR here would race every parallel fixture
    // reader in this test binary; the environment lock serializes writers,
    // not unrelated readers.
    let panic = std::panic::catch_unwind(|| {
        require_test_model_with_hint(
            "definitely_missing_required_fixture.onnx",
            "Generate the required fixture.",
        );
    })
    .expect_err("a missing first-party fixture must fail, never skip");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("panic should carry an actionable string");

    assert!(message.contains("definitely_missing_required_fixture.onnx"));
    assert!(message.contains("Generate the required fixture."));
    assert!(message.contains(TEST_MODELS_ENV));
}

#[test]
fn identifies_only_external_avoice_exports() {
    for name in AVOICE_EXPORT_FILES {
        assert!(is_avoice_export(name));
    }
    assert!(!is_avoice_export("whisper_tiny_encoder.onnx"));
    assert!(!is_avoice_export("transformer_block.onnx"));
}

#[test]
fn concurrent_fixture_publication_never_exposes_partial_bytes() {
    use std::io::ErrorKind;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier,
    };

    const WRITERS: usize = 8;
    const READERS: usize = 4;
    let temp = tempfile::tempdir().expect("tempdir");
    let path = Arc::new(temp.path().join("fixture.onnx"));
    let expected = Arc::new(vec![0xA5; 512 * 1024]);
    let start = Arc::new(Barrier::new(WRITERS + READERS + 1));
    let running = Arc::new(AtomicBool::new(true));

    let mut writers = Vec::new();
    for _ in 0..WRITERS {
        let path = Arc::clone(&path);
        let expected = Arc::clone(&expected);
        let start = Arc::clone(&start);
        writers.push(std::thread::spawn(move || {
            start.wait();
            assert!(publish_test_model_atomically(&path, &expected));
        }));
    }

    let mut readers = Vec::new();
    for _ in 0..READERS {
        let path = Arc::clone(&path);
        let expected = Arc::clone(&expected);
        let start = Arc::clone(&start);
        let running = Arc::clone(&running);
        readers.push(std::thread::spawn(move || {
            start.wait();
            while running.load(Ordering::Acquire) {
                match std::fs::read(path.as_ref()) {
                    Ok(bytes) => assert_eq!(bytes, *expected),
                    Err(error) if error.kind() == ErrorKind::NotFound => {}
                    Err(error) => panic!("fixture read failed: {error}"),
                }
            }
        }));
    }

    start.wait();
    for writer in writers {
        writer.join().expect("fixture publisher");
    }
    running.store(false, Ordering::Release);
    for reader in readers {
        reader.join().expect("fixture reader");
    }
    assert_eq!(std::fs::read(path.as_ref()).expect("fixture"), *expected);
}

#[test]
#[cfg(feature = "onnx-value-info")]
fn test_specialize_kokoro_duration_predictor_for_lstm_unroll_rewrites_packet_contract_3601() {
    let mut proto = kokoro_duration_proto(false);

    specialize_kokoro_duration_predictor_for_lstm_unroll(&mut proto, 4);

    let graph = proto.graph.as_ref().expect("graph");
    assert_eq!(dims(&graph.input[0]), vec![1, 4, 640]);
    assert_eq!(dims(&graph.output[0]), vec![1, 4, 50]);
    assert_eq!(dims(&graph.value_info[0]), vec![1, 4, 50]);

    let transpose_info = graph
        .value_info
        .iter()
        .find(|info| info.name == "/lstm/Transpose_output_0")
        .expect("helper should inject missing transpose value info");
    assert_eq!(dims(transpose_info), vec![4, 1, 640]);

    let lstm_node = graph
        .node
        .iter()
        .find(|node| node.name == "/lstm/LSTM")
        .expect("lstm node should remain");
    assert!(lstm_node.input[5].is_empty(), "initial_h should be cleared");
    assert!(lstm_node.input[6].is_empty(), "initial_c should be cleared");

    let matmul_node = graph
        .node
        .iter()
        .find(|node| node.name == "/duration_proj/linear_layer/MatMul")
        .expect("matmul node should remain");
    assert_eq!(matmul_node.input[0], "/lstm/LSTM_output_0");

    for dead_node in [
        "/lstm/Transpose_1",
        "/lstm/Constant_3",
        "/lstm/Reshape",
        "/lstm/Transpose_2",
    ] {
        assert!(
            graph.node.iter().all(|node| node.name != dead_node),
            "{dead_node} should be pruned"
        );
    }
    assert!(
        graph.node.iter().any(|node| node.name == "live_relu"),
        "helper should preserve unrelated live nodes"
    );
}

#[test]
#[cfg(feature = "onnx-value-info")]
fn test_specialize_kokoro_duration_predictor_for_lstm_unroll_avoids_duplicate_value_info_3601() {
    let mut proto = kokoro_duration_proto(true);

    specialize_kokoro_duration_predictor_for_lstm_unroll(&mut proto, 6);

    let graph = proto.graph.as_ref().expect("graph");
    let transpose_entries = graph
        .value_info
        .iter()
        .filter(|info| info.name == "/lstm/Transpose_output_0")
        .collect::<Vec<_>>();
    assert_eq!(
        transpose_entries.len(),
        1,
        "existing transpose value_info should be reused instead of duplicated"
    );
    assert_eq!(dims(transpose_entries[0]), vec![6, 1, 640]);
}
