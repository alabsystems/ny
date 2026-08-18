// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for bidirectional LSTM (BiLSTM) unrolling and Kokoro duration predictor.
//!
//! Split from `tests.rs` to stay under the 500-line file limit.
//! Part of #3497: Kokoro BiLSTM duration predictor CROWN verification.

use super::node_builder::make_int_attr;
use super::*;
use crate::onnx_proto::{
    self, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use ndarray::{ArrayD, IxDyn};

fn tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
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

fn make_float_tensor(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1, // FLOAT
        name: name.to_string(),
        float_data: data,
        ..Default::default()
    }
}

fn make_bilstm_node(x_input: &str, hidden_size: i64, layout: i64) -> NodeProto {
    NodeProto {
        op_type: "LSTM".to_string(),
        input: vec![
            x_input.to_string(),
            "W".to_string(),
            "R".to_string(),
            "B".to_string(),
            String::new(), // sequence_lens
            String::new(), // initial_hidden_state
            String::new(), // initial_cell_state
        ],
        output: vec!["Y".to_string(), "Y_h".to_string(), "Y_c".to_string()],
        name: "bilstm0".to_string(),
        domain: String::new(),
        attribute: vec![
            make_int_attr("hidden_size", hidden_size),
            make_int_attr("layout", layout),
            onnx_proto::AttributeProto {
                name: "direction".to_string(),
                s: Some(b"bidirectional".to_vec()),
                r#type: onnx_proto::attribute_type::STRING,
                ..Default::default()
            },
        ],
    }
}

/// Build a bidirectional LSTM ONNX model as protobuf bytes.
///
/// Weights have shape `[2, ...]` (num_directions=2). Forward and reverse
/// directions use different weight patterns so the unroller must slice correctly.
fn build_bilstm_onnx_bytes(
    batch: i64,
    seq: i64,
    input_size: i64,
    hidden_size: i64,
    output_names: &[&str],
) -> Vec<u8> {
    use prost::Message;

    let num_directions = 2_i64;

    let w_data: Vec<f32> = (0..num_directions * 4 * hidden_size * input_size)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let r_data: Vec<f32> = (0..num_directions * 4 * hidden_size * hidden_size)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let b_data: Vec<f32> = vec![0.0; (num_directions * 8 * hidden_size) as usize];

    let mut bilstm_node = make_bilstm_node("X", hidden_size, 1);
    bilstm_node.output = output_names.iter().map(|s| s.to_string()).collect();

    let outputs: Vec<ValueInfoProto> = output_names
        .iter()
        .filter(|s| !s.is_empty())
        .map(|name| match *name {
            "Y" => tensor_value_info(name, &[batch, seq, num_directions, hidden_size]),
            "Y_h" => tensor_value_info(name, &[batch, num_directions, hidden_size]),
            "Y_c" => tensor_value_info(name, &[batch, num_directions, hidden_size]),
            _ => tensor_value_info(name, &[batch, num_directions, hidden_size]),
        })
        .collect();

    let graph = GraphProto {
        name: "bilstm_test".to_string(),
        input: vec![tensor_value_info("X", &[batch, seq, input_size])],
        output: outputs,
        node: vec![bilstm_node],
        initializer: vec![
            make_float_tensor("W", &[num_directions, 4 * hidden_size, input_size], w_data),
            make_float_tensor("R", &[num_directions, 4 * hidden_size, hidden_size], r_data),
            make_float_tensor("B", &[num_directions, 8 * hidden_size], b_data),
        ],
        ..Default::default()
    };

    let model_proto = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-bilstm-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    model_proto.encode_to_vec()
}

fn load_bilstm_fixture(name: &str, bytes: &[u8]) -> crate::OnnxModel {
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    crate::loader::load_onnx_bytes_with_config(name, bytes, &config)
        .expect("BiLSTM fixture should load without native shape inference")
}

/// Verify that a bidirectional LSTM node is unrolled into primitive ops.
#[test]
fn test_bilstm_lowering_produces_expected_node_types_3497() {
    let mut bilstm_node = make_bilstm_node("X", 4, 1);
    bilstm_node.output = vec![String::new(), "Y_h".to_string(), "Y_c".to_string()];
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, 3, 2])],
        node: vec![bilstm_node],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[2, 16, 2])));
    weights.insert("R".to_string(), ArrayD::zeros(IxDyn(&[2, 16, 4])));
    weights.insert("B".to_string(), ArrayD::zeros(IxDyn(&[2, 32])));

    let mut nodes = graph.node.clone();
    let graph_vi = graph.value_info().to_vec();
    lower_lstm_nodes(
        &mut nodes,
        &mut weights,
        &graph.input,
        &graph_vi,
        &std::collections::HashMap::new(),
    );

    assert!(
        !nodes.iter().any(|n| n.op_type == "LSTM"),
        "BiLSTM node should be replaced by unrolled primitives"
    );

    assert!(
        nodes.len() > 100,
        "BiLSTM with 3 timesteps x 2 directions should produce 100+ nodes, got {}",
        nodes.len()
    );

    let all_outputs: Vec<&str> = nodes
        .iter()
        .flat_map(|n| n.output.iter().map(|s| s.as_str()))
        .collect();
    assert!(
        all_outputs.contains(&"Y_h"),
        "BiLSTM Y_h output must be preserved"
    );
    assert!(
        all_outputs.contains(&"Y_c"),
        "BiLSTM Y_c output must be preserved"
    );
}

/// End-to-end: load BiLSTM ONNX model, parse, convert to graph, run IBP.
#[test]
fn test_bilstm_ibp_produces_finite_ordered_bounds_3497() {
    let bytes = build_bilstm_onnx_bytes(1, 3, 2, 4, &["", "Y_h", ""]);
    let model = load_bilstm_fixture("bilstm_ibp", &bytes);
    let graph = model
        .to_graph_network()
        .expect("BiLSTM model should convert to graph network");

    let center = ArrayD::zeros(IxDyn(&[3, 2]));
    let input = ny_tensor::BoundedTensor::from_epsilon(center, 0.1).expect("epsilon ball");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on unrolled BiLSTM");

    assert!(
        !output.lower().is_empty(),
        "BiLSTM IBP output must have at least one element"
    );
    for (&lo, &hi) in output.lower().iter().zip(output.upper().iter()) {
        assert!(
            lo.is_finite(),
            "BiLSTM lower bound must be finite, got {lo}"
        );
        assert!(
            hi.is_finite(),
            "BiLSTM upper bound must be finite, got {hi}"
        );
        assert!(lo <= hi, "BiLSTM bounds must be ordered: {lo} <= {hi}");
    }
}

/// BiLSTM duration predictor (Y_h path): Kokoro-style BiLSTM → Reshape → projection →
/// Sigmoid → Sum. Proves positive expected duration bounds.
///
/// Uses Y_h (final hidden state from both directions, shape [2, batch, H])
/// reshaped to [2*H] → projection → sigmoid → sum proof head.
fn build_bilstm_duration_predictor_bytes(
    seq: i64,
    input_size: i64,
    hidden_size: i64,
    num_buckets: i64,
) -> Vec<u8> {
    use prost::Message;

    let batch = 1_i64;
    let num_directions = 2_i64;
    let proj_dim = 2 * hidden_size;

    let w_data: Vec<f32> = (0..num_directions * 4 * hidden_size * input_size)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let r_data: Vec<f32> = (0..num_directions * 4 * hidden_size * hidden_size)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let b_data: Vec<f32> = vec![0.0; (num_directions * 8 * hidden_size) as usize];
    let w_proj_data: Vec<f32> = (0..proj_dim * num_buckets)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    let sum_weight_data: Vec<f32> = vec![1.0; num_buckets as usize];

    let mut bilstm_node = make_bilstm_node("X", hidden_size, 1);
    bilstm_node.output = vec![String::new(), "Y_h".to_string(), String::new()];

    let nodes = vec![
        bilstm_node,
        NodeProto {
            op_type: "Reshape".to_string(),
            input: vec!["Y_h".to_string(), "reshape_yh_target".to_string()],
            output: vec!["flat_h".to_string()],
            name: "reshape_bilstm_yh".to_string(),
            ..Default::default()
        },
        NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["flat_h".to_string(), "W_proj".to_string()],
            output: vec!["logits".to_string()],
            name: "proj_matmul".to_string(),
            ..Default::default()
        },
        NodeProto {
            op_type: "Sigmoid".to_string(),
            input: vec!["logits".to_string()],
            output: vec!["probs".to_string()],
            name: "sigmoid_head".to_string(),
            ..Default::default()
        },
        NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["probs".to_string(), "sum_weight".to_string()],
            output: vec!["expected_duration".to_string()],
            name: "sum_matmul".to_string(),
            ..Default::default()
        },
    ];

    let graph = GraphProto {
        name: "bilstm_duration_predictor".to_string(),
        input: vec![tensor_value_info("X", &[batch, seq, input_size])],
        output: vec![tensor_value_info("expected_duration", &[batch, 1])],
        node: nodes,
        initializer: vec![
            make_float_tensor("W", &[num_directions, 4 * hidden_size, input_size], w_data),
            make_float_tensor("R", &[num_directions, 4 * hidden_size, hidden_size], r_data),
            make_float_tensor("B", &[num_directions, 8 * hidden_size], b_data),
            make_float_tensor("W_proj", &[proj_dim, num_buckets], w_proj_data),
            make_float_tensor("sum_weight", &[num_buckets, 1], sum_weight_data),
            TensorProto {
                dims: vec![1],
                data_type: 7, // INT64
                name: "reshape_yh_target".to_string(),
                raw_data: proj_dim.to_le_bytes().to_vec(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let model_proto = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-bilstm-duration-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    model_proto.encode_to_vec()
}

#[test]
fn test_bilstm_duration_predictor_loads_3497() {
    let bytes = build_bilstm_duration_predictor_bytes(3, 2, 4, 8);
    let model = load_bilstm_fixture("bilstm_dur", &bytes);
    assert!(
        model.network.layers.len() > 10,
        "BiLSTM + projection should produce many layers, got {}",
        model.network.layers.len()
    );

    let graph = model
        .to_graph_network()
        .expect("BiLSTM duration predictor should convert to graph network");
    assert!(
        !graph.output_name().is_empty(),
        "Graph must have a named output"
    );
}

/// End-to-end: BiLSTM (Y_h path) → Reshape → projection → Sigmoid → Sum proves
/// positive expected durations. Uses Y_h (final hidden state from both directions).
#[test]
fn test_bilstm_duration_predictor_positive_bounds_3497() {
    let seq = 3_i64;
    let input_size = 2_i64;
    let hidden_size = 4_i64;
    let num_buckets = 8_i64;

    let bytes = build_bilstm_duration_predictor_bytes(seq, input_size, hidden_size, num_buckets);
    let model = load_bilstm_fixture("bilstm_dur_ibp", &bytes);
    let graph = model
        .to_graph_network()
        .expect("BiLSTM duration predictor should convert to graph network");

    let center = ArrayD::zeros(IxDyn(&[seq as usize, input_size as usize]));
    let input = ny_tensor::BoundedTensor::from_epsilon(center, 0.1).expect("epsilon ball");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on BiLSTM duration predictor");

    assert!(
        !output.lower().is_empty(),
        "BiLSTM duration output must have at least one element"
    );

    for (&lo, &hi) in output.lower().iter().zip(output.upper().iter()) {
        assert!(lo.is_finite(), "Lower bound must be finite, got {lo}");
        assert!(hi.is_finite(), "Upper bound must be finite, got {hi}");
        assert!(lo <= hi, "Bounds must be ordered: {lo} <= {hi}");
        assert!(
            lo > 0.0,
            "BiLSTM expected duration lower bound must be strictly positive \
             (sigmoid+sum proof head), got {lo}"
        );
    }
}

/// Build the legacy fixture that depended on changing ONNX's four-dimensional
/// bidirectional Y into a three-dimensional convenience tensor. It is retained
/// solely to prove that this unsound route now fails closed.
fn build_bilstm_y_output_duration_predictor_bytes(
    seq: i64,
    input_size: i64,
    hidden_size: i64,
    num_buckets: i64,
) -> Vec<u8> {
    use prost::Message;

    let batch = 1_i64;
    let num_directions = 2_i64;
    let proj_dim = num_directions * hidden_size;

    let w_data: Vec<f32> = (0..num_directions * 4 * hidden_size * input_size)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let r_data: Vec<f32> = (0..num_directions * 4 * hidden_size * hidden_size)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let b_data: Vec<f32> = vec![0.0; (num_directions * 8 * hidden_size) as usize];
    let w_proj_data: Vec<f32> = (0..proj_dim * num_buckets)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    let sum_weight_data: Vec<f32> = vec![1.0; num_buckets as usize];

    // Raw ONNX Y is [batch, seq, directions, hidden]. The following MatMul only
    // typechecked after the old lowering silently concatenated the last two
    // axes; admission must reject before that substitution can occur.
    let mut bilstm_node = make_bilstm_node("X", hidden_size, 1);
    bilstm_node.output = vec!["Y".to_string(), String::new(), String::new()];

    let nodes = vec![
        bilstm_node,
        NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["Y".to_string(), "W_proj".to_string()],
            output: vec!["logits".to_string()],
            name: "proj_matmul".to_string(),
            ..Default::default()
        },
        NodeProto {
            op_type: "Sigmoid".to_string(),
            input: vec!["logits".to_string()],
            output: vec!["probs".to_string()],
            name: "sigmoid_head".to_string(),
            ..Default::default()
        },
        NodeProto {
            op_type: "MatMul".to_string(),
            input: vec!["probs".to_string(), "sum_weight".to_string()],
            output: vec!["expected_duration".to_string()],
            name: "sum_matmul".to_string(),
            ..Default::default()
        },
    ];

    let graph = GraphProto {
        name: "bilstm_y_duration_predictor".to_string(),
        input: vec![tensor_value_info("X", &[batch, seq, input_size])],
        output: vec![tensor_value_info("expected_duration", &[batch, seq, 1])],
        node: nodes,
        initializer: vec![
            make_float_tensor("W", &[num_directions, 4 * hidden_size, input_size], w_data),
            make_float_tensor("R", &[num_directions, 4 * hidden_size, hidden_size], r_data),
            make_float_tensor("B", &[num_directions, 8 * hidden_size], b_data),
            make_float_tensor("W_proj", &[proj_dim, num_buckets], w_proj_data),
            make_float_tensor("sum_weight", &[num_buckets, 1], sum_weight_data),
        ],
        ..Default::default()
    };

    let model_proto = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-bilstm-y-duration-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    model_proto.encode_to_vec()
}

/// BiLSTM Y must fail closed until lowering preserves its exact 4-D ONNX
/// layout. The prior 3-D convenience rewrite changed the type and semantics of
/// the value consumed by the projection.
#[test]
fn test_bilstm_y_output_duration_predictor_fails_closed_3497() {
    let bytes = build_bilstm_y_output_duration_predictor_bytes(3, 2, 4, 8);
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    let error = crate::loader::load_onnx_bytes_with_config("bilstm_y_dur", &bytes, &config)
        .expect_err("unsupported bidirectional Y must reject the model");
    assert!(error.to_string().contains("LSTM"), "{error}");
}

/// Refusal is stable across the former end-to-end proof fixture as well: a
/// positive proof over a semantically changed 3-D tensor had no authority.
#[test]
fn test_bilstm_y_output_proof_fixture_never_reaches_propagation_3497() {
    let bytes = build_bilstm_y_output_duration_predictor_bytes(3, 2, 4, 8);
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    assert!(
        crate::loader::load_onnx_bytes_with_config("bilstm_y_dur_ibp", &bytes, &config).is_err(),
        "bidirectional Y cannot reach bound propagation through a changed layout"
    );
}

// --- Weight slicing unit tests (proof_coverage: Re: #3497) ---
// These verify the critical correctness property: direction slicing extracts
// exactly the right weight block. If off by one, all BiLSTM bounds are silently wrong.

/// Verify `slice_direction` extracts the correct weight values for each direction.
///
/// Given W: [2, 4H, I] with forward weights all 1.0 and reverse weights all 2.0,
/// dir=0 must return exactly the forward block and dir=1 exactly the reverse block.
#[test]
fn test_slice_direction_extracts_correct_values_3497() {
    let hidden_size = 2_usize;
    let input_size = 3_usize;
    let fwd_val = 1.0_f32;
    let rev_val = 2.0_f32;

    // W: [2, 4H, I] — forward weights are all fwd_val, reverse are all rev_val
    let fwd_block = vec![fwd_val; 4 * hidden_size * input_size];
    let rev_block = vec![rev_val; 4 * hidden_size * input_size];
    let mut w_data = fwd_block;
    w_data.extend_from_slice(&rev_block);
    let w = ArrayD::from_shape_vec(IxDyn(&[2, 4 * hidden_size, input_size]), w_data)
        .expect("valid W shape");

    // dir=0 -> forward weights (all fwd_val)
    let fwd_slice = node_builder::slice_direction(&w, 0).expect("dir=0 slice");
    assert_eq!(fwd_slice.shape(), &[1, 4 * hidden_size, input_size]);
    for &v in fwd_slice.iter() {
        assert_eq!(
            v, fwd_val,
            "dir=0 should extract forward weights (all {fwd_val})"
        );
    }

    // dir=1 -> reverse weights (all rev_val)
    let rev_slice = node_builder::slice_direction(&w, 1).expect("dir=1 slice");
    assert_eq!(rev_slice.shape(), &[1, 4 * hidden_size, input_size]);
    for &v in rev_slice.iter() {
        assert_eq!(
            v, rev_val,
            "dir=1 should extract reverse weights (all {rev_val})"
        );
    }

    // dir=2 -> out of bounds
    assert!(node_builder::slice_direction(&w, 2).is_err());
}

/// Verify bias slicing extracts correct [8H] bias for each direction.
///
/// ONNX LSTM bias: [num_directions, 8H]. Wb = B[dir, 0..4H], Rb = B[dir, 4H..8H].
#[test]
fn test_slice_direction_bias_correct_values_3497() {
    let hidden_size = 2_usize;

    // B: [2, 8H] — forward bias is all 0.5, reverse bias is all 1.5
    let fwd_bias = vec![0.5_f32; 8 * hidden_size];
    let rev_bias = vec![1.5_f32; 8 * hidden_size];
    let mut b_data = fwd_bias;
    b_data.extend_from_slice(&rev_bias);
    let b = ArrayD::from_shape_vec(IxDyn(&[2, 8 * hidden_size]), b_data).expect("valid B shape");

    // dir=0 -> forward bias
    let fwd_flat = node_builder::slice_direction(&b, 0).expect("dir=0 bias slice");
    assert_eq!(fwd_flat.shape(), &[1, 8 * hidden_size]);
    for &v in fwd_flat.iter() {
        assert_eq!(v, 0.5, "dir=0 bias should be forward values (0.5)");
    }

    // dir=1 -> reverse bias
    let rev_flat = node_builder::slice_direction(&b, 1).expect("dir=1 bias slice");
    assert_eq!(rev_flat.shape(), &[1, 8 * hidden_size]);
    for &v in rev_flat.iter() {
        assert_eq!(v, 1.5, "dir=1 bias should be reverse values (1.5)");
    }
}

/// Verify squeeze_leading_dim correctly removes the leading [1] dimension
/// and preserves all values exactly.
#[test]
fn test_squeeze_leading_dim_removes_direction_3497() {
    let hidden_size = 2_usize;
    let input_size = 3_usize;

    let data: Vec<f32> = (0..4 * hidden_size * input_size)
        .map(|i| i as f32 * 0.1)
        .collect();
    let sliced = ArrayD::from_shape_vec(IxDyn(&[1, 4 * hidden_size, input_size]), data.clone())
        .expect("valid sliced shape");

    let squeezed = node_builder::squeeze_leading_dim(&sliced).expect("squeeze");
    assert_eq!(squeezed.shape(), &[4 * hidden_size, input_size]);

    // Values should be preserved exactly
    let squeezed_flat: Vec<f32> = squeezed.iter().copied().collect();
    assert_eq!(squeezed_flat, data);
}
