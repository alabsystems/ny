// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for unidirectional LSTM node unrolling.
//! BiLSTM tests (integration + weight slicing) are in tests_bilstm.rs (#3497).

use super::node_builder::make_int_attr;
use super::*;
use crate::onnx_proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
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

fn make_lstm_node(x_input: &str, hidden_size: i64, layout: i64) -> NodeProto {
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
        name: "lstm0".to_string(),
        domain: String::new(),
        attribute: vec![
            make_int_attr("hidden_size", hidden_size),
            make_int_attr("layout", layout),
        ],
    }
}

/// Build a minimal ONNX model with a single LSTM node.
/// X: [batch=1, seq=3, input=2], hidden_size=4, layout=1 (batch_first)
fn build_lstm_test_model() -> (Vec<NodeProto>, WeightStore, GraphProto) {
    let batch = 1_i64;
    let seq = 3_i64;
    let input_size = 2_i64;
    let hidden_size = 4_i64;

    let w_data: Vec<f32> = (0..4 * hidden_size * input_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let r_data: Vec<f32> = (0..4 * hidden_size * hidden_size)
        .map(|i| (i as f32) * 0.005)
        .collect();
    let b_data: Vec<f32> = vec![0.0; 8 * hidden_size as usize];

    let mut weights = WeightStore::new();
    weights.insert(
        "W".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[1, (4 * hidden_size) as usize, input_size as usize]),
            w_data,
        )
        .expect("W shape"),
    );
    weights.insert(
        "R".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[1, (4 * hidden_size) as usize, hidden_size as usize]),
            r_data,
        )
        .expect("R shape"),
    );
    weights.insert(
        "B".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, (8 * hidden_size) as usize]), b_data).expect("B shape"),
    );

    let graph = GraphProto {
        name: "lstm_test".to_string(),
        input: vec![tensor_value_info("X", &[batch, seq, input_size])],
        output: vec![
            tensor_value_info("Y", &[batch, seq, 1, hidden_size]),
            tensor_value_info("Y_h", &[1, batch, hidden_size]),
            tensor_value_info("Y_c", &[1, batch, hidden_size]),
        ],
        node: vec![make_lstm_node("X", hidden_size, 1)],
        ..Default::default()
    };

    (graph.node.clone(), weights, graph)
}

#[test]
fn test_lstm_lowering_produces_expected_node_types() {
    let (mut nodes, mut weights, graph) = build_lstm_test_model();
    let inferred_shapes = std::collections::HashMap::new();

    let graph_vi = graph.value_info().to_vec();
    lower_lstm_nodes(
        &mut nodes,
        &mut weights,
        &graph.input,
        &graph_vi,
        &inferred_shapes,
    );

    assert!(
        !nodes.iter().any(|n| n.op_type == "LSTM"),
        "LSTM node should be replaced by unrolled primitives"
    );

    let op_types: Vec<&str> = nodes.iter().map(|n| n.op_type.as_str()).collect();
    assert!(op_types.contains(&"Slice"), "should contain Slice ops");
    assert!(op_types.contains(&"Reshape"), "should contain Reshape ops");
    assert!(op_types.contains(&"MatMul"), "should contain MatMul ops");
    assert!(op_types.contains(&"Add"), "should contain Add ops");
    assert!(op_types.contains(&"Sigmoid"), "should contain Sigmoid ops");
    assert!(op_types.contains(&"Tanh"), "should contain Tanh ops");
    assert!(op_types.contains(&"Mul"), "should contain Mul ops");
}

#[test]
fn test_lstm_lowering_correct_node_count() {
    let (mut nodes, mut weights, graph) = build_lstm_test_model();
    let inferred_shapes = std::collections::HashMap::new();

    let graph_vi = graph.value_info().to_vec();
    lower_lstm_nodes(
        &mut nodes,
        &mut weights,
        &graph.input,
        &graph_vi,
        &inferred_shapes,
    );

    // 3 timesteps × 19 nodes/timestep + output wiring nodes
    assert!(
        nodes.len() > 50,
        "expected 50+ nodes for 3-timestep LSTM, got {}",
        nodes.len()
    );
}

#[test]
fn test_lstm_lowering_preserves_output_names() {
    let (mut nodes, mut weights, graph) = build_lstm_test_model();
    let inferred_shapes = std::collections::HashMap::new();

    let graph_vi = graph.value_info().to_vec();
    lower_lstm_nodes(
        &mut nodes,
        &mut weights,
        &graph.input,
        &graph_vi,
        &inferred_shapes,
    );

    let all_outputs: Vec<&str> = nodes
        .iter()
        .flat_map(|n| n.output.iter().map(|s| s.as_str()))
        .collect();
    assert!(all_outputs.contains(&"Y_h"), "Y_h output must be preserved");
    assert!(all_outputs.contains(&"Y_c"), "Y_c output must be preserved");
    assert!(all_outputs.contains(&"Y"), "Y output must be preserved");
}

#[test]
fn test_lstm_lowering_stores_precomputed_weights() {
    let (mut nodes, mut weights, graph) = build_lstm_test_model();
    let inferred_shapes = std::collections::HashMap::new();

    let graph_vi = graph.value_info().to_vec();
    lower_lstm_nodes(
        &mut nodes,
        &mut weights,
        &graph.input,
        &graph_vi,
        &inferred_shapes,
    );

    let w_t = weights.get("lstm0__lstm_W_T").expect("W_T should exist");
    assert_eq!(
        w_t.shape(),
        &[2, 16],
        "W_T should be [input_size=2, 4*H=16]"
    );

    let r_t = weights.get("lstm0__lstm_R_T").expect("R_T should exist");
    assert_eq!(
        r_t.shape(),
        &[4, 16],
        "R_T should be [hidden_size=4, 4*H=16]"
    );

    let bias = weights.get("lstm0__lstm_bias").expect("bias should exist");
    assert_eq!(bias.shape(), &[16], "bias should be [4*H=16]");

    let h0 = weights.get("lstm0__lstm_h0").expect("h0 should exist");
    assert_eq!(
        h0.shape(),
        &[4],
        "h0 should be [H=4] (batch squeezed for batch=1)"
    );

    let c0 = weights.get("lstm0__lstm_c0").expect("c0 should exist");
    assert_eq!(
        c0.shape(),
        &[4],
        "c0 should be [H=4] (batch squeezed for batch=1)"
    );
}

#[test]
fn test_lstm_lowering_rejects_reverse_only() {
    let mut node = make_lstm_node("X", 4, 1);
    node.attribute.push(onnx_proto::AttributeProto {
        name: "direction".to_string(),
        s: b"reverse".to_vec(),
        r#type: onnx_proto::attribute_type::STRING,
        ..Default::default()
    });

    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, 3, 2])],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 16, 2])));
    weights.insert("R".to_string(), ArrayD::zeros(IxDyn(&[1, 16, 4])));
    weights.insert("B".to_string(), ArrayD::zeros(IxDyn(&[1, 32])));

    let graph_vi = graph.value_info().to_vec();
    let result = unroll_lstm_node(
        &node,
        &mut weights,
        &graph.input,
        &graph_vi,
        &std::collections::HashMap::new(),
    );
    assert!(result.is_err(), "reverse-only LSTM should be rejected");
    assert!(
        result.unwrap_err().contains("unsupported"),
        "error should mention unsupported direction"
    );
}

#[test]
fn test_lstm_lowering_rejects_dynamic_seq_len() {
    let node = make_lstm_node("X", 4, 1);
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, -1, 2])], // dynamic seq
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 16, 2])));
    weights.insert("R".to_string(), ArrayD::zeros(IxDyn(&[1, 16, 4])));
    weights.insert("B".to_string(), ArrayD::zeros(IxDyn(&[1, 32])));

    let graph_vi = graph.value_info().to_vec();
    let result = unroll_lstm_node(
        &node,
        &mut weights,
        &graph.input,
        &graph_vi,
        &std::collections::HashMap::new(),
    );
    assert!(result.is_err(), "dynamic seq length should be rejected");
    assert!(
        result.unwrap_err().contains("concrete"),
        "error should mention concrete sequence length"
    );
}

/// Helper: build an ONNX model proto with a single LSTM node.
/// Returns the serialized protobuf bytes.
fn build_lstm_onnx_bytes(
    batch: i64,
    seq: i64,
    input_size: i64,
    hidden_size: i64,
    output_names: &[&str],
) -> Vec<u8> {
    use prost::Message;

    let w_data: Vec<f32> = (0..4 * hidden_size * input_size)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let r_data: Vec<f32> = (0..4 * hidden_size * hidden_size)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let b_data: Vec<f32> = vec![0.0; 8 * hidden_size as usize];

    let mut lstm_node = make_lstm_node("X", hidden_size, 1);
    lstm_node.output = output_names.iter().map(|s| s.to_string()).collect();

    let outputs: Vec<ValueInfoProto> = output_names
        .iter()
        .filter(|s| !s.is_empty())
        .map(|name| match *name {
            "Y" => tensor_value_info(name, &[batch, seq, 1, hidden_size]),
            "Y_h" => tensor_value_info(name, &[1, batch, hidden_size]),
            "Y_c" => tensor_value_info(name, &[1, batch, hidden_size]),
            _ => tensor_value_info(name, &[1, batch, hidden_size]),
        })
        .collect();

    let graph = GraphProto {
        name: "lstm_ibp_test".to_string(),
        input: vec![tensor_value_info("X", &[batch, seq, input_size])],
        output: outputs,
        node: vec![lstm_node],
        initializer: vec![
            make_float_tensor("W", &[1, 4 * hidden_size, input_size], w_data),
            make_float_tensor("R", &[1, 4 * hidden_size, hidden_size], r_data),
            make_float_tensor("B", &[1, 8 * hidden_size], b_data),
        ],
        ..Default::default()
    };

    let model_proto = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-lstm-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    model_proto.encode_to_vec()
}

/// End-to-end test: build an ONNX model with LSTM, parse it, verify the
/// unrolled model loads and produces the expected number of layers.
#[test]
fn test_lstm_end_to_end_parse_and_ibp() {
    let bytes = build_lstm_onnx_bytes(1, 3, 2, 4, &["Y", "Y_h", "Y_c"]);
    let model = crate::loader::load_onnx_bytes("lstm_test", &bytes)
        .expect("LSTM model should load after unrolling");
    assert!(
        model.network.layers.len() > 10,
        "unrolled LSTM should produce many layers, got {}",
        model.network.layers.len()
    );
}

/// End-to-end IBP test: load an LSTM model, convert to graph, run IBP,
/// verify that output bounds are finite and ordered (lower <= upper).
#[test]
fn test_lstm_ibp_produces_finite_ordered_bounds() {
    let bytes = build_lstm_onnx_bytes(1, 3, 2, 4, &["Y_h", "", ""]);
    let model = crate::loader::load_onnx_bytes("lstm_ibp", &bytes).expect("LSTM model should load");
    let graph = model
        .to_graph_network()
        .expect("LSTM model should convert to graph network");

    let center = ArrayD::zeros(IxDyn(&[3, 2]));
    let input = ny_tensor::BoundedTensor::from_epsilon(center, 0.1).expect("epsilon ball");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on unrolled LSTM");

    assert!(
        !output.lower().is_empty(),
        "IBP output must have at least one element"
    );
    for (&lo, &hi) in output.lower().iter().zip(output.upper().iter()) {
        assert!(lo.is_finite(), "IBP lower bound must be finite, got {lo}");
        assert!(hi.is_finite(), "IBP upper bound must be finite, got {hi}");
        assert!(lo <= hi, "IBP bounds must be ordered: {lo} <= {hi}");
    }
}

/// Build an ONNX model matching the Kokoro duration predictor I/O pattern:
/// LSTM -> Linear projection -> Sigmoid -> Weighted sum.
fn build_lstm_duration_predictor_bytes(
    seq: i64,
    input_size: i64,
    hidden_size: i64,
    num_buckets: i64,
) -> Vec<u8> {
    use prost::Message;

    let batch = 1_i64;

    let w_data: Vec<f32> = (0..4 * hidden_size * input_size)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let r_data: Vec<f32> = (0..4 * hidden_size * hidden_size)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let b_data: Vec<f32> = vec![0.0; 8 * hidden_size as usize];

    let w_proj_data: Vec<f32> = (0..hidden_size * num_buckets)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    let sum_weight_data: Vec<f32> = vec![1.0; num_buckets as usize];

    let mut lstm_node = make_lstm_node("X", hidden_size, 1);
    lstm_node.output = vec![String::new(), "Y_h".to_string(), String::new()];

    let nodes = vec![
        lstm_node,
        NodeProto {
            op_type: "Reshape".to_string(),
            input: vec!["Y_h".to_string(), "reshape_target".to_string()],
            output: vec!["flat_h".to_string()],
            name: "reshape_yh".to_string(),
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
        name: "lstm_duration_predictor".to_string(),
        input: vec![tensor_value_info("X", &[batch, seq, input_size])],
        output: vec![tensor_value_info("expected_duration", &[batch, 1])],
        node: nodes,
        initializer: vec![
            make_float_tensor("W", &[1, 4 * hidden_size, input_size], w_data),
            make_float_tensor("R", &[1, 4 * hidden_size, hidden_size], r_data),
            make_float_tensor("B", &[1, 8 * hidden_size], b_data),
            make_float_tensor("W_proj", &[hidden_size, num_buckets], w_proj_data),
            make_float_tensor("sum_weight", &[num_buckets, 1], sum_weight_data),
            TensorProto {
                dims: vec![1],
                data_type: 7, // INT64
                name: "reshape_target".to_string(),
                raw_data: hidden_size.to_le_bytes().to_vec(),
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
        producer_name: "ny-onnx-duration-predictor-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    model_proto.encode_to_vec()
}

#[test]
fn test_lstm_duration_predictor_loads_3497() {
    let bytes = build_lstm_duration_predictor_bytes(3, 2, 4, 8);
    let model = crate::loader::load_onnx_bytes("duration_pred", &bytes)
        .expect("Duration predictor model should load after LSTM unrolling");
    assert!(
        model.network.layers.len() > 10,
        "LSTM + projection head should produce many layers, got {}",
        model.network.layers.len()
    );

    let graph = model
        .to_graph_network()
        .expect("Duration predictor should convert to graph network");
    assert!(
        !graph.output_name().is_empty(),
        "Graph must have a named output"
    );
}

/// Sigmoid(x) > 0 for all finite x. Sum of positive terms is positive.
/// Therefore expected duration lower bound must be > 0.
#[test]
fn test_lstm_duration_predictor_positive_bounds_3497() {
    let seq = 3_i64;
    let input_size = 2_i64;
    let hidden_size = 4_i64;
    let num_buckets = 8_i64;

    let bytes = build_lstm_duration_predictor_bytes(seq, input_size, hidden_size, num_buckets);
    let model = crate::loader::load_onnx_bytes("duration_pred_ibp", &bytes)
        .expect("Duration predictor model should load");
    let graph = model
        .to_graph_network()
        .expect("Duration predictor should convert to graph network");

    let center = ArrayD::zeros(IxDyn(&[seq as usize, input_size as usize]));
    let input = ny_tensor::BoundedTensor::from_epsilon(center, 0.1).expect("epsilon ball");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on duration predictor model");

    assert!(
        !output.lower().is_empty(),
        "Output must have at least one element"
    );

    for (&lo, &hi) in output.lower().iter().zip(output.upper().iter()) {
        assert!(lo.is_finite(), "Lower bound must be finite, got {lo}");
        assert!(hi.is_finite(), "Upper bound must be finite, got {hi}");
        assert!(lo <= hi, "Bounds must be ordered: {lo} <= {hi}");
        assert!(
            lo > 0.0,
            "Expected duration lower bound must be strictly positive \
             (sigmoid+sum proof head), got {lo}"
        );
    }
}
