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
fn test_lstm_lowering_does_not_rewrite_custom_domain_lookalike() {
    let (mut nodes, mut weights, graph) = build_lstm_test_model();
    nodes[0].domain = "custom.lstm".to_string();
    let original = nodes.clone();

    lower_lstm_nodes(
        &mut nodes,
        &mut weights,
        &graph.input,
        graph.value_info(),
        &std::collections::HashMap::new(),
    );

    assert_eq!(nodes, original, "custom-domain LSTM must remain untouched");
    assert!(
        !weights.contains_key("lstm0__lstm_W_T"),
        "custom-domain LSTM must not materialize standard-ONNX lowering weights"
    );
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
        s: Some(b"reverse".to_vec()),
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
        &authored_value_names(&node, &graph),
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
        &authored_value_names(&node, &graph),
    );
    assert!(result.is_err(), "dynamic seq length should be rejected");
    assert!(
        result.unwrap_err().contains("concrete"),
        "error should mention concrete sequence length"
    );
}

fn unroll_test_node(
    node: &NodeProto,
    weights: &mut WeightStore,
    graph: &GraphProto,
) -> Result<Vec<NodeProto>, String> {
    unroll_lstm_node(
        node,
        weights,
        &graph.input,
        graph.value_info(),
        &std::collections::HashMap::new(),
        &authored_value_names(node, graph),
    )
}

fn authored_value_names(node: &NodeProto, graph: &GraphProto) -> HashSet<String> {
    graph
        .input
        .iter()
        .chain(graph.value_info())
        .map(|value| value.name.clone())
        .chain(node.input.iter().chain(&node.output).cloned())
        .filter(|name| !name.is_empty())
        .collect()
}

fn float_attr(name: &str, value: f32) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        f: Some(value),
        r#type: onnx_proto::attribute_type::FLOAT,
        ..Default::default()
    }
}

fn floats_attr(name: &str, values: &[f32]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        floats: values.to_vec(),
        r#type: onnx_proto::attribute_type::FLOATS,
        ..Default::default()
    }
}

#[test]
fn test_lstm_rejects_unsupported_optional_inputs() {
    let (nodes, mut weights, graph) = build_lstm_test_model();

    let mut sequence_lens = nodes[0].clone();
    sequence_lens.input[4] = "sequence_lens".to_string();
    let error = unroll_test_node(&sequence_lens, &mut weights.clone(), &graph)
        .expect_err("non-empty sequence_lens must be rejected");
    assert!(error.contains("sequence_lens"), "{error}");

    let mut peephole = nodes[0].clone();
    peephole.input.resize(8, String::new());
    peephole.input[7] = "P".to_string();
    let error = unroll_test_node(&peephole, &mut weights, &graph)
        .expect_err("non-empty peephole P must be rejected");
    assert!(error.contains("peephole P"), "{error}");
}

#[test]
fn test_lstm_rejects_unsupported_semantic_attributes() {
    let (nodes, mut weights, graph) = build_lstm_test_model();
    let unsupported = [
        onnx_proto::AttributeProto {
            name: "activations".to_string(),
            r#type: onnx_proto::attribute_type::STRINGS,
            ..Default::default()
        },
        floats_attr("activation_alpha", &[0.2]),
        floats_attr("activation_beta", &[0.5]),
        float_attr("clip", 1.0),
        make_int_attr("input_forget", 1),
    ];

    for attribute in unsupported {
        let mut node = nodes[0].clone();
        let name = attribute.name.clone();
        node.attribute.push(attribute);
        let error = unroll_test_node(&node, &mut weights.clone(), &graph)
            .expect_err("unsupported LSTM semantics must fail closed");
        assert!(error.contains(&name), "attribute={name}, error={error}");
    }

    let mut explicit_default = nodes[0].clone();
    explicit_default
        .attribute
        .push(make_int_attr("input_forget", 0));
    unroll_test_node(&explicit_default, &mut weights, &graph)
        .expect("the explicit default input_forget=0 is exact");
}

#[test]
fn test_lstm_rejects_bad_attribute_schema() {
    let (nodes, mut weights, graph) = build_lstm_test_model();

    let mut duplicate = nodes[0].clone();
    duplicate.attribute.push(make_int_attr("hidden_size", 4));
    let error = unroll_test_node(&duplicate, &mut weights.clone(), &graph)
        .expect_err("duplicate attributes must be rejected");
    assert!(error.contains("duplicate"), "{error}");

    let mut wrong_type = nodes[0].clone();
    wrong_type
        .attribute
        .retain(|attribute| attribute.name != "layout");
    wrong_type.attribute.push(float_attr("layout", 1.0));
    let error = unroll_test_node(&wrong_type, &mut weights.clone(), &graph)
        .expect_err("wrong attribute type must be rejected");
    assert!(error.contains("type"), "{error}");

    let mut unknown = nodes[0].clone();
    unknown.attribute.push(make_int_attr("future_mode", 0));
    let error = unroll_test_node(&unknown, &mut weights.clone(), &graph)
        .expect_err("unknown attributes must be rejected");
    assert!(error.contains("unknown"), "{error}");

    let mut invalid_layout = nodes[0].clone();
    invalid_layout
        .attribute
        .retain(|attribute| attribute.name != "layout");
    invalid_layout.attribute.push(make_int_attr("layout", 2));
    let error = unroll_test_node(&invalid_layout, &mut weights, &graph)
        .expect_err("layout outside {0,1} must be rejected");
    assert!(error.contains("layout"), "{error}");
}

#[test]
fn test_lstm_rejects_nonpositive_dimensions() {
    let (nodes, weights, graph) = build_lstm_test_model();

    for hidden_size in [0, -1] {
        let mut node = nodes[0].clone();
        node.attribute
            .retain(|attribute| attribute.name != "hidden_size");
        node.attribute
            .push(make_int_attr("hidden_size", hidden_size));
        let error = unroll_test_node(&node, &mut weights.clone(), &graph)
            .expect_err("nonpositive hidden_size must be rejected");
        assert!(error.contains("positive"), "{error}");
    }

    for shape in [[0, 3, 2], [1, 3, 0]] {
        let bad_graph = GraphProto {
            input: vec![tensor_value_info("X", &shape)],
            ..Default::default()
        };
        let error = unroll_test_node(&nodes[0], &mut weights.clone(), &bad_graph)
            .expect_err("nonpositive X dimensions must be rejected");
        assert!(error.contains("positive"), "shape={shape:?}, error={error}");
    }
}

#[test]
fn test_lstm_bias_precomputation_rejects_inexact_point_one_plus_point_two() {
    let mut node = make_lstm_node("X", 1, 1);
    node.output = vec![String::new(), "Y_h".to_string()];
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, 1, 1])],
        ..Default::default()
    };
    let mut bias = vec![0.0_f32; 8];
    bias[0] = 0.1;
    bias[4] = 0.2;
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    weights.insert("R".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    weights.insert(
        "B".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 8]), bias).expect("B shape"),
    );

    let error = unroll_test_node(&node, &mut weights, &graph)
        .expect_err("rounded Wb+Rb must not be materialized as an exact constant");
    assert!(error.contains("not exactly representable"), "{error}");
}

#[test]
fn test_lstm_h0_precomputation_rejects_inexact_point_one_times_point_two() {
    let mut node = make_lstm_node("X", 1, 1);
    node.input.resize(7, String::new());
    node.input[5] = "h0".to_string();
    node.output = vec![String::new(), "Y_h".to_string()];
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, 1, 1])],
        ..Default::default()
    };
    let mut recurrent = vec![0.0_f32; 4];
    recurrent[0] = 0.2;
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    weights.insert(
        "R".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), recurrent).expect("R shape"),
    );
    weights.insert("B".to_string(), ArrayD::zeros(IxDyn(&[1, 8])));
    weights.insert(
        "h0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![0.1]).expect("h0 shape"),
    );

    let error = unroll_test_node(&node, &mut weights, &graph)
        .expect_err("rounded h0*R product must not be materialized as an exact constant");
    assert!(error.contains("not exactly representable"), "{error}");
}

#[test]
fn test_lstm_precomputation_keeps_exact_binary_common_values() {
    let mut node = make_lstm_node("X", 1, 1);
    node.input.resize(7, String::new());
    node.input[5] = "h0".to_string();
    node.output = vec![String::new(), "Y_h".to_string()];
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, 1, 1])],
        ..Default::default()
    };
    let mut recurrent = vec![0.0_f32; 4];
    recurrent[0] = 2.0;
    let mut bias = vec![0.0_f32; 8];
    bias[0] = 0.5;
    bias[4] = 0.25;
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    weights.insert(
        "R".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), recurrent).expect("R shape"),
    );
    weights.insert(
        "B".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 8]), bias).expect("B shape"),
    );
    weights.insert(
        "h0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![0.5]).expect("h0 shape"),
    );

    unroll_test_node(&node, &mut weights, &graph)
        .expect("exact binary sums and products should retain LSTM lowering");
    assert_eq!(
        weights.get("lstm0__lstm_bias").expect("combined bias")[[0]],
        0.75
    );
    assert_eq!(
        weights.get("lstm0__lstm_h0_hR").expect("h0 product")[[0]],
        1.0
    );
}

#[test]
fn test_lstm_layout_one_initial_states_use_direction_axis_one() {
    let mut node = make_lstm_node("X", 1, 1);
    node.attribute.push(onnx_proto::AttributeProto {
        name: "direction".to_string(),
        s: Some(b"bidirectional".to_vec()),
        r#type: onnx_proto::attribute_type::STRING,
        ..Default::default()
    });
    node.input.resize(7, String::new());
    node.input[5] = "h0".to_string();
    node.input[6] = "c0".to_string();
    node.output = vec![String::new(), "Y_h".to_string(), "Y_c".to_string()];
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[2, 1, 1])],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[2, 4, 1])));
    weights.insert("R".to_string(), ArrayD::zeros(IxDyn(&[2, 4, 1])));
    weights.insert("B".to_string(), ArrayD::zeros(IxDyn(&[2, 8])));
    weights.insert(
        "h0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2, 1]), vec![1.0, 2.0, 3.0, 4.0]).expect("h0 shape"),
    );
    weights.insert(
        "c0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2, 1]), vec![5.0, 6.0, 7.0, 8.0]).expect("c0 shape"),
    );

    let lowered = unroll_test_node(&node, &mut weights, &graph)
        .expect("layout=1 constant initial states should lower exactly");
    assert_eq!(
        weights
            .get("lstm0_fwd__lstm_h0")
            .expect("forward h0")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1.0, 3.0]
    );
    assert_eq!(
        weights
            .get("lstm0_rev__lstm_h0")
            .expect("reverse h0")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![2.0, 4.0]
    );
    for name in [
        "lstm0__lstm_bidi_yh_fwd_unsq",
        "lstm0__lstm_bidi_yh_rev_unsq",
        "lstm0__lstm_bidi_yc_fwd_unsq",
        "lstm0__lstm_bidi_yc_rev_unsq",
    ] {
        let unsqueeze = lowered
            .iter()
            .find(|candidate| candidate.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(unsqueeze.attribute[0].ints, vec![1]);
    }
}

#[test]
fn test_lstm_rejects_wrong_initial_state_shape_and_integer_parameters() {
    let mut node = make_lstm_node("X", 1, 1);
    node.input.resize(7, String::new());
    node.input[5] = "h0".to_string();
    node.output = vec![String::new(), "Y_h".to_string()];
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[2, 1, 1])],
        ..Default::default()
    };
    let mut wrong_shape = WeightStore::new();
    wrong_shape.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    wrong_shape.insert("R".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    wrong_shape.insert("B".to_string(), ArrayD::zeros(IxDyn(&[1, 8])));
    wrong_shape.insert("h0".to_string(), ArrayD::zeros(IxDyn(&[1, 2, 1])));
    let error = unroll_test_node(&node, &mut wrong_shape, &graph)
        .expect_err("layout=1 h0 must have [batch,directions,H] shape");
    assert!(error.contains("initial_h must have shape"), "{error}");

    let mut integer_w = WeightStore::new();
    integer_w.insert("W".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    integer_w.insert_integers("W".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    integer_w.insert("R".to_string(), ArrayD::zeros(IxDyn(&[1, 4, 1])));
    integer_w.insert("B".to_string(), ArrayD::zeros(IxDyn(&[1, 8])));
    integer_w.insert("h0".to_string(), ArrayD::zeros(IxDyn(&[2, 1, 1])));
    let error = unroll_test_node(&node, &mut integer_w, &graph)
        .expect_err("integer LSTM arithmetic parameters must be rejected");
    assert!(error.contains("integer-valued"), "{error}");
}

#[test]
fn test_bidirectional_sequence_output_fails_closed() {
    let mut node = make_lstm_node("X", 1, 1);
    node.attribute.push(onnx_proto::AttributeProto {
        name: "direction".to_string(),
        s: Some(b"bidirectional".to_vec()),
        r#type: onnx_proto::attribute_type::STRING,
        ..Default::default()
    });
    let graph = GraphProto {
        input: vec![tensor_value_info("X", &[1, 1, 1])],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("W".to_string(), ArrayD::zeros(IxDyn(&[2, 4, 1])));
    weights.insert("R".to_string(), ArrayD::zeros(IxDyn(&[2, 4, 1])));
    weights.insert("B".to_string(), ArrayD::zeros(IxDyn(&[2, 8])));

    let error = unroll_test_node(&node, &mut weights, &graph)
        .expect_err("3-D substitution for 4-D bidirectional Y is unsound");
    assert!(error.contains("four-dimensional"), "{error}");
}

#[test]
fn test_lstm_lowering_never_shadows_authored_value_names() {
    let (nodes, mut weights, graph) = build_lstm_test_model();
    let authored = ArrayD::from_elem(IxDyn(&[1]), 17.0_f32);
    weights.insert("lstm0__lstm_gate_i_start".to_string(), authored.clone());

    let error = unroll_test_node(&nodes[0], &mut weights, &graph)
        .expect_err("synthesized LSTM constants must not replace authored tensors");
    assert!(error.contains("shadow authored value"), "{error}");
    assert_eq!(
        weights
            .get("lstm0__lstm_gate_i_start")
            .expect("authored tensor remains"),
        &authored
    );

    let (nodes, mut weights, mut graph) = build_lstm_test_model();
    graph
        .input
        .push(tensor_value_info("lstm0__lstm_t0_start", &[1]));
    let error = unroll_test_node(&nodes[0], &mut weights, &graph)
        .expect_err("synthesized constants must not shadow runtime inputs");
    assert!(error.contains("shadow authored value"), "{error}");
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
    let model = load_lstm_fixture("lstm_test", &bytes);
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
    let model = load_lstm_fixture("lstm_ibp", &bytes);
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

fn load_lstm_fixture(name: &str, bytes: &[u8]) -> crate::OnnxModel {
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    crate::loader::load_onnx_bytes_with_config(name, bytes, &config)
        .expect("LSTM fixture should load without native shape inference")
}

#[test]
fn test_lstm_duration_predictor_loads_3497() {
    let bytes = build_lstm_duration_predictor_bytes(3, 2, 4, 8);
    let model = load_lstm_fixture("duration_pred", &bytes);
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
    let model = load_lstm_fixture("duration_pred_ibp", &bytes);
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
