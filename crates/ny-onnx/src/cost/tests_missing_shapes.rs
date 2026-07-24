// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::estimate_model_cost;
use super::lookup::ShapeLookup;
use crate::{
    load_onnx_bytes, onnx_proto, AttributeValue, DataType, LayerSpec, Network, OnnxModel,
    TensorSpec, WeightStore,
};
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use prost::Message;
use std::collections::HashMap;

fn tensor_value_info(name: &str, shape: &[i64], elem_type: i32) -> onnx_proto::ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn tensor_i64(name: &str, shape: &[i64], data: &[i64]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 7,
        name: name.to_string(),
        raw_data: data.iter().flat_map(|value| value.to_le_bytes()).collect(),
        float_data: Vec::new(),
        ..Default::default()
    }
}

fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: data.iter().flat_map(|value| value.to_le_bytes()).collect(),
        float_data: Vec::new(),
        ..Default::default()
    }
}

fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn attr_int(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        ints: values.to_vec(),
        r#type: onnx_proto::attribute_type::INTS,
        ..Default::default()
    }
}

fn infer_first_layer_output_shape(model: &OnnxModel) -> Result<Vec<usize>, String> {
    let lookup = ShapeLookup::new(model);
    let layer = model
        .network
        .layers
        .first()
        .expect("inline ONNX fixture should produce one layer");
    lookup
        .infer_output_shape(layer)
        .map_err(|err| err.to_string())
}

fn load_inline_onnx_model_with_opset(
    name: &str,
    opset_version: i64,
    graph: onnx_proto::GraphProto,
) -> OnnxModel {
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: opset_version,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };
    load_onnx_bytes(name, &model.encode_to_vec()).expect("inline ONNX fixture should load")
}

fn load_inline_onnx_model(name: &str, graph: onnx_proto::GraphProto) -> OnnxModel {
    load_inline_onnx_model_with_opset(name, 17, graph)
}

#[test]
fn test_estimate_model_cost_infers_missing_shape_for_pointwise_intermediate_3498() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("relu1", "Relu", &["input"], &["hidden"]),
            node("relu2", "Relu", &["hidden"], &["out"]),
        ],
        name: "pointwise_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 4], 1)],
        output: vec![tensor_value_info("out", &[1, 4], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("pointwise_missing_shape.onnx", graph);

    let cost = estimate_model_cost(&model).expect("pointwise missing-shape fallback should work");

    assert_eq!(cost.layers.len(), 2);
    assert_eq!(cost.layers[0].output_shapes, vec![vec![1, 4]]);
    assert_eq!(cost.layers[1].output_shapes, vec![vec![1, 4]]);
    assert_eq!(cost.layers[0].timing_family, "elementwise");
    assert_eq!(cost.layers[1].timing_family, "elementwise");
}

#[test]
fn test_estimate_model_cost_infers_missing_shape_for_triu_intermediate_4270() {
    let model = OnnxModel::empty_with_network(
        Network {
            name: "triu_missing_shape".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![3, 3],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "out".to_string(),
                shape: vec![3, 3],
                dtype: DataType::Float32,
            }],
            layers: vec![
                LayerSpec {
                    name: "triu".to_string(),
                    layer_type: LayerType::Triu,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["masked".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
                LayerSpec {
                    name: "relu".to_string(),
                    layer_type: LayerType::ReLU,
                    inputs: vec!["masked".to_string()],
                    outputs: vec!["out".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
            ],
            param_count: 0,
        },
        WeightStore::new(),
    )
    .with_tensor_shapes(HashMap::from([
        ("input".to_string(), vec![3, 3]),
        ("out".to_string(), vec![3, 3]),
    ]));

    let cost = estimate_model_cost(&model).expect("Triu missing-shape fallback should work");

    assert_eq!(cost.layers.len(), 2);
    assert_eq!(cost.layers[0].layer_type, "Triu");
    assert_eq!(cost.layers[0].output_shapes, vec![vec![3, 3]]);
    assert_eq!(cost.layers[0].timing_family, "elementwise");
    assert_eq!(cost.layers[1].output_shapes, vec![vec![3, 3]]);
}

#[test]
fn test_estimate_model_cost_ignores_embedded_instance_norm_affine_names_3500() {
    let model = OnnxModel::empty_with_network(
        Network {
            name: "instance_norm_missing_affines".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 2, 3],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "out".to_string(),
                shape: vec![1, 2, 3],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "norm".to_string(),
                layer_type: LayerType::InstanceNorm,
                inputs: vec![
                    "input".to_string(),
                    "missing_scale".to_string(),
                    "missing_bias".to_string(),
                ],
                outputs: vec!["out".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        WeightStore::new(),
    )
    .with_tensor_shapes(HashMap::from([
        ("input".to_string(), vec![1, 2, 3]),
        ("out".to_string(), vec![1, 2, 3]),
    ]));

    let cost =
        estimate_model_cost(&model).expect("InstanceNorm cost should ignore embedded affines");

    assert_eq!(cost.layers.len(), 1);
    assert_eq!(cost.layers[0].activation_input_bytes, 2 * 3 * 4);
    assert_eq!(cost.layers[0].parameter_input_bytes, 0);
}

#[test]
fn test_infer_output_shape_handles_broadcasted_binary_missing_shape_3500() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("add", "Add", &["lhs", "rhs"], &["sum"]),
            node("relu", "Relu", &["sum"], &["out"]),
        ],
        name: "broadcasted_binary_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("lhs", &[1, 1], 1),
            tensor_value_info("rhs", &[1, 4], 1),
        ],
        output: vec![tensor_value_info("out", &[1, 4], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("broadcasted_binary_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("binary fallback should broadcast inputs");
    assert_eq!(inferred_shape, vec![1, 4]);
}

#[test]
fn test_infer_output_shape_rejects_incompatible_broadcast_binary_missing_shape_3500() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("add", "Add", &["lhs", "rhs"], &["sum"]),
            node("relu", "Relu", &["sum"], &["out"]),
        ],
        name: "incompatible_broadcast_binary_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("lhs", &[2, 3], 1),
            tensor_value_info("rhs", &[4], 1),
        ],
        output: vec![tensor_value_info("out", &[2, 3], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("incompatible_broadcast_binary_missing_shape.onnx", graph);

    let message = infer_first_layer_output_shape(&model)
        .expect_err("incompatible binary broadcast fallback must fail closed");
    assert!(
        message.contains("cannot broadcast"),
        "error should explain why binary fallback is rejected, got: {message}",
    );
}

#[test]
fn test_infer_output_shape_handles_missing_shape_for_constant_reshape_intermediate_3498() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("reshape", "Reshape", &["input", "shape"], &["reshaped"]),
            node("relu", "Relu", &["reshaped"], &["out"]),
        ],
        name: "reshape_missing_shape".to_string(),
        initializer: vec![tensor_i64("shape", &[2], &[2, 2])],
        input: vec![tensor_value_info("input", &[1, 4], 1)],
        output: vec![tensor_value_info("out", &[2, 2], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("reshape_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("constant reshape fallback should work");
    assert_eq!(inferred_shape, vec![2, 2]);
}

#[test]
fn test_infer_output_shape_rejects_missing_shape_for_runtime_reshape_intermediate_3498() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("reshape", "Reshape", &["input", "shape"], &["reshaped"]),
            node("relu", "Relu", &["reshaped"], &["out"]),
        ],
        name: "reshape_runtime_shape_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("input", &[1, 4], 1),
            tensor_value_info("shape", &[2], 7),
        ],
        output: vec![tensor_value_info("out", &[2, 2], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("reshape_runtime_shape_missing_shape.onnx", graph);

    let message = infer_first_layer_output_shape(&model)
        .expect_err("runtime-driven reshape fallback must stay rejected");
    assert!(
        message.contains("shape input 'shape' is not a constant tensor"),
        "error should explain why runtime reshape fallback is rejected, got: {message}",
    );
}

#[test]
fn test_infer_output_shape_handles_transpose_intermediate_3498() {
    let mut transpose = node("transpose", "Transpose", &["input"], &["hidden"]);
    transpose.attribute = vec![attr_ints("perm", &[1, 0, 2])];
    let graph = onnx_proto::GraphProto {
        node: vec![transpose, node("relu", "Relu", &["hidden"], &["out"])],
        name: "transpose_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[4, 1, 640], 1)],
        output: vec![tensor_value_info("out", &[1, 4, 640], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("transpose_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("transpose fallback should use perm");
    assert_eq!(inferred_shape, vec![1, 4, 640]);
}

#[test]
fn test_infer_output_shape_handles_reduce_sum_keepdims_intermediate_3498() {
    let mut reduce_sum = node("reduce_sum", "ReduceSum", &["input"], &["reduced"]);
    reduce_sum.attribute = vec![attr_ints("axes", &[-1]), attr_int("keepdims", 1)];
    let graph = onnx_proto::GraphProto {
        node: vec![reduce_sum, node("sqrt", "Sqrt", &["reduced"], &["out"])],
        name: "reduce_sum_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[5, 1024], 1)],
        output: vec![tensor_value_info("out", &[5, 1], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("reduce_sum_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("reduction fallback should use axes");
    assert_eq!(inferred_shape, vec![5, 1]);
}

#[test]
fn test_infer_output_shape_handles_reduce_sum_input_axes_intermediate_3498() {
    let mut reduce_sum = node("reduce_sum", "ReduceSum", &["input", "axes"], &["reduced"]);
    reduce_sum.attribute = vec![attr_int("keepdims", 1)];
    let graph = onnx_proto::GraphProto {
        node: vec![reduce_sum, node("sqrt", "Sqrt", &["reduced"], &["out"])],
        name: "reduce_sum_input_axes_missing_shape".to_string(),
        initializer: vec![tensor_i64("axes", &[1], &[-1])],
        input: vec![tensor_value_info("input", &[5, 1024], 1)],
        output: vec![tensor_value_info("out", &[5, 1], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model =
        load_inline_onnx_model_with_opset("reduce_sum_input_axes_missing_shape.onnx", 18, graph);

    let inferred_shape = infer_first_layer_output_shape(&model)
        .expect("reduction fallback should honor constant axes inputs");
    assert_eq!(inferred_shape, vec![5, 1]);
}

#[test]
fn test_infer_output_shape_handles_slice_intermediate_3498() {
    let slice = node(
        "slice",
        "Slice",
        &["input", "starts", "ends", "axes", "steps"],
        &["window"],
    );
    let graph = onnx_proto::GraphProto {
        node: vec![slice, node("relu", "Relu", &["window"], &["out"])],
        name: "slice_missing_shape".to_string(),
        initializer: vec![
            tensor_i64("starts", &[1], &[0]),
            tensor_i64("ends", &[1], &[1]),
            tensor_i64("axes", &[1], &[1]),
            tensor_i64("steps", &[1], &[1]),
        ],
        input: vec![tensor_value_info("input", &[1, 4, 640], 1)],
        output: vec![tensor_value_info("out", &[1, 1, 640], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("slice_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("slice fallback should use constant bounds");
    assert_eq!(inferred_shape, vec![1, 1, 640]);
}

#[test]
fn test_infer_output_shape_handles_weighted_matmul_intermediate_3498() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("matmul", "MatMul", &["input", "weight"], &["hidden"]),
            node("relu", "Relu", &["hidden"], &["out"]),
        ],
        name: "matmul_missing_shape".to_string(),
        initializer: vec![tensor_f32(
            "weight",
            &[4, 3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0,
            ],
        )],
        input: vec![tensor_value_info("input", &[1, 4], 1)],
        output: vec![tensor_value_info("out", &[1, 3], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("matmul_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("weighted MatMul fallback should work");
    assert_eq!(inferred_shape, vec![1, 3]);
}

#[test]
fn test_infer_output_shape_handles_unsqueeze_intermediate_3498() {
    let mut unsqueeze = node("unsqueeze", "Unsqueeze", &["input"], &["hidden"]);
    unsqueeze.attribute = vec![attr_ints("axes", &[0])];
    let graph = onnx_proto::GraphProto {
        node: vec![unsqueeze, node("relu", "Relu", &["hidden"], &["out"])],
        name: "unsqueeze_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[2, 3], 1)],
        output: vec![tensor_value_info("out", &[1, 2, 3], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("unsqueeze_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("unsqueeze fallback should use axes");
    assert_eq!(inferred_shape, vec![1, 2, 3]);
}

#[test]
fn test_infer_output_shape_handles_unsqueeze_input_axes_intermediate_3498() {
    let unsqueeze = node("unsqueeze", "Unsqueeze", &["input", "axes"], &["hidden"]);
    let graph = onnx_proto::GraphProto {
        node: vec![unsqueeze, node("relu", "Relu", &["hidden"], &["out"])],
        name: "unsqueeze_input_axes_missing_shape".to_string(),
        initializer: vec![tensor_i64("axes", &[1], &[0])],
        input: vec![tensor_value_info("input", &[2, 3], 1)],
        output: vec![tensor_value_info("out", &[1, 2, 3], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model =
        load_inline_onnx_model_with_opset("unsqueeze_input_axes_missing_shape.onnx", 13, graph);

    let inferred_shape = infer_first_layer_output_shape(&model)
        .expect("unsqueeze fallback should honor constant axes inputs");
    assert_eq!(inferred_shape, vec![1, 2, 3]);
}

#[test]
fn test_infer_output_shape_handles_concat_intermediate_3498() {
    let mut concat = node("concat", "Concat", &["lhs", "rhs"], &["hidden"]);
    concat.attribute = vec![attr_int("axis", 1)];
    let graph = onnx_proto::GraphProto {
        node: vec![concat, node("relu", "Relu", &["hidden"], &["out"])],
        name: "concat_missing_shape".to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("lhs", &[1, 2], 1),
            tensor_value_info("rhs", &[1, 3], 1),
        ],
        output: vec![tensor_value_info("out", &[1, 5], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("concat_missing_shape.onnx", graph);

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("concat fallback should use axis");
    assert_eq!(inferred_shape, vec![1, 5]);
}

#[test]
fn test_infer_output_shape_handles_conv1d_intermediate_3500() {
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[4, 2, 3])));
    let model = OnnxModel::empty_with_network(
        Network {
            name: "conv1d_missing_shape".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 2, 8],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "hidden".to_string(),
                shape: vec![1, 4, 4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "conv".to_string(),
                layer_type: LayerType::Conv1d,
                inputs: vec!["input".to_string(), "w".to_string()],
                outputs: vec!["hidden".to_string()],
                weights: None,
                attributes: HashMap::from([
                    ("pads".to_string(), AttributeValue::Ints(vec![1, 1])),
                    ("strides".to_string(), AttributeValue::Ints(vec![2])),
                ]),
            }],
            param_count: 4 * 2 * 3,
        },
        weights,
    )
    .with_tensor_shapes(HashMap::from([("input".to_string(), vec![1, 2, 8])]));

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("Conv1d fallback should use ONNX formula");
    assert_eq!(inferred_shape, vec![1, 4, 4]);
}

#[test]
fn test_infer_output_shape_handles_conv_transpose1d_intermediate_3500() {
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[3, 2, 3])));
    let model = OnnxModel::empty_with_network(
        Network {
            name: "conv_transpose1d_missing_shape".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 4],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "hidden".to_string(),
                shape: vec![1, 2, 8],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "deconv".to_string(),
                layer_type: LayerType::ConvTranspose1d,
                inputs: vec!["input".to_string(), "w".to_string()],
                outputs: vec!["hidden".to_string()],
                weights: None,
                attributes: HashMap::from([
                    ("pads".to_string(), AttributeValue::Ints(vec![1, 1])),
                    ("strides".to_string(), AttributeValue::Ints(vec![2])),
                    ("output_padding".to_string(), AttributeValue::Ints(vec![1])),
                ]),
            }],
            param_count: 3 * 2 * 3,
        },
        weights,
    )
    .with_tensor_shapes(HashMap::from([("input".to_string(), vec![1, 3, 4])]));

    let inferred_shape = infer_first_layer_output_shape(&model)
        .expect("ConvTranspose1d fallback should use ONNX formula");
    assert_eq!(inferred_shape, vec![1, 2, 8]);
}

#[test]
fn test_infer_output_shape_handles_linear_transb1_intermediate_3500() {
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[4, 2])));
    let model = OnnxModel::empty_with_network(
        Network {
            name: "linear_transb1_missing_shape".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 2],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "hidden".to_string(),
                shape: vec![1, 3, 4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "linear".to_string(),
                layer_type: LayerType::Linear,
                inputs: vec!["input".to_string(), "w".to_string()],
                outputs: vec!["hidden".to_string()],
                weights: None,
                attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
            }],
            param_count: 4 * 2,
        },
        weights,
    )
    .with_tensor_shapes(HashMap::from([("input".to_string(), vec![1, 3, 2])]));

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("Linear fallback should honor transB=1");
    assert_eq!(inferred_shape, vec![1, 3, 4]);
}

#[test]
fn test_infer_output_shape_handles_linear_default_transb0_intermediate_3500() {
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[2, 4])));
    let model = OnnxModel::empty_with_network(
        Network {
            name: "linear_default_transb0_missing_shape".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 2],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "hidden".to_string(),
                shape: vec![1, 3, 4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "linear".to_string(),
                layer_type: LayerType::Linear,
                inputs: vec!["input".to_string(), "w".to_string()],
                outputs: vec!["hidden".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 2 * 4,
        },
        weights,
    )
    .with_tensor_shapes(HashMap::from([("input".to_string(), vec![1, 3, 2])]));

    let inferred_shape = infer_first_layer_output_shape(&model)
        .expect("Linear fallback should use ONNX default transB=0");
    assert_eq!(inferred_shape, vec![1, 3, 4]);
}

#[test]
fn test_infer_output_shape_handles_pad_input_pads_intermediate_3500() {
    let mut weights = WeightStore::new();
    weights.insert_integers(
        "pads".to_string(),
        ndarray::arr1(&[0_i64, 0, 2, 0, 0, 2]).into_dyn(),
    );
    let model = OnnxModel::empty_with_network(
        Network {
            name: "pad_missing_shape".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 4],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "hidden".to_string(),
                shape: vec![1, 3, 8],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "pad".to_string(),
                layer_type: LayerType::Pad,
                inputs: vec!["input".to_string(), "pads".to_string()],
                outputs: vec!["hidden".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        weights,
    )
    .with_tensor_shapes(HashMap::from([("input".to_string(), vec![1, 3, 4])]));

    let inferred_shape =
        infer_first_layer_output_shape(&model).expect("Pad fallback should use constant pads");
    assert_eq!(inferred_shape, vec![1, 3, 8]);
}

#[test]
fn test_infer_output_shape_rejects_allowzero_reshape_intermediate_3498() {
    let mut reshape = node("reshape", "Reshape", &["input", "shape"], &["reshaped"]);
    reshape.attribute = vec![attr_int("allowzero", 1)];
    let graph = onnx_proto::GraphProto {
        node: vec![reshape, node("relu", "Relu", &["reshaped"], &["out"])],
        name: "reshape_allowzero_missing_shape".to_string(),
        initializer: vec![tensor_i64("shape", &[2], &[0, -1])],
        input: vec![tensor_value_info("input", &[2, 2], 1)],
        output: vec![tensor_value_info("out", &[2, 2], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = load_inline_onnx_model("reshape_allowzero_missing_shape.onnx", graph);

    let message = infer_first_layer_output_shape(&model)
        .expect_err("allowzero reshape fallback must fail closed");
    assert!(
        message.contains("allowzero=1"),
        "error should explain why allowzero reshape fallback is rejected, got: {message}",
    );
}
