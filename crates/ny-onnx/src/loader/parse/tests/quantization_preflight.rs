// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::parse_onnx_bytes;
use super::super::quantization_preflight::validate_quantization_schemas;
use crate::loader::{
    BatchNormFoldingPolicy, CustomOpRegistry, ShapeInferBackend, ShapeInferencePolicy,
};
use crate::onnx_proto::{
    attribute_type, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;
use std::collections::HashMap;

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT8: i32 = 3;
const UINT16: i32 = 4;
const INT16: i32 = 5;
const INT32: i32 = 6;
const INT64: i32 = 7;
const FLOAT16: i32 = 10;
const UINT4: i32 = 21;
const INT4: i32 = 22;

fn initializer(name: &str, dtype: i32) -> TensorProto {
    let mut tensor = TensorProto {
        dims: vec![1],
        data_type: dtype,
        name: name.to_string(),
        ..Default::default()
    };
    match dtype {
        FLOAT => tensor.float_data.push(1.0),
        UINT8 | INT8 | UINT16 | INT16 | INT32 => tensor.int32_data.push(0),
        INT64 => tensor.int64_data.push(0),
        FLOAT16 => tensor.raw_data.extend_from_slice(&0x3c00_u16.to_le_bytes()),
        _ => {}
    }
    tensor
}

fn value_info(name: &str, dtype: i32) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: dtype,
                shape: Some(TensorShapeProto {
                    dim: vec![crate::onnx_proto::tensor_shape_proto::Dimension {
                        value: Some(
                            crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(1),
                        ),
                    }],
                }),
            }),
        }),
    }
}

fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: Some(value),
        r#type: attribute_type::INT,
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values.to_vec(),
        r#type: attribute_type::INTS,
        ..Default::default()
    }
}

fn int64_initializer(name: &str, values: &[i64]) -> TensorProto {
    TensorProto {
        dims: vec![values.len() as i64],
        data_type: INT64,
        name: name.to_string(),
        int64_data: values.to_vec(),
        ..Default::default()
    }
}

fn node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    output: &str,
    attrs: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: attrs,
    }
}

fn opsets(version: i64) -> HashMap<String, i64> {
    HashMap::from([(String::new(), version), ("ai.onnx".to_string(), version)])
}

fn q_graph(op_type: &str, x_dtype: i32, attrs: Vec<AttributeProto>) -> GraphProto {
    let mut initializers = vec![initializer("x", x_dtype), initializer("scale", FLOAT)];
    // Most tests exercise per-tensor quantization.  Encode that as a true
    // scalar so a length-one per-axis vector cannot accidentally stand in for
    // scalar semantics.
    initializers[1].dims.clear();
    let inputs = if op_type == "QuantizeLinear" {
        initializers.push(initializer("zp", UINT8));
        initializers[2].dims.clear();
        vec!["x", "scale", "zp"]
    } else if x_dtype == INT32 {
        vec!["x", "scale"]
    } else {
        initializers.push(initializer("zp", x_dtype));
        initializers[2].dims.clear();
        vec!["x", "scale", "zp"]
    };
    GraphProto {
        node: vec![node("qdq", op_type, &inputs, "y", attrs)],
        initializer: initializers,
        ..Default::default()
    }
}

fn validation_error(graph: &GraphProto, version: i64) -> String {
    validate_quantization_schemas(graph, &opsets(version))
        .expect_err("quantization preflight should reject this graph")
        .to_string()
}

fn set_initializer_shape(graph: &mut GraphProto, name: &str, shape: &[i64]) {
    graph
        .initializer
        .iter_mut()
        .find(|initializer| initializer.name == name)
        .unwrap_or_else(|| panic!("missing initializer {name}"))
        .dims = shape.to_vec();
}

#[test]
fn accepts_canonical_float32_qdq_and_infers_quantize_output_dtype() {
    let graph = GraphProto {
        node: vec![
            node(
                "quantize",
                "QuantizeLinear",
                &["x", "scale", "zp"],
                "q",
                vec![attr_int("axis", 0)],
            ),
            node(
                "dequantize",
                "DequantizeLinear",
                &["q", "scale", "zp"],
                "y",
                vec![attr_int("axis", 0)],
            ),
        ],
        initializer: vec![
            initializer("x", FLOAT),
            initializer("scale", FLOAT),
            initializer("zp", UINT8),
        ],
        output: vec![value_info("y", FLOAT)],
        ..Default::default()
    };

    validate_quantization_schemas(&graph, &opsets(23)).expect("canonical Q/DQ should validate");
}

#[test]
fn accepts_empty_optional_zero_point_and_int32_special_cases() {
    let q = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["x", "scale", ""],
            "q",
            vec![attr_int("axis", 0)],
        )],
        initializer: vec![initializer("x", INT32), initializer("scale", FLOAT)],
        ..Default::default()
    };
    validate_quantization_schemas(&q, &opsets(23))
        .expect("INT32 Q input and an empty optional input are canonical");

    let dq = q_graph("DequantizeLinear", INT32, vec![]);
    validate_quantization_schemas(&dq, &opsets(23))
        .expect("INT32 DQ input without a zero point is canonical");
}

#[test]
fn opset10_accepts_only_true_scalar_quantization_parameters() {
    for (op_type, x_dtype) in [("QuantizeLinear", FLOAT), ("DequantizeLinear", UINT8)] {
        let mut graph = q_graph(op_type, x_dtype, vec![]);
        set_initializer_shape(&mut graph, "scale", &[]);
        set_initializer_shape(&mut graph, "zp", &[]);
        validate_quantization_schemas(&graph, &opsets(10))
            .unwrap_or_else(|error| panic!("scalar opset-10 {op_type} should validate: {error}"));
    }

    let mut graph = q_graph("DequantizeLinear", INT32, vec![]);
    set_initializer_shape(&mut graph, "scale", &[]);
    validate_quantization_schemas(&graph, &opsets(10))
        .expect("opset-10 DQ permits INT32 without a zero point");
}

#[test]
fn rejects_unmodeled_quantization_parameter_granularity_before_folding() {
    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    set_initializer_shape(&mut graph, "scale", &[1]);
    set_initializer_shape(&mut graph, "zp", &[1]);
    let error = validation_error(&graph, 10);
    assert!(
        error.contains("true rank-0 scalar scale") && error.contains("[1]"),
        "{error}"
    );

    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    set_initializer_shape(&mut graph, "scale", &[]);
    set_initializer_shape(&mut graph, "zp", &[1]);
    let error = validation_error(&graph, 13);
    assert!(error.contains("must exactly match scale"), "{error}");

    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    set_initializer_shape(&mut graph, "scale", &[1, 1]);
    set_initializer_shape(&mut graph, "zp", &[1, 1]);
    let error = validation_error(&graph, 23);
    assert!(error.contains("blocked same-rank parameters"), "{error}");
}

#[test]
fn per_axis_parameters_require_a_proven_axis_and_exact_extent() {
    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![attr_int("axis", -1)]);
    set_initializer_shape(&mut graph, "x", &[2, 3]);
    set_initializer_shape(&mut graph, "scale", &[3]);
    set_initializer_shape(&mut graph, "zp", &[3]);
    validate_quantization_schemas(&graph, &opsets(23))
        .expect("negative last-axis selection with matching extent is canonical");

    set_initializer_shape(&mut graph, "scale", &[2]);
    set_initializer_shape(&mut graph, "zp", &[2]);
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("extent 2") && error.contains("axis -1 extent 3"),
        "{error}"
    );

    graph.node[0].attribute = vec![attr_int("axis", 2)];
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("outside input") && error.contains("rank 2"),
        "{error}"
    );
}

#[test]
fn integer_quantization_dtype_families_are_versioned() {
    let mut quantize = q_graph("QuantizeLinear", FLOAT, vec![]);
    quantize
        .initializer
        .iter_mut()
        .find(|initializer| initializer.name == "zp")
        .expect("zero point")
        .data_type = UINT16;
    let error = validation_error(&quantize, 20);
    assert!(
        error.contains("dtype 4") && error.contains("opset 20"),
        "{error}"
    );
    validate_quantization_schemas(&quantize, &opsets(21))
        .expect("UINT16 zero points enter QuantizeLinear at opset 21");

    let dequantize = q_graph("DequantizeLinear", INT16, vec![]);
    let error = validation_error(&dequantize, 20);
    assert!(
        error.contains("dtype 5") && error.contains("opset 20"),
        "{error}"
    );
    validate_quantization_schemas(&dequantize, &opsets(21))
        .expect("INT16 inputs enter DequantizeLinear at opset 21");
}

#[test]
fn quantize_opset21_versions_block_size_and_output_dtype() {
    let attributes = [
        attr_int("block_size", 0),
        attr_int("output_dtype", i64::from(UINT8)),
    ];
    for attribute in attributes {
        let graph = q_graph("QuantizeLinear", FLOAT, vec![attribute.clone()]);
        let error = validation_error(&graph, 20);
        assert!(
            error.contains("opset 21"),
            "{} at opset 20: {error}",
            attribute.name
        );
        validate_quantization_schemas(&graph, &opsets(21)).unwrap_or_else(|error| {
            panic!(
                "QuantizeLinear {} should be accepted at opset 21: {error}",
                attribute.name
            )
        });
    }
}

#[test]
fn packed_four_bit_quantization_is_rejected_until_nibble_decoding_is_implemented() {
    for dtype in [UINT4, INT4] {
        let graph = q_graph("DequantizeLinear", dtype, vec![]);
        let error = validation_error(&graph, 23);
        assert!(
            error.contains(&format!("dtype {dtype}")) && error.contains("unsupported"),
            "{error}"
        );

        let graph = q_graph(
            "QuantizeLinear",
            FLOAT,
            vec![attr_int("output_dtype", i64::from(dtype))],
        );
        let error = validation_error(&graph, 23);
        assert!(error.contains("supported integer dtype"), "{error}");
    }
}

#[test]
fn proves_constant_cast_and_dtype_preserving_producers_conservatively() {
    let scale_constant = NodeProto {
        input: vec![],
        output: vec!["base_scale".to_string()],
        name: "scale_constant".to_string(),
        op_type: "Constant".to_string(),
        domain: String::new(),
        attribute: vec![AttributeProto {
            name: "value".to_string(),
            t: Some(initializer("", FLOAT)),
            r#type: attribute_type::TENSOR,
            ..Default::default()
        }],
    };
    let identity = node(
        "scale_identity",
        "Identity",
        &["base_scale"],
        "scale",
        vec![],
    );
    let cast = node(
        "x_cast",
        "Cast",
        &["raw_x"],
        "x",
        vec![attr_int("to", i64::from(FLOAT))],
    );
    let quantize = node(
        "quantize",
        "QuantizeLinear",
        &["x", "scale", "zp"],
        "q",
        vec![attr_int("axis", 0)],
    );
    let graph = GraphProto {
        node: vec![scale_constant, identity, cast, quantize],
        initializer: vec![initializer("raw_x", FLOAT16), initializer("zp", UINT8)],
        ..Default::default()
    };

    validate_quantization_schemas(&graph, &opsets(23))
        .expect("the Cast target and safe producer cone prove FLOAT32 Q inputs");
}

#[test]
fn proves_static_parameter_shapes_through_shape_operator_cones_without_value_info() {
    let mut x = initializer("x", FLOAT);
    x.dims = vec![2, 3];
    let mut scale_base = initializer("scale_base", FLOAT);
    scale_base.dims = vec![1, 3];
    let mut zero_point = initializer("zp", UINT8);
    zero_point.dims = vec![3];

    let graph = GraphProto {
        node: vec![
            node("identity", "Identity", &["scale_base"], "scale_i", vec![]),
            node(
                "cast",
                "Cast",
                &["scale_i"],
                "scale_c",
                vec![attr_int("to", i64::from(FLOAT))],
            ),
            node(
                "reshape",
                "Reshape",
                &["scale_c", "reshape_shape"],
                "scale_r",
                vec![],
            ),
            node(
                "unsqueeze",
                "Unsqueeze",
                &["scale_r", "unsqueeze_axes"],
                "scale_u",
                vec![],
            ),
            node(
                "transpose",
                "Transpose",
                &["scale_u"],
                "scale_t",
                vec![attr_ints("perm", &[1, 0])],
            ),
            node(
                "squeeze",
                "Squeeze",
                &["scale_t", "squeeze_axes"],
                "scale",
                vec![],
            ),
            node(
                "quantize",
                "QuantizeLinear",
                &["x", "scale", "zp"],
                "q",
                vec![attr_int("axis", 1)],
            ),
        ],
        initializer: vec![
            x,
            scale_base,
            zero_point,
            int64_initializer("reshape_shape", &[3]),
            int64_initializer("unsqueeze_axes", &[0]),
            int64_initializer("squeeze_axes", &[1]),
        ],
        ..Default::default()
    };

    validate_quantization_schemas(&graph, &opsets(23))
        .expect("the exact parameter shape cone should need no intermediate value_info");
}

#[test]
fn static_parameter_shape_controls_may_use_exact_identity_and_integer_cast_cones() {
    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![attr_int("axis", 0)]);
    set_initializer_shape(&mut graph, "x", &[3]);
    set_initializer_shape(&mut graph, "scale", &[3]);
    set_initializer_shape(&mut graph, "zp", &[3]);

    // Replace the direct scale with a reshape whose target is an exact INT32
    // initializer promoted to the schema-required INT64 through Cast/Identity.
    graph
        .initializer
        .iter_mut()
        .find(|initializer| initializer.name == "scale")
        .expect("scale")
        .name = "scale_base".to_string();
    graph.initializer.push(TensorProto {
        dims: vec![1],
        data_type: INT32,
        name: "shape_i32".to_string(),
        int32_data: vec![3],
        ..Default::default()
    });
    graph.node.insert(
        0,
        node(
            "shape_cast",
            "Cast",
            &["shape_i32"],
            "shape_i64",
            vec![attr_int("to", i64::from(INT64))],
        ),
    );
    graph.node.insert(
        1,
        node(
            "shape_identity",
            "Identity",
            &["shape_i64"],
            "shape",
            vec![],
        ),
    );
    graph.node.insert(
        2,
        node(
            "reshape_scale",
            "Reshape",
            &["scale_base", "shape"],
            "scale",
            vec![],
        ),
    );

    validate_quantization_schemas(&graph, &opsets(23))
        .expect("exact integer control cones should remain statically provable");
}

#[test]
fn proves_scalar_parameter_reshape_from_an_exact_empty_shape_tensor() {
    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    let scale = graph
        .initializer
        .iter_mut()
        .find(|initializer| initializer.name == "scale")
        .expect("scale");
    scale.name = "scale_base".to_string();
    scale.dims = vec![1];
    graph
        .initializer
        .push(int64_initializer("scalar_shape", &[]));
    graph.node.insert(
        0,
        node(
            "reshape_scalar_scale",
            "Reshape",
            &["scale_base", "scalar_shape"],
            "scale",
            vec![],
        ),
    );

    validate_quantization_schemas(&graph, &opsets(23))
        .expect("an exact empty INT64 shape tensor should prove a true scalar parameter reshape");
}

#[test]
fn rejects_attributes_before_their_schema_versions() {
    let cases = [
        ("QuantizeLinear", 9, None, "opset 10"),
        ("QuantizeLinear", 12, Some(("axis", 0)), "opset 13"),
        ("QuantizeLinear", 18, Some(("saturate", 1)), "opset 19"),
        ("QuantizeLinear", 22, Some(("precision", 1)), "opset 23"),
        ("DequantizeLinear", 20, Some(("block_size", 0)), "opset 21"),
        (
            "DequantizeLinear",
            22,
            Some(("output_dtype", 1)),
            "opset 23",
        ),
    ];

    for (op_type, version, attr, expected) in cases {
        let attrs = attr
            .map(|(name, value)| vec![attr_int(name, value)])
            .unwrap_or_default();
        let x_dtype = if op_type == "QuantizeLinear" {
            FLOAT
        } else {
            UINT8
        };
        let error = validation_error(&q_graph(op_type, x_dtype, attrs), version);
        assert!(
            error.contains(expected),
            "{op_type} opset {version}: {error}"
        );
    }
}

#[test]
fn rejects_malformed_and_unsupported_attributes() {
    let mut non_int = attr_int("axis", 0);
    non_int.r#type = attribute_type::FLOAT;
    let cases = [
        (vec![attr_int("axis", 0), attr_int("axis", 0)], "duplicate"),
        (vec![non_int], "must have ONNX INT type"),
        (vec![attr_int("mystery", 0)], "unsupported attribute"),
        (vec![attr_int("block_size", 2)], "blocked quantization"),
        (vec![attr_int("saturate", 2)], "must be 0 or 1"),
        (vec![attr_int("precision", 10)], "default/0 or FLOAT"),
        (
            vec![attr_int("output_dtype", INT32.into())],
            "supported integer dtype",
        ),
    ];
    for (attrs, expected) in cases {
        let error = validation_error(&q_graph("QuantizeLinear", FLOAT, attrs), 23);
        assert!(error.contains(expected), "{error}");
    }

    let error = validation_error(
        &q_graph(
            "DequantizeLinear",
            UINT8,
            vec![attr_int("output_dtype", 10)],
        ),
        23,
    );
    assert!(error.contains("default/0 or FLOAT"), "{error}");
}

#[test]
fn rejects_noncanonical_inputs_and_outputs() {
    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    graph.node[0].input[0].clear();
    let error = validation_error(&graph, 23);
    assert!(error.contains("two required inputs"), "{error}");

    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    graph.node[0].input.push("extra".to_string());
    let error = validation_error(&graph, 23);
    assert!(error.contains("two required inputs"), "{error}");

    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    graph.node[0].output.push("extra".to_string());
    let error = validation_error(&graph, 23);
    assert!(error.contains("one non-empty output"), "{error}");
}

#[test]
fn rejects_unmodeled_authored_dtypes_and_zero_point_mismatches() {
    let mut graph = q_graph("QuantizeLinear", FLOAT, vec![]);
    graph.initializer[1].data_type = FLOAT16;
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("scale") && error.contains("got 10"),
        "{error}"
    );

    let graph = q_graph("DequantizeLinear", FLOAT, vec![]);
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("quantized dtype 1") && error.contains("unsupported"),
        "{error}"
    );

    let mut graph = q_graph("DequantizeLinear", UINT8, vec![]);
    graph.initializer[2].data_type = INT8;
    let error = validation_error(&graph, 23);
    assert!(error.contains("must match input"), "{error}");

    let mut graph = q_graph("DequantizeLinear", INT32, vec![]);
    graph.initializer.push(initializer("zp", INT32));
    graph.node[0].input.push("zp".to_string());
    let error = validation_error(&graph, 23);
    assert!(error.contains("must not specify a zero point"), "{error}");

    let graph = q_graph(
        "QuantizeLinear",
        FLOAT,
        vec![attr_int("output_dtype", i64::from(INT8))],
    );
    let error = validation_error(&graph, 23);
    assert!(error.contains("must match zero-point dtype"), "{error}");
}

#[test]
fn fails_closed_when_a_quantization_input_dtype_is_unknown() {
    let custom = NodeProto {
        input: vec![],
        output: vec!["x".to_string()],
        name: "custom_source".to_string(),
        op_type: "Source".to_string(),
        domain: "vendor.example".to_string(),
        attribute: vec![],
    };
    let graph = GraphProto {
        node: vec![
            custom,
            node(
                "quantize",
                "QuantizeLinear",
                &["x", "scale", "zp"],
                "q",
                vec![],
            ),
        ],
        initializer: vec![initializer("scale", FLOAT), initializer("zp", UINT8)],
        ..Default::default()
    };
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("cannot prove") && error.contains("'x'"),
        "{error}"
    );
}

#[test]
fn fails_closed_when_a_quantization_parameter_shape_is_unknown() {
    let mut scale_info = value_info("scale", FLOAT);
    scale_info
        .r#type
        .as_mut()
        .expect("tensor type")
        .tensor_type
        .as_mut()
        .expect("tensor type")
        .shape = None;
    let graph = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["x", "scale", "zp"],
            "q",
            vec![],
        )],
        initializer: vec![initializer("x", FLOAT), initializer("zp", UINT8)],
        input: vec![scale_info],
        ..Default::default()
    };
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("cannot prove") && error.contains("shape") && error.contains("'scale'"),
        "{error}"
    );
}

#[test]
fn fails_closed_when_reshape_parameter_shape_values_are_dynamic() {
    let mut shape_info = value_info("dynamic_shape", INT64);
    // Its rank and extent are known, but its value is not. Guessing that value
    // would let a dynamic Reshape masquerade as a scalar/per-axis parameter.
    shape_info
        .r#type
        .as_mut()
        .expect("tensor type")
        .tensor_type
        .as_mut()
        .expect("tensor type")
        .shape
        .as_mut()
        .expect("shape")
        .dim[0]
        .value = Some(crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(1));

    let graph = GraphProto {
        node: vec![
            node(
                "reshape_scale",
                "Reshape",
                &["scale_base", "dynamic_shape"],
                "scale",
                vec![],
            ),
            node(
                "quantize",
                "QuantizeLinear",
                &["x", "scale", "zp"],
                "q",
                vec![attr_int("axis", 0)],
            ),
        ],
        initializer: vec![
            initializer("x", FLOAT),
            initializer("scale_base", FLOAT),
            initializer("zp", UINT8),
        ],
        input: vec![shape_info],
        ..Default::default()
    };
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("cannot prove") && error.contains("shape") && error.contains("'scale'"),
        "{error}"
    );
}

#[test]
fn fails_closed_when_squeeze_axes_values_are_dynamic() {
    let axes_info = value_info("dynamic_axes", INT64);
    let mut scale_base = initializer("scale_base", FLOAT);
    scale_base.dims = vec![1, 1];
    let graph = GraphProto {
        node: vec![
            node(
                "squeeze_scale",
                "Squeeze",
                &["scale_base", "dynamic_axes"],
                "scale",
                vec![],
            ),
            node(
                "quantize",
                "QuantizeLinear",
                &["x", "scale", "zp"],
                "q",
                vec![],
            ),
        ],
        initializer: vec![
            initializer("x", FLOAT),
            scale_base,
            initializer("zp", UINT8),
        ],
        input: vec![axes_info],
        ..Default::default()
    };
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("cannot prove") && error.contains("shape") && error.contains("'scale'"),
        "{error}"
    );
}

#[test]
fn fails_closed_when_a_per_axis_input_shape_is_unknown() {
    let mut x_info = value_info("x", FLOAT);
    x_info
        .r#type
        .as_mut()
        .expect("tensor type")
        .tensor_type
        .as_mut()
        .expect("tensor type")
        .shape = None;
    let graph = GraphProto {
        node: vec![node(
            "quantize",
            "QuantizeLinear",
            &["x", "scale", "zp"],
            "q",
            vec![attr_int("axis", 0)],
        )],
        initializer: vec![initializer("scale", FLOAT), initializer("zp", UINT8)],
        input: vec![x_info],
        ..Default::default()
    };
    let error = validation_error(&graph, 23);
    assert!(
        error.contains("cannot prove")
            && error.contains("shape")
            && error.contains("input tensor 'x'"),
        "{error}"
    );
}

#[test]
fn parse_rejects_invalid_constant_only_quantization_before_it_can_fold_away() {
    let graph = GraphProto {
        node: vec![node(
            "constant_quantize",
            "QuantizeLinear",
            &["x", "scale"],
            "q",
            vec![attr_int("precision", i64::from(FLOAT))],
        )],
        initializer: vec![initializer("x", FLOAT), initializer("scale", FLOAT)],
        output: vec![value_info("q", UINT8)],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 22,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    let error = parse_onnx_bytes(
        &model.encode_to_vec(),
        &CustomOpRegistry::default(),
        ShapeInferencePolicy::Skip,
        &ShapeInferBackend::InProcess,
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
        false,
    )
    .expect_err("raw opset-invalid Q must fail before constant folding");
    assert!(
        error.to_string().contains("requires ONNX opset 23"),
        "{error}"
    );
}

#[test]
fn parse_rejects_opset10_vector_scale_before_constant_quantization_can_fold() {
    let graph = GraphProto {
        node: vec![node(
            "constant_quantize",
            "QuantizeLinear",
            &["x", "scale", "zp"],
            "q",
            vec![],
        )],
        initializer: vec![
            initializer("x", FLOAT),
            initializer("scale", FLOAT),
            initializer("zp", UINT8),
        ],
        output: vec![value_info("q", UINT8)],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 10,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    let error = parse_onnx_bytes(
        &model.encode_to_vec(),
        &CustomOpRegistry::default(),
        ShapeInferencePolicy::Skip,
        &ShapeInferBackend::InProcess,
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
        false,
    )
    .expect_err("opset-10 vector scale must fail before constant folding");
    assert!(
        error.to_string().contains("true rank-0 scalar scale"),
        "{error}"
    );
}
