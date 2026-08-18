// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::reduction_schema::validate_reduction_comparison_schemas_unfolded;
use crate::onnx_proto::{
    attribute_type, tensor_shape_proto, AttributeProto, GraphProto, NodeProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use std::collections::HashMap;

const FLOAT: i32 = 1;
const INT8: i32 = 3;
const INT32: i32 = 6;
const INT64: i32 = 7;
const STRING: i32 = 8;
const BOOL: i32 = 9;
const FLOAT16: i32 = 10;
const BFLOAT16: i32 = 16;

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INT,
        i: Some(value),
        ..Default::default()
    }
}

fn ints_attr(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INTS,
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn node(op_type: &str, inputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        name: format!("{op_type}_node"),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: vec!["out".to_string()],
        attribute: attrs,
        ..Default::default()
    }
}

fn info(name: &str, dtype: i32, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: dtype,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|dimension| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(*dimension)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn tensor(name: &str, dtype: i32, shape: &[i64]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: dtype,
        ..Default::default()
    }
}

fn validate(
    node: NodeProto,
    opset: i64,
    inputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
) -> ny_core::Result<()> {
    validate_reduction_comparison_schemas_unfolded(
        &GraphProto {
            node: vec![node],
            input: inputs,
            initializer: initializers,
            ..Default::default()
        },
        &HashMap::from([(String::new(), opset)]),
    )
}

fn reduction_inputs() -> Vec<ValueInfoProto> {
    vec![info("data", FLOAT, &[2, 3])]
}

#[test]
fn reduction_dtype_proves_authentic_split_elementwise_producer_cone() {
    let split = NodeProto {
        name: "split".to_string(),
        op_type: "Split".to_string(),
        input: vec!["data".to_string()],
        output: vec!["left".to_string(), "right".to_string()],
        ..Default::default()
    };
    let product = NodeProto {
        name: "product".to_string(),
        op_type: "Mul".to_string(),
        input: vec!["left".to_string(), "right".to_string()],
        output: vec!["product_out".to_string()],
        ..Default::default()
    };
    let reduction = node("ReduceSum", &["product_out"], vec![ints_attr("axes", &[1])]);
    validate_reduction_comparison_schemas_unfolded(
        &GraphProto {
            node: vec![split, product, reduction],
            input: reduction_inputs(),
            ..Default::default()
        },
        &HashMap::from([(String::new(), 10)]),
    )
    .expect("Split outputs preserve FLOAT dtype through the Mul/ReduceSum cone");
}

#[test]
fn reduction_axes_encoding_is_versioned_per_operator() {
    validate(
        node("ReduceSum", &["data"], vec![ints_attr("axes", &[1])]),
        12,
        reduction_inputs(),
        vec![],
    )
    .expect("ReduceSum-12 uses an axes attribute");
    let error = validate(
        node("ReduceSum", &["data", "axes"], vec![]),
        12,
        reduction_inputs(),
        vec![tensor("axes", INT64, &[1])],
    )
    .expect_err("ReduceSum input axes were introduced at opset 13");
    assert!(error.to_string().contains("signature"), "{error}");

    validate(
        node("ReduceSum", &["data", "axes"], vec![]),
        13,
        reduction_inputs(),
        vec![tensor("axes", INT64, &[1])],
    )
    .expect("ReduceSum-13 uses an INT64 vector input");
    let error = validate(
        node("ReduceSum", &["data"], vec![ints_attr("axes", &[1])]),
        13,
        reduction_inputs(),
        vec![],
    )
    .expect_err("the obsolete attribute must not survive at opset 13");
    assert!(error.to_string().contains("obsolete axes"), "{error}");

    for op_type in ["ReduceMean", "ReduceMax", "ReduceMin"] {
        validate(
            node(op_type, &["data"], vec![ints_attr("axes", &[1])]),
            17,
            reduction_inputs(),
            vec![],
        )
        .unwrap_or_else(|error| panic!("{op_type}-17 attribute form: {error}"));
        validate(
            node(op_type, &["data", "axes"], vec![]),
            18,
            reduction_inputs(),
            vec![tensor("axes", INT64, &[1])],
        )
        .unwrap_or_else(|error| panic!("{op_type}-18 input form: {error}"));
        assert!(
            validate(
                node(op_type, &["data", "axes"], vec![]),
                17,
                reduction_inputs(),
                vec![tensor("axes", INT64, &[1])],
            )
            .is_err(),
            "{op_type}-17 must reject input axes"
        );
    }
}

#[test]
fn reductions_validate_boolean_attributes_and_axes_tensor_schema() {
    validate(
        node(
            "ReduceMean",
            &["data", ""],
            vec![int_attr("keepdims", 0), int_attr("noop_with_empty_axes", 1)],
        ),
        18,
        reduction_inputs(),
        vec![],
    )
    .expect("an omitted modern axes input can request identity semantics");

    for (attribute, value) in [("keepdims", 2), ("noop_with_empty_axes", -1)] {
        let error = validate(
            node("ReduceMean", &["data"], vec![int_attr(attribute, value)]),
            18,
            reduction_inputs(),
            vec![],
        )
        .expect_err("boolean attributes must be exactly zero or one");
        assert!(error.to_string().contains("0 or 1"), "{error}");
    }

    let error = validate(
        node("ReduceSum", &["data", "axes"], vec![]),
        13,
        reduction_inputs(),
        vec![tensor("axes", INT32, &[1])],
    )
    .expect_err("reduction axes inputs are INT64, unlike CumSum axis");
    assert!(error.to_string().contains("dtype [7]"), "{error}");
    let error = validate(
        node("ReduceSum", &["data", "axes"], vec![]),
        13,
        reduction_inputs(),
        vec![tensor("axes", INT64, &[])],
    )
    .expect_err("a reduction axes input is a vector, not a scalar");
    assert!(error.to_string().contains("rank 1"), "{error}");

    let error = validate(
        node("ReduceSum", &["data"], vec![int_attr("mystery", 0)]),
        13,
        reduction_inputs(),
        vec![],
    )
    .expect_err("unknown attributes must fail before folding");
    assert!(error.to_string().contains("mystery"), "{error}");
}

#[test]
fn reduction_dtype_expansions_are_audited_but_ny_float_subset_fails_closed() {
    for (op_type, old_opset, new_opset, dtype) in [
        ("ReduceSum", 12, 13, BFLOAT16),
        ("ReduceMax", 11, 12, INT8),
        ("ReduceMin", 19, 20, BOOL),
    ] {
        let inputs = || vec![info("data", dtype, &[2, 3])];
        assert!(
            validate(
                node(op_type, &["data"], vec![]),
                old_opset,
                inputs(),
                vec![]
            )
            .is_err(),
            "{op_type} must reject dtype {dtype} at opset {old_opset}"
        );
        let error = validate(
            node(op_type, &["data"], vec![]),
            new_opset,
            inputs(),
            vec![],
        )
        .expect_err("a dtype becoming ONNX-valid must not bypass ny's FLOAT-only representation");
        assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");
    }

    let error = validate(
        node("ReduceMean", &["data"], vec![]),
        18,
        vec![info("data", INT32, &[2, 3])],
        vec![],
    )
    .expect_err("integer reductions cannot be represented by ny's f32 arithmetic");
    assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");
}

#[test]
fn cumsum_requires_opset_scalar_integer_axis_and_boolean_flags() {
    let inputs = || vec![info("data", FLOAT, &[2, 3])];
    let axis = || vec![tensor("axis", INT32, &[])];
    assert!(validate(
        node("CumSum", &["data", "axis"], vec![]),
        10,
        inputs(),
        axis()
    )
    .is_err());
    validate(
        node(
            "CumSum",
            &["data", "axis"],
            vec![int_attr("exclusive", 1), int_attr("reverse", 0)],
        ),
        11,
        inputs(),
        axis(),
    )
    .expect("CumSum-11 accepts a scalar INT32 axis");

    let error = validate(
        node("CumSum", &["data", "axis"], vec![]),
        14,
        inputs(),
        vec![tensor("axis", INT64, &[1])],
    )
    .expect_err("a one-element vector is not CumSum's required scalar axis");
    assert!(error.to_string().contains("rank 0"), "{error}");
    let error = validate(
        node("CumSum", &["data", "axis"], vec![int_attr("reverse", 2)]),
        14,
        inputs(),
        axis(),
    )
    .expect_err("reverse=2 is outside the represented boolean semantics");
    assert!(error.to_string().contains("0 or 1"), "{error}");

    let error = validate(
        node("CumSum", &["integer_data", "axis"], vec![]),
        14,
        vec![info("integer_data", INT32, &[2, 3])],
        axis(),
    )
    .expect_err("integer CumSum cannot be represented by ny's f32 scan");
    assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");
}

#[test]
fn comparison_minimum_opsets_attributes_and_dtypes_are_versioned() {
    let float_inputs = || vec![info("a", FLOAT, &[2, 3]), info("b", FLOAT, &[2, 3])];
    assert!(validate(
        node("GreaterOrEqual", &["a", "b"], vec![]),
        11,
        float_inputs(),
        vec![],
    )
    .is_err());
    validate(
        node("GreaterOrEqual", &["a", "b"], vec![]),
        12,
        float_inputs(),
        vec![],
    )
    .expect("GreaterOrEqual was introduced at opset 12");

    let error = validate(
        node("Greater", &["a", "b"], vec![int_attr("broadcast", 1)]),
        7,
        float_inputs(),
        vec![],
    )
    .expect_err("legacy broadcast attributes disappeared at opset 7");
    assert!(error.to_string().contains("broadcast"), "{error}");

    assert!(
        validate(
            node("Equal", &["a", "b"], vec![]),
            7,
            float_inputs(),
            vec![],
        )
        .is_err(),
        "Equal-7 accepts only BOOL/INT32/INT64"
    );
    validate(
        node("Equal", &["a", "b"], vec![]),
        11,
        float_inputs(),
        vec![],
    )
    .expect("Equal-11 added FLOAT");

    let string_inputs = || vec![tensor("a", STRING, &[1]), tensor("b", STRING, &[1])];
    assert!(validate(
        node("Equal", &["a", "b"], vec![]),
        18,
        vec![],
        string_inputs(),
    )
    .is_err());
    let error = validate(
        node("Equal", &["a", "b"], vec![]),
        19,
        vec![],
        string_inputs(),
    )
    .expect_err("Equal-19 added STRING, but ny cannot represent string comparison");
    assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");

    let error = validate(
        node("Equal", &["a", "b"], vec![]),
        13,
        vec![info("a", INT32, &[2]), info("b", INT32, &[2])],
        vec![],
    )
    .expect_err("integer comparison cannot be normalized to f32 without semantic loss");
    assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");
}

#[test]
fn legacy_comparison_broadcast_is_admitted_only_when_numpy_equivalent() {
    validate(
        node("Greater", &["a", "b"], vec![]),
        6,
        vec![info("a", FLOAT, &[2, 3]), info("b", FLOAT, &[2, 3])],
        vec![],
    )
    .expect("broadcast=0 with equal shapes is ordinary elementwise comparison");
    let error = validate(
        node("Greater", &["a", "b"], vec![]),
        6,
        vec![info("a", FLOAT, &[2, 3]), info("b", FLOAT, &[3])],
        vec![],
    )
    .expect_err("ny must not invent NumPy broadcasting when legacy broadcast=0");
    assert!(error.to_string().contains("provably equal"), "{error}");

    validate(
        node(
            "Greater",
            &["a", "b"],
            vec![int_attr("broadcast", 1), int_attr("axis", 1)],
        ),
        6,
        vec![info("a", FLOAT, &[2, 3]), info("b", FLOAT, &[3])],
        vec![],
    )
    .expect("trailing legacy placement is exactly NumPy equivalent");
    let error = validate(
        node(
            "Greater",
            &["a", "b"],
            vec![int_attr("broadcast", 1), int_attr("axis", 0)],
        ),
        6,
        vec![info("a", FLOAT, &[3, 3]), info("b", FLOAT, &[3])],
        vec![],
    )
    .expect_err("non-trailing legacy placement has different semantics");
    assert!(error.to_string().contains("trailing placement"), "{error}");

    let error = validate(
        node(
            "Greater",
            &["a", "b"],
            vec![int_attr("broadcast", 1), int_attr("axis", 0)],
        ),
        6,
        vec![info("a", FLOAT, &[1]), info("b", FLOAT, &[2])],
        vec![],
    )
    .expect_err("legacy broadcasting may expand B to A, but may not grow A like NumPy");
    assert!(error.to_string().contains("incompatible"), "{error}");
}

#[test]
fn comparisons_and_where_authenticate_cross_input_and_output_dtypes() {
    let mismatch = validate(
        node("Greater", &["a", "b"], vec![]),
        13,
        vec![info("a", FLOAT, &[1]), info("b", INT32, &[1])],
        vec![],
    )
    .expect_err("comparison operands share one schema type variable");
    assert!(
        mismatch.to_string().contains("one tensor type"),
        "{mismatch}"
    );

    let where_inputs = || {
        vec![
            info("condition", BOOL, &[2]),
            info("x", FLOAT, &[2]),
            info("y", FLOAT, &[2]),
        ]
    };
    assert!(validate(
        node("Where", &["condition", "x", "y"], vec![]),
        8,
        where_inputs(),
        vec![],
    )
    .is_err());
    validate(
        node("Where", &["condition", "x", "y"], vec![]),
        9,
        where_inputs(),
        vec![],
    )
    .expect("Where was introduced at opset 9");

    let error = validate(
        node("Where", &["condition", "x", "y"], vec![]),
        16,
        vec![
            info("condition", FLOAT, &[2]),
            info("x", FLOAT, &[2]),
            info("y", FLOAT, &[2]),
        ],
        vec![],
    )
    .expect_err("Where condition must be BOOL");
    assert!(error.to_string().contains("dtype [9]"), "{error}");

    let selected = || {
        vec![
            info("condition", BOOL, &[2]),
            info("x", BFLOAT16, &[2]),
            info("y", BFLOAT16, &[2]),
        ]
    };
    assert!(validate(
        node("Where", &["condition", "x", "y"], vec![]),
        15,
        selected(),
        vec![],
    )
    .is_err());
    let error = validate(
        node("Where", &["condition", "x", "y"], vec![]),
        16,
        selected(),
        vec![],
    )
    .expect_err("Where-16 added BFLOAT16, but ny represents only FLOAT branches");
    assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");

    let error = validate(
        node("Where", &["condition", "x", "y"], vec![]),
        13,
        vec![
            info("condition", BOOL, &[2]),
            info("x", FLOAT16, &[2]),
            info("y", FLOAT16, &[2]),
        ],
        vec![],
    )
    .expect_err("FLOAT16 selection cannot be silently normalized to FLOAT");
    assert!(error.to_string().contains("only ONNX FLOAT"), "{error}");
}
