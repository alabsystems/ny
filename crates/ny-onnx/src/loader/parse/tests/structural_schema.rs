// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::structural_schema::validate_structural_schemas_unfolded;
use crate::onnx_proto::{
    attribute_type, tensor_shape_proto, AttributeProto, GraphProto, NodeProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use std::collections::HashMap;

const FLOAT: i32 = 1;
const INT32: i32 = 6;
const INT64: i32 = 7;
const FLOAT16: i32 = 10;

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

fn node(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        name: format!("{op_type}_node"),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        attribute: attributes,
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
    nodes: Vec<NodeProto>,
    opset: i64,
    inputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
) -> ny_core::Result<()> {
    validate_structural_schemas_unfolded(
        &GraphProto {
            node: nodes,
            input: inputs,
            initializer: initializers,
            ..Default::default()
        },
        &HashMap::from([(String::new(), opset)]),
    )
}

fn float_input() -> Vec<ValueInfoProto> {
    vec![info("data", FLOAT, &[1, 3, 4])]
}

#[test]
fn range_schema_and_exact_int64_control_path_are_authenticated() {
    let int64_range = node("Range", &["start", "limit", "delta"], &["shape"], vec![]);
    let reshape = node("Reshape", &["data", "shape"], &["out"], vec![]);
    let scalar_controls = vec![
        tensor("start", INT64, &[]),
        tensor("limit", INT64, &[]),
        tensor("delta", INT64, &[]),
    ];
    validate(
        vec![int64_range.clone(), reshape],
        13,
        vec![info("data", FLOAT, &[6])],
        scalar_controls.clone(),
    )
    .expect("an exact static INT64 Range may terminate at a Reshape shape input");

    assert!(validate(
        vec![int64_range.clone()],
        10,
        vec![],
        scalar_controls.clone(),
    )
    .is_err());
    assert!(validate(
        vec![node("Range", &["start", "limit"], &["shape"], vec![],)],
        13,
        vec![],
        scalar_controls.clone(),
    )
    .is_err());
    assert!(validate(
        vec![int64_range.clone()],
        13,
        vec![],
        vec![
            tensor("start", INT64, &[1]),
            tensor("limit", INT64, &[]),
            tensor("delta", INT64, &[]),
        ],
    )
    .is_err());
    assert!(validate(
        vec![int64_range.clone()],
        13,
        vec![],
        vec![
            tensor("start", INT64, &[]),
            tensor("limit", INT64, &[]),
            tensor("delta", FLOAT, &[]),
        ],
    )
    .is_err());
    assert!(validate(
        vec![node(
            "Range",
            &["start", "limit", "delta"],
            &["shape"],
            vec![int_attr("stash_type", 1)],
        )],
        26,
        vec![],
        scalar_controls.clone(),
    )
    .is_err());
    validate(
        vec![node(
            "Range",
            &["start", "limit", "delta"],
            &["shape"],
            vec![int_attr("stash_type", 1)],
        )],
        27,
        vec![],
        scalar_controls.clone(),
    )
    .expect("Range-27 stash_type=FLOAT is represented");
    assert!(validate(
        vec![node(
            "Range",
            &["start", "limit", "delta"],
            &["shape"],
            vec![int_attr("stash_type", INT64.into())],
        )],
        27,
        vec![],
        scalar_controls,
    )
    .is_err());

    let graph = GraphProto {
        node: vec![int64_range],
        initializer: vec![
            tensor("start", INT64, &[]),
            tensor("limit", INT64, &[]),
            tensor("delta", INT64, &[]),
        ],
        output: vec![info("shape", INT64, &[1])],
        ..Default::default()
    };
    let error = validate_structural_schemas_unfolded(&graph, &HashMap::from([(String::new(), 13)]))
        .expect_err("INT64 Range values must not escape through the FLOAT32 verifier API");
    assert!(error.to_string().contains("graph output"), "{error}");
}

#[test]
fn shape_schema_versions_attributes_and_int64_result() {
    validate(
        vec![node("Shape", &["data"], &["out"], vec![])],
        13,
        float_input(),
        vec![],
    )
    .expect("full Shape is valid before opset 15");

    let sliced = node(
        "Shape",
        &["data"],
        &["out"],
        vec![int_attr("start", 1), int_attr("end", 2)],
    );
    assert!(validate(vec![sliced.clone()], 14, float_input(), vec![]).is_err());
    validate(vec![sliced], 15, float_input(), vec![]).expect("Shape-15 sliced shape");

    let error = validate(
        vec![node("Shape", &["data"], &["out"], vec![])],
        15,
        vec![info("data", FLOAT16, &[1, 3])],
        vec![],
    )
    .expect_err("unrepresented data dtype must fail before folding");
    assert!(error.to_string().contains("dtype 10"), "{error}");
}

#[test]
fn reshape_schema_authenticates_control_and_rejects_allowzero_one() {
    let reshape = node("Reshape", &["data", "shape"], &["out"], vec![]);
    validate(
        vec![reshape.clone()],
        13,
        float_input(),
        vec![tensor("shape", INT64, &[2])],
    )
    .expect("INT64 vector shape is represented");
    assert!(validate(
        vec![reshape.clone()],
        13,
        float_input(),
        vec![tensor("shape", INT32, &[2])],
    )
    .is_err());
    assert!(validate(
        vec![reshape],
        13,
        float_input(),
        vec![tensor("shape", INT64, &[])],
    )
    .is_err());

    let allowzero = node(
        "Reshape",
        &["data", "shape"],
        &["out"],
        vec![int_attr("allowzero", 1)],
    );
    assert!(validate(
        vec![allowzero.clone()],
        13,
        float_input(),
        vec![tensor("shape", INT64, &[2])],
    )
    .is_err());
    let error = validate(
        vec![allowzero],
        14,
        float_input(),
        vec![tensor("shape", INT64, &[2])],
    )
    .expect_err("allowzero=1 is not represented by the propagated layer");
    assert!(error.to_string().contains("allowzero=1"), "{error}");
}

#[test]
fn squeeze_and_unsqueeze_control_location_is_opset_aware() {
    let old_squeeze = node(
        "Squeeze",
        &["data"],
        &["out"],
        vec![ints_attr("axes", &[1])],
    );
    validate(vec![old_squeeze.clone()], 12, float_input(), vec![])
        .expect("Squeeze-12 axes attribute");
    assert!(validate(vec![old_squeeze], 13, float_input(), vec![]).is_err());

    let new_squeeze = node("Squeeze", &["data", "axes"], &["out"], vec![]);
    validate(
        vec![new_squeeze],
        13,
        float_input(),
        vec![tensor("axes", INT64, &[1])],
    )
    .expect("Squeeze-13 axes input");

    let old_unsqueeze = node(
        "Unsqueeze",
        &["data"],
        &["out"],
        vec![ints_attr("axes", &[1])],
    );
    validate(vec![old_unsqueeze.clone()], 12, float_input(), vec![])
        .expect("Unsqueeze-12 axes attribute");
    assert!(validate(vec![old_unsqueeze], 13, float_input(), vec![]).is_err());
    assert!(validate(
        vec![node("Unsqueeze", &["data", "axes"], &["out"], vec![])],
        13,
        float_input(),
        vec![tensor("axes", INT32, &[1])],
    )
    .is_err());
}

#[test]
fn slice_attribute_and_input_forms_are_not_interchanged() {
    let old = node(
        "Slice",
        &["data"],
        &["out"],
        vec![
            ints_attr("starts", &[0]),
            ints_attr("ends", &[2]),
            ints_attr("axes", &[1]),
        ],
    );
    validate(vec![old.clone()], 9, float_input(), vec![]).expect("Slice-9 attributes");
    assert!(validate(vec![old], 10, float_input(), vec![]).is_err());

    let modern = node(
        "Slice",
        &["data", "starts", "ends", "", "steps"],
        &["out"],
        vec![],
    );
    validate(
        vec![modern.clone()],
        10,
        float_input(),
        vec![
            tensor("starts", INT32, &[1]),
            tensor("ends", INT32, &[1]),
            tensor("steps", INT32, &[1]),
        ],
    )
    .expect("optional axes placeholder with uniform INT32 controls");
    let error = validate(
        vec![modern],
        10,
        float_input(),
        vec![
            tensor("starts", INT32, &[1]),
            tensor("ends", INT64, &[1]),
            tensor("steps", INT32, &[1]),
        ],
    )
    .expect_err("Slice control tensors share one Tind");
    assert!(error.to_string().contains("ends input"), "{error}");
}

#[test]
fn concat_requires_modern_axis_and_uniform_data_type() {
    let concat = node("Concat", &["data", "other"], &["out"], vec![]);
    assert!(validate(
        vec![concat],
        4,
        vec![info("data", FLOAT, &[1, 2]), info("other", FLOAT, &[1, 2])],
        vec![],
    )
    .is_err());
    let concat = node(
        "Concat",
        &["data", "other"],
        &["out"],
        vec![int_attr("axis", 1)],
    );
    validate(
        vec![concat.clone()],
        4,
        vec![info("data", FLOAT, &[1, 2]), info("other", FLOAT, &[1, 2])],
        vec![],
    )
    .expect("Concat-4 requires axis");
    validate(
        vec![concat.clone()],
        1,
        vec![info("data", FLOAT, &[1, 2]), info("other", FLOAT, &[1, 2])],
        vec![],
    )
    .expect("Concat-1 is represented when its non-ny default axis is explicit");
    assert!(validate(
        vec![concat],
        13,
        vec![
            info("data", FLOAT, &[1, 2]),
            info("other", FLOAT16, &[1, 2])
        ],
        vec![],
    )
    .is_err());
}

#[test]
fn split_partition_encoding_is_versioned_and_num_outputs_fails_closed() {
    let old = node(
        "Split",
        &["data"],
        &["left", "right"],
        vec![int_attr("axis", 1), ints_attr("split", &[1, 2])],
    );
    validate(vec![old.clone()], 12, float_input(), vec![]).expect("Split-12 attributes");
    assert!(validate(vec![old], 13, float_input(), vec![]).is_err());

    let modern = node(
        "Split",
        &["data", "parts"],
        &["left", "right"],
        vec![int_attr("axis", 1)],
    );
    validate(
        vec![modern],
        13,
        float_input(),
        vec![tensor("parts", INT64, &[2])],
    )
    .expect("Split-13 partition input");

    let divisible = node(
        "Split",
        &["data"],
        &["a", "b", "c"],
        vec![int_attr("axis", 1), int_attr("num_outputs", 3)],
    );
    validate(vec![divisible], 18, float_input(), vec![])
        .expect("Split-18 num_outputs is exact when the authenticated extent divides evenly");

    let uneven = node(
        "Split",
        &["data"],
        &["left", "right"],
        vec![int_attr("axis", 1), int_attr("num_outputs", 2)],
    );
    let error = validate(vec![uneven], 18, float_input(), vec![])
        .expect_err("Split-18 uneven num_outputs semantics are not represented");
    assert!(error.to_string().contains("num_outputs"), "{error}");
}

#[test]
fn expand_tile_flatten_and_transpose_authenticate_raw_schema() {
    let expand = node("Expand", &["data", "shape"], &["out"], vec![]);
    validate(
        vec![expand],
        13,
        float_input(),
        vec![tensor("shape", INT64, &[3])],
    )
    .expect("Expand-8+ shape vector");

    let tile = node("Tile", &["data", "repeats"], &["out"], vec![]);
    assert!(validate(
        vec![tile.clone()],
        5,
        float_input(),
        vec![tensor("repeats", INT64, &[3])],
    )
    .is_err());
    validate(
        vec![tile],
        6,
        float_input(),
        vec![tensor("repeats", INT64, &[3])],
    )
    .expect("Tile-6 repeats-vector schema");

    let flatten = node("Flatten", &["data"], &["out"], vec![int_attr("axis", 4)]);
    assert!(validate(vec![flatten], 13, float_input(), vec![]).is_err());

    let transpose = node(
        "Transpose",
        &["data"],
        &["out"],
        vec![ints_attr("perm", &[0, 0, 2])],
    );
    assert!(validate(vec![transpose], 13, float_input(), vec![]).is_err());
}

#[test]
fn sliced_shape_extent_drives_downstream_control_rank() {
    let shape = node(
        "Shape",
        &["shape_source"],
        &["repeats"],
        vec![int_attr("start", 1), int_attr("end", 3)],
    );
    let tile = node("Tile", &["tile_data", "repeats"], &["out"], vec![]);
    validate(
        vec![shape, tile],
        15,
        vec![
            info("shape_source", FLOAT, &[1, 3, 4]),
            info("tile_data", FLOAT, &[3, 4]),
        ],
        vec![],
    )
    .expect("Shape(start=1,end=3) produces a length-2 repeats vector, not the full rank-3 shape");

    let empty_shape = node(
        "Shape",
        &["shape_source"],
        &["empty_repeats"],
        vec![int_attr("start", 2), int_attr("end", 1)],
    );
    let scalar_tile = node(
        "Tile",
        &["scalar", "empty_repeats"],
        &["scalar_out"],
        vec![],
    );
    validate(
        vec![empty_shape, scalar_tile],
        15,
        vec![
            info("shape_source", FLOAT, &[1, 3, 4]),
            info("scalar", FLOAT, &[]),
        ],
        vec![],
    )
    .expect("Shape(start > end) has authenticated vector length zero");
}

#[test]
fn int64_structural_values_are_sealed_to_control_paths() {
    let shape = node("Shape", &["data"], &["shape_out"], vec![]);
    let identity = node("Identity", &["shape_out"], &["identity_out"], vec![]);
    let reshape = node("Reshape", &["payload", "identity_out"], &["out"], vec![]);
    validate(
        vec![shape.clone(), identity.clone(), reshape],
        15,
        vec![
            info("data", FLOAT, &[1, 2]),
            info("payload", FLOAT, &[1, 2]),
        ],
        vec![],
    )
    .expect("INT64 Shape/Identity chain terminates at Reshape's control input");

    let relu = node("Relu", &["identity_out"], &["out"], vec![]);
    let error = validate(
        vec![shape.clone(), identity, relu],
        15,
        vec![info("data", FLOAT, &[1, 2])],
        vec![],
    )
    .expect_err("INT64 shape values must not enter f32 activation propagation");
    assert!(error.to_string().contains("f32 runtime graph"), "{error}");

    let graph = GraphProto {
        node: vec![shape],
        input: vec![info("data", FLOAT, &[1, 2])],
        output: vec![info("shape_out", INT64, &[2])],
        ..Default::default()
    };
    let error = validate_structural_schemas_unfolded(&graph, &HashMap::from([(String::new(), 15)]))
        .expect_err("INT64 Shape output must not be exposed through ny's FLOAT32 verifier API");
    assert!(error.to_string().contains("graph output"), "{error}");

    let gather = node(
        "Gather",
        &["integer_data", "indices"],
        &["gathered"],
        vec![],
    );
    let relu = node("Relu", &["gathered"], &["relu_out"], vec![]);
    let error = validate(
        vec![gather.clone(), relu],
        15,
        vec![],
        vec![
            tensor("integer_data", INT64, &[2]),
            tensor("indices", INT64, &[1]),
        ],
    )
    .expect_err("INT64 initializer/Gather data must not evade the f32-runtime seal");
    assert!(error.to_string().contains("f32 runtime graph"), "{error}");

    let control = node("Reshape", &["payload", "gathered"], &["reshaped"], vec![]);
    validate(
        vec![gather, control],
        15,
        vec![info("payload", FLOAT, &[1, 2])],
        vec![
            tensor("integer_data", INT64, &[2]),
            tensor("indices", INT64, &[1]),
        ],
    )
    .expect("static INT64 Gather result may terminate at a structural control input");

    let shape = node("Shape", &["data"], &["shape_out"], vec![]);
    let gather = node("Gather", &["shape_out", "indices"], &["selected"], vec![]);
    let reshape = node("Reshape", &["payload", "selected"], &["out"], vec![]);
    validate(
        vec![shape, gather, reshape],
        15,
        vec![
            info("data", FLOAT, &[1, 2]),
            info("payload", FLOAT, &[1, 2]),
        ],
        vec![tensor("indices", INT64, &[1])],
    )
    .expect("Shape-to-Gather metadata chain remains an authenticated control path");
}
