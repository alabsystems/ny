// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::parse_onnx_bytes;
use super::super::prepare::prepare_graph;
use super::super::schema_preflight::validate_standard_schemas_unfolded;
use crate::loader::{
    BatchNormFoldingPolicy, CustomOpRegistry, ShapeInferBackend, ShapeInferencePolicy,
};
use crate::onnx_proto::{
    attribute_type, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto,
};
use crate::WeightStore;
use prost::Message;
use std::collections::HashMap;

const FLOAT: i32 = 1;

fn tensor_attr(name: &str, dims: Vec<i64>, values: Vec<f32>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::TENSOR,
        t: Some(TensorProto {
            dims,
            data_type: FLOAT,
            float_data: values,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn shape_initializer(name: &str, dtype: i32) -> TensorProto {
    let mut tensor = TensorProto {
        name: name.to_string(),
        dims: vec![1],
        data_type: dtype,
        ..Default::default()
    };
    match dtype {
        1 => tensor.float_data = vec![2.0],
        6 => tensor.int32_data = vec![2],
        7 => tensor.int64_data = vec![2],
        10 => tensor.raw_data = 0x4000_u16.to_le_bytes().to_vec(),
        _ => panic!("unsupported test dtype {dtype}"),
    }
    tensor
}

fn float_attr(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::FLOAT,
        f: Some(value),
        ..Default::default()
    }
}

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INT,
        i: Some(value),
        ..Default::default()
    }
}

fn ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INTS,
        ints: values,
        ..Default::default()
    }
}

fn string_attr(name: &str, value: &[u8]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::STRING,
        s: Some(value.to_vec()),
        ..Default::default()
    }
}

fn node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        attribute: attributes,
        ..Default::default()
    }
}

fn opsets(version: i64) -> HashMap<String, i64> {
    HashMap::from([(String::new(), version), ("ai.onnx".to_string(), version)])
}

#[test]
fn constant_payload_variants_are_opset_aware_before_folding() {
    let tensor_constant = GraphProto {
        node: vec![node(
            "tensor",
            "Constant",
            &[],
            &["x"],
            vec![tensor_attr("value", vec![], vec![1.0])],
        )],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&tensor_constant, &opsets(11))
        .expect("tensor-valued Constant predates opset 12");

    for (name, attribute) in [
        ("value_float", float_attr("value_float", 1.0)),
        (
            "value_floats",
            AttributeProto {
                name: "value_floats".to_string(),
                r#type: attribute_type::FLOATS,
                floats: vec![1.0],
                ..Default::default()
            },
        ),
    ] {
        let graph = GraphProto {
            node: vec![node("scalar", "Constant", &[], &["x"], vec![attribute])],
            ..Default::default()
        };
        let error = validate_standard_schemas_unfolded(&graph, &opsets(11))
            .expect_err("scalar/list Constant payloads were introduced in opset 12");
        assert!(
            error.to_string().contains(name) && error.to_string().contains("opset 12"),
            "{error}"
        );
        validate_standard_schemas_unfolded(&graph, &opsets(12))
            .expect("the same payload is valid at opset 12");
    }
}

#[test]
fn standard_attribute_storage_ambiguity_is_rejected_before_folding() {
    let graphs = [
        GraphProto {
            node: vec![node(
                "quantize",
                "QuantizeLinear",
                &["x", "scale"],
                &["y"],
                vec![AttributeProto {
                    name: "axis".to_string(),
                    i: Some(0),
                    floats: vec![1.0],
                    r#type: attribute_type::INT,
                    ..Default::default()
                }],
            )],
            ..Default::default()
        },
        GraphProto {
            node: vec![node(
                "constant",
                "Constant",
                &[],
                &["x"],
                vec![AttributeProto {
                    name: "value_float".to_string(),
                    f: Some(1.0),
                    ints: vec![2],
                    r#type: attribute_type::FLOAT,
                    ..Default::default()
                }],
            )],
            ..Default::default()
        },
    ];
    for graph in graphs {
        let error = validate_standard_schemas_unfolded(&graph, &opsets(23))
            .expect_err("inactive raw AttributeProto storage must not be discarded");
        assert!(error.to_string().contains("ambiguous"), "{error}");
    }
}

#[test]
fn duplicate_standard_attribute_names_are_rejected_globally() {
    let graph = GraphProto {
        node: vec![node(
            "activation",
            "LeakyRelu",
            &["x"],
            &["y"],
            vec![float_attr("alpha", 0.1), float_attr("alpha", 0.2)],
        )],
        ..Default::default()
    };

    let error = validate_standard_schemas_unfolded(&graph, &opsets(23))
        .expect_err("duplicate attributes on every standard operator must fail closed");
    assert!(
        error.to_string().contains("duplicate 'alpha' attributes"),
        "{error}"
    );
}

#[test]
fn custom_domain_attributes_obey_global_name_and_storage_invariants() {
    let mut duplicate = node(
        "custom",
        "VendorOp",
        &["x"],
        &["y"],
        vec![float_attr("alpha", 0.1), float_attr("alpha", 0.2)],
    );
    duplicate.domain = "vendor.example".to_string();
    let error = validate_standard_schemas_unfolded(
        &GraphProto {
            node: vec![duplicate],
            ..Default::default()
        },
        &opsets(23),
    )
    .expect_err("custom handlers must not receive duplicate attribute names");
    assert!(error.to_string().contains("duplicate 'alpha'"), "{error}");

    let mut ambiguous = node(
        "custom",
        "VendorOp",
        &["x"],
        &["y"],
        vec![AttributeProto {
            name: "axis".to_string(),
            r#type: attribute_type::INT,
            i: Some(0),
            f: Some(0.0),
            ..Default::default()
        }],
    );
    ambiguous.domain = "vendor.example".to_string();
    let error = validate_standard_schemas_unfolded(
        &GraphProto {
            node: vec![ambiguous],
            ..Default::default()
        },
        &opsets(23),
    )
    .expect_err("custom handlers must not receive ambiguous AttributeProto storage");
    assert!(error.to_string().contains("ambiguous"), "{error}");
}

#[test]
fn constant_of_shape_schema_rejects_malformed_signatures_and_values() {
    let valid = node(
        "fill",
        "ConstantOfShape",
        &["shape"],
        &["y"],
        vec![tensor_attr("value", vec![1], vec![2.0])],
    );
    validate_standard_schemas_unfolded(
        &GraphProto {
            node: vec![valid.clone()],
            initializer: vec![shape_initializer("shape", 7)],
            ..Default::default()
        },
        &opsets(9),
    )
    .expect("one scalar tensor value is canonical");
    let error = validate_standard_schemas_unfolded(
        &GraphProto {
            node: vec![valid],
            initializer: vec![shape_initializer("shape", 7)],
            ..Default::default()
        },
        &opsets(8),
    )
    .expect_err("ConstantOfShape was introduced in opset 9");
    assert!(error.to_string().contains("opset 9"), "{error}");

    let malformed = [
        node("missing_input", "ConstantOfShape", &[], &["y"], vec![]),
        node("empty_input", "ConstantOfShape", &[""], &["y"], vec![]),
        node(
            "extra_input",
            "ConstantOfShape",
            &["shape", "other"],
            &["y"],
            vec![],
        ),
        node(
            "extra_output",
            "ConstantOfShape",
            &["shape"],
            &["y", "z"],
            vec![],
        ),
        node(
            "wrong_attribute",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![float_attr("value", 1.0)],
        ),
        node(
            "unknown_attribute",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![tensor_attr("other", vec![1], vec![1.0])],
        ),
        node(
            "duplicate_attribute",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![
                tensor_attr("value", vec![1], vec![1.0]),
                tensor_attr("value", vec![1], vec![2.0]),
            ],
        ),
        node(
            "nonscalar_value",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![tensor_attr("value", vec![2], vec![1.0, 2.0])],
        ),
        node(
            "empty_value",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![tensor_attr("value", vec![0], vec![])],
        ),
        node(
            "negative_value_dimension",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![tensor_attr("value", vec![-1], vec![])],
        ),
    ];
    for malformed_node in malformed {
        let graph = GraphProto {
            node: vec![malformed_node.clone()],
            ..Default::default()
        };
        let error = validate_standard_schemas_unfolded(&graph, &opsets(17))
            .expect_err("malformed ConstantOfShape must fail before folding");
        assert!(
            error.to_string().contains("ConstantOfShape"),
            "{}: {error}",
            malformed_node.name
        );
    }
}

#[test]
fn constant_of_shape_requires_authenticated_int64_shape_input_before_folding() {
    for dtype in [1, 6] {
        let graph = GraphProto {
            node: vec![node("fill", "ConstantOfShape", &["shape"], &["y"], vec![])],
            initializer: vec![shape_initializer("shape", dtype)],
            ..Default::default()
        };
        let error = validate_standard_schemas_unfolded(&graph, &opsets(17))
            .expect_err("FLOAT/INT32 shape tensors must not fold as INT64 ConstantOfShape inputs");
        assert!(
            error.to_string().contains("requires INT64")
                && error.to_string().contains(&format!("dtype {dtype}")),
            "{error}"
        );
    }

    let direct = GraphProto {
        node: vec![node("fill", "ConstantOfShape", &["shape"], &["y"], vec![])],
        initializer: vec![shape_initializer("shape", 7)],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&direct, &opsets(17)).expect("direct INT64 shape is valid");

    let via_identity = GraphProto {
        node: vec![
            node("identity", "Identity", &["shape_raw"], &["shape"], vec![]),
            node("fill", "ConstantOfShape", &["shape"], &["y"], vec![]),
        ],
        initializer: vec![shape_initializer("shape_raw", 7)],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&via_identity, &opsets(17))
        .expect("an exact Identity cone preserves INT64 dtype");

    let via_cast = GraphProto {
        node: vec![
            node(
                "cast",
                "Cast",
                &["shape_i32"],
                &["shape"],
                vec![AttributeProto {
                    name: "to".to_string(),
                    r#type: attribute_type::INT,
                    i: Some(7),
                    ..Default::default()
                }],
            ),
            node("fill", "ConstantOfShape", &["shape"], &["y"], vec![]),
        ],
        initializer: vec![shape_initializer("shape_i32", 6)],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&via_cast, &opsets(17))
        .expect("an exact Cast-to-INT64 cone establishes the required dtype");
}

#[test]
fn arg_extrema_schemas_are_versioned_and_strict() {
    for op_type in ["ArgMax", "ArgMin"] {
        let valid = GraphProto {
            node: vec![node(op_type, op_type, &["x"], &["y"], vec![])],
            initializer: vec![shape_initializer("x", FLOAT)],
            ..Default::default()
        };
        validate_standard_schemas_unfolded(&valid, &opsets(1))
            .expect("axis=0, keepdims=1 defaults");

        let select_last = GraphProto {
            node: vec![node(
                op_type,
                op_type,
                &["x"],
                &["y"],
                vec![int_attr("select_last_index", 1)],
            )],
            initializer: vec![shape_initializer("x", FLOAT)],
            ..Default::default()
        };
        assert!(validate_standard_schemas_unfolded(&select_last, &opsets(11)).is_err());
        validate_standard_schemas_unfolded(&select_last, &opsets(12))
            .expect("select_last_index starts at opset 12");

        for malformed in [
            node(
                "bad_axis",
                op_type,
                &["x"],
                &["y"],
                vec![float_attr("axis", 0.0)],
            ),
            node(
                "unknown",
                op_type,
                &["x"],
                &["y"],
                vec![int_attr("bogus", 0)],
            ),
            node("extra_output", op_type, &["x"], &["y", "z"], vec![]),
        ] {
            assert!(validate_standard_schemas_unfolded(
                &GraphProto {
                    node: vec![malformed],
                    initializer: vec![shape_initializer("x", FLOAT)],
                    ..Default::default()
                },
                &opsets(13),
            )
            .is_err());
        }
    }
}

#[test]
fn topk_schemas_authenticate_versions_semantics_and_k_dtype() {
    let legacy = GraphProto {
        node: vec![node(
            "topk",
            "TopK",
            &["x"],
            &["values", "indices"],
            vec![int_attr("k", 2)],
        )],
        initializer: vec![shape_initializer("x", FLOAT)],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&legacy, &opsets(9)).expect("legacy k attribute");
    assert!(validate_standard_schemas_unfolded(&legacy, &opsets(10)).is_err());

    let modern = |dtype, attrs| GraphProto {
        node: vec![node(
            "topk",
            "TopK",
            &["x", "k"],
            &["values", "indices"],
            attrs,
        )],
        initializer: vec![shape_initializer("x", FLOAT), shape_initializer("k", dtype)],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&modern(7, vec![]), &opsets(10)).expect("modern INT64 K");
    assert!(validate_standard_schemas_unfolded(&modern(6, vec![]), &opsets(10)).is_err());
    for unsupported in [int_attr("largest", 0), int_attr("sorted", 0)] {
        assert!(
            validate_standard_schemas_unfolded(&modern(7, vec![unsupported]), &opsets(11)).is_err()
        );
    }
    assert!(validate_standard_schemas_unfolded(
        &modern(7, vec![int_attr("largest", 1)]),
        &opsets(10)
    )
    .is_err());
}

#[test]
fn dropout_is_erased_only_when_identity_is_authored_for_that_opset() {
    let dropout = |attrs: Vec<AttributeProto>, inputs: &[&str], outputs: &[&str]| GraphProto {
        node: vec![node("dropout", "Dropout", inputs, outputs, attrs)],
        initializer: vec![shape_initializer("x", FLOAT)],
        ..Default::default()
    };

    assert!(
        validate_standard_schemas_unfolded(&dropout(vec![], &["x"], &["y"]), &opsets(6)).is_err()
    );
    validate_standard_schemas_unfolded(
        &dropout(vec![int_attr("is_test", 1)], &["x"], &["y"]),
        &opsets(6),
    )
    .expect("legacy is_test=1 is identity");
    validate_standard_schemas_unfolded(
        &dropout(vec![float_attr("ratio", 0.0)], &["x"], &["y"]),
        &opsets(10),
    )
    .expect("ratio=0 is identity in every ambient mode");
    // Opsets 7..=11 removed `is_test` and had not yet gained `training_mode`,
    // so no authored control can select training. ONNX's own version adapter
    // upgrades such a node to Dropout-12 with `training_mode` ABSENT, i.e. at
    // its schema default `false`, under which the spec says "ratio is ignored
    // and the operation mimics inference mode where nothing will be dropped
    // from the input data". The identity is therefore the only reachable
    // meaning, for every ratio (vggnet16_2022's `vgg16-7.onnx`).
    validate_standard_schemas_unfolded(&dropout(vec![], &["x"], &["y"]), &opsets(10))
        .expect("opset 7..=11 has no authored training-mode control");
    validate_standard_schemas_unfolded(
        &dropout(vec![float_attr("ratio", 0.5)], &["x"], &["y"]),
        &opsets(8),
    )
    .expect("an authored non-zero ratio is still the inference identity at opset 8");

    validate_standard_schemas_unfolded(&dropout(vec![], &["x", "ratio"], &["y"]), &opsets(13))
        .expect("modern omitted training_mode defaults to inference");
    assert!(validate_standard_schemas_unfolded(
        &dropout(vec![], &["x", "", "training"], &["y"]),
        &opsets(13)
    )
    .is_err());
    assert!(validate_standard_schemas_unfolded(
        &dropout(vec![], &["x"], &["y", "mask"]),
        &opsets(13)
    )
    .is_err());
}

#[test]
fn identity_gather_and_nonzero_fail_closed_at_the_raw_schema_boundary() {
    let bad_identity = GraphProto {
        node: vec![node(
            "identity",
            "Identity",
            &["x"],
            &["y"],
            vec![int_attr("axis", 0)],
        )],
        ..Default::default()
    };
    assert!(validate_standard_schemas_unfolded(&bad_identity, &opsets(13)).is_err());

    let gather = |dtype, attrs| GraphProto {
        node: vec![node(
            "gather",
            "Gather",
            &["data", "indices"],
            &["y"],
            attrs,
        )],
        initializer: vec![
            shape_initializer("data", FLOAT),
            shape_initializer("indices", dtype),
        ],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&gather(6, vec![]), &opsets(13)).expect("INT32 indices");
    validate_standard_schemas_unfolded(&gather(7, vec![int_attr("axis", 0)]), &opsets(13))
        .expect("INT64 indices");
    assert!(validate_standard_schemas_unfolded(&gather(1, vec![]), &opsets(13)).is_err());
    assert!(validate_standard_schemas_unfolded(
        &gather(7, vec![float_attr("axis", 0.0)]),
        &opsets(13)
    )
    .is_err());

    let integer_runtime_gather = GraphProto {
        node: vec![node(
            "gather",
            "Gather",
            &["data", "indices"],
            &["y"],
            vec![],
        )],
        initializer: vec![
            shape_initializer("data", 6),
            shape_initializer("indices", 7),
        ],
        ..Default::default()
    };
    assert!(validate_standard_schemas_unfolded(&integer_runtime_gather, &opsets(13)).is_err());

    let argmax_gather = GraphProto {
        node: vec![
            node("argmax", "ArgMax", &["scores"], &["indices"], vec![]),
            node("gather", "Gather", &["data", "indices"], &["y"], vec![]),
        ],
        initializer: vec![
            shape_initializer("scores", FLOAT),
            shape_initializer("data", FLOAT),
        ],
        ..Default::default()
    };
    validate_standard_schemas_unfolded(&argmax_gather, &opsets(13))
        .expect("ArgMax output is schema-authenticated INT64");

    let nonzero = GraphProto {
        node: vec![node("nonzero", "NonZero", &["x"], &["y"], vec![])],
        ..Default::default()
    };
    assert!(validate_standard_schemas_unfolded(&nonzero, &opsets(8)).is_err());
    let error = validate_standard_schemas_unfolded(&nonzero, &opsets(9))
        .expect_err("variable output shapes are unsupported");
    assert!(
        error.to_string().contains("data-dependent output shape"),
        "{error}"
    );
}

#[test]
fn pooling_schemas_preserve_defaults_and_reject_unrepresented_semantics() {
    let pool = |op_type: &str, attrs: Vec<AttributeProto>, outputs: &[&str]| GraphProto {
        node: vec![node("pool", op_type, &["x"], outputs, attrs)],
        initializer: vec![shape_initializer("x", FLOAT)],
        ..Default::default()
    };
    let kernel = || ints_attr("kernel_shape", vec![2, 3]);

    validate_standard_schemas_unfolded(&pool("AveragePool", vec![kernel()], &["y"]), &opsets(1))
        .expect("opset-1 AveragePool defaults strides to [1,1]");
    validate_standard_schemas_unfolded(&pool("MaxPool", vec![kernel()], &["y"]), &opsets(1))
        .expect("opset-1 MaxPool defaults strides to [1,1]");
    validate_standard_schemas_unfolded(&pool("GlobalAveragePool", vec![], &["y"]), &opsets(1))
        .expect("GlobalAveragePool has no attributes");

    for malformed in [
        pool("AveragePool", vec![], &["y"]),
        pool(
            "AveragePool",
            vec![kernel(), ints_attr("strides", vec![2])],
            &["y"],
        ),
        pool(
            "AveragePool",
            vec![kernel(), ints_attr("pads", vec![1, 0, 0, 0])],
            &["y"],
        ),
        pool(
            "AveragePool",
            vec![kernel(), int_attr("count_include_pad", 2)],
            &["y"],
        ),
        pool(
            "AveragePool",
            vec![kernel(), ints_attr("dilations", vec![2, 1])],
            &["y"],
        ),
        pool(
            "MaxPool",
            vec![kernel(), string_attr("auto_pad", b"SAME_UPPER")],
            &["y"],
        ),
        pool("MaxPool", vec![kernel()], &["values", "indices"]),
        pool(
            "GlobalAveragePool",
            vec![ints_attr("kernel_shape", vec![2, 2])],
            &["y"],
        ),
    ] {
        assert!(
            validate_standard_schemas_unfolded(&malformed, &opsets(19)).is_err(),
            "unsupported pooling form should fail closed: {:?}",
            malformed.node[0]
        );
    }

    assert!(validate_standard_schemas_unfolded(
        &pool(
            "AveragePool",
            vec![kernel(), int_attr("count_include_pad", 1)],
            &["y"],
        ),
        &opsets(6),
    )
    .is_err());
    validate_standard_schemas_unfolded(
        &pool(
            "AveragePool",
            vec![
                kernel(),
                ints_attr("strides", vec![1, 1]),
                ints_attr("pads", vec![1, 0, 1, 0]),
                int_attr("count_include_pad", 1),
                int_attr("ceil_mode", 0),
                ints_attr("dilations", vec![1, 1]),
                string_attr("auto_pad", b"NOTSET"),
            ],
            &["y"],
        ),
        &opsets(19),
    )
    .expect("the represented 2D AveragePool subset should pass");

    let non_float = GraphProto {
        node: vec![node("pool", "MaxPool", &["x"], &["y"], vec![kernel()])],
        initializer: vec![shape_initializer("x", 10)],
        ..Default::default()
    };
    let error = validate_standard_schemas_unfolded(&non_float, &opsets(19))
        .expect_err("FLOAT16 pooling is outside ny's f32 arithmetic model");
    assert!(error.to_string().contains("tensor(float)"), "{error}");
}

#[test]
fn standard_attention_is_not_silently_remapped_to_simplified_self_attention() {
    let attention = |op_type: &str| GraphProto {
        node: vec![node("attention", op_type, &["q", "k", "v"], &["y"], vec![])],
        ..Default::default()
    };

    let before_introduction =
        validate_standard_schemas_unfolded(&attention("Attention"), &opsets(22))
            .expect_err("Attention was introduced in opset 23");
    assert!(
        before_introduction.to_string().contains("opset 23"),
        "{before_introduction}"
    );

    for version in [23, 24] {
        let error = validate_standard_schemas_unfolded(&attention("Attention"), &opsets(version))
            .expect_err("the standard Attention function is not the simplified ternary layer");
        assert!(
            error.to_string().contains("simplified SelfAttention")
                && error.to_string().contains(&format!("Attention-{version}")),
            "{error}"
        );
    }

    let alias = validate_standard_schemas_unfolded(&attention("MultiHeadAttention"), &opsets(24))
        .expect_err("MultiHeadAttention is not a standard main-domain operator");
    assert!(
        alias
            .to_string()
            .contains("not a registered main-domain operator"),
        "{alias}"
    );
}

#[test]
fn vendor_aliases_are_not_accepted_in_the_standard_domain() {
    for op_type in [
        "ArgSort",
        "Snake",
        "RoPE",
        "RotaryPositionEmbedding",
        "SimplifiedLayerNormalization",
        "AdaIN",
        "AdaptiveInstanceNorm",
        "AdaptiveInstanceNormalization",
    ] {
        let graph = GraphProto {
            node: vec![node("vendor_alias", op_type, &["x"], &["y"], vec![])],
            ..Default::default()
        };
        let error = validate_standard_schemas_unfolded(&graph, &opsets(24))
            .expect_err("vendor alias in the standard domain must fail closed");
        assert!(
            error
                .to_string()
                .contains("not a registered main-domain operator"),
            "{op_type}: {error}"
        );
    }
}

#[test]
fn direct_normalization_rejects_non_float_data_before_weight_normalization() {
    let graph = GraphProto {
        node: vec![node(
            "batch_norm",
            "BatchNormalization",
            &["x", "scale", "bias", "mean", "variance"],
            &["y"],
            vec![],
        )],
        initializer: vec![
            shape_initializer("x", 7),
            shape_initializer("scale", FLOAT),
            shape_initializer("bias", FLOAT),
            shape_initializer("mean", FLOAT),
            shape_initializer("variance", FLOAT),
        ],
        ..Default::default()
    };
    let error = validate_standard_schemas_unfolded(&graph, &opsets(14))
        .expect_err("integer BatchNormalization data must not become f32 arithmetic");
    assert!(
        (error.to_string().contains("tensor(float)") && error.to_string().contains("input 0"))
            || error.to_string().contains("f32 runtime graph"),
        "{error}"
    );
}

#[test]
fn prepare_rejects_malformed_constant_of_shape_before_it_can_fold() {
    let mut graph = GraphProto {
        node: vec![node(
            "bad_fill",
            "ConstantOfShape",
            &["shape"],
            &["y"],
            vec![tensor_attr("value", vec![2], vec![1.0, 2.0])],
        )],
        initializer: vec![TensorProto {
            name: "shape".to_string(),
            dims: vec![1],
            data_type: 7,
            int64_data: vec![2],
            ..Default::default()
        }],
        ..Default::default()
    };
    let error = prepare_graph(
        &mut graph,
        &mut WeightStore::new(),
        &mut HashMap::new(),
        false,
        None,
    )
    .expect_err("prepare must reject malformed ConstantOfShape before folding");
    assert!(
        error.to_string().contains("exactly one scalar element"),
        "{error}"
    );
}

#[test]
fn invalid_constant_producer_into_dequantize_cannot_disappear() {
    let graph = GraphProto {
        node: vec![
            node(
                "invalid_integer_constant",
                "Constant",
                &[],
                &["q"],
                vec![AttributeProto {
                    name: "value_int".to_string(),
                    r#type: attribute_type::INT,
                    i: Some(1),
                    ..Default::default()
                }],
            ),
            node(
                "dequantize",
                "DequantizeLinear",
                &["q", "scale"],
                &["y"],
                vec![],
            ),
        ],
        initializer: vec![TensorProto {
            name: "scale".to_string(),
            data_type: FLOAT,
            float_data: vec![1.0],
            ..Default::default()
        }],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 11,
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
    .expect_err("invalid Constant must fail before Q/DQ inspection or folding");
    assert!(
        error.to_string().contains("value_int") && error.to_string().contains("opset 12"),
        "{error}"
    );
}
