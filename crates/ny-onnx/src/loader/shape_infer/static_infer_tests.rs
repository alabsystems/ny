// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fixture-free rule tests for the ORT-free static shape-inference fallback.
//!
//! Every graph is built in Rust. Each rule is pinned on its exact output, and
//! every refusal case is pinned as UNKNOWN (absent from the table and listed
//! as unresolved) — the fallback's contract is that it never asserts a shape
//! it cannot derive exactly.

use super::static_infer::{infer_tensor_shapes_static, StaticShapeTable};
use crate::onnx_proto::{
    attribute_type, tensor_shape_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;

/// Marks a symbolic (`dim_param`) dimension in [`G::input`].
const SYM: i64 = -1;

const DT_FLOAT: i32 = 1;
const DT_INT64: i32 = 7;

fn value_info(name: &str, dims: Option<&[i64]>, elem_type: i32) -> ValueInfoProto {
    let shape = dims.map(|dims| TensorShapeProto {
        dim: dims
            .iter()
            .map(|&dim| tensor_shape_proto::Dimension {
                value: Some(if dim == SYM {
                    tensor_shape_proto::dimension::Value::DimParam("batch".to_string())
                } else {
                    tensor_shape_proto::dimension::Value::DimValue(dim)
                }),
            })
            .collect(),
    });
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto { elem_type, shape }),
        }),
    }
}

fn element_count(dims: &[i64]) -> usize {
    dims.iter()
        .map(|&dim| usize::try_from(dim).expect("test dims are non-negative"))
        .product()
}

fn f32_tensor(name: &str, dims: &[i64]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: DT_FLOAT,
        raw_data: vec![0u8; element_count(dims) * 4],
        ..Default::default()
    }
}

fn i64_tensor(name: &str, dims: &[i64], values: &[i64]) -> TensorProto {
    assert_eq!(
        element_count(dims),
        values.len(),
        "test tensor is consistent"
    );
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: DT_INT64,
        raw_data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        ..Default::default()
    }
}

fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INT,
        i: Some(value),
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::INTS,
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::FLOAT,
        f: Some(value),
        ..Default::default()
    }
}

fn attr_floats(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::FLOATS,
        floats: values.to_vec(),
        ..Default::default()
    }
}

fn attr_str(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::STRING,
        s: Some(value.as_bytes().to_vec()),
        ..Default::default()
    }
}

fn attr_tensor(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_type::TENSOR,
        t: Some(tensor),
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|name| (*name).to_string()).collect(),
        output: outputs.iter().map(|name| (*name).to_string()).collect(),
        name: format!("{op}_{}", outputs.first().copied().unwrap_or("")),
        op_type: op.to_string(),
        attribute: attrs,
        ..Default::default()
    }
}

/// An INT64 `Constant` node.
fn int_const(output: &str, dims: &[i64], values: &[i64]) -> NodeProto {
    node(
        "Constant",
        &[],
        &[output],
        vec![attr_tensor("value", i64_tensor("", dims, values))],
    )
}

/// Minimal graph builder.
#[derive(Default)]
struct G {
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    inits: Vec<TensorProto>,
    nodes: Vec<NodeProto>,
}

impl G {
    fn input(mut self, name: &str, dims: &[i64]) -> Self {
        self.inputs.push(value_info(name, Some(dims), DT_FLOAT));
        self
    }

    fn int_input(mut self, name: &str, dims: &[i64]) -> Self {
        self.inputs.push(value_info(name, Some(dims), DT_INT64));
        self
    }

    fn output(mut self, name: &str) -> Self {
        self.outputs.push(value_info(name, None, DT_FLOAT));
        self
    }

    fn declared_output(mut self, name: &str, dims: &[i64]) -> Self {
        self.outputs.push(value_info(name, Some(dims), DT_FLOAT));
        self
    }

    fn weight(mut self, name: &str, dims: &[i64]) -> Self {
        self.inits.push(f32_tensor(name, dims));
        self
    }

    fn int_weight(mut self, name: &str, dims: &[i64], values: &[i64]) -> Self {
        self.inits.push(i64_tensor(name, dims, values));
        self
    }

    fn raw_init(mut self, tensor: TensorProto) -> Self {
        self.inits.push(tensor);
        self
    }

    fn node(mut self, node: NodeProto) -> Self {
        self.nodes.push(node);
        self
    }

    fn model(&self, opset: i64) -> ModelProto {
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: opset,
            }],
            graph: Some(GraphProto {
                name: "static_infer_test".to_string(),
                node: self.nodes.clone(),
                initializer: self.inits.clone(),
                input: self.inputs.clone(),
                output: self.outputs.clone(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn bytes(&self, opset: i64) -> Vec<u8> {
        encode(&self.model(opset))
    }

    fn infer(&self, opset: i64) -> StaticShapeTable {
        infer_tensor_shapes_static(&self.bytes(opset))
    }
}

fn encode(model: &ModelProto) -> Vec<u8> {
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode test model");
    bytes
}

#[track_caller]
fn has(table: &StaticShapeTable, name: &str, dims: &[i64]) {
    assert_eq!(
        table.shapes.get(name).map(Vec::as_slice),
        Some(dims),
        "shape of '{name}'; table = {:?}, unresolved = {:?}, refusal = {:?}",
        table.shapes,
        table.unresolved,
        table.refusal
    );
}

#[track_caller]
fn unknown(table: &StaticShapeTable, name: &str) {
    assert!(
        !table.shapes.contains_key(name),
        "'{name}' must stay unknown, got {:?}",
        table.shapes.get(name)
    );
    assert!(
        table.unresolved.iter().any(|entry| entry == name),
        "'{name}' must be reported unresolved; unresolved = {:?}",
        table.unresolved
    );
}

// ---------------------------------------------------------------------------
// Elementwise and broadcasting.
// ---------------------------------------------------------------------------

#[test]
fn unary_elementwise_ops_preserve_input_shape() {
    let g = G::default()
        .input("x", &[1, 3, 4])
        .node(node("Relu", &["x"], &["r"], vec![]))
        .node(node("Sigmoid", &["r"], &["s"], vec![]))
        .node(node("Tanh", &["s"], &["t"], vec![]))
        .node(node("Neg", &["t"], &["n"], vec![]))
        .node(node(
            "LeakyRelu",
            &["n"],
            &["l"],
            vec![attr_float("alpha", 0.1)],
        ))
        .node(node("Softmax", &["l"], &["sm"], vec![attr_int("axis", -1)]))
        .node(node("Clip", &["sm", "", ""], &["c"], vec![]))
        .output("c");
    let table = g.infer(13);
    for name in ["x", "r", "s", "t", "n", "l", "sm", "c"] {
        has(&table, name, &[1, 3, 4]);
    }
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
    assert!(table.refusal.is_none());
}

#[test]
fn binary_ops_broadcast_multidirectionally() {
    let g = G::default()
        .input("a", &[2, 1, 4])
        .input("s", &[])
        .input("v", &[5])
        .input("m", &[2, 5])
        .weight("b", &[3, 1])
        .node(node("Add", &["a", "b"], &["ab"], vec![]))
        .node(node("Sub", &["ab", "s"], &["abs"], vec![]))
        .node(node("Mul", &["v", "m"], &["vm"], vec![]))
        .node(node("Div", &["m", "v"], &["mv"], vec![]))
        .node(node("Pow", &["ab", "b"], &["p"], vec![]))
        .node(node("Greater", &["ab", "b"], &["g"], vec![]))
        .output("abs")
        .output("vm")
        .output("mv")
        .output("p")
        .output("g");
    let table = g.infer(13);
    has(&table, "s", &[]);
    has(&table, "ab", &[2, 3, 4]);
    has(&table, "abs", &[2, 3, 4]);
    has(&table, "vm", &[2, 5]);
    has(&table, "mv", &[2, 5]);
    has(&table, "p", &[2, 3, 4]);
    has(&table, "g", &[2, 3, 4]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

#[test]
fn incompatible_broadcast_stays_unknown_and_poisons_consumers() {
    let g = G::default()
        .input("a", &[2, 3])
        .input("b", &[4])
        .node(node("Add", &["a", "b"], &["bad"], vec![]))
        .node(node("Relu", &["bad"], &["after"], vec![]))
        .output("after");
    let table = g.infer(13);
    unknown(&table, "bad");
    unknown(&table, "after");
    has(&table, "a", &[2, 3]);
}

#[test]
fn legacy_broadcast_attribute_arithmetic_is_refused() {
    // Before opset 7 Add broadcast only through `broadcast`/`axis`
    // attributes; the fallback does not model that legacy rule.
    let g = G::default()
        .input("a", &[1, 3])
        .input("b", &[1, 3])
        .node(node("Add", &["a", "b"], &["sum"], vec![]))
        .node(node("Relu", &["a"], &["r"], vec![]))
        .output("sum")
        .output("r");
    let table = g.infer(6);
    unknown(&table, "sum");
    has(&table, "r", &[1, 3]);
}

#[test]
fn variadic_ops_and_where_broadcast() {
    let g = G::default()
        .input("a", &[2, 1])
        .input("b", &[1, 3])
        .input("c", &[3])
        .node(node("Max", &["a", "b", "c"], &["mx"], vec![]))
        .node(node("Sum", &["a"], &["one"], vec![]))
        .node(node("Greater", &["a", "b"], &["cond"], vec![]))
        .node(node("Where", &["cond", "a", "c"], &["w"], vec![]))
        .output("mx")
        .output("one")
        .output("w");
    let table = g.infer(13);
    has(&table, "mx", &[2, 3]);
    has(&table, "one", &[2, 1]);
    has(&table, "cond", &[2, 3]);
    has(&table, "w", &[2, 3]);
    // Variadic broadcasting only exists from opset 8.
    let legacy = g.infer(7);
    unknown(&legacy, "mx");
}

// ---------------------------------------------------------------------------
// MatMul / Gemm / Conv.
// ---------------------------------------------------------------------------

#[test]
fn matmul_follows_numpy_rules_including_vectors_and_batch_broadcast() {
    let g = G::default()
        .input("x", &[1, 2])
        .input("v6", &[6])
        .input("m23", &[2, 3])
        .input("v3", &[3])
        .input("batch_a", &[2, 1, 3, 4])
        .input("batch_b", &[5, 4, 6])
        .weight("w68", &[6, 8])
        .weight("w21", &[2, 1])
        .node(node("MatMul", &["x", "w21"], &["col"], vec![]))
        .node(node("MatMul", &["v6", "w68"], &["vec_left"], vec![]))
        .node(node("MatMul", &["m23", "v3"], &["vec_right"], vec![]))
        .node(node("MatMul", &["v3", "v3"], &["dot"], vec![]))
        .node(node(
            "MatMul",
            &["batch_a", "batch_b"],
            &["batched"],
            vec![],
        ))
        .output("col")
        .output("vec_left")
        .output("vec_right")
        .output("dot")
        .output("batched");
    let table = g.infer(13);
    has(&table, "col", &[1, 1]);
    has(&table, "vec_left", &[8]);
    has(&table, "vec_right", &[2]);
    has(&table, "dot", &[]);
    has(&table, "batched", &[2, 5, 3, 6]);
}

#[test]
fn matmul_inner_dimension_mismatch_stays_unknown() {
    let g = G::default()
        .input("a", &[2, 3])
        .weight("w", &[4, 5])
        .input("p", &[2, 2, 3])
        .input("q", &[3, 3, 4])
        .node(node("MatMul", &["a", "w"], &["bad"], vec![]))
        .node(node("MatMul", &["p", "q"], &["bad_batch"], vec![]))
        .output("bad")
        .output("bad_batch");
    let table = g.infer(13);
    unknown(&table, "bad");
    unknown(&table, "bad_batch");
}

#[test]
fn gemm_honours_transposes_and_bias_broadcast() {
    let g = G::default()
        .input("x", &[1, 4])
        .input("xt", &[4, 1])
        .weight("w_t", &[8, 4])
        .weight("w", &[4, 8])
        .weight("bias", &[8])
        .weight("bias_row", &[1, 8])
        .node(node(
            "Gemm",
            &["x", "w_t", "bias"],
            &["y_tb"],
            vec![attr_int("transB", 1), attr_float("alpha", 1.0)],
        ))
        .node(node(
            "Gemm",
            &["xt", "w", "bias_row"],
            &["y_ta"],
            vec![attr_int("transA", 1)],
        ))
        .node(node("Gemm", &["x", "w"], &["y_no_c"], vec![]))
        .output("y_tb")
        .output("y_ta")
        .output("y_no_c");
    let table = g.infer(13);
    has(&table, "y_tb", &[1, 8]);
    has(&table, "y_ta", &[1, 8]);
    has(&table, "y_no_c", &[1, 8]);
    // C was mandatory before opset 11.
    unknown(&g.infer(9), "y_no_c");
}

#[test]
fn gemm_rejects_non_matrix_operands_and_bad_bias() {
    let g = G::default()
        .input("x3", &[1, 1, 4])
        .input("x", &[1, 4])
        .weight("w", &[4, 8])
        .weight("w_bad_k", &[5, 8])
        .weight("bad_bias", &[3])
        .node(node("Gemm", &["x3", "w"], &["rank3"], vec![]))
        .node(node("Gemm", &["x", "w", "bad_bias"], &["bias"], vec![]))
        .node(node("Gemm", &["x", "w_bad_k"], &["k"], vec![]))
        .node(node(
            "Gemm",
            &["x", "w"],
            &["legacy_attr"],
            vec![attr_int("broadcast", 1)],
        ))
        .output("rank3")
        .output("bias")
        .output("k")
        .output("legacy_attr");
    let table = g.infer(13);
    unknown(&table, "rank3");
    unknown(&table, "bias");
    unknown(&table, "k");
    // `broadcast` is not a Gemm attribute from opset 7.
    unknown(&table, "legacy_attr");
}

#[test]
fn conv_explicit_pads_strides_dilations_and_groups() {
    let g = G::default()
        .input("x", &[1, 3, 10, 12])
        .input("xg", &[1, 6, 5, 5])
        .input("image", &[SYM, 1, 20, 20])
        .weight("w", &[8, 3, 3, 5])
        .weight("b", &[8])
        .weight("wg", &[9, 2, 3, 3])
        .weight("w_col", &[5, 1, 5, 1])
        .node(node(
            "Conv",
            &["x", "w", "b"],
            &["y"],
            vec![
                attr_ints("pads", &[1, 2, 1, 2]),
                attr_ints("strides", &[2, 1]),
                attr_ints("dilations", &[1, 2]),
                attr_ints("kernel_shape", &[3, 5]),
            ],
        ))
        .node(node(
            "Conv",
            &["xg", "wg"],
            &["yg"],
            vec![attr_int("group", 3)],
        ))
        .node(node(
            "Conv",
            &["image", "w_col"],
            &["collins"],
            vec![
                attr_int("group", 1),
                attr_ints("dilations", &[1, 1]),
                attr_ints("kernel_shape", &[5, 1]),
                attr_ints("pads", &[0, 0, 0, 0]),
                attr_ints("strides", &[1, 1]),
            ],
        ))
        .output("y")
        .output("yg")
        .output("collins");
    let table = g.infer(11);
    // H: (10 + 1 + 1 - 3) / 2 + 1 = 5; W: (12 + 2 + 2 - 9) / 1 + 1 = 8.
    has(&table, "y", &[1, 8, 5, 8]);
    has(&table, "yg", &[1, 9, 3, 3]);
    has(&table, "image", &[1, 1, 20, 20]);
    has(&table, "collins", &[1, 5, 16, 20]);
}

#[test]
fn conv_auto_pad_modes() {
    let base = || {
        G::default()
            .input("x", &[1, 1, 7, 7])
            .input("x_even", &[1, 1, 8, 6])
            .weight("w", &[2, 1, 3, 3])
    };
    let strides = || attr_ints("strides", &[2, 2]);
    let g = base()
        .node(node(
            "Conv",
            &["x", "w"],
            &["same_upper"],
            vec![attr_str("auto_pad", "SAME_UPPER"), strides()],
        ))
        .node(node(
            "Conv",
            &["x", "w"],
            &["same_lower"],
            vec![attr_str("auto_pad", "SAME_LOWER"), strides()],
        ))
        .node(node(
            "Conv",
            &["x_even", "w"],
            &["same_even"],
            vec![
                attr_str("auto_pad", "SAME_UPPER"),
                attr_ints("strides", &[2, 3]),
            ],
        ))
        .node(node(
            "Conv",
            &["x", "w"],
            &["valid"],
            vec![attr_str("auto_pad", "VALID"), strides()],
        ))
        .node(node(
            "Conv",
            &["x", "w"],
            &["notset"],
            vec![attr_str("auto_pad", "NOTSET"), strides()],
        ))
        .node(node(
            "Conv",
            &["x", "w"],
            &["valid_with_pads"],
            vec![
                attr_str("auto_pad", "VALID"),
                attr_ints("pads", &[1, 1, 1, 1]),
            ],
        ))
        .node(node(
            "Conv",
            &["x", "w"],
            &["bogus_mode"],
            vec![attr_str("auto_pad", "FOO")],
        ))
        .output("same_upper")
        .output("same_lower")
        .output("same_even")
        .output("valid")
        .output("notset")
        .output("valid_with_pads")
        .output("bogus_mode");
    let table = g.infer(13);
    has(&table, "same_upper", &[1, 2, 4, 4]);
    has(&table, "same_lower", &[1, 2, 4, 4]);
    // ceil(8 / 2) = 4 and ceil(6 / 3) = 2: exact multiples must not round up.
    has(&table, "same_even", &[1, 2, 4, 2]);
    has(&table, "valid", &[1, 2, 3, 3]);
    has(&table, "notset", &[1, 2, 3, 3]);
    unknown(&table, "valid_with_pads");
    unknown(&table, "bogus_mode");
}

#[test]
fn conv_inconsistent_weights_stay_unknown() {
    let g = G::default()
        .input("x", &[1, 4, 5, 5])
        .input("tiny", &[1, 1, 2, 2])
        .weight("w_bad_c", &[2, 3, 3, 3])
        .weight("w", &[2, 4, 3, 3])
        .weight("bad_bias", &[3])
        .weight("w_big", &[1, 1, 3, 3])
        .node(node("Conv", &["x", "w_bad_c"], &["channels"], vec![]))
        .node(node(
            "Conv",
            &["x", "w"],
            &["kernel_attr"],
            vec![attr_ints("kernel_shape", &[2, 2])],
        ))
        .node(node("Conv", &["x", "w", "bad_bias"], &["bias"], vec![]))
        .node(node("Conv", &["tiny", "w_big"], &["too_small"], vec![]))
        .output("channels")
        .output("kernel_attr")
        .output("bias")
        .output("too_small");
    let table = g.infer(13);
    unknown(&table, "channels");
    unknown(&table, "kernel_attr");
    unknown(&table, "bias");
    unknown(&table, "too_small");
}

// ---------------------------------------------------------------------------
// Structural ops with constant parameters.
// ---------------------------------------------------------------------------

#[test]
fn flatten_axis_rules_including_negative_axis() {
    let g = G::default()
        .input("x", &[2, 3, 4, 5])
        .node(node("Flatten", &["x"], &["default"], vec![]))
        .node(node(
            "Flatten",
            &["x"],
            &["zero"],
            vec![attr_int("axis", 0)],
        ))
        .node(node(
            "Flatten",
            &["x"],
            &["rank"],
            vec![attr_int("axis", 4)],
        ))
        .node(node(
            "Flatten",
            &["x"],
            &["negative"],
            vec![attr_int("axis", -1)],
        ))
        .node(node(
            "Flatten",
            &["x"],
            &["too_far"],
            vec![attr_int("axis", 5)],
        ))
        .output("default")
        .output("zero")
        .output("rank")
        .output("negative")
        .output("too_far");
    let table = g.infer(13);
    has(&table, "default", &[2, 60]);
    has(&table, "zero", &[1, 120]);
    has(&table, "rank", &[120, 1]);
    has(&table, "negative", &[24, 5]);
    unknown(&table, "too_far");
    // Negative Flatten axes are only defined from opset 11.
    let legacy = g.infer(9);
    has(&legacy, "default", &[2, 60]);
    unknown(&legacy, "negative");
}

#[test]
fn concat_positive_and_negative_axis() {
    let g = G::default()
        .input("a", &[1, 2])
        .input("b", &[1, 3])
        .input("c", &[2, 3])
        .node(node(
            "Concat",
            &["a", "b"],
            &["pos"],
            vec![attr_int("axis", 1)],
        ))
        .node(node(
            "Concat",
            &["a", "b"],
            &["neg"],
            vec![attr_int("axis", -1)],
        ))
        .node(node(
            "Concat",
            &["c", "c"],
            &["rows"],
            vec![attr_int("axis", 0)],
        ))
        .output("pos")
        .output("neg")
        .output("rows");
    let table = g.infer(13);
    has(&table, "pos", &[1, 5]);
    has(&table, "neg", &[1, 5]);
    has(&table, "rows", &[4, 3]);
    unknown(&g.infer(9), "neg");
}

#[test]
fn concat_mismatch_or_unknown_input_stays_unknown() {
    let g = G::default()
        .input("a", &[1, 2])
        .input("c", &[2, 3])
        .input("sym", &[1, SYM])
        .node(node(
            "Concat",
            &["a", "c"],
            &["mismatch"],
            vec![attr_int("axis", 1)],
        ))
        .node(node(
            "Concat",
            &["a", "sym"],
            &["partial"],
            vec![attr_int("axis", 1)],
        ))
        .node(node("Concat", &["a", "a"], &["no_axis"], vec![]))
        .output("mismatch")
        .output("partial")
        .output("no_axis");
    let table = g.infer(13);
    unknown(&table, "mismatch");
    unknown(&table, "sym");
    unknown(&table, "partial");
    unknown(&table, "no_axis");
}

#[test]
fn slice_constant_parameters_with_int64_max_sentinel_and_negative_axes() {
    let g = G::default()
        .input("x", &[1, 6])
        .input("m", &[4, 5])
        .node(int_const("zero", &[1], &[0]))
        .node(int_const("one", &[1], &[1]))
        .node(int_const("three", &[1], &[3]))
        .node(int_const("max", &[1], &[i64::MAX]))
        .node(int_const("minus_two", &[1], &[-2]))
        .node(int_const("minus_one", &[1], &[-1]))
        .node(int_const("starts2", &[2], &[1, 0]))
        .node(int_const("ends2", &[2], &[3, 100]))
        .node(node(
            "Slice",
            &["x", "zero", "three", "one", "one"],
            &["head"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["x", "three", "max", "one", "one"],
            &["tail"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["x", "minus_two", "max", "minus_one"],
            &["last_two"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["m", "starts2", "ends2"],
            &["no_axes"],
            vec![],
        ))
        .output("head")
        .output("tail")
        .output("last_two")
        .output("no_axes");
    let table = g.infer(13);
    has(&table, "max", &[1]);
    has(&table, "head", &[1, 3]);
    has(&table, "tail", &[1, 3]);
    has(&table, "last_two", &[1, 2]);
    has(&table, "no_axes", &[2, 5]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

#[test]
fn slice_negative_steps_and_int64_min_sentinel() {
    let g = G::default()
        .input("x", &[1, 6])
        .input("y", &[1, 7])
        .node(int_const("axis1", &[1], &[1]))
        .node(int_const("minus_one", &[1], &[-1]))
        .node(int_const("min", &[1], &[i64::MIN]))
        .node(int_const("zero", &[1], &[0]))
        .node(int_const("six", &[1], &[6]))
        .node(int_const("seven", &[1], &[7]))
        .node(int_const("two", &[1], &[2]))
        .node(int_const("minus_two", &[1], &[-2]))
        .node(node(
            "Slice",
            &["x", "minus_one", "min", "axis1", "minus_one"],
            &["reversed"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["y", "zero", "seven", "axis1", "two"],
            &["every_other"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["y", "six", "zero", "axis1", "minus_two"],
            &["back_by_two"],
            vec![],
        ))
        .output("reversed")
        .output("every_other")
        .output("back_by_two");
    let table = g.infer(13);
    has(&table, "reversed", &[1, 6]);
    has(&table, "every_other", &[1, 4]);
    // Indices 6, 4, 2 (end 0 is exclusive).
    has(&table, "back_by_two", &[1, 3]);
}

#[test]
fn slice_legacy_attribute_form() {
    let g = G::default()
        .input("x", &[2, 5])
        .node(node(
            "Slice",
            &["x"],
            &["y"],
            vec![
                attr_ints("starts", &[1]),
                attr_ints("ends", &[3]),
                attr_ints("axes", &[1]),
            ],
        ))
        .output("y");
    has(&g.infer(9), "y", &[2, 2]);
    // The attribute form is not Slice's schema from opset 10.
    unknown(&g.infer(13), "y");
}

#[test]
fn slice_empty_result_and_non_constant_parameters_stay_unknown() {
    let g = G::default()
        .input("x", &[1, 6])
        .int_input("runtime_starts", &[1])
        .node(int_const("axis1", &[1], &[1]))
        .node(int_const("three", &[1], &[3]))
        .node(int_const("zero", &[1], &[0]))
        .node(int_const("dup_axes", &[2], &[1, -1]))
        .node(int_const("starts_pair", &[2], &[0, 0]))
        .node(int_const("ends_pair", &[2], &[1, 1]))
        .node(node(
            "Slice",
            &["x", "three", "three", "axis1"],
            &["empty"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["x", "runtime_starts", "three", "axis1"],
            &["runtime"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["x", "zero", "three", "axis1", "zero"],
            &["zero_step"],
            vec![],
        ))
        .node(node(
            "Slice",
            &["x", "starts_pair", "ends_pair", "dup_axes"],
            &["duplicate_axes"],
            vec![],
        ))
        .output("empty")
        .output("runtime")
        .output("zero_step")
        .output("duplicate_axes");
    let table = g.infer(13);
    has(&table, "runtime_starts", &[1]);
    unknown(&table, "empty");
    unknown(&table, "runtime");
    unknown(&table, "zero_step");
    unknown(&table, "duplicate_axes");
}

#[test]
fn gather_shapes_and_negative_axis() {
    let g = G::default()
        .input("x", &[1, 6])
        .input("m", &[4, 5])
        .node(int_const("picks", &[2], &[2, 5]))
        .node(int_const("scalar", &[], &[3]))
        .node(int_const("grid", &[2, 3], &[0, 1, 2, 0, 1, 2]))
        .node(node(
            "Gather",
            &["x", "picks"],
            &["cols"],
            vec![attr_int("axis", 1)],
        ))
        .node(node(
            "Gather",
            &["x", "picks"],
            &["cols_neg"],
            vec![attr_int("axis", -1)],
        ))
        .node(node("Gather", &["m", "scalar"], &["row"], vec![]))
        .node(node(
            "Gather",
            &["m", "grid"],
            &["grid_out"],
            vec![attr_int("axis", 1)],
        ))
        .output("cols")
        .output("cols_neg")
        .output("row")
        .output("grid_out");
    let table = g.infer(13);
    has(&table, "cols", &[1, 2]);
    has(&table, "cols_neg", &[1, 2]);
    has(&table, "row", &[5]);
    has(&table, "grid_out", &[4, 2, 3]);
    let legacy = g.infer(9);
    has(&legacy, "cols", &[1, 2]);
    unknown(&legacy, "cols_neg");
}

#[test]
fn squeeze_and_unsqueeze_with_negative_axes_in_both_opset_forms() {
    let modern = G::default()
        .input("x", &[1, 3, 1, 4])
        .input("y", &[3, 4])
        .node(int_const("minus_two", &[1], &[-2]))
        .node(int_const("one", &[1], &[1]))
        .node(int_const("ends", &[2], &[0, -1]))
        .node(int_const("dup", &[2], &[0, 0]))
        .node(node("Squeeze", &["x", "minus_two"], &["sq_neg"], vec![]))
        .node(node("Squeeze", &["x"], &["sq_all"], vec![]))
        .node(node("Squeeze", &["x", "one"], &["sq_not_one"], vec![]))
        .node(node("Unsqueeze", &["y", "ends"], &["unsq"], vec![]))
        .node(node("Unsqueeze", &["y", "dup"], &["unsq_dup"], vec![]))
        .node(node(
            "Unsqueeze",
            &["y"],
            &["unsq_attr"],
            vec![attr_ints("axes", &[0])],
        ))
        .output("sq_neg")
        .output("sq_all")
        .output("sq_not_one")
        .output("unsq")
        .output("unsq_dup")
        .output("unsq_attr");
    let table = modern.infer(13);
    has(&table, "sq_neg", &[1, 3, 4]);
    has(&table, "sq_all", &[3, 4]);
    unknown(&table, "sq_not_one");
    has(&table, "unsq", &[1, 3, 4, 1]);
    unknown(&table, "unsq_dup");
    // Opset 13 moved `axes` to an input; the attribute form is refused there.
    unknown(&table, "unsq_attr");

    let legacy = G::default()
        .input("x", &[1, 3, 1, 4])
        .input("y", &[3, 4])
        .node(node(
            "Squeeze",
            &["x"],
            &["sq"],
            vec![attr_ints("axes", &[0])],
        ))
        .node(node(
            "Squeeze",
            &["x"],
            &["sq_neg"],
            vec![attr_ints("axes", &[-2])],
        ))
        .node(node(
            "Unsqueeze",
            &["y"],
            &["unsq"],
            vec![attr_ints("axes", &[1])],
        ))
        .node(node(
            "Unsqueeze",
            &["y"],
            &["unsq_neg"],
            vec![attr_ints("axes", &[-1])],
        ))
        .output("sq")
        .output("sq_neg")
        .output("unsq")
        .output("unsq_neg");
    let table = legacy.infer(11);
    has(&table, "sq", &[3, 1, 4]);
    has(&table, "sq_neg", &[1, 3, 4]);
    has(&table, "unsq", &[3, 1, 4]);
    has(&table, "unsq_neg", &[3, 4, 1]);
    // ...and negative axes do not exist before opset 11.
    let older = legacy.infer(10);
    has(&older, "sq", &[3, 1, 4]);
    unknown(&older, "sq_neg");
    unknown(&older, "unsq_neg");
}

#[test]
fn reshape_zero_copy_minus_one_and_allowzero() {
    let g = G::default()
        .input("x", &[2, 3, 4])
        .int_weight("copy_infer", &[2], &[0, -1])
        .int_weight("infer_middle", &[3], &[4, -1, 3])
        .int_weight("flat", &[1], &[-1])
        .int_weight("explicit", &[2], &[2, 12])
        .node(node("Reshape", &["x", "copy_infer"], &["a"], vec![]))
        .node(node("Reshape", &["x", "infer_middle"], &["b"], vec![]))
        .node(node("Reshape", &["x", "flat"], &["c"], vec![]))
        .node(node(
            "Reshape",
            &["x", "explicit"],
            &["d"],
            vec![attr_int("allowzero", 1)],
        ))
        .output("a")
        .output("b")
        .output("c")
        .output("d");
    let table = g.infer(14);
    has(&table, "a", &[2, 12]);
    has(&table, "b", &[4, 2, 3]);
    has(&table, "c", &[24]);
    has(&table, "d", &[2, 12]);
    // `allowzero` is not a Reshape attribute before opset 14.
    unknown(&g.infer(13), "d");
    has(&g.infer(13), "a", &[2, 12]);

    let legacy = G::default()
        .input("x", &[2, 3, 4])
        .node(node(
            "Reshape",
            &["x"],
            &["y"],
            vec![attr_ints("shape", &[6, 4])],
        ))
        .output("y");
    has(&legacy.infer(4), "y", &[6, 4]);
}

#[test]
fn reshape_inconsistent_or_non_constant_stays_unknown() {
    let g = G::default()
        .input("x", &[2, 3, 4])
        .int_input("runtime_shape", &[2])
        .int_weight("wrong_count", &[2], &[5, 5])
        .int_weight("two_infers", &[2], &[-1, -1])
        .int_weight("zero_past_rank", &[4], &[0, 0, 0, 0])
        .int_weight("literal_zero", &[2], &[0, 24])
        .int_weight("negative", &[2], &[-2, 12])
        .node(node("Reshape", &["x", "wrong_count"], &["count"], vec![]))
        .node(node("Reshape", &["x", "two_infers"], &["infers"], vec![]))
        .node(node(
            "Reshape",
            &["x", "zero_past_rank"],
            &["past_rank"],
            vec![],
        ))
        .node(node(
            "Reshape",
            &["x", "runtime_shape"],
            &["runtime"],
            vec![],
        ))
        .node(node("Reshape", &["x", "negative"], &["neg"], vec![]))
        .node(node(
            "Reshape",
            &["x", "literal_zero"],
            &["empty"],
            vec![attr_int("allowzero", 1)],
        ))
        .output("count")
        .output("infers")
        .output("past_rank")
        .output("runtime")
        .output("neg")
        .output("empty");
    let table = g.infer(14);
    for name in ["count", "infers", "past_rank", "runtime", "neg", "empty"] {
        unknown(&table, name);
    }
}

#[test]
fn transpose_default_and_explicit_perm() {
    let g = G::default()
        .input("x", &[2, 3, 4])
        .node(node("Transpose", &["x"], &["rev"], vec![]))
        .node(node(
            "Transpose",
            &["x"],
            &["swap"],
            vec![attr_ints("perm", &[0, 2, 1])],
        ))
        .node(node(
            "Transpose",
            &["x"],
            &["short"],
            vec![attr_ints("perm", &[0, 1])],
        ))
        .node(node(
            "Transpose",
            &["x"],
            &["repeat"],
            vec![attr_ints("perm", &[0, 0, 1])],
        ))
        .output("rev")
        .output("swap")
        .output("short")
        .output("repeat");
    let table = g.infer(13);
    has(&table, "rev", &[4, 3, 2]);
    has(&table, "swap", &[2, 4, 3]);
    unknown(&table, "short");
    unknown(&table, "repeat");
}

#[test]
fn identity_cast_and_dropout_preserve_shape() {
    let g = G::default()
        .input("x", &[2, 3])
        .node(node("Identity", &["x"], &["id"], vec![]))
        .node(node("Cast", &["id"], &["cast"], vec![attr_int("to", 11)]))
        .node(node(
            "Dropout",
            &["cast"],
            &["drop", "mask"],
            vec![attr_float("ratio", 0.5)],
        ))
        .output("drop")
        .output("mask");
    let table = g.infer(13);
    has(&table, "id", &[2, 3]);
    has(&table, "cast", &[2, 3]);
    has(&table, "drop", &[2, 3]);
    has(&table, "mask", &[2, 3]);
}

#[test]
fn cast_keeps_integer_values_only_when_exactly_representable() {
    let g = G::default()
        .input("x", &[2, 3])
        .node(int_const("target", &[2], &[1, -1]))
        .node(int_const("huge", &[2], &[1 << 40, -1]))
        .node(node(
            "Cast",
            &["target"],
            &["as_i32"],
            vec![attr_int("to", 6)],
        ))
        .node(node(
            "Cast",
            &["as_i32"],
            &["back"],
            vec![attr_int("to", 7)],
        ))
        .node(node("Reshape", &["x", "back"], &["flat"], vec![]))
        .node(node(
            "Cast",
            &["huge"],
            &["huge_i32"],
            vec![attr_int("to", 6)],
        ))
        .node(node(
            "Cast",
            &["huge_i32"],
            &["huge_back"],
            vec![attr_int("to", 7)],
        ))
        .node(node("Reshape", &["x", "huge_back"], &["huge_flat"], vec![]))
        .node(node(
            "Cast",
            &["target"],
            &["as_float"],
            vec![attr_int("to", 1)],
        ))
        .node(node(
            "Cast",
            &["as_float"],
            &["float_back"],
            vec![attr_int("to", 7)],
        ))
        .node(node(
            "Reshape",
            &["x", "float_back"],
            &["float_flat"],
            vec![],
        ))
        .output("flat")
        .output("huge_flat")
        .output("float_flat");
    let table = g.infer(13);
    has(&table, "flat", &[1, 6]);
    // The cast shapes are known; only the VALUES are withheld.
    has(&table, "huge_back", &[2]);
    unknown(&table, "huge_flat");
    has(&table, "float_back", &[2]);
    unknown(&table, "float_flat");
}

#[test]
fn reduce_ops_in_axes_input_and_attribute_forms() {
    let g = G::default()
        .input("x", &[1, 6])
        .input("w", &[2, 6])
        .node(int_const("axis1", &[1], &[1]))
        .node(int_const("last", &[1], &[-1]))
        .node(int_const("dup", &[2], &[1, -1]))
        .node(node(
            "ReduceSum",
            &["x", "axis1"],
            &["keep"],
            vec![attr_int("keepdims", 1)],
        ))
        .node(node(
            "ReduceSum",
            &["x", "last"],
            &["drop"],
            vec![attr_int("keepdims", 0)],
        ))
        .node(node(
            "ReduceSum",
            &["x"],
            &["noop"],
            vec![attr_int("noop_with_empty_axes", 1)],
        ))
        .node(node("ReduceSum", &["x"], &["all"], vec![]))
        .node(node("ReduceSum", &["x", "dup"], &["dup_axes"], vec![]))
        .node(node(
            "ReduceSum",
            &["x"],
            &["attr_form"],
            vec![attr_ints("axes", &[1])],
        ))
        .node(node(
            "ReduceMean",
            &["w"],
            &["mean"],
            vec![attr_ints("axes", &[-1]), attr_int("keepdims", 0)],
        ))
        .output("keep")
        .output("drop")
        .output("noop")
        .output("all")
        .output("dup_axes")
        .output("attr_form")
        .output("mean");
    let table = g.infer(13);
    has(&table, "keep", &[1, 1]);
    has(&table, "drop", &[1]);
    has(&table, "noop", &[1, 6]);
    has(&table, "all", &[1, 1]);
    unknown(&table, "dup_axes");
    // ReduceSum took `axes` as an input from opset 13; ReduceMean at 18.
    unknown(&table, "attr_form");
    has(&table, "mean", &[2]);

    let legacy = G::default()
        .input("x", &[1, 6])
        .node(node(
            "ReduceSum",
            &["x"],
            &["y"],
            vec![attr_ints("axes", &[1]), attr_int("keepdims", 1)],
        ))
        .output("y");
    has(&legacy.infer(11), "y", &[1, 1]);
}

/// The dist_shift `mnist_prior` flatten idiom, with a symbolic batch that the
/// fallback pins to 1 exactly as the ORT path does.
fn mnist_prior_chain() -> G {
    G::default()
        .input("input", &[SYM, 1, 28, 28])
        .int_weight("minus_one", &[1], &[-1])
        .weight("fc_w", &[8, 784])
        .weight("fc_b", &[8])
        .node(node("Shape", &["input"], &["shape"], vec![]))
        .node(int_const("zero", &[], &[0]))
        .node(node(
            "Gather",
            &["shape", "zero"],
            &["batch"],
            vec![attr_int("axis", 0)],
        ))
        .node(node(
            "Unsqueeze",
            &["batch"],
            &["batch1"],
            vec![attr_ints("axes", &[0])],
        ))
        .node(node(
            "Concat",
            &["batch1", "minus_one"],
            &["target"],
            vec![attr_int("axis", 0)],
        ))
        .node(node("Reshape", &["input", "target"], &["flat"], vec![]))
        .node(node(
            "Gemm",
            &["flat", "fc_w", "fc_b"],
            &["logits"],
            vec![attr_int("transB", 1)],
        ))
        .declared_output("logits", &[SYM, 8])
}

#[test]
fn shape_gather_unsqueeze_concat_reshape_chain_resolves_on_the_pinned_batch() {
    let table = mnist_prior_chain().infer(11);
    has(&table, "input", &[1, 1, 28, 28]);
    has(&table, "shape", &[4]);
    has(&table, "batch", &[]);
    has(&table, "batch1", &[1]);
    has(&table, "target", &[2]);
    has(&table, "flat", &[1, 784]);
    has(&table, "logits", &[1, 8]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

#[test]
fn shape_start_end_attributes_from_opset_15() {
    let g = G::default()
        .input("x", &[2, 3, 4])
        .node(node(
            "Shape",
            &["x"],
            &["tail"],
            vec![attr_int("start", -2)],
        ))
        .node(node("Reshape", &["x", "tail"], &["y"], vec![]))
        .output("y");
    let table = g.infer(15);
    has(&table, "tail", &[2]);
    // Reshape to [3, 4] needs 24 == 12 elements: inconsistent, so unknown.
    unknown(&table, "y");
    unknown(&g.infer(14), "tail");
}

#[test]
fn constant_node_forms() {
    let g = G::default()
        .input("x", &[1])
        .node(node(
            "Constant",
            &[],
            &["tensor"],
            vec![attr_tensor("value", f32_tensor("", &[2, 3]))],
        ))
        .node(node(
            "Constant",
            &[],
            &["int"],
            vec![attr_int("value_int", 4)],
        ))
        .node(node(
            "Constant",
            &[],
            &["ints"],
            vec![attr_ints("value_ints", &[1, 2, 3])],
        ))
        .node(node(
            "Constant",
            &[],
            &["float"],
            vec![attr_float("value_float", 0.5)],
        ))
        .node(node(
            "Constant",
            &[],
            &["floats"],
            vec![attr_floats("value_floats", &[0.5, 1.5])],
        ))
        .node(node(
            "Constant",
            &[],
            &["empty_ints"],
            vec![attr_ints("value_ints", &[])],
        ))
        .node(node(
            "Constant",
            &[],
            &["two_attrs"],
            vec![attr_int("value_int", 1), attr_float("value_float", 1.0)],
        ))
        .node(node(
            "Constant",
            &[],
            &["mistyped"],
            vec![attr_float("value_int", 1.0)],
        ))
        .output("tensor");
    let table = g.infer(13);
    has(&table, "tensor", &[2, 3]);
    has(&table, "int", &[]);
    has(&table, "ints", &[3]);
    has(&table, "float", &[]);
    has(&table, "floats", &[2]);
    unknown(&table, "empty_ints");
    unknown(&table, "two_attrs");
    unknown(&table, "mistyped");
}

#[test]
fn constant_of_shape_and_expand_use_constant_shape_values() {
    let g = G::default()
        .input("x", &[3, 1])
        .node(int_const("dims", &[2], &[2, 5]))
        .node(int_const("expand_to", &[2], &[3, 4]))
        .node(node("ConstantOfShape", &["dims"], &["filled"], vec![]))
        .node(node("Expand", &["x", "expand_to"], &["expanded"], vec![]))
        .output("filled")
        .output("expanded");
    let table = g.infer(13);
    has(&table, "filled", &[2, 5]);
    has(&table, "expanded", &[3, 4]);
}

// ---------------------------------------------------------------------------
// Fail-closed cases: never assert what cannot be derived.
// ---------------------------------------------------------------------------

#[test]
fn unmodelled_op_and_foreign_domain_poison_downstream() {
    let mut foreign = node("Relu", &["x"], &["foreign"], vec![]);
    foreign.domain = "com.example".to_string();
    let g = G::default()
        .input("x", &[1, 3])
        .node(node("NotARealOp", &["x"], &["mystery"], vec![]))
        .node(node("Relu", &["mystery"], &["after_mystery"], vec![]))
        .node(foreign)
        .node(node("Relu", &["foreign"], &["after_foreign"], vec![]))
        .output("after_mystery")
        .output("after_foreign");
    let table = g.infer(13);
    for name in ["mystery", "after_mystery", "foreign", "after_foreign"] {
        unknown(&table, name);
    }
    has(&table, "x", &[1, 3]);
}

#[test]
fn ssa_violations_stay_unknown() {
    let g = G::default()
        .input("x", &[1, 4])
        .weight("w", &[4, 2])
        // Shadows the initializer `w`.
        .node(node("Relu", &["x"], &["w"], vec![]))
        .node(node("MatMul", &["x", "w"], &["uses_shadowed"], vec![]))
        // Two producers of one name.
        .node(node("Relu", &["x"], &["dup"], vec![]))
        .node(node("Sigmoid", &["x"], &["dup"], vec![]))
        .node(node("Relu", &["dup"], &["uses_dup"], vec![]))
        // Consumed before it is produced (non-topological order).
        .node(node("Relu", &["late"], &["uses_late"], vec![]))
        .node(node("Relu", &["x"], &["late"], vec![]))
        .output("uses_shadowed")
        .output("uses_dup")
        .output("uses_late");
    let table = g.infer(13);
    for name in ["w", "uses_shadowed", "dup", "uses_dup", "uses_late"] {
        unknown(&table, name);
    }
    has(&table, "late", &[1, 4]);
}

#[test]
fn declared_terminal_output_conflict_is_withdrawn() {
    let g = G::default()
        .input("x", &[1, 3])
        .node(node("Relu", &["x"], &["wrong_decl"], vec![]))
        .node(node("Sigmoid", &["x"], &["symbolic_decl"], vec![]))
        .node(node("Tanh", &["x"], &["rank_decl"], vec![]))
        .declared_output("wrong_decl", &[1, 4])
        .declared_output("symbolic_decl", &[SYM, 3])
        .declared_output("rank_decl", &[3]);
    let table = g.infer(13);
    unknown(&table, "wrong_decl");
    has(&table, "symbolic_decl", &[1, 3]);
    unknown(&table, "rank_decl");
}

#[test]
fn only_the_leading_symbolic_dim_is_pinned() {
    let g = G::default()
        .input("lead", &[SYM, 3])
        .input("inner", &[1, SYM])
        .node(node("Relu", &["lead"], &["lead_out"], vec![]))
        .node(node("Relu", &["inner"], &["inner_out"], vec![]))
        .output("lead_out")
        .output("inner_out");
    let table = g.infer(13);
    has(&table, "lead", &[1, 3]);
    has(&table, "lead_out", &[1, 3]);
    unknown(&table, "inner");
    unknown(&table, "inner_out");
}

#[test]
fn inconsistent_initializer_payload_is_not_trusted() {
    let mut short = f32_tensor("short", &[4, 2]);
    short.raw_data.truncate(4);
    let mut both = i64_tensor("both", &[1], &[2]);
    both.int64_data = vec![2];
    let mut external = f32_tensor("external", &[4, 2]);
    external.data_location = 1;
    let g = G::default()
        .input("x", &[1, 4])
        .input("y", &[1, 6])
        .raw_init(short)
        .raw_init(both)
        .raw_init(external)
        .node(node("MatMul", &["x", "short"], &["from_short"], vec![]))
        .node(node(
            "MatMul",
            &["x", "external"],
            &["from_external"],
            vec![],
        ))
        .node(node("Reshape", &["y", "both"], &["from_both"], vec![]))
        .output("from_short")
        .output("from_external")
        .output("from_both");
    let table = g.infer(13);
    unknown(&table, "from_short");
    unknown(&table, "from_external");
    unknown(&table, "from_both");
}

#[test]
fn undecodable_graphless_and_ambiguous_models_are_refused() {
    let garbage = infer_tensor_shapes_static(b"\xff\xfe definitely not an onnx model");
    assert!(garbage.refusal.is_some());
    assert!(garbage.shapes.is_empty());

    let graphless = infer_tensor_shapes_static(&encode(&ModelProto {
        ir_version: 8,
        ..Default::default()
    }));
    assert!(graphless.refusal.is_some());
    assert!(graphless.shapes.is_empty());

    let g = G::default()
        .input("x", &[1, 3])
        .node(node("Relu", &["x"], &["y"], vec![]))
        .output("y");
    let mut conflicting = g.model(13);
    conflicting.opset_import.push(OperatorSetIdProto {
        domain: String::new(),
        version: 11,
    });
    let refused = infer_tensor_shapes_static(&encode(&conflicting));
    assert!(
        refused.refusal.is_some(),
        "conflicting opsets must be refused"
    );
    assert!(refused.shapes.is_empty());

    // A model whose every input is an initializer has no runtime inputs; the
    // ORT path returns an empty table for it, and so does the fallback.
    let constant_only = G::default()
        .weight("w", &[2, 2])
        .node(node("Relu", &["w"], &["y"], vec![]))
        .output("y");
    let mut model = constant_only.model(13);
    if let Some(graph) = model.graph.as_mut() {
        graph.input.push(value_info("w", Some(&[2, 2]), DT_FLOAT));
    }
    let no_runtime = infer_tensor_shapes_static(&encode(&model));
    assert!(no_runtime.shapes.is_empty());
    assert!(no_runtime.refusal.is_some());
}

/// `y = MatMul(w, x)` with the weight `w` an initializer that is ALSO a graph
/// input, optionally plus `r = Reshape(a, t)` whose INT64 shape `t` is the
/// same kind of initializer-backed input.
fn overridable_initializer_model(
    w_declared: Option<&[i64]>,
    t_declared: Option<&[i64]>,
    ir_version: i64,
    with_reshape: bool,
) -> ModelProto {
    let mut g = G::default()
        .input("x", &[5, 2])
        .weight("w", &[3, 5])
        .node(node("MatMul", &["w", "x"], &["y"], vec![]))
        .output("y");
    if with_reshape {
        g = g
            .input("a", &[6])
            .int_weight("t", &[2], &[2, 3])
            .node(node("Reshape", &["a", "t"], &["r"], vec![]))
            .output("r");
    }
    let mut model = g.model(13);
    model.ir_version = ir_version;
    let graph = model.graph.as_mut().expect("test model has a graph");
    graph.input.push(value_info("w", w_declared, DT_FLOAT));
    if with_reshape {
        graph.input.push(value_info("t", t_declared, DT_INT64));
    }
    model
}

#[test]
fn overridable_initializer_is_typed_from_its_declaration_like_ort() {
    let infer = |w: Option<&[i64]>, t: Option<&[i64]>, ir: i64| {
        infer_tensor_shapes_static(&encode(&overridable_initializer_model(w, t, ir, true)))
    };
    // IR 3: an initializer listed as an input is a constant, so its dims and
    // values seed exactly as ORT uses them (ORT: y [3, 2], r [2, 3]).
    let constant = infer(Some(&[SYM, 5]), None, 3);
    has(&constant, "y", &[3, 2]);
    has(&constant, "r", &[2, 3]);
    // IR 8: an overridable default. A declaration pinning exactly the payload
    // dims gives the shape (ORT: y [3, 2]); the values never fold (ORT: r
    // has no shape).
    let pinned = infer(Some(&[3, 5]), Some(&[2]), 8);
    has(&pinned, "y", &[3, 2]);
    unknown(&pinned, "r");
    // A symbolic or absent declaration leaves the tensor unknown (ORT: y is
    // [-1, 2] / has no shape) rather than taking the payload's dims.
    let symbolic = infer(Some(&[SYM, 5]), None, 8);
    unknown(&symbolic, "y");
    unknown(&symbolic, "r");
    let undeclared = infer(None, None, 8);
    unknown(&undeclared, "y");
    // An unset IR version is normalised to 9 by the ORT path: overridable.
    let unset = infer(Some(&[SYM, 5]), Some(&[2]), 0);
    unknown(&unset, "y");
    unknown(&unset, "r");
}

#[test]
fn opset_newer_than_the_audited_schemas_is_refused() {
    let g = G::default()
        .input("x", &[1, 3])
        .node(node("Relu", &["x"], &["y"], vec![]))
        .output("y");
    has(&g.infer(26), "y", &[1, 3]);
    let future = g.infer(27);
    assert!(
        future
            .refusal
            .as_deref()
            .is_some_and(|reason| reason.contains("newer than the newest audited")),
        "an unaudited opset must be refused, got {:?}",
        future.refusal
    );
    assert!(future.shapes.is_empty());
}

#[test]
fn missing_opset_import_defaults_like_the_ort_normalisation() {
    let g = G::default()
        .input("x", &[1, 3])
        .input("y", &[1, 3])
        .node(node("Add", &["x", "y"], &["sum"], vec![]))
        .output("sum");
    let mut model = g.model(13);
    model.opset_import.clear();
    // No default-domain import normalises to opset 13, where Add broadcasts.
    has(&infer_tensor_shapes_static(&encode(&model)), "sum", &[1, 3]);
}

// ---------------------------------------------------------------------------
// The two measured regressions (x86 under qemu, 2026-09-28).
// ---------------------------------------------------------------------------

/// The linearizenn `AllInOne_*` tail: the Concat whose input
/// `/controller/MatMul_output_0` had "no recorded ONNX shape".
fn linearizenn_allinone_tail() -> G {
    G::default()
        .input("input", &[1, 4])
        .weight("main.0.weight", &[256, 4])
        .weight("main.0.bias", &[256])
        .weight("main.2.weight", &[2, 256])
        .weight("main.2.bias", &[2])
        .weight("onnx::MatMul_47", &[2, 1])
        .weight("onnx::MatMul_48", &[2, 1])
        .weight("onnx::MatMul_49", &[2, 2])
        .node(node(
            "Gemm",
            &["input", "main.0.weight", "main.0.bias"],
            &["/main/main.0/Gemm_output_0"],
            vec![
                attr_float("alpha", 1.0),
                attr_float("beta", 1.0),
                attr_int("transB", 1),
            ],
        ))
        .node(node(
            "Relu",
            &["/main/main.0/Gemm_output_0"],
            &["/main/main.1/Relu_output_0"],
            vec![],
        ))
        .node(node(
            "Gemm",
            &["/main/main.1/Relu_output_0", "main.2.weight", "main.2.bias"],
            &["/main/main.2/Gemm_output_0"],
            vec![
                attr_float("alpha", 1.0),
                attr_float("beta", 1.0),
                attr_int("transB", 1),
            ],
        ))
        .node(node(
            "MatMul",
            &["/main/main.2/Gemm_output_0", "onnx::MatMul_47"],
            &["/controller/MatMul_output_0"],
            vec![],
        ))
        .node(int_const("/Constant_output_0", &[1], &[1]))
        .node(int_const("/Constant_1_output_0", &[1], &[2]))
        .node(int_const("/Constant_2_output_0", &[1], &[i64::MAX]))
        .node(int_const("/Constant_3_output_0", &[1], &[1]))
        .node(node(
            "Slice",
            &[
                "input",
                "/Constant_1_output_0",
                "/Constant_2_output_0",
                "/Constant_output_0",
                "/Constant_3_output_0",
            ],
            &["/Slice_output_0"],
            vec![],
        ))
        .node(node(
            "MatMul",
            &["/Slice_output_0", "onnx::MatMul_48"],
            &["/linear_weight/MatMul_output_0"],
            vec![],
        ))
        .node(node(
            "Concat",
            &[
                "/controller/MatMul_output_0",
                "/linear_weight/MatMul_output_0",
            ],
            &["/Concat_output_0"],
            vec![attr_int("axis", 1)],
        ))
        .node(node(
            "MatMul",
            &["/Concat_output_0", "onnx::MatMul_49"],
            &["output"],
            vec![],
        ))
        .declared_output("output", &[1, 2])
}

#[test]
fn linearizenn_allinone_concat_inputs_are_recovered() {
    let table = linearizenn_allinone_tail().infer(14);
    has(&table, "/controller/MatMul_output_0", &[1, 1]);
    has(&table, "/Slice_output_0", &[1, 2]);
    has(&table, "/linear_weight/MatMul_output_0", &[1, 1]);
    has(&table, "/Concat_output_0", &[1, 2]);
    has(&table, "output", &[1, 2]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

/// The lsnc `relu_quadrotor2d_state` idioms: rank-1 MatMul branches, a
/// Gather on axis 1, Concat on axis -1, and ReduceSum over axes [-1].
fn lsnc_quadrotor_idioms() -> G {
    G::default()
        .input("onnx::Mul_0", &[1, 6])
        .weight("net.0.weight", &[8, 6])
        .weight("net.0.bias", &[8])
        .weight("net.2.weight", &[2, 8])
        .weight("net.2.bias", &[2])
        .weight("onnx::MatMul_124", &[6, 8])
        .weight("onnx::MatMul_125", &[8, 2])
        .node(node(
            "Constant",
            &[],
            &["/Constant_output_0"],
            vec![attr_tensor("value", f32_tensor("", &[]))],
        ))
        .node(node(
            "Mul",
            &["onnx::Mul_0", "/Constant_output_0"],
            &["/Mul_output_0"],
            vec![],
        ))
        .node(node(
            "Gemm",
            &["/Mul_output_0", "net.0.weight", "net.0.bias"],
            &["/net.0/Gemm_output_0"],
            vec![attr_int("transB", 1)],
        ))
        .node(node(
            "Relu",
            &["/net.0/Gemm_output_0"],
            &["/net.1/Relu_output_0"],
            vec![],
        ))
        .node(node(
            "Gemm",
            &["/net.1/Relu_output_0", "net.2.weight", "net.2.bias"],
            &["/net.2/Gemm_output_0"],
            vec![attr_int("transB", 1)],
        ))
        .node(node(
            "Constant",
            &[],
            &["/net.0_1/Constant_output_0"],
            vec![attr_tensor("value", f32_tensor("", &[6]))],
        ))
        .node(node(
            "MatMul",
            &["/net.0_1/Constant_output_0", "onnx::MatMul_124"],
            &["/net.0_1/MatMul_output_0"],
            vec![],
        ))
        .node(node(
            "Add",
            &["net.0.bias", "/net.0_1/MatMul_output_0"],
            &["/net.0_1/Add_output_0"],
            vec![],
        ))
        .node(node(
            "Relu",
            &["/net.0_1/Add_output_0"],
            &["/net.1_1/Relu_output_0"],
            vec![],
        ))
        .node(node(
            "MatMul",
            &["/net.1_1/Relu_output_0", "onnx::MatMul_125"],
            &["/net.2_1/MatMul_output_0"],
            vec![],
        ))
        .node(node(
            "Add",
            &["net.2.bias", "/net.2_1/MatMul_output_0"],
            &["/net.2_1/Add_output_0"],
            vec![],
        ))
        .node(node(
            "Sub",
            &["/net.2/Gemm_output_0", "/net.2_1/Add_output_0"],
            &["/controller/Sub_output_0"],
            vec![],
        ))
        .node(node(
            "Cast",
            &["/controller/Sub_output_0"],
            &["/Cast_output_0"],
            vec![attr_int("to", 1)],
        ))
        .node(int_const("onnx::Gather_65", &[2], &[2, 5]))
        .node(node(
            "Gather",
            &["/Mul_output_0", "onnx::Gather_65"],
            &["/Gather_output_0"],
            vec![attr_int("axis", 1)],
        ))
        .node(node(
            "Concat",
            &["/Gather_output_0", "/Cast_output_0"],
            &["/Concat_output_0"],
            vec![attr_int("axis", -1)],
        ))
        .node(node(
            "Mul",
            &["/Mul_output_0", "/Mul_output_0"],
            &["/Mul_3_output_0"],
            vec![],
        ))
        .node(int_const("onnx::ReduceSum_112", &[1], &[-1]))
        .node(node(
            "ReduceSum",
            &["/Mul_3_output_0", "onnx::ReduceSum_112"],
            &["/ReduceSum_output_0"],
            vec![attr_int("keepdims", 1)],
        ))
        .node(node(
            "Concat",
            &["/Concat_output_0", "/ReduceSum_output_0"],
            &["123"],
            vec![attr_int("axis", -1)],
        ))
        .declared_output("123", &[1, 5])
}

#[test]
fn lsnc_quadrotor_idioms_are_recovered() {
    let table = lsnc_quadrotor_idioms().infer(17);
    has(&table, "/Constant_output_0", &[]);
    has(&table, "/Mul_output_0", &[1, 6]);
    has(&table, "/net.2/Gemm_output_0", &[1, 2]);
    has(&table, "/net.0_1/MatMul_output_0", &[8]);
    has(&table, "/net.2_1/MatMul_output_0", &[2]);
    has(&table, "/net.2_1/Add_output_0", &[2]);
    has(&table, "/controller/Sub_output_0", &[1, 2]);
    has(&table, "/Gather_output_0", &[1, 2]);
    has(&table, "/Concat_output_0", &[1, 4]);
    has(&table, "/ReduceSum_output_0", &[1, 1]);
    has(&table, "123", &[1, 5]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

/// The collins_rul_cnn stack at its authored opset 8: explicit-pad Conv with
/// a column kernel, a full-extent "fc" Conv, Dropout, and Flatten.
fn collins_conv_stack() -> G {
    let conv_attrs = |kernel: &[i64]| {
        vec![
            attr_int("group", 1),
            attr_ints("dilations", &[1, 1]),
            attr_ints("kernel_shape", kernel),
            attr_ints("pads", &[0, 0, 0, 0]),
            attr_ints("strides", &[1, 1]),
        ]
    };
    G::default()
        .input("imageinput", &[SYM, 1, 20, 20])
        .weight("conv_1_W", &[5, 1, 5, 1])
        .weight("conv_1_B", &[5])
        .weight("fc_1_W", &[100, 5, 16, 20])
        .weight("fc_1_B", &[100])
        .weight("fc_2_W", &[1, 100, 1, 1])
        .weight("fc_2_B", &[1])
        .node(node(
            "Conv",
            &["imageinput", "conv_1_W", "conv_1_B"],
            &["conv_1"],
            conv_attrs(&[5, 1]),
        ))
        .node(node("Relu", &["conv_1"], &["relu_1"], vec![]))
        .node(node(
            "Conv",
            &["relu_1", "fc_1_W", "fc_1_B"],
            &["fc_1"],
            conv_attrs(&[16, 20]),
        ))
        .node(node("Relu", &["fc_1"], &["relu_5"], vec![]))
        .node(node(
            "Dropout",
            &["relu_5"],
            &["dropout"],
            vec![attr_float("ratio", 0.5)],
        ))
        .node(node(
            "Conv",
            &["dropout", "fc_2_W", "fc_2_B"],
            &["fc_2"],
            conv_attrs(&[1, 1]),
        ))
        .node(node(
            "Flatten",
            &["fc_2"],
            &["fc_2_Flatten"],
            vec![attr_int("axis", 1)],
        ))
        .declared_output("fc_2_Flatten", &[SYM, 1])
}

#[test]
fn collins_conv_stack_resolves_at_opset_8() {
    let table = collins_conv_stack().infer(8);
    has(&table, "imageinput", &[1, 1, 20, 20]);
    has(&table, "conv_1", &[1, 5, 16, 20]);
    has(&table, "fc_1", &[1, 100, 1, 1]);
    has(&table, "dropout", &[1, 100, 1, 1]);
    has(&table, "fc_2", &[1, 1, 1, 1]);
    has(&table, "fc_2_Flatten", &[1, 1]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

/// One graph exercising every broadcast / negative-axis rule at once, all
/// parameters as INT64 initializers.
fn broadcast_zoo() -> G {
    G::default()
        .input("a", &[2, 1, 4])
        .input("bmat", &[5, 4, 6])
        .input("x", &[1, 3, 1, 4])
        .weight("b", &[3, 1])
        .int_weight("shape_ab", &[2], &[0, -1])
        .int_weight("expand_to", &[3], &[2, 3, 4])
        .int_weight("sq_axes", &[1], &[-2])
        .int_weight("unsq_axes", &[2], &[0, -1])
        .int_weight("red_axes", &[1], &[-1])
        .int_weight("picks", &[2], &[0, 2])
        .int_weight("sl_starts", &[1], &[-1])
        .int_weight("sl_ends", &[1], &[i64::MIN])
        .int_weight("sl_axes", &[1], &[-1])
        .int_weight("sl_steps", &[1], &[-1])
        .node(node("Add", &["a", "b"], &["ab"], vec![]))
        .node(node("Reshape", &["ab", "shape_ab"], &["ab2"], vec![]))
        .node(node("Expand", &["a", "expand_to"], &["ex"], vec![]))
        .node(node("Greater", &["ab", "ex"], &["gt"], vec![]))
        .node(node("Where", &["gt", "ab", "ex"], &["wh"], vec![]))
        .node(node("Unsqueeze", &["wh", "unsq_axes"], &["wu"], vec![]))
        .node(node(
            "Flatten",
            &["wu"],
            &["fl"],
            vec![attr_int("axis", -2)],
        ))
        .node(node("Squeeze", &["x", "sq_axes"], &["xs"], vec![]))
        .node(node("MatMul", &["xs", "bmat"], &["mm"], vec![]))
        .node(node(
            "Transpose",
            &["mm"],
            &["tr"],
            vec![attr_ints("perm", &[0, 2, 1])],
        ))
        .node(node(
            "ReduceSum",
            &["tr", "red_axes"],
            &["rs"],
            vec![attr_int("keepdims", 0)],
        ))
        .node(node(
            "Gather",
            &["rs", "picks"],
            &["ga"],
            vec![attr_int("axis", -1)],
        ))
        .node(node(
            "Slice",
            &["ga", "sl_starts", "sl_ends", "sl_axes", "sl_steps"],
            &["sl"],
            vec![],
        ))
        .node(node(
            "Concat",
            &["ga", "sl"],
            &["cc"],
            vec![attr_int("axis", -1)],
        ))
        .output("ab2")
        .output("fl")
        .output("cc")
}

#[test]
fn broadcast_zoo_resolves_every_rule() {
    let table = broadcast_zoo().infer(13);
    has(&table, "ab", &[2, 3, 4]);
    has(&table, "ab2", &[2, 12]);
    has(&table, "ex", &[2, 3, 4]);
    has(&table, "gt", &[2, 3, 4]);
    has(&table, "wh", &[2, 3, 4]);
    has(&table, "wu", &[1, 2, 3, 4, 1]);
    has(&table, "fl", &[6, 4]);
    has(&table, "xs", &[1, 3, 4]);
    has(&table, "mm", &[5, 3, 6]);
    has(&table, "tr", &[5, 6, 3]);
    has(&table, "rs", &[5, 6]);
    has(&table, "ga", &[5, 2]);
    has(&table, "sl", &[5, 2]);
    has(&table, "cc", &[5, 4]);
    assert!(table.unresolved.is_empty(), "{:?}", table.unresolved);
}

/// The two measured regressions, for the loader-level tests.
pub(super) fn regression_models() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "linearizenn_allinone_tail",
            linearizenn_allinone_tail().bytes(14),
        ),
        ("lsnc_quadrotor_idioms", lsnc_quadrotor_idioms().bytes(17)),
    ]
}

/// Every synthetic graph the fallback resolves completely, for the
/// differential test against in-process ORT.
#[cfg(feature = "ort")]
pub(super) fn differential_models() -> Vec<(&'static str, Vec<u8>)> {
    let mut models = regression_models();
    models.push(("mnist_prior_chain", mnist_prior_chain().bytes(11)));
    models.push(("collins_conv_stack", collins_conv_stack().bytes(8)));
    models.push(("broadcast_zoo", broadcast_zoo().bytes(13)));
    // An IR-8 initializer-backed input whose declaration pins its dims: the
    // one overridable form the fallback resolves, so ny's own ORT must agree.
    models.push((
        "overridable_initializer_pinned",
        encode(&overridable_initializer_model(
            Some(&[3, 5]),
            None,
            8,
            false,
        )),
    ));
    models
}
