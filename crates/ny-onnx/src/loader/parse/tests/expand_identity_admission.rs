// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cctsdb-expand-identity / #cctsdb-trunc-gated-index admission tests.
//!
//! The positive cases mirror cctsdb_yolo_2023's patch models in miniature: the
//! runtime-shape `Tensor.expand(-1, ...)` idiom (constant-false `Equal` over a
//! `Shape` output, pass-through `Where`) and the trunc-gated runtime INT64
//! index chain (patch coordinates read from the FLOAT input via
//! `Gather -> Cast(->INT64)` and consumed as Slice bounds / index arithmetic).
//!
//! The negatives pin the admission's edges: a comparison against ZERO (a dim
//! CAN be zero), a genuinely runtime-vs-runtime comparison, a `Where` whose
//! pass-through is not the compared shape, and a runtime INT64 root that is
//! NOT a guarded FLOAT truncation all keep their historical refusals verbatim.

use super::super::expand_identity::canonicalize_expand_identity_cones;
use super::super::schema_preflight::{
    validate_standard_schemas, validate_standard_schemas_unfolded,
};
use crate::onnx_proto::{
    attribute_type, tensor_shape_proto, AttributeProto, GraphProto, NodeProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use crate::WeightStore;
use std::collections::HashMap;

const FLOAT: i32 = 1;
const INT64: i32 = 7;

fn opset13() -> HashMap<String, i64> {
    HashMap::from([(String::new(), 13)])
}

fn node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|value| (*value).to_string()).collect(),
        output: outputs.iter().map(|value| (*value).to_string()).collect(),
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

fn int64_tensor(name: &str, dims: &[i64], values: &[i64]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: INT64,
        int64_data: values.to_vec(),
        ..Default::default()
    }
}

fn float_tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: FLOAT,
        float_data: values.to_vec(),
        ..Default::default()
    }
}

fn int64_value_attr(values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: "value".to_string(),
        r#type: attribute_type::TENSOR,
        t: Some(TensorProto {
            dims: vec![values.len() as i64],
            data_type: INT64,
            int64_data: values.to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn constant(name: &str, output: &str, values: &[i64]) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: "Constant".to_string(),
        output: vec![output.to_string()],
        attribute: vec![int64_value_attr(values)],
        ..Default::default()
    }
}

/// The runtime-shape expand idiom in miniature:
///
///     s    = Shape(x)
///     s2   = Shape(s)
///     ones = ConstantOfShape(s2, value=INT64 [1])
///     neg  = Mul(ones, Constant([-1]))
///     eq   = Equal(s, neg)                 -- constant all-false
///     w    = Where(eq, ones, s)            -- pass-through of s
///     out  = Gather(wdata, w)              -- keeps the chain live
fn expand_identity_graph(negative_value: i64, where_pass_through: &str) -> GraphProto {
    let mut constant_of_shape = node("ConstantOfShape", "make_ones", &["s2"], &["ones"]);
    constant_of_shape.attribute = vec![int64_value_attr(&[1])];
    GraphProto {
        input: vec![info("x", FLOAT, &[2, 3])],
        initializer: vec![float_tensor("wdata", &[5], &[0.0, 0.1, 0.2, 0.3, 0.4])],
        node: vec![
            node("Shape", "shape_x", &["x"], &["s"]),
            node("Shape", "shape_s", &["s"], &["s2"]),
            constant_of_shape,
            constant("neg_one", "negk", &[negative_value]),
            node("Mul", "neg_ones", &["ones", "negk"], &["neg"]),
            node("Equal", "expand_eq", &["s", "neg"], &["eq"]),
            node(
                "Where",
                "expand_where",
                &["eq", "ones", where_pass_through],
                &["w"],
            ),
            node("Gather", "gather_out", &["wdata", "w"], &["out"]),
        ],
        output: vec![info("out", FLOAT, &[2])],
        ..Default::default()
    }
}

#[test]
fn expand_identity_idiom_certifies_defers_and_canonicalizes() {
    let graph = expand_identity_graph(-1, "s");
    let exemptions = validate_standard_schemas(&graph, &opset13(), false)
        .expect("the certified expand-identity cone must pass the preflight");

    // Without canonicalization the deferral re-raises the ORIGINAL refusal:
    // the admission cannot outlive its certificate.
    let error = exemptions
        .require_all_folded(&WeightStore::new(), &graph)
        .expect_err("an uncanonicalized certified cone must re-raise");
    assert!(
        error.to_string().contains("dtype 7"),
        "re-raise must carry the original dtype refusal, got: {error}"
    );

    let mut canonical = graph;
    canonicalize_expand_identity_cones(&mut canonical);
    let op_types: Vec<&str> = canonical
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(
        op_types,
        ["Shape", "Gather"],
        "the Where and the whole Equal cone must be deleted after consumer rewiring"
    );
    let gather = &canonical.node[1];
    assert_eq!(gather.name, "gather_out");
    assert_eq!(
        gather.input,
        vec!["wdata".to_string(), "s".to_string()],
        "the Where consumer must be rewired to the pass-through Shape output"
    );
    assert_tensor_names_are_continuous(&canonical);

    exemptions
        .require_all_folded(&WeightStore::new(), &canonical)
        .expect("canonicalization discharges every deferred refusal");
}

/// Every non-empty node input in `graph` must have a producer: a graph input,
/// an initializer, or another node's output.  This is exactly the continuity
/// the graph builder's `tensor_to_node` map later requires — a dangling name
/// here is the "Skipped op ... not found in tensor_to_node" defect.
fn assert_tensor_names_are_continuous(graph: &GraphProto) {
    let mut known: std::collections::HashSet<&str> = graph
        .input
        .iter()
        .map(|input| input.name.as_str())
        .chain(graph.initializer.iter().map(|init| init.name.as_str()))
        .collect();
    known.extend(
        graph
            .node
            .iter()
            .flat_map(|node| node.output.iter().map(String::as_str)),
    );
    for node in &graph.node {
        for input in node.input.iter().filter(|input| !input.is_empty()) {
            assert!(
                known.contains(input.as_str()),
                "node '{}' ({}) reads '{}', which has no producer after canonicalization",
                node.name,
                node.op_type,
                input
            );
        }
    }
}

/// #cctsdb wiring pin (patch-1's `Expand_94`/`Concat` cones): consumers of a
/// canonicalized `Where` — an `Expand` reading the erased output as its SHAPE
/// input, and a `Concat` — must resolve to the pass-through `Shape` output,
/// and no node may be left reading the erased name.
#[test]
fn expand_and_concat_consumers_of_canonicalized_where_read_the_pass_through() {
    let mut graph = expand_identity_graph(-1, "s");
    // Replace the Gather consumer with the shipped consumers: Expand takes the
    // Where output as its SHAPE input (input 1), Concat as a data input.
    graph.node.pop();
    graph.node.push(node(
        "Expand",
        "expand_consumer",
        &["src", "w"],
        &["expanded"],
    ));
    graph
        .node
        .push(node("Concat", "concat_consumer", &["w", "w"], &["stacked"]));
    graph.input.push(info("src", FLOAT, &[1]));
    graph.output = vec![info("expanded", FLOAT, &[2]), info("stacked", INT64, &[4])];

    canonicalize_expand_identity_cones(&mut graph);

    let expand = graph
        .node
        .iter()
        .find(|node| node.name == "expand_consumer")
        .expect("the Expand consumer must survive canonicalization");
    assert_eq!(
        expand.input,
        vec!["src".to_string(), "s".to_string()],
        "Expand's SHAPE input must be rewired from the erased Where output to the \
         pass-through Shape output"
    );
    let concat = graph
        .node
        .iter()
        .find(|node| node.name == "concat_consumer")
        .expect("the Concat consumer must survive canonicalization");
    assert_eq!(
        concat.input,
        vec!["s".to_string(), "s".to_string()],
        "every Concat reference to the erased Where output must be rewired"
    );
    assert!(
        graph.node.iter().all(|node| node.name != "expand_where"),
        "the rewired Where must be deleted"
    );
    assert!(
        graph
            .node
            .iter()
            .flat_map(|node| node.input.iter())
            .all(|input| input != "w"),
        "no consumer may still read the erased Where output"
    );
    assert_tensor_names_are_continuous(&graph);
}

/// A certified `Where` whose output is an authored graph output cannot be
/// erased (the name cannot move), so it must survive canonicalization intact
/// — and a surviving node is exactly what `require_all_folded` re-raises on
/// (pinned by `literal_cone`'s discharge test).  In the shipped pipeline the
/// structural preflight already refuses an INT64 graph output outright, so
/// this guard is defense-in-depth; it is exercised here directly.
#[test]
fn where_feeding_a_graph_output_survives_canonicalization() {
    let mut graph = expand_identity_graph(-1, "s");
    graph.node.pop(); // drop the Gather; 'w' becomes the authored output
    graph.output = vec![info("w", INT64, &[2])];

    canonicalize_expand_identity_cones(&mut graph);

    let surviving = graph
        .node
        .iter()
        .find(|node| node.name == "expand_where")
        .expect("a Where feeding an authored graph output must not be erased");
    assert_eq!(
        surviving.op_type, "Where",
        "the surviving node must stay a Where"
    );
    assert_eq!(
        surviving.input,
        vec!["eq".to_string(), "ones".to_string(), "s".to_string()],
        "the surviving Where must keep its authored inputs"
    );
    assert!(
        graph.node.iter().any(|node| node.name == "expand_eq"),
        "the Equal stays: the surviving Where still consumes it"
    );
    assert_tensor_names_are_continuous(&graph);
}

#[test]
fn equal_against_zero_stays_refused_verbatim() {
    // A dimension CAN be zero, so Equal(Shape, 0) is a real runtime comparison
    // and must keep the historical refusal.
    let graph = expand_identity_graph(0, "s");
    let error = validate_standard_schemas(&graph, &opset13(), false)
        .expect_err("Equal(Shape, 0) must not be certified");
    assert!(
        error
            .to_string()
            .contains("has comparison operands dtype 7"),
        "expected the verbatim dtype-7 comparison refusal, got: {error}"
    );
}

#[test]
fn runtime_versus_runtime_equal_stays_refused_verbatim() {
    let graph = GraphProto {
        input: vec![info("x", FLOAT, &[4])],
        node: vec![
            {
                let mut cast = node("Cast", "cast_a", &["x"], &["a"]);
                cast.attribute = vec![AttributeProto {
                    name: "to".to_string(),
                    r#type: attribute_type::INT,
                    i: Some(i64::from(INT64)),
                    ..Default::default()
                }];
                cast
            },
            {
                let mut cast = node("Cast", "cast_b", &["x"], &["b"]);
                cast.attribute = vec![AttributeProto {
                    name: "to".to_string(),
                    r#type: attribute_type::INT,
                    i: Some(i64::from(INT64)),
                    ..Default::default()
                }];
                cast
            },
            node("Equal", "runtime_eq", &["a", "b"], &["eq"]),
        ],
        ..Default::default()
    };
    let error = validate_standard_schemas(&graph, &opset13(), false)
        .expect_err("a genuinely runtime integer comparison must stay refused");
    assert!(
        error
            .to_string()
            .contains("has comparison operands dtype 7"),
        "expected the verbatim dtype-7 comparison refusal, got: {error}"
    );
}

#[test]
fn where_pass_through_must_be_the_compared_shape_operand() {
    // Where(eq, ones, ones): the condition is certified all-false, but the
    // selected value is NOT the compared Shape output, so the node is not a
    // pass-through of a certified operand and must keep its refusal.
    let graph = expand_identity_graph(-1, "ones");
    let error = validate_standard_schemas(&graph, &opset13(), false)
        .expect_err("a Where that does not pass through the compared Shape must stay refused");
    assert!(
        error.to_string().contains("has selected values dtype 7"),
        "expected the verbatim dtype-7 Where refusal, got: {error}"
    );
}

/// The trunc-gated runtime index chain in miniature: patch coordinates read
/// from the FLOAT input, truncated to INT64 (a guarded `Trunc` at lowering),
/// used as Slice bounds, then combined by exact integer arithmetic and
/// consumed at an integer-control position (Gather indices).
fn trunc_gated_index_graph(starts_producer: NodeProto, starts_value: &str) -> GraphProto {
    GraphProto {
        input: vec![info("x", FLOAT, &[12])],
        initializer: vec![
            int64_tensor("gidx", &[1], &[9]),
            int64_tensor("slice_data", &[4], &[0, 1, 2, 3]),
            int64_tensor("ends", &[1], &[3]),
            int64_tensor("axes", &[1], &[0]),
            int64_tensor("rshape", &[1], &[2]),
            int64_tensor("addk", &[1], &[1]),
            float_tensor("wdata", &[8], &[0.0; 8]),
        ],
        node: vec![
            starts_producer,
            node(
                "Slice",
                "coord_slice",
                &["slice_data", starts_value, "ends", "axes"],
                &["sl"],
            ),
            node("Reshape", "coord_reshape", &["sl", "rshape"], &["resh"]),
            node("Add", "coord_add", &["resh", "addk"], &["idx"]),
            node("Gather", "gather_out", &["wdata", "idx"], &["out"]),
        ],
        // Authored metadata for the runtime intermediates, as ORT shape
        // inference provides for the shipped models.
        #[cfg(feature = "onnx-value-info")]
        value_info: vec![info(starts_value, INT64, &[1]), info("sl", INT64, &[2])],
        output: vec![info("out", FLOAT, &[2])],
        ..Default::default()
    }
}

fn float_to_int64_cast_chain() -> Vec<NodeProto> {
    let mut cast = node("Cast", "coord_cast", &["coord_float"], &["starts"]);
    cast.attribute = vec![AttributeProto {
        name: "to".to_string(),
        r#type: attribute_type::INT,
        i: Some(i64::from(INT64)),
        ..Default::default()
    }];
    vec![
        node("Gather", "coord_read", &["x", "gidx"], &["coord_float"]),
        cast,
    ]
}

#[test]
fn trunc_gated_runtime_int64_index_chain_is_admitted() {
    let mut graph = trunc_gated_index_graph(
        node("Identity", "starts_identity", &["starts_pre"], &["starts"]),
        "starts",
    );
    // Replace the placeholder producer with the real Gather -> Cast chain.
    graph.node.remove(0);
    for (position, chain_node) in float_to_int64_cast_chain().into_iter().enumerate() {
        graph.node.insert(position, chain_node);
    }
    validate_standard_schemas_unfolded(&graph, &opset13()).expect(
        "a runtime INT64 chain rooted only in a guarded FLOAT truncation and terminating \
         at integer-control positions must be admitted",
    );
}

#[test]
fn runtime_int64_root_without_float_truncation_stays_refused_verbatim() {
    // Same chain, but the Slice starts come from ArgMax — a runtime INT64
    // producer that is NOT a guarded FLOAT truncation.  The static-source
    // audit must keep its historical refusal.
    let argmax = {
        let mut argmax = node("ArgMax", "coord_argmax", &["x"], &["starts"]);
        argmax.attribute = vec![AttributeProto {
            name: "keepdims".to_string(),
            r#type: attribute_type::INT,
            i: Some(1),
            ..Default::default()
        }];
        argmax
    };
    let graph = trunc_gated_index_graph(argmax, "starts");
    let error = validate_standard_schemas_unfolded(&graph, &opset13())
        .expect_err("a non-trunc-gated runtime INT64 root must stay refused");
    // Without authored value_info, Slice must refuse the unauthenticated
    // starts rank before the later static-source audit can run.
    let expected = if cfg!(feature = "onnx-value-info") {
        "is not an exact static shape-control producer"
    } else {
        "standard ONNX Slice node 'coord_slice' cannot authenticate rank of starts input 'starts' before folding"
    };
    assert!(
        error.to_string().contains(expected),
        "expected the verbatim refusal containing {expected:?}, got: {error}"
    );
}
