// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::metadata::validate_matmul_shapes;
use crate::onnx_proto::{AttributeProto, NodeProto};
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

fn make_node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        name: String::new(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: Some(value),
        ..Default::default()
    }
}

#[test]
fn test_validate_matmul_corrects_wrong_output_dim() {
    let nodes = vec![make_node("MatMul", &["a", "w"], &["y"])];
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[30, 98])));
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    shapes.insert("y".to_string(), vec![1, 30]);

    validate_matmul_shapes(&nodes, &weights, &mut shapes);

    assert_eq!(shapes["y"], vec![1, 98]);
}

#[test]
fn test_validate_gemm_trans_b_corrects_wrong_output_dim() {
    let mut node = make_node("Gemm", &["a", "w", "b"], &["y"]);
    node.attribute.push(make_int_attr("transB", 1));
    let nodes = vec![node];
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[98, 30])));
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    shapes.insert("y".to_string(), vec![1, 30]);

    validate_matmul_shapes(&nodes, &weights, &mut shapes);

    assert_eq!(shapes["y"], vec![1, 98]);
}

#[test]
fn test_validate_gemm_no_trans_b_corrects_wrong_output_dim() {
    let nodes = vec![make_node("Gemm", &["a", "w", "b"], &["y"])];
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[30, 98])));
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    shapes.insert("y".to_string(), vec![1, 30]);

    validate_matmul_shapes(&nodes, &weights, &mut shapes);

    assert_eq!(shapes["y"], vec![1, 98]);
}

#[test]
fn test_validate_matmul_no_change_when_correct() {
    let nodes = vec![make_node("MatMul", &["a", "w"], &["y"])];
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), ArrayD::zeros(IxDyn(&[30, 98])));
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    shapes.insert("y".to_string(), vec![1, 98]);

    validate_matmul_shapes(&nodes, &weights, &mut shapes);

    assert_eq!(shapes["y"], vec![1, 98]);
}

#[test]
fn test_validate_matmul_skips_non_weight_input() {
    let nodes = vec![make_node("MatMul", &["a", "b"], &["y"])];
    let weights = WeightStore::new();
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    shapes.insert("y".to_string(), vec![1, 30]);

    validate_matmul_shapes(&nodes, &weights, &mut shapes);

    assert_eq!(shapes["y"], vec![1, 30]);
}

#[test]
fn test_validate_matmul_corrects_multiple_layers() {
    let nodes = vec![
        make_node("MatMul", &["input", "w1"], &["h1"]),
        make_node("MatMul", &["h1", "w2"], &["output"]),
    ];
    let mut weights = WeightStore::new();
    weights.insert("w1".to_string(), ArrayD::zeros(IxDyn(&[30, 98])));
    weights.insert("w2".to_string(), ArrayD::zeros(IxDyn(&[98, 2])));
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    shapes.insert("h1".to_string(), vec![1, 30]);
    shapes.insert("output".to_string(), vec![1, 30]);

    validate_matmul_shapes(&nodes, &weights, &mut shapes);

    assert_eq!(shapes["h1"], vec![1, 98]);
    assert_eq!(shapes["output"], vec![1, 2]);
}
