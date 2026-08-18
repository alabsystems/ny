// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::{
    assert_folded_tensor, attr_int, attr_tensor, fold, node, tensor_f32, tensor_value_info,
};
use crate::onnx_proto::GraphProto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_lsnc_chain_stops_before_inexact_constant_add() {
    // Tests the chain: Constant -> MatMul(with weight) -> Add(with bias) -> Relu
    // This is the exact pattern from the lsnc quadrotor2d model
    let const_tensor = tensor_f32("const_val", &[1, 2], &[1.0, -1.0]);
    let graph = GraphProto {
        node: vec![
            node(
                "const",
                "Constant",
                &[],
                &["const_out"],
                vec![attr_tensor("value", const_tensor)],
            ),
            node(
                "matmul",
                "MatMul",
                &["const_out", "weight"],
                &["mm_out"],
                Vec::new(),
            ),
            node("add", "Add", &["bias", "mm_out"], &["add_out"], Vec::new()),
            node("relu", "Relu", &["add_out"], &["relu_out"], Vec::new()),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    // Weight: 2x3 matrix
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.5, 0.0, 1.0, -0.5]).unwrap();
    // Bias: 3-element vector
    let bias = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap();
    weights.insert("weight".to_string(), weight);
    weights.insert("bias".to_string(), bias);

    fold(&graph, &mut weights);

    assert!(
        weights.contains_key("const_out"),
        "Constant should be folded"
    );
    assert!(weights.contains_key("mm_out"), "MatMul should be folded");
    assert!(
        !weights.contains_key("add_out"),
        "1.0 + authored f32(0.1) is not exactly binary32 and must remain explicit"
    );
    assert!(
        !weights.contains_key("relu_out"),
        "the dependent ReLU cannot fold after the certified chain stops"
    );
}

#[test]
fn test_ml4acopf_chain_equal_where_expand() {
    // Tests the chain: ConstantOfShape -> Mul -> Equal -> Where -> Expand
    // This is the exact pattern from the ml4acopf model
    let shape_tensor = tensor_f32("shape_val", &[2], &[1.0, 3.0]);
    let graph = GraphProto {
        node: vec![
            node(
                "shape_const",
                "Constant",
                &[],
                &["shape_out"],
                vec![attr_tensor("value", shape_tensor)],
            ),
            node(
                "cos",
                "ConstantOfShape",
                &["shape_out"],
                &["cos_out"],
                Vec::new(),
            ),
            node(
                "mul",
                "Mul",
                &["cos_out", "scale"],
                &["mul_out"],
                Vec::new(),
            ),
            node(
                "eq",
                "Equal",
                &["ref_val", "mul_out"],
                &["eq_out"],
                Vec::new(),
            ),
            node(
                "wh",
                "Where",
                &["eq_out", "cos_out", "ref_val"],
                &["where_out"],
                Vec::new(),
            ),
            node(
                "expand",
                "Expand",
                &["data_to_expand", "where_out"],
                &["expand_out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    // Scale of 2.0: ConstantOfShape(zeros)*2 = zeros. ref_val != zeros -> Where selects ref_val.
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    // ref_val represents the desired shape dimensions [1, 2, 3]
    let ref_val = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    let data = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![42.0]).unwrap();
    weights.insert("scale".to_string(), scale);
    weights.insert("ref_val".to_string(), ref_val);
    weights.insert("data_to_expand".to_string(), data);

    fold(&graph, &mut weights);

    assert!(
        weights.contains_key("cos_out"),
        "ConstantOfShape should fold"
    );
    assert!(weights.contains_key("mul_out"), "Mul should fold");
    assert!(weights.contains_key("eq_out"), "Equal should fold");
    assert!(weights.contains_key("where_out"), "Where should fold");
    // Where selects ref_val [1,2,3] since condition is false (0!=1, 0!=2, 0!=3).
    // Expand broadcasts [1,1,1] to shape [1,2,3] → squeeze leading-1 → [2,3].
    let expand_out = weights.get("expand_out").expect("Expand should fold");
    assert_eq!(expand_out.shape(), &[2, 3]);
}

/// Build the ml4acopf ConstantOfShape→Mul→Equal→Where→Expand graph chain.
/// Constant([2])→COS→[0,0]→Mul(×-1)→[0,0]→Equal([1,-1])→Where→[1,-1]→Expand
fn ml4acopf_neg1_expand_graph() -> (GraphProto, WeightStore) {
    let shape_tensor = tensor_f32("shape_val", &[1], &[2.0]);
    let graph = GraphProto {
        node: vec![
            node(
                "shape_const",
                "Constant",
                &[],
                &["shape_out"],
                vec![attr_tensor("value", shape_tensor)],
            ),
            node(
                "cos",
                "ConstantOfShape",
                &["shape_out"],
                &["cos_out"],
                Vec::new(),
            ),
            node(
                "mul",
                "Mul",
                &["cos_out", "neg_scale"],
                &["mul_out"],
                Vec::new(),
            ),
            node(
                "eq",
                "Equal",
                &["ref_shape", "mul_out"],
                &["eq_out"],
                Vec::new(),
            ),
            node(
                "wh",
                "Where",
                &["eq_out", "cos_out", "ref_shape"],
                &["where_out"],
                Vec::new(),
            ),
            node(
                "expand",
                "Expand",
                &["data_to_expand", "where_out"],
                &["expand_out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "neg_scale".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
    );
    weights.insert(
        "ref_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).unwrap(),
    );
    let data: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    weights.insert(
        "data_to_expand".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[20]), data).unwrap(),
    );
    (graph, weights)
}

/// Regression: full ml4acopf chain Where→[1,-1]→Expand→squeeze to [20] (#3602).
#[test]
fn test_ml4acopf_chain_negative_one_expand_3602() {
    let (graph, mut weights) = ml4acopf_neg1_expand_graph();
    fold(&graph, &mut weights);

    assert!(
        weights.contains_key("cos_out"),
        "ConstantOfShape should fold"
    );
    assert!(weights.contains_key("mul_out"), "Mul should fold");
    assert!(weights.contains_key("eq_out"), "Equal should fold");
    assert!(weights.contains_key("where_out"), "Where should fold");
    // Where selects ref_shape [1,-1] since condition is false (0!=1, 0!=-1).
    let where_out = weights.get("where_out").expect("Where should fold");
    assert_eq!(where_out.shape(), &[2]);
    assert!((where_out[[0]] - 1.0).abs() < 1.0e-6);
    assert!((where_out[[1]] - (-1.0)).abs() < 1.0e-6);
    // Expand: data [20] + shape [1, -1] → broadcast [1, 20] → squeeze [20]
    let expand_out = weights
        .get("expand_out")
        .expect("Expand with -1 should fold (#3602)");
    assert_eq!(expand_out.shape(), &[20]);
}

#[test]
fn test_reduce_prod_constant_fold_avoice_shape_count_chain_3499() {
    let gather_index = tensor_f32("gather_index", &[], &[2.0]);
    let graph = GraphProto {
        node: vec![
            node("shape", "Shape", &["activation"], &["shape_out"], vec![]),
            node(
                "index_const",
                "Constant",
                &[],
                &["gather_index"],
                vec![attr_tensor("value", gather_index)],
            ),
            node(
                "gather",
                "Gather",
                &["shape_out", "gather_index"],
                &["axis_size"],
                vec![attr_int("axis", 0)],
            ),
            node(
                "reduce_prod",
                "ReduceProd",
                &["axis_size"],
                &["frame_count"],
                vec![],
            ),
            node(
                "cast",
                "Cast",
                &["frame_count"],
                &["frame_count_f32"],
                vec![attr_int("to", 1)],
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("activation".to_string(), ArrayD::zeros(IxDyn(&[5, 7, 11])));

    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "axis_size", &[], &[11.0]);
    assert_folded_tensor(&weights, "frame_count", &[], &[11.0]);
    assert_folded_tensor(&weights, "frame_count_f32", &[], &[11.0]);
}

#[test]
fn test_shape_gather_constant_fold_through_reshape_shape_inference_3500() {
    let reshape_shape = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 2.0, 2.0]).unwrap();
    let gather_index = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    let graph = GraphProto {
        input: vec![tensor_value_info("activation", &[1, 4])],
        node: vec![
            node(
                "reshape",
                "Reshape",
                &["activation", "reshape_shape"],
                &["reshape_out"],
                vec![],
            ),
            node("shape", "Shape", &["reshape_out"], &["shape_out"], vec![]),
            node(
                "gather",
                "Gather",
                &["shape_out", "gather_index"],
                &["axis_size"],
                vec![attr_int("axis", 0)],
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("reshape_shape".to_string(), reshape_shape);
    weights.insert("gather_index".to_string(), gather_index);

    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "shape_out", &[3], &[1.0, 2.0, 2.0]);
    assert_folded_tensor(&weights, "axis_size", &[], &[2.0]);
}

#[test]
fn test_shape_gather_constant_fold_through_gemm_reshape_shape_inference_3500() {
    let gemm_weight = ArrayD::zeros(IxDyn(&[512, 128]));
    let gemm_bias = ArrayD::zeros(IxDyn(&[512]));
    let reshape_shape = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, -1.0, 1.0]).unwrap();
    let gather_index = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    let graph = GraphProto {
        input: vec![tensor_value_info("style", &[1, 128])],
        node: vec![
            node(
                "gemm",
                "Gemm",
                &["style", "fc_weight", "fc_bias"],
                &["gemm_out"],
                vec![attr_int("transB", 1)],
            ),
            node(
                "reshape",
                "Reshape",
                &["gemm_out", "reshape_shape"],
                &["reshape_out"],
                vec![],
            ),
            node("shape", "Shape", &["reshape_out"], &["shape_out"], vec![]),
            node(
                "gather",
                "Gather",
                &["shape_out", "gather_index"],
                &["axis_size"],
                vec![attr_int("axis", 0)],
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert("fc_weight".to_string(), gemm_weight);
    weights.insert("fc_bias".to_string(), gemm_bias);
    weights.insert("reshape_shape".to_string(), reshape_shape);
    weights.insert("gather_index".to_string(), gather_index);

    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "shape_out", &[3], &[1.0, 512.0, 1.0]);
    assert_folded_tensor(&weights, "axis_size", &[], &[512.0]);
}
