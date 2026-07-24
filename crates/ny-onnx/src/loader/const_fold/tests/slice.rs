// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::{assert_folded_tensor, attr_int, attr_ints, fold, node, tensor_value_info};
use crate::onnx_proto::GraphProto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

/// Slice with all-constant inputs should be folded. This eliminates
/// shape-computation Slice ops that would otherwise pass through to the
/// propagation engine with invalid axis-adjusted ranges.
#[test]
fn test_slice_constant_fold_basic() {
    // data = [10, 20, 30, 40, 50], starts=[1], ends=[4], axes=[0]
    // Expected output: [20, 30, 40]
    let data = ArrayD::from_shape_vec(IxDyn(&[5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let starts = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let ends = ArrayD::from_shape_vec(IxDyn(&[1]), vec![4.0]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_0",
            "Slice",
            &["data", "starts", "ends", "axes"],
            &["out"],
            vec![],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);
    weights.insert("starts".to_string(), starts);
    weights.insert("ends".to_string(), ends);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Slice with all-constant inputs should be folded (#3206)");
    assert_eq!(out.shape(), &[3]);
    assert!((out[0] - 20.0).abs() < 1.0e-6);
    assert!((out[1] - 30.0).abs() < 1.0e-6);
    assert!((out[2] - 40.0).abs() < 1.0e-6);
}

/// Opset 1-9 Slice encodes starts/ends/axes as plural INTS attributes rather
/// than separate input tensors. Constant folding should still eliminate shape
/// chains that use the older attribute format.
#[test]
fn test_slice_constant_fold_opset9_attributes() {
    let data = ArrayD::from_shape_vec(IxDyn(&[5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_attr",
            "Slice",
            &["data"],
            &["out"],
            vec![
                attr_ints("starts", &[1]),
                attr_ints("ends", &[4]),
                attr_ints("axes", &[0]),
            ],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("opset 9 Slice attributes should fold");
    assert_eq!(out.shape(), &[3]);
    assert!((out[0] - 20.0).abs() < 1.0e-6);
    assert!((out[1] - 30.0).abs() < 1.0e-6);
    assert!((out[2] - 40.0).abs() < 1.0e-6);
}

/// vit_2023's opset-9 Slice sits on top of `Shape(Conv(...))`, so const-fold
/// needs basic Conv shape inference to materialize the Shape output first.
#[test]
fn test_shape_slice_constant_fold_through_conv_shape_inference_vit_chain() {
    let graph = GraphProto {
        input: vec![tensor_value_info("input", &[1, 3, 32, 32])],
        node: vec![
            node(
                "projection",
                "Conv",
                &["input", "projection_weight"],
                &["conv_out"],
                vec![
                    attr_ints("kernel_shape", &[8, 8]),
                    attr_ints("pads", &[0, 0, 0, 0]),
                    attr_ints("strides", &[8, 8]),
                    attr_ints("dilations", &[1, 1]),
                ],
            ),
            node("shape", "Shape", &["conv_out"], &["shape_out"], vec![]),
            node(
                "slice_attr",
                "Slice",
                &["shape_out"],
                &["out"],
                vec![
                    attr_ints("starts", &[0]),
                    attr_ints("ends", &[2]),
                    attr_ints("axes", &[0]),
                ],
            ),
        ],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert(
        "projection_weight".to_string(),
        ArrayD::zeros(IxDyn(&[48, 3, 8, 8])),
    );

    fold(&graph, &mut weights);

    let shape_out = weights
        .get("shape_out")
        .expect("Shape(conv_out) should fold through Conv shape inference");
    assert_eq!(shape_out.shape(), &[4]);
    assert!((shape_out[0] - 1.0).abs() < 1.0e-6);
    assert!((shape_out[1] - 48.0).abs() < 1.0e-6);
    assert!((shape_out[2] - 4.0).abs() < 1.0e-6);
    assert!((shape_out[3] - 4.0).abs() < 1.0e-6);

    let out = weights
        .get("out")
        .expect("opset 9 Slice on Shape(conv_out) should fold");
    assert_eq!(out.shape(), &[2]);
    assert!((out[0] - 1.0).abs() < 1.0e-6);
    assert!((out[1] - 48.0).abs() < 1.0e-6);
}

/// Slice with end > dim should clamp (ONNX INT64_MAX sentinel).
#[test]
fn test_slice_constant_fold_end_clamp() {
    // data = [10, 20, 30], starts=[1], ends=[i64::MAX as f32], axes=[0]
    // Expected: clamp end to 3, output = [20, 30]
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let starts = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    // i64::MAX as f32 saturates to a large float
    let ends = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_0",
            "Slice",
            &["data", "starts", "ends", "axes"],
            &["out"],
            vec![],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);
    weights.insert("starts".to_string(), starts);
    weights.insert("ends".to_string(), ends);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Slice with +inf end should fold and clamp to the input length");
    assert_eq!(out.shape(), &[2]);
    assert!((out[0] - 20.0).abs() < 1.0e-6);
    assert!((out[1] - 30.0).abs() < 1.0e-6);
}

/// Slice bounds that lost their original integer type during constant folding
/// should still truncate the same way the converter does (#3500).
#[test]
fn test_slice_constant_fold_truncates_float_bounds_3500() {
    let data = ArrayD::from_shape_vec(IxDyn(&[5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let starts = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let ends = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.5]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_truncate",
            "Slice",
            &["data", "starts", "ends", "axes"],
            &["out"],
            vec![],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);
    weights.insert("starts".to_string(), starts);
    weights.insert("ends".to_string(), ends);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Slice with truncated float bounds should fold");
    assert_eq!(out.shape(), &[2]);
    assert!((out[0] - 10.0).abs() < 1.0e-6);
    assert!((out[1] - 20.0).abs() < 1.0e-6);
}

/// Slice in a 2D tensor along axis 1.
#[test]
fn test_slice_constant_fold_2d_axis1() {
    // data = [[1, 2, 3, 4], [5, 6, 7, 8]], starts=[1], ends=[3], axes=[1]
    // Expected: [[2, 3], [6, 7]]
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap();
    let starts = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let ends = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_0",
            "Slice",
            &["data", "starts", "ends", "axes"],
            &["out"],
            vec![],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);
    weights.insert("starts".to_string(), starts);
    weights.insert("ends".to_string(), ends);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("Slice 2D axis=1 should fold");
    assert_eq!(out.shape(), &[2, 2]);
    assert!((out[[0, 0]] - 2.0).abs() < 1.0e-6);
    assert!((out[[0, 1]] - 3.0).abs() < 1.0e-6);
    assert!((out[[1, 0]] - 6.0).abs() < 1.0e-6);
    assert!((out[[1, 1]] - 7.0).abs() < 1.0e-6);
}

/// Slice with negative step should fold a reversed constant range.
#[test]
fn test_slice_constant_fold_reverse_step() {
    // data = [10, 20, 30], starts=[-1], ends=[-4], axes=[0], steps=[-1]
    // Expected output: [30, 20, 10]
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let starts = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap();
    let ends = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-4.0]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let steps = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_reverse",
            "Slice",
            &["data", "starts", "ends", "axes", "steps"],
            &["out"],
            vec![],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);
    weights.insert("starts".to_string(), starts);
    weights.insert("ends".to_string(), ends);
    weights.insert("axes".to_string(), axes);
    weights.insert("steps".to_string(), steps);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("reverse Slice should fold");
    assert_eq!(out.shape(), &[3]);
    assert!((out[0] - 30.0).abs() < 1.0e-6);
    assert!((out[1] - 20.0).abs() < 1.0e-6);
    assert!((out[2] - 10.0).abs() < 1.0e-6);
}

fn avoice_pad_shape_chain_graph() -> GraphProto {
    GraphProto {
        node: vec![
            node(
                "constant_of_shape",
                "ConstantOfShape",
                &["shape_seed"],
                &["constant_fill"],
                Vec::new(),
            ),
            node(
                "concat",
                "Concat",
                &["prefix", "constant_fill"],
                &["concat_out"],
                vec![attr_int("axis", 0)],
            ),
            node(
                "reshape",
                "Reshape",
                &["concat_out", "reshape_shape"],
                &["reshape_out"],
                Vec::new(),
            ),
            node(
                "reverse_slice",
                "Slice",
                &["reshape_out", "starts", "ends", "axes", "steps"],
                &["slice_out"],
                Vec::new(),
            ),
            node(
                "transpose",
                "Transpose",
                &["slice_out"],
                &["transpose_out"],
                vec![attr_ints("perm", &[1, 0])],
            ),
            node(
                "flatten",
                "Reshape",
                &["transpose_out", "flatten_shape"],
                &["out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    }
}

fn avoice_pad_shape_chain_weights() -> WeightStore {
    let mut weights = WeightStore::new();
    weights.insert(
        "shape_seed".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![4.0]).unwrap(),
    );
    weights.insert(
        "prefix".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap(),
    );
    weights.insert(
        "reshape_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap(),
    );
    weights.insert(
        "starts".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
    );
    weights.insert(
        "ends".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![i64::MIN as f32]).unwrap(),
    );
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );
    weights.insert(
        "steps".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
    );
    weights.insert(
        "flatten_shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
    );
    weights
}

#[test]
fn test_slice_constant_fold_avoice_pad_shape_chain() {
    let graph = avoice_pad_shape_chain_graph();
    let mut weights = avoice_pad_shape_chain_weights();
    fold(&graph, &mut weights);

    assert_folded_tensor(&weights, "constant_fill", &[4], &[0.0, 0.0, 0.0, 0.0]);
    assert_folded_tensor(
        &weights,
        "concat_out",
        &[6],
        &[2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert_folded_tensor(
        &weights,
        "reshape_out",
        &[3, 2],
        &[2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert_folded_tensor(
        &weights,
        "slice_out",
        &[3, 2],
        &[0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
    );
    assert_folded_tensor(
        &weights,
        "transpose_out",
        &[2, 3],
        &[0.0, 0.0, 2.0, 0.0, 0.0, 2.0],
    );
    assert!(
        weights
            .get("transpose_out")
            .expect("transpose_out missing")
            .clone()
            .into_shape_with_order(IxDyn(&[6]))
            .is_ok(),
        "transpose_out should reshape to [6] directly"
    );
    let flatten = graph.node.last().expect("flatten node missing");
    assert!(
        super::super::ops::try_fold_all_const_node(flatten, &weights, false).is_some(),
        "final flatten node should be directly constant foldable"
    );
    assert_folded_tensor(&weights, "out", &[6], &[0.0, 0.0, 2.0, 0.0, 0.0, 2.0]);
}

/// Slice with start clamped to dim produces empty result.
/// This is the #3206 scenario: Slice(start=2) on dim of size 1.
#[test]
fn test_slice_constant_fold_empty_3206() {
    // data = [42.0] (size 1), starts=[2], ends=[10], axes=[0]
    // After clamping: start=1, end=1, length=0 → empty
    let data = ArrayD::from_shape_vec(IxDyn(&[1]), vec![42.0]).unwrap();
    let starts = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    let ends = ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();

    let graph = GraphProto {
        node: vec![node(
            "slice_0",
            "Slice",
            &["data", "starts", "ends", "axes"],
            &["out"],
            vec![],
        )],
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), data);
    weights.insert("starts".to_string(), starts);
    weights.insert("ends".to_string(), ends);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Slice with clamped-empty range should still fold (#3206)");
    assert_eq!(out.shape(), &[0], "empty slice should produce size-0 dim");
}
