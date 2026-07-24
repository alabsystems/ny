// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::{attr_ints, fold, node};
use crate::onnx_proto::GraphProto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_transpose_rejects_negative_perm() {
    let graph = GraphProto {
        node: vec![node(
            "t",
            "Transpose",
            &["data"],
            &["out"],
            vec![attr_ints("perm", &[-1, 0])],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    weights.insert("data".to_string(), data);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_transpose_rejects_out_of_range_perm() {
    let graph = GraphProto {
        node: vec![node(
            "t",
            "Transpose",
            &["data"],
            &["out"],
            vec![attr_ints("perm", &[0, 5])],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    weights.insert("data".to_string(), data);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_unsqueeze_sorts_axes_before_insertion() {
    // axes=[2, 0] unsorted: output rank = 2 + 2 = 4.
    // Sorted axes=[0, 2] → insert dim 1 at position 0 then at position 2.
    // Result: [1, 2, 1, 3]
    let graph = GraphProto {
        node: vec![node(
            "unsq",
            "Unsqueeze",
            &["data"],
            &["out"],
            vec![attr_ints("axes", &[2, 0])],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    weights.insert("data".to_string(), data);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Unsqueeze output");
    assert_eq!(out.shape(), &[1, 2, 1, 3]);
}

#[test]
fn test_unsqueeze_rejects_duplicate_axes() {
    let graph = GraphProto {
        node: vec![node(
            "unsq",
            "Unsqueeze",
            &["data"],
            &["out"],
            vec![attr_ints("axes", &[0, 0])],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    weights.insert("data".to_string(), data);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_unsqueeze_attribute_negative_axis_uses_output_rank() {
    // Opset 11 style: axes as attribute.
    // Input rank=2, axes=[-1, 0] => output rank=4.
    // Resolve -1 against output rank => 3, sort to [0, 3].
    // Result shape: [1, 2, 3, 1]
    let graph = GraphProto {
        node: vec![node(
            "unsq",
            "Unsqueeze",
            &["data"],
            &["out"],
            vec![attr_ints("axes", &[-1, 0])],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    weights.insert("data".to_string(), data);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Unsqueeze output");
    assert_eq!(out.shape(), &[1, 2, 3, 1]);
}

#[test]
fn test_unsqueeze_input_axes_negative_axis_opset13() {
    // Opset 13+ style: axes as second input tensor.
    // Input rank=2, axes=[-1, 0] => output rank=4.
    // Resolve to [3, 0], sort to [0, 3], shape => [1, 2, 3, 1].
    let graph = GraphProto {
        node: vec![node(
            "unsq",
            "Unsqueeze",
            &["data", "axes"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Unsqueeze output");
    assert_eq!(out.shape(), &[1, 2, 3, 1]);
}

#[test]
fn test_unsqueeze_input_axes_rejects_out_of_range_negative_axis() {
    // Input rank=2, one axis => output rank=3; valid range is [-3, 2].
    // Axis -4 is out of range and must be rejected.
    let graph = GraphProto {
        node: vec![node(
            "unsq",
            "Unsqueeze",
            &["data", "axes"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let axes = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-4.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("axes".to_string(), axes);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_expand_constant_fold() {
    let graph = GraphProto {
        node: vec![node(
            "expand",
            "Expand",
            &["data", "shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Expand output");
    assert_eq!(out.shape(), &[2, 3]);
    assert!((out[[0, 0]] - 1.0).abs() < 1.0e-6);
    assert!((out[[1, 2]] - 3.0).abs() < 1.0e-6);
}

#[test]
fn test_expand_onnx_semantics_shape_mismatch() {
    // ONNX Expand: output_shape = element-wise max of right-aligned(data_shape, target_shape)
    // data shape (20,) + target shape (1, 1) → output shape (1, 20)
    // This is the actual ml4acopf pattern where Expand receives a shape that
    // is smaller than the data in some dimensions.
    let graph = GraphProto {
        node: vec![node(
            "expand",
            "Expand",
            &["data", "shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Expand should fold with ONNX semantics");
    // output_shape = max(right-aligned (1,3), (1,1)) = (1, 3) → squeeze → (3,)
    assert_eq!(out.shape(), &[3]);
    assert!((out[[0]] - 10.0).abs() < 1.0e-6);
    assert!((out[[1]] - 20.0).abs() < 1.0e-6);
    assert!((out[[2]] - 30.0).abs() < 1.0e-6);
}

/// Regression test for #3602: Expand with -1 in target shape (ONNX convention
/// meaning "use data dimension"). The ml4acopf model produces target shape
/// [1, -1] via ConstantOfShape→Mul→Equal→Where chain. After broadcast [1, 20]
/// the leading-1 is squeezed to [20] for unbatched-mode compatibility.
#[test]
fn test_expand_negative_one_target_shape_3602() {
    // data shape [20] + target shape [1, -1] → broadcast [1, 20] → squeeze [20]
    let graph = GraphProto {
        node: vec![node(
            "expand",
            "Expand",
            &["data", "shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    let data = ArrayD::from_shape_vec(IxDyn(&[20]), data).unwrap();
    // Target shape [1, -1] stored as f32: -1.0 means "use data dimension"
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights
        .get("out")
        .expect("Expand with -1 target should fold (#3602)");
    // Leading-1 squeezed: [1, 20] → [20] for unbatched mode
    assert_eq!(out.shape(), &[20]);
    for i in 0..20 {
        assert!(
            (out[[i]] - (i + 1) as f32).abs() < 1.0e-6,
            "element [{}] = {} (expected {})",
            i,
            out[[i]],
            i + 1
        );
    }
}
