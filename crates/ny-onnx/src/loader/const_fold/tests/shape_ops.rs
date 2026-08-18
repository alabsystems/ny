// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::{attr_float, attr_int, attr_tensor, fold, node, tensor_f32};
use crate::onnx_proto::GraphProto;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_constant_of_shape_default_fill() {
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing ConstantOfShape output");
    assert_eq!(out.shape(), &[2, 3]);
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn test_constant_of_shape_fill_value() {
    let value_tensor = tensor_f32("fill", &[], &[3.5]);
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            vec![attr_tensor("value", value_tensor)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing ConstantOfShape output");
    // Shape [1, 2] is squeezed to [2] by the unbatched-mode leading-1 squeeze
    // in fold_constant_nodes (added in W4-1151 for #3194). The fill value and
    // element count are preserved.
    assert_eq!(out.shape(), &[2]);
    assert!(out.iter().all(|v| (*v - 3.5).abs() < 1.0e-6));
}

#[test]
fn test_constant_of_shape_preserves_non_batch_singleton_axis_for_vit_cls_token_3760() {
    let graph = GraphProto {
        node: vec![
            node(
                "cos",
                "ConstantOfShape",
                &["shape"],
                &["cos_out"],
                Vec::new(),
            ),
            node(
                "add",
                "Add",
                &["cos_out", "cls_token"],
                &["add_out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.0, 48.0]).unwrap();
    let cls_token =
        ArrayD::from_shape_vec(IxDyn(&[48]), (0..48).map(|v| v as f32).collect()).unwrap();
    weights.insert("shape".to_string(), shape);
    weights.insert("cls_token".to_string(), cls_token);

    fold(&graph, &mut weights);

    let cos_out = weights.get("cos_out").expect("ConstantOfShape should fold");
    assert_eq!(cos_out.shape(), &[1, 48]);
    assert!(cos_out.iter().all(|v| *v == 0.0));

    let add_out = weights.get("add_out").expect("Add should fold");
    assert_eq!(add_out.shape(), &[1, 48]);
    assert!((add_out[[0, 0]] - 0.0).abs() < 1.0e-6);
    assert!((add_out[[0, 47]] - 47.0).abs() < 1.0e-6);
}

#[test]
fn test_constant_of_shape_rejects_non_schema_float_attribute() {
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            vec![attr_float("value", 2.25)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_constant_of_shape_integer_fill_is_not_lossily_folded() {
    let value_tensor = crate::onnx_proto::TensorProto {
        dims: Vec::new(),
        data_type: 7,
        name: "integer_fill".to_string(),
        int64_data: vec![16_777_217],
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            vec![attr_tensor("value", value_tensor)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    );
    fold(&graph, &mut weights);
    assert!(!weights.contains_key("out"));
}

/// The `Tensor.expand` idiom in ml4acopf_2024's plain models opens with
/// `ConstantOfShape([2], value = INT64 1)`. The fold must materialize it with
/// an EXACT i64 sidecar and the INT64 range, because the whole cone downstream
/// (`Mul` by -1, `Equal`, `Where`) is evaluated on the typed integer path and
/// declines rather than falling back to the lossy f32 view.
#[test]
fn test_constant_of_shape_exact_integer_fill_keeps_its_i64_payload() {
    let value_tensor = crate::onnx_proto::TensorProto {
        dims: Vec::new(),
        data_type: 7,
        name: "integer_fill".to_string(),
        int64_data: vec![1],
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            vec![attr_tensor("value", value_tensor)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    weights.insert(
        "shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    );
    fold(&graph, &mut weights);

    assert_eq!(
        weights
            .get("out")
            .map(|values| values.iter().copied().collect::<Vec<_>>()),
        Some(vec![1.0_f32, 1.0])
    );
    assert_eq!(
        weights
            .get_integers("out")
            .map(|values| values.iter().copied().collect::<Vec<_>>()),
        Some(vec![1_i64, 1]),
        "an INT64 fill must publish its exact integer sidecar, not only an f32 mirror"
    );
    assert_eq!(
        weights.get_integer_range("out"),
        Some((i64::MIN, i64::MAX)),
        "the authored INT64 range is what selects the exact integer fold path downstream"
    );
}

#[test]
fn test_constant_of_shape_from_constant_node() {
    let shape_tensor = tensor_f32("shape", &[2], &[2.0, 1.0]);
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
                &["out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let mut weights = WeightStore::new();

    fold(&graph, &mut weights);

    assert!(weights.contains_key("shape_out"));
    let out = weights.get("out").expect("missing ConstantOfShape output");
    assert_eq!(out.shape(), &[2, 1]);
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn test_constant_of_shape_rejects_negative_shape() {
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_constant_of_shape_rejects_non_scalar_value() {
    let value_tensor = tensor_f32("fill", &[2], &[1.0, 2.0]);
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            vec![attr_tensor("value", value_tensor)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_constant_of_shape_rejects_non_integer_shape() {
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.25]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_constant_fold_skips_multi_output_nodes() {
    let graph = GraphProto {
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out", "extra"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 1.0]).unwrap();
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
    assert!(!weights.contains_key("extra"));
}

#[test]
fn test_reshape_constant_fold_infers_and_copies_zero() {
    let graph = GraphProto {
        node: vec![node(
            "reshape",
            "Reshape",
            &["data", "shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, -1.0]).unwrap();
    weights.insert("data".to_string(), data.clone());
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Reshape output");
    assert_eq!(out.shape(), &[2, 3]);
    assert!(out
        .iter()
        .zip(data.iter())
        .all(|(a, b)| (*a - *b).abs() < 1.0e-6));
}

#[test]
fn test_reshape_constant_fold_rejects_double_infer() {
    let graph = GraphProto {
        node: vec![node(
            "reshape",
            "Reshape",
            &["data", "shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_reshape_constant_fold_rejects_allowzero_literal_zero() {
    let graph = GraphProto {
        node: vec![node(
            "reshape",
            "Reshape",
            &["data", "shape"],
            &["out"],
            vec![attr_int("allowzero", 1)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, -1.0]).unwrap();
    weights.insert("data".to_string(), data);
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    assert!(!weights.contains_key("out"));
}

#[test]
fn test_reshape_constant_fold_infers_zero_when_total_elems_zero() {
    let graph = GraphProto {
        node: vec![node(
            "reshape",
            "Reshape",
            &["data", "shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[0, 3]), Vec::new()).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, -1.0]).unwrap();
    weights.insert("data".to_string(), data.clone());
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Reshape output");
    assert_eq!(out.shape(), &[0, 0]);
    assert_eq!(out.len(), data.len());
}

#[test]
fn test_reshape_constant_fold_allowzero_allows_extra_zero_dims() {
    let graph = GraphProto {
        node: vec![node(
            "reshape",
            "Reshape",
            &["data", "shape"],
            &["out"],
            vec![attr_int("allowzero", 1)],
        )],
        ..Default::default()
    };
    let mut weights = WeightStore::new();
    let data = ArrayD::from_shape_vec(IxDyn(&[0, 3]), Vec::new()).unwrap();
    let shape = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, -1.0]).unwrap();
    weights.insert("data".to_string(), data.clone());
    weights.insert("shape".to_string(), shape);

    fold(&graph, &mut weights);

    let out = weights.get("out").expect("missing Reshape output");
    assert_eq!(out.shape(), &[0, 0, 0]);
    assert_eq!(out.len(), data.len());
}
