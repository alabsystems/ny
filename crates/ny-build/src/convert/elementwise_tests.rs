// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::ConvertContext;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::layers::{BoundPropagation, CompareOp};
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};

fn make_context() -> ConvertContext<'static> {
    make_context_with_weights(WeightStore::new())
}

fn make_context_with_weights(weights: WeightStore) -> ConvertContext<'static> {
    let weights = Box::leak(Box::new(weights));
    let tensor_shapes = Box::leak(Box::new(HashMap::new()));
    let constant_tensors = Box::leak(Box::new(HashSet::new()));
    ConvertContext::new(weights, tensor_shapes, constant_tensors)
}

fn make_context_with_shapes(tensor_shapes: HashMap<String, Vec<i64>>) -> ConvertContext<'static> {
    let weights = Box::leak(Box::new(WeightStore::new()));
    let tensor_shapes = Box::leak(Box::new(tensor_shapes));
    let constant_tensors = Box::leak(Box::new(HashSet::new()));
    ConvertContext::new(weights, tensor_shapes, constant_tensors)
}

#[test]
fn convert_softmax_adjusts_positive_axis_for_unbatched_mode_3499() {
    // Trailing-relative remap: ONNX axis=2 on a recorded rank-3 tensor maps
    // to -1, correct whether the runtime tensor kept its ONNX rank or had
    // its leading batch dim stripped (#pensieve ReduceSum no-op class).
    let shapes = HashMap::from([("x".to_string(), vec![1_i64, 4, 6])]);
    let ctx = make_context_with_shapes(shapes);
    let spec = LayerSpec {
        name: "/asp/Softmax".to_string(),
        layer_type: LayerType::Softmax,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(2))]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("softmax conversion should succeed")
        .expect("softmax layer should be converted");

    match layer {
        Layer::Softmax(softmax) => assert_eq!(softmax.axis, -1),
        other => panic!("expected Softmax layer, got {:?}", other),
    }
}

#[test]
fn convert_softmax_positive_axis_unknown_rank_keeps_legacy() {
    // Unknown recorded rank keeps the legacy `axis - 1` adjustment
    // (ny-synthesized-subgraph compatibility; see remap_axis_trailing).
    let ctx = make_context();
    let spec = LayerSpec {
        name: "/asp/Softmax".to_string(),
        layer_type: LayerType::Softmax,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(2))]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("softmax conversion should succeed")
        .expect("softmax layer should be converted");
    match layer {
        Layer::Softmax(softmax) => assert_eq!(softmax.axis, 1),
        other => panic!("expected Softmax layer, got {:?}", other),
    }
}

#[test]
fn convert_softmax_rejects_batch_axis_in_unbatched_mode_3499() {
    let shapes = HashMap::from([("x".to_string(), vec![1_i64, 4])]);
    let ctx = make_context_with_shapes(shapes);
    let spec = LayerSpec {
        name: "softmax".to_string(),
        layer_type: LayerType::Softmax,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(0))]),
    };

    let err = ctx
        .convert_elementwise(&spec)
        .expect_err("axis=0 should be rejected in unbatched mode");
    assert!(
        err.to_string().contains("batch dimension"),
        "expected batch-dimension error, got {err}"
    );
}

#[test]
fn convert_softmax_axis_0_rank1_genuine_data_axis_loads_and_shapes_correctly() {
    // A rank-1 (no batch axis) input `[4]` with ONNX axis=0 loads WITHOUT the
    // unbatched-mode error: `data_had_batch_axis == Some(false)` → SoftmaxLayer(0),
    // which normalizes over the genuine sole data axis (shape preserved).
    let shapes = HashMap::from([("x".to_string(), vec![4_i64])]);
    let ctx = make_context_with_shapes(shapes);
    let spec = LayerSpec {
        name: "softmax".to_string(),
        layer_type: LayerType::Softmax,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(0))]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("axis=0 Softmax on a genuine rank-1 data axis must load without error")
        .expect("softmax layer should be converted");

    let Layer::Softmax(softmax) = &layer else {
        panic!("expected Softmax layer, got {:?}", layer);
    };
    // Trailing-relative encoding: axis 0 of a rank-1 tensor is stored as -1,
    // which resolves to the same sole data axis at propagation time.
    assert_eq!(softmax.axis, -1);

    // SoftmaxLayer(-1) on rank-1 [4] normalizes over axis 0; output shape == input shape.
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0_f32; 4]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0_f32; 4]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let out = softmax.propagate_ibp(&input).expect("softmax ibp");
    assert_eq!(out.shape(), &[4]);
}

#[test]
fn convert_snake_reads_per_channel_alpha_tensor_4117() {
    let mut weights = WeightStore::new();
    weights.insert(
        "snake_alpha".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![0.5, 2.0]).expect("valid alpha"),
    );
    let ctx = make_context_with_weights(weights);
    let spec = LayerSpec {
        name: "snake".to_string(),
        layer_type: LayerType::Snake,
        inputs: vec!["x".to_string(), "snake_alpha".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("a".to_string(), AttributeValue::Float(99.0))]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("snake conversion should succeed")
        .expect("snake layer should be converted");

    match layer {
        Layer::Snake(snake) => {
            assert_eq!(snake.alpha().len(), 2);
            assert!((snake.alpha()[0] - 0.5).abs() < 1e-6);
            assert!((snake.alpha()[1] - 2.0).abs() < 1e-6);
        }
        other => panic!("expected Snake layer, got {:?}", other),
    }
}

#[test]
fn convert_triu_3x3_default_diagonal_4270() {
    let ctx = make_context_with_shapes(HashMap::from([("x".to_string(), vec![3, 3])]));
    let spec = LayerSpec {
        name: "triu".to_string(),
        layer_type: LayerType::Triu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("triu conversion should succeed")
        .expect("triu should convert to a layer");

    let Layer::MulConstant(layer) = layer else {
        panic!("expected MulConstant layer");
    };

    let expected_mask = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![
            1.0, 1.0, 1.0, //
            0.0, 1.0, 1.0, //
            0.0, 0.0, 1.0,
        ],
    )
    .expect("valid triangular mask");
    assert_eq!(layer.constant(), &expected_mask);
    assert_eq!(layer.input_shape(), Some(&[3, 3][..]));

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 3]), -1.0),
        ArrayD::from_elem(IxDyn(&[3, 3]), 1.0),
    )
    .expect("valid interval");
    let output = layer.propagate_ibp(&input).expect("IBP should succeed");

    let expected_lower = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![
            -1.0, -1.0, -1.0, //
            0.0, -1.0, -1.0, //
            0.0, 0.0, -1.0,
        ],
    )
    .expect("valid lower bounds");
    let expected_upper = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![
            1.0, 1.0, 1.0, //
            0.0, 1.0, 1.0, //
            0.0, 0.0, 1.0,
        ],
    )
    .expect("valid upper bounds");
    assert_eq!(output.lower(), &expected_lower);
    assert_eq!(output.upper(), &expected_upper);
}

#[test]
fn convert_tril_3x3_diagonal_neg1_4270() {
    let ctx = make_context_with_shapes(HashMap::from([("x".to_string(), vec![3, 3])]));
    let spec = LayerSpec {
        name: "tril".to_string(),
        layer_type: LayerType::Tril,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("diagonal".to_string(), AttributeValue::Int(-1))]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("tril conversion should succeed")
        .expect("tril should convert to a layer");

    let Layer::MulConstant(layer) = layer else {
        panic!("expected MulConstant layer");
    };

    let expected_mask = ArrayD::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0,
        ],
    )
    .expect("valid strict-lower mask");
    assert_eq!(layer.constant(), &expected_mask);
    assert_eq!(layer.input_shape(), Some(&[3, 3][..]));
}

#[test]
fn convert_triu_batched_4d_4270() {
    let ctx = make_context_with_shapes(HashMap::from([("x".to_string(), vec![2, 1, 3, 3])]));
    let spec = LayerSpec {
        name: "triu_batched".to_string(),
        layer_type: LayerType::Triu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("batched triu conversion should succeed")
        .expect("triu should convert to a layer");

    let Layer::MulConstant(layer) = layer else {
        panic!("expected MulConstant layer");
    };

    let base_mask = vec![
        1.0, 1.0, 1.0, //
        0.0, 1.0, 1.0, //
        0.0, 0.0, 1.0,
    ];
    let expected_mask = ArrayD::from_shape_vec(IxDyn(&[2, 1, 3, 3]), base_mask.repeat(2))
        .expect("valid batched triangular mask");
    assert_eq!(layer.constant(), &expected_mask);
    assert_eq!(layer.input_shape(), Some(&[2, 1, 3, 3][..]));
}

#[test]
fn convert_compare_with_scalar_rhs_constant_4269() {
    let mut weights = WeightStore::new();
    weights.insert(
        "threshold".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![3.5]).expect("scalar threshold"),
    );
    let ctx = make_context_with_weights(weights);
    let spec = LayerSpec {
        name: "greater_than".to_string(),
        layer_type: LayerType::Compare,
        inputs: vec!["x".to_string(), "threshold".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([(
            "compare_op".to_string(),
            AttributeValue::String("Gt".to_string()),
        )]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("compare conversion should succeed")
        .expect("compare should convert to a layer");

    let Layer::Compare(compare) = layer else {
        panic!("expected Compare layer");
    };
    assert!((compare.threshold - 3.5).abs() < 1.0e-6);
    assert_eq!(compare.op, CompareOp::Gt);
}

#[test]
fn convert_compare_flips_left_scalar_constant_4269() {
    let mut weights = WeightStore::new();
    weights.insert(
        "lhs".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![2.0]).expect("scalar lhs"),
    );
    let ctx = make_context_with_weights(weights);
    let spec = LayerSpec {
        name: "constant_gt_x".to_string(),
        layer_type: LayerType::Compare,
        inputs: vec!["lhs".to_string(), "x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([(
            "compare_op".to_string(),
            AttributeValue::String("Gt".to_string()),
        )]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("compare conversion should succeed")
        .expect("compare should convert to a layer");

    let Layer::Compare(compare) = layer else {
        panic!("expected Compare layer");
    };
    assert!((compare.threshold - 2.0).abs() < 1.0e-6);
    assert_eq!(compare.op, CompareOp::Lt);
}

#[test]
fn convert_compare_without_constants_uses_compare_tensor_4269() {
    let ctx = make_context();
    let spec = LayerSpec {
        name: "equal_tensor".to_string(),
        layer_type: LayerType::Compare,
        inputs: vec!["lhs".to_string(), "rhs".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([(
            "compare_op".to_string(),
            AttributeValue::String("Eq".to_string()),
        )]),
    };

    let layer = ctx
        .convert_elementwise(&spec)
        .expect("compare conversion should succeed")
        .expect("compare should convert to a layer");

    let Layer::CompareTensor(compare) = layer else {
        panic!("expected CompareTensor layer");
    };
    assert_eq!(compare.op, CompareOp::Eq);
}
