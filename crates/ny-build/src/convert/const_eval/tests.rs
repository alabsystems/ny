// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::ConvertContext;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

#[test]
fn evaluate_constant_layer_concatenates_constant_inputs_3499() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "a".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    );
    weights.insert(
        "b".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );

    let spec = LayerSpec {
        name: "concat".to_string(),
        layer_type: LayerType::Concat,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(0))]),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("concat constant evaluation should succeed");
    assert_eq!(output.shape(), &[4]);
    assert_eq!(
        output.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn evaluate_constant_layer_slices_constant_input_3499() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    );
    weights.insert(
        "starts".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    );
    weights.insert(
        "ends".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    );
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );

    let spec = LayerSpec {
        name: "slice".to_string(),
        layer_type: LayerType::Slice,
        inputs: vec![
            "data".to_string(),
            "starts".to_string(),
            "ends".to_string(),
            "axes".to_string(),
        ],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("slice constant evaluation should succeed");
    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![20.0, 30.0]);
}

#[test]
fn evaluate_constant_layer_adds_both_frozen_constants_3937() {
    let tensor_shapes = HashMap::new();
    let weights = WeightStore::new();
    let constant_tensors = HashSet::from(["cos".to_string(), "sin".to_string()]);
    let evaluated = HashMap::from([
        (
            "cos".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ),
        (
            "sin".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
        ),
    ]);
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );

    let spec = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["cos".to_string(), "sin".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("Add with two frozen constant values should be folded");
    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![4.0, 6.0]);
}

fn transpose_spec(input: &str, perm: Vec<i64>) -> LayerSpec {
    LayerSpec {
        name: "transpose".to_string(),
        layer_type: LayerType::Transpose,
        inputs: vec![input.to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::from([("perm".to_string(), AttributeValue::Ints(perm))]),
    }
}

/// vit_2023: a rank-1 {48}-style constant fed to a `Transpose perm={0,2,1}`
/// previously PANICKED (`permuted_axes` index out of bounds). It must now
/// normalize the over-ranked perm to the identity and return the input
/// unchanged (transpose of a rank-1 tensor is the identity).
#[test]
fn evaluate_transpose_rank1_overranked_perm_is_identity() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    let data = vec![10.0, 20.0, 30.0];
    weights.insert(
        "emb".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), data.clone()).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );
    let spec = transpose_spec("emb", vec![0, 2, 1]);
    let out = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("rank-1 transpose must fold (identity), not panic");
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.iter().copied().collect::<Vec<_>>(), data);
}

/// A `perm={0,2,1}` authored for a batched rank-3 tensor, applied to the
/// unbatched rank-2 constant, must normalize to `{1,0}` and transpose correctly.
#[test]
fn evaluate_transpose_rank2_batchstripped_perm_transposes() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    // [[1,2,3],[4,5,6]] shape [2,3] -> transpose -> [[1,4],[2,5],[3,6]] shape [3,2].
    weights.insert(
        "m".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );
    let spec = transpose_spec("m", vec![0, 2, 1]);
    let out = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("rank-3 perm on rank-2 constant must normalize to {1,0} and fold");
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(
        out.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}
