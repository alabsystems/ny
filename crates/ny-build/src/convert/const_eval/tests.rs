// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::ConvertContext;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

fn constant_eval_context<'a>(
    weights: &'a WeightStore,
    tensor_shapes: &'a HashMap<String, Vec<i64>>,
    constant_tensors: &'a HashSet<String>,
    evaluated: &'a HashMap<String, ArrayD<f32>>,
) -> ConvertContext<'a> {
    ConvertContext::with_evaluated_constants(weights, tensor_shapes, constant_tensors, evaluated)
}

#[test]
fn exact_linear_constant_evaluation_accepts_certified_integer_dot_products() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![10.0, 20.0]).unwrap(),
    );
    weights.insert(
        "weight".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 0.0, 1.0]).unwrap(),
    );
    weights.insert(
        "bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 2.0]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = constant_eval_context(&weights, &tensor_shapes, &constant_tensors, &evaluated);
    let spec = LayerSpec {
        name: "exact_linear".to_string(),
        layer_type: LayerType::Linear,
        inputs: vec!["x".to_string(), "weight".to_string(), "bias".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("an exactly representable affine map should materialize");
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.as_slice().unwrap(), &[50.0, 22.0]);
}

#[test]
fn exact_linear_constant_evaluation_rejects_rounded_products() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.1]).unwrap(),
    );
    weights.insert(
        "weight".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.1]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = constant_eval_context(&weights, &tensor_shapes, &constant_tensors, &evaluated);
    let spec = LayerSpec {
        name: "inexact_linear".to_string(),
        layer_type: LayerType::Linear,
        inputs: vec!["x".to_string(), "weight".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("transB".to_string(), AttributeValue::Int(1))]),
    };

    assert!(ctx.evaluate_constant_layer(&spec, &evaluated).is_none());
}

#[test]
fn exact_conv_constant_evaluation_accepts_certified_integer_convolution() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );
    weights.insert(
        "kernel".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![2.0, 4.0, 6.0]).unwrap(),
    );
    weights.insert(
        "bias".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = constant_eval_context(&weights, &tensor_shapes, &constant_tensors, &evaluated);
    let spec = LayerSpec {
        name: "exact_conv".to_string(),
        // ONNX's rank-generic Conv is mapped to this compatibility variant;
        // convert_layer dispatches the rank-3 kernel to Conv1d.
        layer_type: LayerType::Conv2d,
        inputs: vec!["x".to_string(), "kernel".to_string(), "bias".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("an exactly representable convolution should materialize");
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.as_slice().unwrap(), &[28.0, 40.0]);
}

#[test]
fn exact_conv_constant_evaluation_rejects_rounded_products() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.1]).unwrap(),
    );
    weights.insert(
        "kernel".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![0.1]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = constant_eval_context(&weights, &tensor_shapes, &constant_tensors, &evaluated);
    let spec = LayerSpec {
        name: "inexact_conv".to_string(),
        layer_type: LayerType::Conv2d,
        inputs: vec!["x".to_string(), "kernel".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    assert!(ctx.evaluate_constant_layer(&spec, &evaluated).is_none());
}

#[test]
fn constant_erf_is_not_materialized_from_an_outward_ibp_enclosure() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
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
        name: "constant_erf".to_string(),
        layer_type: LayerType::Erf,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    assert!(
        ctx.evaluate_constant_layer(&spec, &evaluated).is_none(),
        "constant Erf must remain a graph operation unless a dedicated exact concrete evaluator is used"
    );
}

#[test]
fn constant_cast_is_identity_only_for_authenticated_float32_target() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "x".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.25]).unwrap(),
    );
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = constant_eval_context(&weights, &tensor_shapes, &constant_tensors, &evaluated);
    let mut spec = LayerSpec {
        name: "cast".to_string(),
        layer_type: LayerType::Cast,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("to".to_string(), AttributeValue::Int(1))]),
    };

    assert_eq!(
        ctx.evaluate_constant_layer(&spec, &evaluated)
            .expect("FLOAT32 Cast is identity")
            .as_slice()
            .unwrap(),
        &[1.25]
    );
    spec.attributes
        .insert("to".to_string(), AttributeValue::Int(7));
    assert!(ctx.evaluate_constant_layer(&spec, &evaluated).is_none());
    spec.attributes.clear();
    assert!(ctx.evaluate_constant_layer(&spec, &evaluated).is_none());
}

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
fn evaluate_constant_layer_rejects_fractional_discrete_operands() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    );
    weights.insert(
        "starts".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );
    weights.insert(
        "ends".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::from_bits(2.0_f32.to_bits() + 1)]).unwrap(),
    );
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
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
        name: "slice_fractional".to_string(),
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
    assert!(
        ctx.evaluate_constant_layer(&spec, &evaluated).is_none(),
        "an adjacent fractional Slice end must not be truncated"
    );
}

#[test]
fn evaluate_gather_prefers_exact_integer_payload() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap(),
    );
    weights.insert(
        "indices".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );
    weights.insert_integers(
        "indices".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1_i64]).unwrap(),
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
        name: "gather_exact".to_string(),
        layer_type: LayerType::Gather,
        inputs: vec!["data".to_string(), "indices".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(0))]),
    };
    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("exact integer Gather should fold");
    assert_eq!(output.as_slice().unwrap(), &[20.0]);
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

#[test]
fn evaluate_shape_normalizes_negative_end() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::from([("data".to_string(), vec![2, 3, 4])]);
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );
    let spec = LayerSpec {
        name: "shape_prefix".to_string(),
        layer_type: LayerType::Shape,
        inputs: vec!["data".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::from([("end".to_string(), AttributeValue::Int(-1))]),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("negative Shape end must normalize relative to rank");
    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
}

#[test]
fn evaluate_shape_declines_unresolved_dimensions() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::from([("data".to_string(), vec![1, -1])]);
    let constant_tensors = HashSet::new();
    let evaluated = HashMap::new();
    let ctx = ConvertContext::with_evaluated_constants(
        &weights,
        &tensor_shapes,
        &constant_tensors,
        &evaluated,
    );
    let spec = LayerSpec {
        name: "dynamic_shape".to_string(),
        layer_type: LayerType::Shape,
        inputs: vec!["data".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    assert!(ctx.evaluate_constant_layer(&spec, &evaluated).is_none());
}

#[test]
fn evaluate_squeeze_preserves_scalar_rank() {
    let tensor_shapes = HashMap::new();
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![7.0]).unwrap(),
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
        name: "squeeze_scalar".to_string(),
        layer_type: LayerType::Squeeze,
        inputs: vec!["data".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::from([("axes".to_string(), AttributeValue::Ints(vec![0]))]),
    };

    let output = ctx
        .evaluate_constant_layer(&spec, &evaluated)
        .expect("squeezing the sole unit dimension should produce a scalar");
    assert!(output.shape().is_empty());
    assert_eq!(output.first(), Some(&7.0));
}
