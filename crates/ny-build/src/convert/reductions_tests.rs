// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for reduction op conversion: the opset 13+/18+
//! axes-as-input-tensor path (Part of #3499) and the TRAILING-RELATIVE
//! batch-squeeze axis remap (#pensieve ReduceSum no-op miscompile).
//!
//! ny's internal runtime tensor for an ONNX tensor of rank `r` either kept
//! its ONNX rank (leading size-1 retained) or had its leading batch dim
//! stripped (rank `r-1`). Reduction axes are therefore stored NEGATIVE
//! (from-the-end): `onnx_axis - r`, resolved against the actual runtime rank
//! at propagation time — correct under both layouts. Out-of-range axes and
//! the batch axis 0 of a rank>1 tensor REFUSE conversion (fail-closed);
//! unknown recorded ranks keep the legacy `axis - 1` adjustment
//! (ny-synthesized-subgraph compatibility).

use super::{AttributeValue, ConvertContext, LayerSpec};
use crate::WeightStore;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::{LayerType, NyError};
use ny_propagate::Layer;
use std::collections::{HashMap, HashSet};

/// Build a minimal ConvertContext for testing (no recorded shapes).
fn test_ctx(weights: &WeightStore) -> ConvertContext<'_> {
    static EMPTY_SHAPES: std::sync::OnceLock<HashMap<String, Vec<i64>>> =
        std::sync::OnceLock::new();
    static EMPTY_CONSTANTS: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    ConvertContext::new(
        weights,
        EMPTY_SHAPES.get_or_init(HashMap::new),
        EMPTY_CONSTANTS.get_or_init(HashSet::new),
    )
}

/// ConvertContext with a recorded ONNX shape for the data tensor "x".
fn test_ctx_with_x_shape<'a>(
    weights: &'a WeightStore,
    shapes: &'a HashMap<String, Vec<i64>>,
) -> ConvertContext<'a> {
    static EMPTY_CONSTANTS: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    ConvertContext::new(weights, shapes, EMPTY_CONSTANTS.get_or_init(HashSet::new))
}

fn x_shape(shape: &[i64]) -> HashMap<String, Vec<i64>> {
    HashMap::from([("x".to_string(), shape.to_vec())])
}

fn test_ctx_with_eval<'a>(
    weights: &'a WeightStore,
    shapes: &'a HashMap<String, Vec<i64>>,
    evaluated: &'a HashMap<String, ArrayD<f32>>,
) -> ConvertContext<'a> {
    static EMPTY_CONSTANTS: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    ConvertContext::with_evaluated_constants(
        weights,
        shapes,
        EMPTY_CONSTANTS.get_or_init(HashSet::new),
        evaluated,
    )
}

fn reduce_sum_spec_with_attr_axes(axes: Vec<i64>) -> LayerSpec {
    LayerSpec {
        name: "reduce_sum".to_string(),
        layer_type: LayerType::ReduceSum,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("axes".to_string(), AttributeValue::Ints(axes)),
            ("keepdims".to_string(), AttributeValue::Int(1)),
        ]),
    }
}

fn reduce_sum_spec_with_input_axes(axes_tensor_name: &str) -> LayerSpec {
    LayerSpec {
        name: "reduce_sum".to_string(),
        layer_type: LayerType::ReduceSum,
        inputs: vec!["x".to_string(), axes_tensor_name.to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("keepdims".to_string(), AttributeValue::Int(1))]),
    }
}

fn reduce_sum_spec_no_axes() -> LayerSpec {
    LayerSpec {
        name: "reduce_sum".to_string(),
        layer_type: LayerType::ReduceSum,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("keepdims".to_string(), AttributeValue::Int(1))]),
    }
}

fn reduce_mean_spec_with_input_axes(axes_tensor_name: &str) -> LayerSpec {
    LayerSpec {
        name: "reduce_mean".to_string(),
        layer_type: LayerType::ReduceMean,
        inputs: vec!["x".to_string(), axes_tensor_name.to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([("keepdims".to_string(), AttributeValue::Int(1))]),
    }
}

fn cumsum_spec_with_axis_input(
    axis_tensor_name: &str,
    exclusive: bool,
    reverse: bool,
) -> LayerSpec {
    LayerSpec {
        name: "cumsum".to_string(),
        layer_type: LayerType::CumSum,
        inputs: vec!["x".to_string(), axis_tensor_name.to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            (
                "exclusive".to_string(),
                AttributeValue::Int(exclusive as i64),
            ),
            ("reverse".to_string(), AttributeValue::Int(reverse as i64)),
        ]),
    }
}

fn cumsum_spec_missing_axis_input() -> LayerSpec {
    LayerSpec {
        name: "cumsum".to_string(),
        layer_type: LayerType::CumSum,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    }
}

fn logsumexp_spec_with_attr_axes(axes: Vec<i64>) -> LayerSpec {
    LayerSpec {
        name: "logsumexp".to_string(),
        layer_type: LayerType::LogSumExp,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("axes".to_string(), AttributeValue::Ints(axes)),
            ("keepdims".to_string(), AttributeValue::Int(1)),
        ]),
    }
}

// ---------------------------------------------------------------------------
// Path 1: Attribute-based axes (opset < 13 for ReduceSum, < 18 for others)
// ---------------------------------------------------------------------------

#[test]
fn convert_reduce_sum_attribute_axes_positive() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_attr_axes(vec![2]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    // ONNX axis=2 on rank-3 [1,4,6] → trailing-relative -1
    assert_eq!(reduce.axes, vec![-1i64]);
    assert!(reduce.keepdims);
}

#[test]
fn convert_reduce_sum_attribute_axes_multiple() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 2, 3, 4]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_attr_axes(vec![1, 3]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    // ONNX axes [1, 3] on rank-4 → trailing-relative [-3, -1]
    assert_eq!(reduce.axes, vec![-3i64, -1]);
}

#[test]
fn convert_reduce_sum_attribute_axes_negative() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_attr_axes(vec![-1]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    // Negative axes pass through unchanged (already trailing-relative)
    assert_eq!(reduce.axes, vec![-1i64]);
}

/// THE pensieve defect shape: ONNX `ReduceSum(axes=[1])` on a `[1, n]`
/// tensor must reduce the `n` elements under BOTH runtime layouts
/// (`[1, n]` retained or `[n]` stripped) — trailing-relative `-1`, NOT the
/// legacy internal `0` (a size-1-axis no-op on the retained layout).
#[test]
fn convert_reduce_sum_axis_one_on_rank2_is_trailing() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_attr_axes(vec![1]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![-1i64]);
}

// ---------------------------------------------------------------------------
// Fail-closed refusals
// ---------------------------------------------------------------------------

/// A positive axis with no recorded ONNX shape keeps the LEGACY `axis - 1`
/// adjustment: unrecorded tensors come from ny-synthesized internal
/// subgraphs (LSTM unrolling, ReduceL2 lowering) that were authored against
/// the stripped-batch convention — for them legacy is correct by
/// construction. Real ONNX models get recorded shapes from load-time shape
/// inference and take the trailing-relative path.
#[test]
fn convert_reduce_sum_positive_axis_unknown_rank_keeps_legacy() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_attr_axes(vec![2]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![1i64]);
}

/// ONNX axis 0 of a rank>1 tensor is the (possibly stripped) batch axis —
/// no single encoding is correct for both runtime layouts; refused.
#[test]
fn convert_reduce_sum_axis_zero_rank2_refused() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_attr_axes(vec![0]);

    let err = ctx.convert_reduce_sum(&spec).unwrap_err();
    assert!(
        matches!(err, NyError::UnsupportedOp(ref msg) if msg.contains("batch dimension")),
        "expected batch-axis refusal, got {err:?}"
    );
}

/// ONNX axis 0 of a rank-1 tensor is a genuine data axis → trailing -1.
#[test]
fn convert_reduce_sum_axis_zero_rank1_is_trailing() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_attr_axes(vec![0]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![-1i64]);
}

/// Out-of-range axis for the recorded rank — refused.
#[test]
fn convert_reduce_sum_axis_out_of_range_refused() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_attr_axes(vec![5]);

    let err = ctx.convert_reduce_sum(&spec).unwrap_err();
    assert!(
        err.to_string().contains("out of range"),
        "expected out-of-range error, got {err}"
    );
}

// ---------------------------------------------------------------------------
// Path 2: Input tensor axes (opset 13+ for ReduceSum, 18+ for others)
// ---------------------------------------------------------------------------

#[test]
fn convert_reduce_sum_opset13_input_tensor_via_weights() {
    let mut weights = WeightStore::new();
    weights.insert("axes_tensor".to_string(), arr1(&[2.0f32]).into_dyn());
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_sum_spec_with_input_axes("axes_tensor");

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    // ONNX axis=2 on rank-3 → trailing-relative -1
    assert_eq!(reduce.axes, vec![-1i64]);
    assert!(reduce.keepdims);
}

#[test]
fn convert_reduce_sum_opset13_input_tensor_via_evaluated_constants() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 2, 3, 4]);
    let mut evaluated = HashMap::new();
    evaluated.insert(
        "axes_const".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 3.0]).unwrap(),
    );
    let ctx = test_ctx_with_eval(&weights, &shapes, &evaluated);
    let spec = reduce_sum_spec_with_input_axes("axes_const");

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    // ONNX axes [1, 3] on rank-4 → trailing-relative [-3, -1]
    assert_eq!(reduce.axes, vec![-3i64, -1]);
}

#[test]
fn convert_reduce_sum_opset13_input_tensor_negative_axis() {
    let mut weights = WeightStore::new();
    weights.insert("neg_axes".to_string(), arr1(&[-1.0f32]).into_dyn());
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_input_axes("neg_axes");

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![-1i64]);
}

#[test]
fn convert_reduce_mean_opset18_input_tensor() {
    let mut weights = WeightStore::new();
    weights.insert("axes_tensor".to_string(), arr1(&[2.0f32]).into_dyn());
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_mean_spec_with_input_axes("axes_tensor");

    let layer = ctx.convert_reduce_mean(&spec).unwrap();
    let Layer::ReduceMean(reduce) = layer else {
        panic!("expected ReduceMean, got {layer:?}");
    };
    // ONNX axis=2 on rank-3 → trailing-relative -1
    assert_eq!(reduce.axes, vec![-1i64]);
    assert!(reduce.keepdims);
}

#[test]
fn convert_reduce_mean_axis_zero_rank2_refused() {
    let mut weights = WeightStore::new();
    weights.insert("axes_tensor".to_string(), arr1(&[0.0f32]).into_dyn());
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_mean_spec_with_input_axes("axes_tensor");

    let err = ctx.convert_reduce_mean(&spec).unwrap_err();
    assert!(
        matches!(err, NyError::UnsupportedOp(ref msg) if msg.contains("batch dimension")),
        "expected batch-axis refusal, got {err:?}"
    );
}

#[test]
fn convert_cumsum_input_tensor_positive_axis_and_flags() {
    let mut weights = WeightStore::new();
    weights.insert("axis_tensor".to_string(), arr1(&[2.0f32]).into_dyn());
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = cumsum_spec_with_axis_input("axis_tensor", true, false);

    let layer = ctx.convert_cumsum(&spec).unwrap();
    let Layer::CumSum(cumsum) = layer else {
        panic!("expected CumSum, got {layer:?}");
    };
    // ONNX axis=2 on rank-3 → trailing-relative -1
    assert_eq!(cumsum.axis, -1);
    assert!(cumsum.exclusive);
    assert!(!cumsum.reverse);
}

#[test]
fn convert_cumsum_input_tensor_negative_axis_passthrough() {
    let mut weights = WeightStore::new();
    weights.insert("axis_tensor".to_string(), arr1(&[-1.0f32]).into_dyn());
    let ctx = test_ctx(&weights);
    let spec = cumsum_spec_with_axis_input("axis_tensor", false, true);

    let layer = ctx.convert_cumsum(&spec).unwrap();
    let Layer::CumSum(cumsum) = layer else {
        panic!("expected CumSum, got {layer:?}");
    };
    assert_eq!(cumsum.axis, -1);
    assert!(!cumsum.exclusive);
    assert!(cumsum.reverse);
}

#[test]
fn convert_cumsum_missing_axis_input_returns_model_load() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = cumsum_spec_missing_axis_input();

    let err = ctx
        .convert_cumsum(&spec)
        .expect_err("missing CumSum axis should error");
    assert!(
        matches!(err, NyError::ModelLoad(ref msg) if msg.contains("requires a constant axis input")),
        "expected ModelLoad for missing axis input, got {err:?}"
    );
}

#[test]
fn convert_cumsum_dynamic_axis_input_returns_unsupported_configuration() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = cumsum_spec_with_axis_input("dynamic_axis", false, false);

    let err = ctx
        .convert_cumsum(&spec)
        .expect_err("dynamic CumSum axis should error");
    assert!(
        matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("constant tensor")),
        "expected UnsupportedConfiguration for dynamic axis input, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Path 2 edge: attributes take priority over input tensor
// ---------------------------------------------------------------------------

#[test]
fn convert_reduce_sum_attributes_take_priority_over_input() {
    let mut weights = WeightStore::new();
    weights.insert("axes_tensor".to_string(), arr1(&[99.0f32]).into_dyn());
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);

    // Spec has BOTH attributes AND input tensor — attributes should win.
    let mut spec = reduce_sum_spec_with_input_axes("axes_tensor");
    spec.attributes
        .insert("axes".to_string(), AttributeValue::Ints(vec![2]));

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    // Attribute axis=2 on rank-3 → -1, not input tensor axis=99
    assert_eq!(reduce.axes, vec![-1i64]);
}

// ---------------------------------------------------------------------------
// Path 3: No axes (reduce over all dimensions)
// ---------------------------------------------------------------------------

#[test]
fn convert_reduce_sum_no_axes_reduces_all() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_no_axes();

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert!(reduce.axes.is_empty(), "empty axes = reduce all dimensions");
}

#[test]
fn convert_reduce_sum_empty_attribute_axes_reduces_all() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_attr_axes(vec![]);

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert!(reduce.axes.is_empty(), "empty attribute axes = reduce all");
}

#[test]
fn convert_reduce_sum_missing_input_tensor_reduces_all() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_input_axes("nonexistent");

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert!(
        reduce.axes.is_empty(),
        "missing input tensor should fall through to reduce-all"
    );
}

// ---------------------------------------------------------------------------
// ReduceMax/ReduceMin/LogSumExp share the same axis remap. These tests catch
// divergence.
// ---------------------------------------------------------------------------

fn reduce_max_spec_with_attr_axes(axes: Vec<i64>) -> LayerSpec {
    LayerSpec {
        name: "reduce_max".to_string(),
        layer_type: LayerType::ReduceMax,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("axes".to_string(), AttributeValue::Ints(axes)),
            ("keepdims".to_string(), AttributeValue::Int(1)),
        ]),
    }
}

fn reduce_min_spec_with_attr_axes(axes: Vec<i64>) -> LayerSpec {
    LayerSpec {
        name: "reduce_min".to_string(),
        layer_type: LayerType::ReduceMin,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("axes".to_string(), AttributeValue::Ints(axes)),
            ("keepdims".to_string(), AttributeValue::Int(1)),
        ]),
    }
}

#[test]
fn convert_reduce_max_positive_axis_adjusted() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_max_spec_with_attr_axes(vec![2]);

    let layer = ctx.convert_reduce_max(&spec).unwrap();
    let Layer::ReduceMax(reduce) = layer else {
        panic!("expected ReduceMax, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![-1i64]);
    assert!(reduce.keepdims);
}

#[test]
fn convert_reduce_min_negative_axis_passthrough() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);
    let spec = reduce_min_spec_with_attr_axes(vec![-1]);

    let layer = ctx.convert_reduce_min(&spec).unwrap();
    let Layer::ReduceMin(reduce) = layer else {
        panic!("expected ReduceMin, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![-1i64]);
}

#[test]
fn convert_reduce_max_axis_zero_rank2_refused() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_max_spec_with_attr_axes(vec![0]);

    let err = ctx.convert_reduce_max(&spec).unwrap_err();
    assert!(
        err.to_string().contains("batch dimension"),
        "expected batch-axis refusal, got {err}"
    );
}

#[test]
fn convert_reduce_min_axis_zero_rank2_refused() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = reduce_min_spec_with_attr_axes(vec![0]);

    let err = ctx.convert_reduce_min(&spec).unwrap_err();
    assert!(
        err.to_string().contains("batch dimension"),
        "expected batch-axis refusal, got {err}"
    );
}

#[test]
fn convert_logsumexp_positive_axis_adjusted() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 4, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = logsumexp_spec_with_attr_axes(vec![2]);

    let layer = ctx.convert_logsumexp(&spec).unwrap();
    let Layer::LogSumExp(reduce) = layer else {
        panic!("expected LogSumExp, got {layer:?}");
    };
    assert_eq!(reduce.axes, vec![-1i64]);
    assert!(reduce.keepdims);
}

#[test]
fn convert_logsumexp_axis_zero_rank2_refused() {
    let weights = WeightStore::new();
    let shapes = x_shape(&[1, 6]);
    let ctx = test_ctx_with_x_shape(&weights, &shapes);
    let spec = logsumexp_spec_with_attr_axes(vec![0]);

    let err = ctx.convert_logsumexp(&spec).unwrap_err();
    assert!(
        err.to_string().contains("batch dimension"),
        "expected batch-axis refusal, got {err}"
    );
}

// ---------------------------------------------------------------------------
// keepdims flag
// ---------------------------------------------------------------------------

#[test]
fn convert_reduce_sum_keepdims_false() {
    let weights = WeightStore::new();
    let ctx = test_ctx(&weights);

    let mut spec = reduce_sum_spec_with_attr_axes(vec![-1]);
    spec.attributes
        .insert("keepdims".to_string(), AttributeValue::Int(0));

    let layer = ctx.convert_reduce_sum(&spec).unwrap();
    let Layer::ReduceSum(reduce) = layer else {
        panic!("expected ReduceSum, got {layer:?}");
    };
    assert!(!reduce.keepdims, "keepdims=0 should produce false");
}

// ---- #2360 regression: NaN/Inf/non-integer axis rejection ----

#[test]
fn reduce_sum_nan_axis_rejected_2360() {
    let mut weights = WeightStore::new();
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap(),
    );
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_input_axes("axes");
    let result = ctx.convert_reduce_sum(&spec);
    assert!(result.is_err(), "NaN axis must be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("non-finite"),
        "error should mention non-finite: {msg}"
    );
}

#[test]
fn reduce_sum_inf_axis_rejected_2360() {
    let mut weights = WeightStore::new();
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap(),
    );
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_input_axes("axes");
    let result = ctx.convert_reduce_sum(&spec);
    assert!(result.is_err(), "Inf axis must be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("non-finite"),
        "error should mention non-finite: {msg}"
    );
}

#[test]
fn reduce_sum_non_integer_axis_rejected_2360() {
    let mut weights = WeightStore::new();
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.5]).unwrap(),
    );
    let ctx = test_ctx(&weights);
    let spec = reduce_sum_spec_with_input_axes("axes");
    let result = ctx.convert_reduce_sum(&spec);
    assert!(result.is_err(), "non-integer axis must be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("non-integer"),
        "error should mention non-integer: {msg}"
    );
}
