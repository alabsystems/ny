// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{legacy_axis_authority, ConvertContext, LegacyBatchAxisPolicy};
use crate::{AttributeValue, LayerSpec, WeightStore, INTERNAL_LEGACY_AXIS_AUTHORITY_ATTR};
use ny_core::{LayerType, NyError};
use std::collections::{HashMap, HashSet};

fn context<'a>(
    weights: &'a WeightStore,
    tensor_shapes: &'a HashMap<String, Vec<i64>>,
    constant_tensors: &'a HashSet<String>,
) -> ConvertContext<'a> {
    ConvertContext::new(weights, tensor_shapes, constant_tensors)
}

/// A positive ONNX axis on a tensor with no recorded shape used to take the
/// legacy `axis - 1` guess. On a runtime tensor that kept its ONNX rank that
/// guess selects the wrong dimension, so an ONNX-authored node now refuses.
#[test]
fn remap_axis_unknown_rank_refuses_onnx_authored_positive_axis() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::new();
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);

    let error = ctx
        .remap_axis_trailing(
            "ReduceSum",
            "/pensieve/ReduceSum",
            "activation",
            1,
            LegacyBatchAxisPolicy::KeepZeroWarn,
            false,
        )
        .expect_err("unknown recorded rank must refuse an ONNX-authored positive axis");
    assert!(
        matches!(
            &error,
            NyError::UnsupportedOp(msg)
                if msg.contains("no recorded ONNX shape") && msg.contains("refusing conversion")
        ),
        "expected an UnsupportedOp refusal naming the missing shape, got {error:?}"
    );
}

/// The `RejectZero` policy only ever governed axis 0; it cannot admit a
/// positive axis whose source rank is unknown.
#[test]
fn remap_axis_unknown_rank_refuses_reject_zero_family_positive_axis() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::new();
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);

    let error = ctx
        .remap_axis_trailing(
            "Softmax",
            "/asp/Softmax",
            "activation",
            2,
            LegacyBatchAxisPolicy::RejectZero,
            false,
        )
        .expect_err("unknown recorded rank must refuse an ONNX-authored positive axis");
    assert!(
        matches!(&error, NyError::UnsupportedOp(msg) if msg.contains("no recorded ONNX shape")),
        "expected an UnsupportedOp refusal, got {error:?}"
    );
}

/// Keeping ONNX axis 0 at unknown rank is also a layout guess and refuses on
/// nodes without the internal source-semantics marker.
#[test]
fn remap_axis_unknown_rank_refuses_onnx_authored_axis_zero_keep_policy() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::new();
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);

    let error = ctx
        .remap_axis_trailing(
            "Concat",
            "/backbone/Concat",
            "activation",
            0,
            LegacyBatchAxisPolicy::KeepZeroWarn,
            false,
        )
        .expect_err("unknown recorded rank must refuse an ONNX-authored axis 0");
    assert!(
        matches!(&error, NyError::UnsupportedOp(msg) if msg.contains("no recorded ONNX shape")),
        "expected an UnsupportedOp refusal, got {error:?}"
    );
}

/// The `RejectZero` axis-0 case already failed closed. Its error is pinned so
/// the provenance rewrite cannot accidentally open that path.
#[test]
fn remap_axis_unknown_rank_keeps_historic_reject_zero_axis_zero_error() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::new();
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);

    let error = ctx
        .remap_axis_trailing(
            "Squeeze",
            "/head/Squeeze",
            "activation",
            0,
            LegacyBatchAxisPolicy::RejectZero,
            false,
        )
        .expect_err("axis 0 has always been refused for this op family");
    assert!(
        matches!(
            &error,
            NyError::ModelLoad(msg)
                if msg.contains("targets batch dimension which does not exist")
        ),
        "expected the historic ModelLoad refusal, got {error:?}"
    );
}

/// The public `LayerSpec` marker selects only format-neutral internal-axis
/// semantics. It does not represent solver, proof, certificate, or verdict
/// authority; this conversion helper only consumes it to select an axis map.
#[test]
fn layer_spec_legacy_axis_marker_selects_internal_axis_semantics_only() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::new();
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);
    let spec = LayerSpec {
        name: "rnn0__lstm_bidi_concat_yh".to_string(),
        layer_type: LayerType::Concat,
        inputs: vec!["rnn0__lstm_bidi_yh_fwd_us".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::from([(
            INTERNAL_LEGACY_AXIS_AUTHORITY_ATTR.to_string(),
            AttributeValue::Int(1),
        )]),
    };

    let axis = ctx
        .remap_axis_trailing(
            "Concat",
            &spec.name,
            "rnn0__lstm_bidi_yh_fwd_us",
            1,
            LegacyBatchAxisPolicy::KeepZeroWarn,
            legacy_axis_authority(&spec).expect("exact internal source semantics"),
        )
        .expect("ny-lowered LSTM nodes keep the legacy adjustment");
    assert_eq!(axis, 0);
}

/// A lookalike node name grants nothing, and malformed provenance fails
/// closed rather than becoming either permission or a silent absence.
#[test]
fn legacy_axis_authority_requires_exact_internal_attribute() {
    let mut spec = LayerSpec {
        name: "authored__lstm_lookalike".to_string(),
        layer_type: LayerType::Concat,
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    assert!(!legacy_axis_authority(&spec).expect("missing authority is false"));

    spec.attributes.insert(
        INTERNAL_LEGACY_AXIS_AUTHORITY_ATTR.to_string(),
        AttributeValue::Int(1),
    );
    assert!(legacy_axis_authority(&spec).expect("exact authority is accepted"));

    spec.attributes.insert(
        INTERNAL_LEGACY_AXIS_AUTHORITY_ATTR.to_string(),
        AttributeValue::Int(0),
    );
    assert!(legacy_axis_authority(&spec).is_err());
}

/// A known recorded rank is unaffected and still yields the
/// trailing-relative (negative) encoding.
#[test]
fn remap_axis_known_rank_still_maps_trailing_relative() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::from([("activation".to_string(), vec![1_i64, 8])]);
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);

    let axis = ctx
        .remap_axis_trailing(
            "ReduceSum",
            "/pensieve/ReduceSum",
            "activation",
            1,
            LegacyBatchAxisPolicy::KeepZeroWarn,
            false,
        )
        .expect("a recorded rank-2 shape resolves the axis");
    assert_eq!(axis, -1);
}

/// An already-negative ONNX axis is trailing-relative by definition and still
/// passes through when the recorded rank is unknown.
#[test]
fn remap_axis_unknown_rank_still_passes_through_negative_axis() {
    let weights = WeightStore::new();
    let tensor_shapes = HashMap::new();
    let constant_tensors = HashSet::new();
    let ctx = context(&weights, &tensor_shapes, &constant_tensors);

    let axis = ctx
        .remap_axis_trailing(
            "Gather",
            "/head/Gather",
            "activation",
            -1,
            LegacyBatchAxisPolicy::RejectZero,
            false,
        )
        .expect("negative axes are already trailing-relative");
    assert_eq!(axis, -1);
}
