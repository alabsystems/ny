// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::WeightStore;
use ndarray::{ArrayD, Axis, IxDyn};
use tracing::{debug, warn};

use crate::loader::tensor::tensor_proto_to_array;

use super::super::common::parse_shape_usize;
use super::super::FoldedTensor;

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    model_unbatched: bool,
) -> Option<FoldedTensor> {
    match node.op_type.as_str() {
        "Constant" if node.input.is_empty() => match node.attribute.as_slice() {
            [attr] if attr.name == "value" => (attr.r#type == attribute_type::TENSOR)
                .then_some(attr.t.as_ref())
                .flatten()
                .and_then(parse_constant_tensor)
                .map(FoldedTensor::from_float),
            // Constant has a one-of payload schema. Never select one tensor
            // from duplicate, competing, or otherwise malformed attributes.
            _ => None,
        },
        "ConstantOfShape" if !node.input.is_empty() => {
            let shape = weights.get(&node.input[0])?;
            try_fold_constant_of_shape(node, shape, model_unbatched)
        }
        _ => None,
    }
}

/// The authored `value` payload of a ConstantOfShape, kept in its authored
/// numeric domain.
enum FillValue {
    Float(f32),
    /// An exact integer fill and the range implied by its authored dtype.
    Integer {
        value: i64,
        range: (i64, i64),
    },
}

fn try_fold_constant_of_shape(
    node: &onnx_proto::NodeProto,
    shape_arr: &ArrayD<f32>,
    model_unbatched: bool,
) -> Option<FoldedTensor> {
    let shape = parse_shape_usize(shape_arr)?;
    let fill = constant_of_shape_fill_value(node)?;

    const MAX_CONST_ELEMENTS: usize = 10_000_000;
    let total_elements = ny_core::checked_shape_product(&shape)?;
    if total_elements > MAX_CONST_ELEMENTS {
        warn!(
            "ConstantOfShape: refusing to allocate {} elements (shape {:?}, limit {})",
            total_elements, shape, MAX_CONST_ELEMENTS
        );
        return None;
    }

    let (float_fill, integer) = match fill {
        FillValue::Float(value) => (value, None),
        FillValue::Integer { value, range } => {
            // The leading-axis rewrite further down would silently reindex a
            // structural INT64 vector, so the exact-integer payload is admitted
            // only for the rank-0/rank-1 forms where that rewrite provably
            // cannot fire (`result.ndim() > 1` is false). Richer integer fills
            // keep failing closed rather than acquiring a rewritten shape.
            if shape.len() > 1 {
                return None;
            }
            // `constant_of_shape_fill_value` already proved the value exactly
            // representable in f32, so the mirror built here is not a rounding.
            (
                crate::loader::numeric_cast::i64_to_f32_warned(
                    value,
                    "ConstantOfShape exact integer fill",
                ),
                Some((value, range)),
            )
        }
    };

    let mut result = ArrayD::from_elem(IxDyn(&shape), float_fill);
    // Unbatched mode removes the synthetic leading batch axis, but later
    // singleton axes can still carry real data semantics (for example ViT's
    // CLS token path uses [1, 1, H] and must become [1, H], not [H]).
    // Globally-unbatched models (#cctsdb B5) have NO batch axis anywhere:
    // keep the ONNX shape verbatim (e.g. cctsdb mask base `253` must stay
    // rank-4 [1,3,64,64] so depth-4 ScatterND indices line up).
    if !model_unbatched && result.ndim() > 1 && result.shape()[0] == 1 {
        result = result.index_axis_move(Axis(0), 0);
    }
    debug!(
        "ConstantOfShape: created tensor shape {:?} (from {:?}) filled with {}",
        result.shape(),
        shape,
        float_fill
    );
    let (integer_data, integer_range) = match integer {
        Some((value, range)) => (
            Some(ArrayD::from_elem(IxDyn(result.shape()), value)),
            Some(range),
        ),
        None => (None, None),
    };
    Some(FoldedTensor {
        float_data: result,
        integer_data,
        integer_range,
    })
}

fn parse_constant_tensor(tensor: &onnx_proto::TensorProto) -> Option<ArrayD<f32>> {
    tensor_proto_to_array(tensor)
        .map_err(|e| {
            warn!("constant fold: Constant tensor parse failed: {e}");
            e
        })
        .ok()
}

fn constant_of_shape_fill_value(node: &onnx_proto::NodeProto) -> Option<FillValue> {
    match node.attribute.as_slice() {
        [attr] if attr.name == "value" && attr.r#type == attribute_type::TENSOR => {
            let tensor = attr.t.as_ref()?;
            match tensor.data_type {
                1 => tensor_proto_to_array(tensor)
                    .map_err(|e| {
                        warn!("constant fold: fill value tensor parse failed: {e}");
                        e
                    })
                    .ok()
                    .and_then(|array: ArrayD<f32>| {
                        (array.len() == 1)
                            .then(|| array.iter().next().copied())
                            .flatten()
                    })
                    .map(FillValue::Float),
                // INT32/INT64 fills carry an exact integer payload that
                // `FoldedTensor` preserves as an i64 sidecar plus the range
                // implied by the authored dtype, so the value ny propagates is
                // the value the model authored — not an f32 reinterpretation.
                // Every other dtype (f16/bf16/DOUBLE, unsigned, BOOL, STRING)
                // stays refused: those either round or have no exact i64 range
                // here.
                6 | 7 => {
                    let range = if tensor.data_type == 6 {
                        (i32::MIN as i64, i32::MAX as i64)
                    } else {
                        (i64::MIN, i64::MAX)
                    };
                    let value = exact_scalar_int64(tensor)?;
                    // `FoldedTensor` publishes BOTH an i64 sidecar and an f32
                    // mirror, and consumers are free to read either. A value
                    // above f32's consecutive-integer range would make the two
                    // views disagree (16_777_217 becomes 16_777_216), so it
                    // stays refused rather than folding into a lossy constant.
                    if !super::super::integer_is_exactly_representable_as_f32(value) {
                        return None;
                    }
                    // The internal Reshape copy-axis sentinel must never be
                    // manufactured from an authored payload.
                    if ny_core::reshape_copy_axis_from_sentinel(value).is_some() {
                        return None;
                    }
                    Some(FillValue::Integer { value, range })
                }
                _ => None,
            }
        }
        [] => Some(FillValue::Float(0.0)),
        // ConstantOfShape accepts only an optional tensor-valued `value`.
        // Reject duplicates, unknowns, and historical scalar lookalikes.
        _ => None,
    }
}

/// Read the single authored integer element of an INT32/INT64 `value` tensor.
///
/// TensorProto permits exactly one storage representation, so a payload that
/// populates more than one field is ambiguous and is refused rather than
/// resolved by precedence.
fn exact_scalar_int64(tensor: &onnx_proto::TensorProto) -> Option<i64> {
    // An external payload is not readable here, and `data_location != 0` leaves
    // `raw_data` empty, so both fall through to `None` below.
    if tensor.data_location != 0 || tensor.dims.iter().product::<i64>() != 1 {
        return None;
    }
    let populated = [
        (!tensor.raw_data.is_empty()).then_some(0u8),
        (!tensor.int32_data.is_empty()).then_some(1),
        (!tensor.int64_data.is_empty()).then_some(2),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    match (populated.as_slice(), tensor.data_type) {
        ([0], 6) => {
            let bytes: [u8; 4] = tensor.raw_data.as_slice().try_into().ok()?;
            Some(i64::from(i32::from_le_bytes(bytes)))
        }
        ([0], 7) => {
            let bytes: [u8; 8] = tensor.raw_data.as_slice().try_into().ok()?;
            Some(i64::from_le_bytes(bytes))
        }
        ([1], 6) => (tensor.int32_data.len() == 1).then(|| i64::from(tensor.int32_data[0])),
        ([2], 7) => (tensor.int64_data.len() == 1).then(|| tensor.int64_data[0]),
        _ => None,
    }
}
