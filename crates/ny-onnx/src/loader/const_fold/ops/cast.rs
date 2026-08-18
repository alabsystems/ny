// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::ArrayD;
use tracing::warn;

use super::super::FoldedTensor;

const ONNX_INT32_DATA_TYPE: i64 = 6;
const ONNX_INT64_DATA_TYPE: i64 = 7;
const ONNX_BOOL_DATA_TYPE: i64 = 9;

pub(super) fn try_fold_cast(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    let input_name = &node.input[0];
    let float_data = weights.get(input_name)?.clone();
    let target_type = node
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .map(|attr| attr.i_value())?;
    // Cast to BOOL is the indicator `x != 0` — neither truncation nor an
    // identity. Folding Cast(2.0 -> BOOL) as 2.0 bakes a wrong constant into
    // the network exactly as folding Cast(0.7 -> INT64) as 0.7 would, and
    // trunc is also wrong here (trunc(0.5) = 0 but bool(0.5) = 1). Materialize
    // the indicator, which is exact for every input including NaN
    // (`NaN != 0` is true, matching ONNX Runtime).
    if target_type == ONNX_BOOL_DATA_TYPE {
        let indicator = float_data.mapv(|v| if v != 0.0 { 1.0_f32 } else { 0.0_f32 });
        let integer_data = indicator.mapv(|v| i64::from(v != 0.0));
        return Some(FoldedTensor {
            float_data: indicator,
            integer_data: Some(integer_data),
            integer_range: Some((0, 1)),
        });
    }
    let integer_data = node
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .and_then(|attr| cast_integer_payload(weights, input_name, attr.i_value()));
    // When Cast produces integer data, derive float_data from the casted values
    // so both views are consistent. This is critical for narrowing casts
    // (INT64→INT32) where wrapping changes the value.
    //
    // When the input has only a FLOAT payload and the target is an integer
    // dtype, apply the cast's trunc-toward-zero semantics to the float view
    // (#cctsdb B1): folding Cast(0.7 -> INT64) as 0.7 would bake a wrong
    // constant into the network. trunc is a no-op for the common
    // integer-valued shape/index chains.
    let float_data = match &integer_data {
        Some(ints) if target_type == ONNX_INT64_DATA_TYPE && ints.shape() == float_data.shape() => {
            // INT64→INT64 is an exact identity.  In particular, an internally
            // generated dynamic Shape sentinel carries an authenticated 0.0
            // compatibility marker that must survive redundant Cast nodes;
            // rebuilding it with `i64 as f32` would destroy that marker.
            float_data
        }
        Some(ints) => ints.mapv(|v| v as f32),
        None if is_integer_dtype(target_type) => float_data.mapv(f32::trunc),
        None => float_data,
    };
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: cast_integer_range(target_type),
    })
}

fn cast_integer_payload(
    weights: &WeightStore,
    input_name: &str,
    target_type: i64,
) -> Option<ArrayD<i64>> {
    let integer_data = weights.get_integers(input_name)?;
    match target_type {
        ONNX_INT64_DATA_TYPE => Some(integer_data.clone()),
        ONNX_INT32_DATA_TYPE => {
            // Materialize the actual INT32 cast: narrow to i32 (wrapping on
            // overflow per C++ static_cast semantics used by ONNX Runtime),
            // then widen back to i64 for storage.
            let has_overflow = integer_data.iter().any(|&v| i32::try_from(v).is_err());
            if has_overflow {
                warn!(
                    "Cast INT64→INT32 overflow in const-fold for '{input_name}': \
                     values outside i32 range will wrap"
                );
            }
            Some(integer_data.mapv(|v| (v as i32) as i64))
        }
        _ => None,
    }
}

/// ONNX TensorProto.DataType integer targets: UINT8=2, INT8=3, UINT16=4,
/// INT16=5, INT32=6, INT64=7, UINT32=12, UINT64=13. BOOL(9) excluded —
/// cast-to-bool is `x != 0`, not truncation.
fn is_integer_dtype(dtype: i64) -> bool {
    matches!(dtype, 2 | 3 | 4 | 5 | 6 | 7 | 12 | 13)
}

fn cast_integer_range(target_type: i64) -> Option<(i64, i64)> {
    match target_type {
        ONNX_INT64_DATA_TYPE => Some((i64::MIN, i64::MAX)),
        ONNX_INT32_DATA_TYPE => Some((i32::MIN as i64, i32::MAX as i64)),
        _ => None,
    }
}
