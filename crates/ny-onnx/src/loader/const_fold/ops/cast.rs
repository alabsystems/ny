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
        .map(|attr| attr.i)?;
    let integer_data = node
        .attribute
        .iter()
        .find(|attr| attr.name == "to")
        .and_then(|attr| cast_integer_payload(weights, input_name, attr.i));
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
        Some(ints) => ints.mapv(|v| v as f32),
        None if is_integer_dtype(target_type) => float_data.mapv(f32::trunc),
        None => float_data,
    };
    Some(FoldedTensor {
        float_data,
        integer_data,
        integer_range: cast_integer_range(input_name, target_type, weights),
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

fn cast_integer_range(
    input_name: &str,
    target_type: i64,
    weights: &WeightStore,
) -> Option<(i64, i64)> {
    match target_type {
        ONNX_INT64_DATA_TYPE => weights.get_integer_range(input_name),
        ONNX_INT32_DATA_TYPE => Some((i32::MIN as i64, i32::MAX as i64)),
        _ => None,
    }
}
