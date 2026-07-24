// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX node construction helpers and weight manipulation utilities
//! for LSTM unrolling.

use crate::onnx_proto;
use ndarray::{Array1, Array2, ArrayD};

pub(super) fn make_node(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    name: &str,
    domain: &str,
    attribute: Vec<onnx_proto::AttributeProto>,
) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: domain.to_string(),
        attribute,
    }
}

pub(super) fn make_node_variadic(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    name: &str,
    domain: &str,
    attribute: Vec<onnx_proto::AttributeProto>,
) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: domain.to_string(),
        attribute,
    }
}

pub(super) fn make_int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

pub(super) fn make_ints_attr(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        ints: values.to_vec(),
        r#type: onnx_proto::attribute_type::INTS,
        ..Default::default()
    }
}

// --- Weight manipulation ---

/// Slice a `[num_directions, ...]` tensor to `[1, ...]` for a given direction.
///
/// Used by bidirectional LSTM unrolling to extract forward (dir=0) and
/// reverse (dir=1) weights from the combined `[2, 4H, I]` weight tensors.
pub(super) fn slice_direction(arr: &ArrayD<f32>, direction: usize) -> Result<ArrayD<f32>, String> {
    let shape = arr.shape();
    if shape.is_empty() || direction >= shape[0] {
        return Err(format!(
            "cannot slice direction {direction} from shape {:?}",
            shape
        ));
    }
    Ok(arr
        .slice_axis(
            ndarray::Axis(0),
            ndarray::Slice::from(direction..direction + 1),
        )
        .to_owned())
}

pub(super) fn squeeze_leading_dim(arr: &ArrayD<f32>) -> Result<Array2<f32>, String> {
    let shape = arr.shape();
    if shape.len() != 3 || shape[0] != 1 {
        return Err(format!(
            "expected [1, M, N] shape for squeeze, got {:?}",
            shape
        ));
    }
    arr.to_owned()
        .into_shape_with_order((shape[1], shape[2]))
        .map_err(|e| format!("reshape failed: {e}"))
}

pub(super) fn flatten_to_1d(arr: &ArrayD<f32>, expected_len: usize) -> Result<Array1<f32>, String> {
    let flat = arr.to_owned().into_raw_vec_and_offset().0;
    if flat.len() != expected_len {
        return Err(format!(
            "expected {expected_len} elements in bias, got {}",
            flat.len()
        ));
    }
    Ok(Array1::from_vec(flat))
}

pub(super) fn transpose_2d(arr: &Array2<f32>) -> Array2<f32> {
    arr.t().to_owned()
}
