// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use tracing::warn;

fn fits_exact_i64_f32(value: f32) -> bool {
    value >= i64::MIN as f32 && value < i64::MAX as f32
}

pub(crate) fn parse_scalar_i64(value: f32) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1.0e-4 {
        return None;
    }
    if !fits_exact_i64_f32(rounded) {
        return None;
    }
    Some(rounded as i64)
}

pub(crate) fn parse_shape_i64(shape_arr: &ArrayD<f32>) -> Option<Vec<i64>> {
    shape_arr.iter().map(|&v| parse_scalar_i64(v)).collect()
}

pub(crate) fn parse_shape_usize(shape_arr: &ArrayD<f32>) -> Option<Vec<usize>> {
    parse_shape_i64(shape_arr)?
        .into_iter()
        .map(|dim| usize::try_from(dim).ok())
        .collect()
}

pub(crate) fn normalize_axis(axis: i64, ndim: usize) -> Option<usize> {
    if ndim == 0 {
        return None;
    }
    let axis = if axis < 0 {
        let axis = axis + ndim as i64;
        if axis < 0 {
            return None;
        }
        axis
    } else {
        axis
    };
    usize::try_from(axis).ok().filter(|&a| a < ndim)
}

pub(crate) fn reshape_allowzero(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "allowzero")
        .and_then(|attr| match attr.r#type {
            attribute_type::INT => Some(attr.i != 0),
            attribute_type::FLOAT => Some(attr.f != 0.0),
            attribute_type::INTS => attr.ints.first().map(|&v| v != 0),
            attribute_type::FLOATS => attr.floats.first().map(|&v| v != 0.0),
            _ => None,
        })
        .unwrap_or(false)
}

pub(crate) fn parse_attribute_or_input_ints(
    node: &onnx_proto::NodeProto,
    attribute_name: &str,
    input_index: usize,
    weights: &WeightStore,
) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|attr| attr.name == attribute_name)
        .map(|attr| attr.ints.clone())
        .or_else(|| {
            node.input
                .get(input_index)
                .filter(|name| !name.is_empty())
                .and_then(|name| read_tensor_i64s(weights, name))
        })
}

pub(crate) fn read_tensor_i64s(weights: &WeightStore, name: &str) -> Option<Vec<i64>> {
    weights
        .get_integers(name)
        .map(|values| values.iter().copied().collect())
        .or_else(|| weights.get(name).and_then(parse_shape_i64))
}

pub(crate) fn read_tensor_i64s_and_shape(
    weights: &WeightStore,
    name: &str,
) -> Option<(Vec<i64>, Vec<usize>)> {
    weights
        .get_integers(name)
        .map(|values| (values.iter().copied().collect(), values.shape().to_vec()))
        .or_else(|| {
            weights.get(name).and_then(|values| {
                parse_shape_i64(values).map(|parsed| (parsed, values.shape().to_vec()))
            })
        })
}

pub(crate) fn reshape_with_warning<T: Clone>(
    data: ArrayD<T>,
    shape: &[usize],
    context: &str,
) -> Option<ArrayD<T>> {
    data.into_shape_with_order(IxDyn(shape))
        .map_err(|e| {
            warn!("constant fold {context} reshape failed: {e}");
            e
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use super::parse_scalar_i64;

    #[test]
    fn parse_scalar_i64_rejects_rounded_i64_max_float_2360() {
        assert_eq!(
            parse_scalar_i64(i64::MAX as f32),
            None,
            "f32(i64::MAX) rounds to 2^63 and must not saturate back to i64::MAX"
        );
        assert_eq!(
            parse_scalar_i64(i64::MIN as f32),
            Some(i64::MIN),
            "i64::MIN remains exactly representable in f32"
        );
    }
}
