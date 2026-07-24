// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{ArrayD, Axis, IxDyn};

use super::super::common::{normalize_axis, parse_attribute_or_input_ints};

pub(super) fn try_fold(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<ArrayD<f32>> {
    match node.op_type.as_str() {
        "ReduceSum" if !node.input.is_empty() => try_fold_reduce_sum(node, weights),
        "ReduceMean" if !node.input.is_empty() => None,
        "ReduceProd" if !node.input.is_empty() => try_fold_reduce_prod(node, weights),
        _ => None,
    }
}

fn try_fold_reduce_sum(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<ArrayD<f32>> {
    let data = weights.get(&node.input[0])?;
    let axes = parse_attribute_or_input_ints(node, "axes", 1, weights);
    let keepdims = node
        .attribute
        .iter()
        .find(|attr| attr.name == "keepdims")
        .map(|attr| attr.i != 0)
        .unwrap_or(true);

    if let Some(axes) = axes {
        let ndim = data.ndim();
        let mut resolved_axes: Vec<usize> = axes
            .iter()
            .map(|&axis| normalize_axis(axis, ndim))
            .collect::<Option<Vec<_>>>()?;
        resolved_axes.sort_unstable();
        resolved_axes.dedup();

        let mut result = data.clone();
        for &axis in resolved_axes.iter().rev() {
            result = result.sum_axis(Axis(axis));
            if keepdims {
                result = result.insert_axis(Axis(axis));
            }
        }
        Some(result)
    } else {
        let sum = data.iter().sum();
        if keepdims {
            Some(ArrayD::from_elem(IxDyn(&vec![1; data.ndim()]), sum))
        } else {
            Some(ArrayD::from_elem(IxDyn(&[]), sum))
        }
    }
}

fn try_fold_reduce_prod(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<ArrayD<f32>> {
    let data = weights.get(&node.input[0])?;
    let axes = parse_attribute_or_input_ints(node, "axes", 1, weights);
    let keepdims = node
        .attribute
        .iter()
        .find(|attr| attr.name == "keepdims")
        .map(|attr| attr.i != 0)
        .unwrap_or(true);

    if let Some(axes) = axes {
        let ndim = data.ndim();
        let mut resolved_axes: Vec<usize> = axes
            .iter()
            .map(|&axis| normalize_axis(axis, ndim))
            .collect::<Option<Vec<_>>>()?;
        resolved_axes.sort_unstable();
        resolved_axes.dedup();

        let mut result = data.clone();
        for &axis in resolved_axes.iter().rev() {
            result = result.fold_axis(Axis(axis), 1.0f32, |acc, value| *acc * *value);
            if keepdims {
                result = result.insert_axis(Axis(axis));
            }
        }
        Some(result)
    } else {
        let product = data.iter().copied().product();
        if keepdims {
            Some(ArrayD::from_elem(IxDyn(&vec![1; data.ndim()]), product))
        } else {
            Some(ArrayD::from_elem(IxDyn(&[]), product))
        }
    }
}
