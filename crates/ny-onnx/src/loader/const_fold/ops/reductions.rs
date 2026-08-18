// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::WeightStore;
use ndarray::{ArrayD, Axis, IxDyn};
use std::collections::HashSet;

use super::super::common::{
    exact_f32_product, exact_f32_sum, normalize_axis, parse_attribute_or_input_ints,
};
use super::super::FoldedTensor;

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    if !matches!(node.input.len(), 1 | 2)
        || node.input.first().is_none_or(String::is_empty)
        || !valid_reduction_attributes(node)
    {
        return None;
    }

    let integer_evidence = weights.get_integers(&node.input[0]).is_some()
        || weights.get_integer_range(&node.input[0]).is_some();
    let integer_operation: fn(i64, i64) -> Option<i64> = match node.op_type.as_str() {
        "ReduceSum" => i64::checked_add,
        "ReduceProd" => i64::checked_mul,
        "ReduceMean" => return None,
        _ => return None,
    };
    if integer_evidence {
        // Never fall through to a lossy f32 compatibility mirror once integer
        // provenance is present.
        return try_fold_integer_reduction(node, weights, integer_operation);
    }

    match node.op_type.as_str() {
        "ReduceSum" => try_fold_float_reduction(node, weights, 0.0, exact_f32_sum)
            .map(FoldedTensor::from_float),
        "ReduceProd" => try_fold_float_reduction(node, weights, 1.0, exact_f32_product)
            .map(FoldedTensor::from_float),
        _ => None,
    }
}

fn valid_reduction_attributes(node: &onnx_proto::NodeProto) -> bool {
    let mut names = HashSet::new();
    node.attribute.iter().all(|attribute| {
        names.insert(attribute.name.as_str())
            && match attribute.name.as_str() {
                "axes" => attribute.r#type == attribute_type::INTS,
                "keepdims" | "noop_with_empty_axes" => {
                    attribute.r#type == attribute_type::INT && matches!(attribute.i_value(), 0 | 1)
                }
                _ => false,
            }
    })
}

#[derive(Clone)]
enum ReductionAxes {
    All,
    Noop,
    Selected(Vec<usize>),
}

fn reduction_axes(node: &onnx_proto::NodeProto, weights: &WeightStore) -> Option<ReductionAxes> {
    let axes_attribute_present = node
        .attribute
        .iter()
        .any(|attribute| attribute.name == "axes");
    let axes_input_present = node.input.get(1).is_some_and(|name| !name.is_empty());
    // These are alternative schema encodings from different opsets, never two
    // competing authorities on the same node.
    if axes_attribute_present && axes_input_present {
        return None;
    }
    let axes_are_explicit = axes_attribute_present || axes_input_present;
    let axes = parse_attribute_or_input_ints(node, "axes", 1, weights);
    if axes_are_explicit && axes.is_none() {
        return None;
    }
    let empty_axes_semantics = || {
        let noop = node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "noop_with_empty_axes")
            .is_some_and(|attribute| attribute.i_value() == 1);
        if noop {
            ReductionAxes::Noop
        } else {
            ReductionAxes::All
        }
    };
    let Some(axes) = axes else {
        // In input-form reduction schemas, an omitted optional axes input has
        // the same semantics as an explicitly empty axes tensor.  In
        // particular, noop_with_empty_axes=1 makes both forms an identity.
        return Some(empty_axes_semantics());
    };
    if axes.is_empty() {
        return Some(empty_axes_semantics());
    }
    let ndim = weights.get(&node.input[0])?.ndim();
    let mut resolved = axes
        .into_iter()
        .map(|axis| normalize_axis(axis, ndim))
        .collect::<Option<Vec<_>>>()?;
    resolved.sort_unstable();
    if resolved.windows(2).any(|axes| axes[0] == axes[1]) {
        return None;
    }
    Some(ReductionAxes::Selected(resolved))
}

fn keepdims(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == "keepdims")
        .is_none_or(|attribute| attribute.i_value() == 1)
}

fn try_fold_float_reduction(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    identity: f32,
    operation: fn(f32, f32) -> Option<f32>,
) -> Option<ArrayD<f32>> {
    let data = weights.get(&node.input[0])?;
    match reduction_axes(node, weights)? {
        ReductionAxes::Noop => data
            .iter()
            .all(|value| value.is_finite())
            .then(|| data.clone()),
        ReductionAxes::Selected(axes) => {
            let mut result = data.clone();
            for &axis in axes.iter().rev() {
                result = reduce_axis_float(&result, axis, identity, operation)?;
                if keepdims(node) {
                    result = result.insert_axis(Axis(axis));
                }
            }
            Some(result)
        }
        ReductionAxes::All => {
            let reduced = data.iter().copied().try_fold(identity, operation)?;
            if keepdims(node) {
                Some(ArrayD::from_elem(IxDyn(&vec![1; data.ndim()]), reduced))
            } else {
                Some(ArrayD::from_elem(IxDyn(&[]), reduced))
            }
        }
    }
}

fn reduce_axis_float(
    data: &ArrayD<f32>,
    axis: usize,
    identity: f32,
    operation: fn(f32, f32) -> Option<f32>,
) -> Option<ArrayD<f32>> {
    let mut output_shape = data.shape().to_vec();
    output_shape.remove(axis);
    let values = data
        .lanes(Axis(axis))
        .into_iter()
        .map(|lane| lane.iter().copied().try_fold(identity, operation))
        .collect::<Option<Vec<_>>>()?;
    ArrayD::from_shape_vec(IxDyn(&output_shape), values).ok()
}

fn try_fold_integer_reduction(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    operation: fn(i64, i64) -> Option<i64>,
) -> Option<FoldedTensor> {
    let data = weights.get_integers(&node.input[0])?;
    let float_mirror = weights.get(&node.input[0])?;
    if data.shape() != float_mirror.shape() {
        return None;
    }
    let range = weights.get_integer_range(&node.input[0])?;
    let identity = if node.op_type == "ReduceSum" { 0 } else { 1 };
    let checked_operation = |lhs: i64, rhs: i64| {
        operation(lhs, rhs).filter(|value| range.0 <= *value && *value <= range.1)
    };
    let integer_data = match reduction_axes(node, weights)? {
        ReductionAxes::Noop => data.clone(),
        ReductionAxes::Selected(axes) => {
            let mut result = data.clone();
            for &axis in axes.iter().rev() {
                result = reduce_axis_integer(&result, axis, identity, &checked_operation)?;
                if keepdims(node) {
                    result = result.insert_axis(Axis(axis));
                }
            }
            result
        }
        ReductionAxes::All => {
            let reduced = data
                .iter()
                .copied()
                .try_fold(identity, &checked_operation)?;
            if keepdims(node) {
                ArrayD::from_elem(IxDyn(&vec![1; data.ndim()]), reduced)
            } else {
                ArrayD::from_elem(IxDyn(&[]), reduced)
            }
        }
    };
    let float_data = integer_data.mapv(|value| {
        crate::loader::numeric_cast::i64_to_f32_warned(value, "integer reduction constant fold")
    });
    Some(FoldedTensor {
        float_data,
        integer_data: Some(integer_data),
        integer_range: Some(range),
    })
}

fn reduce_axis_integer(
    data: &ArrayD<i64>,
    axis: usize,
    identity: i64,
    operation: &impl Fn(i64, i64) -> Option<i64>,
) -> Option<ArrayD<i64>> {
    let mut output_shape = data.shape().to_vec();
    output_shape.remove(axis);
    let values = data
        .lanes(Axis(axis))
        .into_iter()
        .map(|lane| lane.iter().copied().try_fold(identity, operation))
        .collect::<Option<Vec<_>>>()?;
    ArrayD::from_shape_vec(IxDyn(&output_shape), values).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reduction_node(op_type: &str, inputs: &[&str]) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|value| value.to_string()).collect(),
            output: vec!["out".to_string()],
            name: "reduction".to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    #[test]
    fn inexact_float_sum_is_not_materialized() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0_f32.powi(-24)]).unwrap(),
        );
        assert!(try_fold(&reduction_node("ReduceSum", &["data"]), &weights).is_none());
    }

    #[test]
    fn empty_axes_default_reduces_all_dimensions() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        );
        weights.insert(
            "axes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap(),
        );
        let output = try_fold(&reduction_node("ReduceSum", &["data", "axes"]), &weights)
            .expect("exact reduce-all should fold");
        assert_eq!(output.float_data.shape(), &[1]);
        assert_eq!(output.float_data.as_slice().unwrap(), &[3.0]);
    }

    #[test]
    fn omitted_axes_with_noop_attribute_is_identity_for_sum_and_product() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        );

        for op_type in ["ReduceSum", "ReduceProd"] {
            let mut node = reduction_node(op_type, &["data"]);
            node.attribute.push(onnx_proto::AttributeProto {
                name: "noop_with_empty_axes".to_string(),
                i: Some(1),
                r#type: attribute_type::INT,
                ..Default::default()
            });
            let output = try_fold(&node, &weights).expect("identity reduction should fold");
            assert_eq!(output.float_data.shape(), &[2]);
            assert_eq!(output.float_data.as_slice().unwrap(), &[1.0, 2.0]);
        }
    }

    #[test]
    fn integer_reduction_uses_exact_sidecar_and_preserves_range() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![16_777_216.0, -16_777_216.0]).unwrap(),
        );
        weights.insert_integers(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![16_777_217, -16_777_216]).unwrap(),
        );
        weights.insert_integer_range("data".to_string(), i64::MIN, i64::MAX);
        let output = try_fold(&reduction_node("ReduceSum", &["data"]), &weights).unwrap();
        assert_eq!(output.integer_data.unwrap().as_slice().unwrap(), &[1]);
        assert_eq!(output.float_data.as_slice().unwrap(), &[1.0]);
        assert_eq!(output.integer_range, Some((i64::MIN, i64::MAX)));
    }

    #[test]
    fn duplicate_or_competing_axes_authorities_are_rejected() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        weights.insert(
            "axes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        );

        let mut duplicate = reduction_node("ReduceSum", &["data"]);
        duplicate.attribute.push(onnx_proto::AttributeProto {
            name: "axes".to_string(),
            r#type: attribute_type::INTS,
            ints: vec![0, -2],
            ..Default::default()
        });
        assert!(try_fold(&duplicate, &weights).is_none());

        let mut competing = reduction_node("ReduceSum", &["data", "axes"]);
        competing.attribute.push(onnx_proto::AttributeProto {
            name: "axes".to_string(),
            r#type: attribute_type::INTS,
            ints: vec![1],
            ..Default::default()
        });
        assert!(try_fold(&competing, &weights).is_none());
    }
}
