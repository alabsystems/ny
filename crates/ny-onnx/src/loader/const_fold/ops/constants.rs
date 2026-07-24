// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::WeightStore;
use ndarray::{ArrayD, Axis, IxDyn};
use tracing::{debug, warn};

use crate::loader::numeric_cast::i64_to_f32_warned;
use crate::loader::tensor::tensor_proto_to_array;

use super::super::common::parse_shape_usize;

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    model_unbatched: bool,
) -> Option<ArrayD<f32>> {
    match node.op_type.as_str() {
        "Constant" => node
            .attribute
            .iter()
            .find(|attr| attr.name == "value")
            .and_then(|attr| {
                (attr.r#type == attribute_type::TENSOR)
                    .then_some(attr.t.as_ref())
                    .flatten()
            })
            .and_then(parse_constant_tensor),
        "ConstantOfShape" if !node.input.is_empty() => {
            let shape = weights.get(&node.input[0])?;
            try_fold_constant_of_shape(node, shape, model_unbatched)
        }
        _ => None,
    }
}

fn try_fold_constant_of_shape(
    node: &onnx_proto::NodeProto,
    shape_arr: &ArrayD<f32>,
    model_unbatched: bool,
) -> Option<ArrayD<f32>> {
    let shape = parse_shape_usize(shape_arr)?;
    let fill_value = constant_of_shape_fill_value(node)?;

    const MAX_CONST_ELEMENTS: usize = 10_000_000;
    let total_elements = ny_core::checked_shape_product(&shape)?;
    if total_elements > MAX_CONST_ELEMENTS {
        warn!(
            "ConstantOfShape: refusing to allocate {} elements (shape {:?}, limit {})",
            total_elements, shape, MAX_CONST_ELEMENTS
        );
        return None;
    }

    let mut result = ArrayD::from_elem(IxDyn(&shape), fill_value);
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
        fill_value
    );
    Some(result)
}

fn parse_constant_tensor(tensor: &onnx_proto::TensorProto) -> Option<ArrayD<f32>> {
    tensor_proto_to_array(tensor)
        .map_err(|e| {
            warn!("constant fold: Constant tensor parse failed: {e}");
            e
        })
        .ok()
}

fn constant_of_shape_fill_value(node: &onnx_proto::NodeProto) -> Option<f32> {
    match node.attribute.iter().find(|attr| attr.name == "value") {
        Some(attr) => match attr.r#type {
            attribute_type::TENSOR => attr
                .t
                .as_ref()
                .and_then(|tensor| {
                    tensor_proto_to_array(tensor)
                        .map_err(|e| {
                            warn!("constant fold: fill value tensor parse failed: {e}");
                            e
                        })
                        .ok()
                })
                .and_then(|array: ArrayD<f32>| {
                    if array.len() == 1 {
                        array.iter().next().copied()
                    } else {
                        None
                    }
                }),
            attribute_type::FLOAT => Some(attr.f),
            attribute_type::INT => Some(i64_to_f32_warned(attr.i, "ConstantOfShape fill INT")),
            attribute_type::FLOATS => {
                if attr.floats.len() == 1 {
                    attr.floats.first().copied()
                } else {
                    None
                }
            }
            attribute_type::INTS => {
                if attr.ints.len() == 1 {
                    attr.ints
                        .first()
                        .map(|value| i64_to_f32_warned(*value, "ConstantOfShape fill INTS"))
                } else {
                    None
                }
            }
            _ => Some(0.0),
        },
        None => Some(0.0),
    }
}
