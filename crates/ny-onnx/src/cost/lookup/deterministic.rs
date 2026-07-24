// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{AttributeValue, LayerSpec};

use super::super::CostError;
use super::common::{
    checked_shape_mul, normalize_index, normalize_indices, shape_elements, shape_inference_error,
};
use super::ShapeLookup;
use std::collections::BTreeSet;

pub(super) fn infer_concat_shape(
    layer: &LayerSpec,
    runtime_shapes: &[(String, Vec<usize>)],
) -> Result<Vec<usize>, CostError> {
    let first_shape = &runtime_shapes[0].1;
    let rank = first_shape.len();
    let axis = concat_axis(layer, rank)?;
    let mut output_shape = first_shape.clone();

    for (_, shape) in runtime_shapes.iter().skip(1) {
        if shape.len() != rank {
            return Err(shape_inference_error(
                layer,
                format!("Concat inputs must share rank {rank}, got {shape:?}"),
            ));
        }
        for (idx, (&lhs, &rhs)) in output_shape.iter().zip(shape.iter()).enumerate() {
            if idx != axis && lhs != rhs {
                return Err(shape_inference_error(
                    layer,
                    format!(
                        "Concat inputs must match on non-axis dims; axis={axis}, dim {idx} had {lhs} vs {rhs}"
                    ),
                ));
            }
        }
        output_shape[axis] = output_shape[axis].checked_add(shape[axis]).ok_or_else(|| {
            shape_inference_error(
                layer,
                format!(
                    "Concat output axis overflow while adding {} and {}",
                    output_shape[axis], shape[axis]
                ),
            )
        })?;
    }

    Ok(output_shape)
}

pub(super) fn infer_unsqueeze_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let axes = normalized_unsqueeze_axes(lookup, layer, input_shape.len())?;
    let output_rank = input_shape.len() + axes.len();
    let mut output_shape = Vec::with_capacity(output_rank);
    let mut input_iter = input_shape.iter();
    let mut axis_iter = axes.into_iter().peekable();

    for output_idx in 0..output_rank {
        if axis_iter.peek().copied() == Some(output_idx) {
            output_shape.push(1);
            axis_iter.next();
        } else {
            output_shape.push(*input_iter.next().ok_or_else(|| {
                shape_inference_error(layer, "unsqueeze input rank underflow".to_string())
            })?);
        }
    }

    Ok(output_shape)
}

pub(super) fn infer_weighted_matmul_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let weight_name = layer
        .inputs
        .get(1)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            shape_inference_error(layer, "MatMul is missing its right-hand input".to_string())
        })?;
    let weight = lookup.model.weights.get(weight_name).ok_or_else(|| {
        shape_inference_error(
            layer,
            format!("MatMul rhs '{weight_name}' is not a constant tensor"),
        )
    })?;
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        return Err(shape_inference_error(
            layer,
            format!(
                "MatMul rhs '{weight_name}' must be rank-2 for missing-shape fallback, got {weight_shape:?}"
            ),
        ));
    }

    let lhs_last = *input_shape.last().ok_or_else(|| {
        shape_inference_error(layer, "MatMul lhs must have rank >= 1".to_string())
    })?;
    let rhs_k = weight_shape[0];
    let rhs_n = weight_shape[1];
    if lhs_last != rhs_k {
        return Err(shape_inference_error(
            layer,
            format!("MatMul lhs last dim {lhs_last} does not match rhs inner dim {rhs_k}"),
        ));
    }

    let mut output_shape = input_shape.to_vec();
    *output_shape
        .last_mut()
        .expect("validated MatMul lhs rank >= 1") = rhs_n;
    Ok(output_shape)
}

pub(super) fn infer_reshape_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let raw_shape = lookup.read_i64_tensor(layer, 1, false, "reshape shape")?;
    if reshape_allowzero(layer)? && raw_shape.contains(&0) {
        return Err(shape_inference_error(
            layer,
            "reshape fallback does not support allowzero=1 with literal zero dimensions"
                .to_string(),
        ));
    }
    if raw_shape.is_empty() {
        return Err(shape_inference_error(
            layer,
            "reshape shape input must not be empty".to_string(),
        ));
    }

    let input_elements = shape_elements(input_shape, layer, "reshape input")?;
    let mut output_shape = Vec::with_capacity(raw_shape.len());
    let mut known_product = 1usize;
    let mut infer_index = None;

    for (idx, dim) in raw_shape.into_iter().enumerate() {
        match dim {
            0 => {
                let copied = *input_shape.get(idx).ok_or_else(|| {
                    shape_inference_error(
                        layer,
                        format!(
                            "reshape zero-copy dimension {idx} is out of range for input rank {}",
                            input_shape.len()
                        ),
                    )
                })?;
                known_product = checked_shape_mul(known_product, copied, layer)?;
                output_shape.push(copied);
            }
            -1 => {
                if infer_index.replace(idx).is_some() {
                    return Err(shape_inference_error(
                        layer,
                        "reshape shape input may contain at most one -1".to_string(),
                    ));
                }
                output_shape.push(1);
            }
            value if value > 0 => {
                let value = value as usize;
                known_product = checked_shape_mul(known_product, value, layer)?;
                output_shape.push(value);
            }
            value => {
                return Err(shape_inference_error(
                    layer,
                    format!("reshape shape input must use positive dims, 0, or -1; got {value}"),
                ));
            }
        }
    }

    if let Some(index) = infer_index {
        if known_product == 0 || input_elements % known_product != 0 {
            return Err(shape_inference_error(
                layer,
                format!(
                    "reshape cannot infer dimension because {input_elements} elements are not divisible by {known_product}"
                ),
            ));
        }
        output_shape[index] = input_elements / known_product;
    } else if known_product != input_elements {
        return Err(shape_inference_error(
            layer,
            format!("reshape target has {known_product} elements but input has {input_elements}"),
        ));
    }

    Ok(output_shape)
}

fn concat_axis(layer: &LayerSpec, rank: usize) -> Result<usize, CostError> {
    let axis = match layer.attributes.get("axis") {
        Some(AttributeValue::Int(value)) => *value,
        Some(other) => {
            return Err(shape_inference_error(
                layer,
                format!("expected integer concat axis, got {other:?}"),
            ));
        }
        None => 0,
    };
    normalize_index(layer, rank, axis, "axis")
}

fn normalized_unsqueeze_axes(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_rank: usize,
) -> Result<Vec<usize>, CostError> {
    let Some(raw_axes) =
        lookup.attribute_or_input_i64s(layer, "axes", 1, false, "unsqueeze axes")?
    else {
        return Err(shape_inference_error(
            layer,
            "unsqueeze fallback requires explicit axes".to_string(),
        ));
    };
    let output_rank = input_rank + raw_axes.len();
    let axes = normalize_indices(layer, output_rank, &raw_axes, "axes")?;
    if axes.iter().collect::<BTreeSet<_>>().len() != axes.len() {
        return Err(shape_inference_error(
            layer,
            format!("unsqueeze axes must be unique, got {raw_axes:?}"),
        ));
    }
    let mut axes = axes;
    axes.sort_unstable();
    Ok(axes)
}

fn reshape_allowzero(layer: &LayerSpec) -> Result<bool, CostError> {
    match layer.attributes.get("allowzero") {
        Some(AttributeValue::Int(0)) | None => Ok(false),
        Some(AttributeValue::Int(1)) => Ok(true),
        Some(AttributeValue::Int(value)) => Err(shape_inference_error(
            layer,
            format!("expected allowzero to be 0 or 1, got {value}"),
        )),
        Some(other) => Err(shape_inference_error(
            layer,
            format!("expected integer allowzero attribute, got {other:?}"),
        )),
    }
}
