// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{AttributeValue, LayerSpec};

use super::super::CostError;
use super::slice::parse_scalar_i64;
use super::ShapeLookup;

pub(super) fn shape_inference_error(layer: &LayerSpec, detail: String) -> CostError {
    CostError::propagation_msg(
        "static cost estimate",
        format!(
            "cannot infer output shape for layer '{}' (type {}): {detail}",
            layer.name, layer.layer_type
        ),
    )
}

impl ShapeLookup<'_> {
    pub(super) fn read_i64_tensor(
        &self,
        layer: &LayerSpec,
        input_index: usize,
        allow_positive_infinity: bool,
        label: &str,
    ) -> Result<Vec<i64>, CostError> {
        let tensor_name = layer
            .inputs
            .get(input_index)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| shape_inference_error(layer, format!("{label} input is missing")))?;
        if let Some(tensor) = self.model.weights.get_integers(tensor_name) {
            return tensor
                .iter()
                .copied()
                .map(|value| {
                    if value == i64::MAX && !allow_positive_infinity {
                        return Err(shape_inference_error(
                            layer,
                            format!(
                                "{label} input '{tensor_name}' contains unsupported +Inf sentinel"
                            ),
                        ));
                    }
                    Ok(value)
                })
                .collect();
        }
        let tensor = self.model.weights.get(tensor_name).ok_or_else(|| {
            shape_inference_error(
                layer,
                format!("{label} input '{tensor_name}' is not a constant tensor"),
            )
        })?;

        tensor
            .iter()
            .map(|value| {
                parse_scalar_i64(*value, allow_positive_infinity).ok_or_else(|| {
                    shape_inference_error(
                        layer,
                        format!("{label} input '{tensor_name}' must contain integer values"),
                    )
                })
            })
            .collect()
    }

    pub(super) fn attribute_or_input_i64s(
        &self,
        layer: &LayerSpec,
        attribute_name: &str,
        input_index: usize,
        allow_positive_infinity: bool,
        label: &str,
    ) -> Result<Option<Vec<i64>>, CostError> {
        match layer.attributes.get(attribute_name) {
            Some(AttributeValue::Ints(values)) => Ok(Some(values.clone())),
            Some(AttributeValue::Int(value)) => Ok(Some(vec![*value])),
            Some(other) => Err(shape_inference_error(
                layer,
                format!("expected integer {label}, got {other:?}"),
            )),
            None => {
                if layer
                    .inputs
                    .get(input_index)
                    .is_some_and(|name| !name.is_empty())
                {
                    self.read_i64_tensor(layer, input_index, allow_positive_infinity, label)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        }
    }
}

pub(super) fn normalized_axes(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    rank: usize,
) -> Result<Vec<usize>, CostError> {
    let Some(axes) = lookup.attribute_or_input_i64s(layer, "axes", 1, false, "reduction axes")?
    else {
        return Ok((0..rank).collect());
    };
    normalize_indices(layer, rank, &axes, "axes")
}

pub(super) fn reduction_keepdims(layer: &LayerSpec) -> Result<bool, CostError> {
    match layer.attributes.get("keepdims") {
        Some(AttributeValue::Int(0)) => Ok(false),
        Some(AttributeValue::Int(1)) | None => Ok(true),
        Some(AttributeValue::Int(value)) => Err(shape_inference_error(
            layer,
            format!("expected keepdims to be 0 or 1, got {value}"),
        )),
        Some(other) => Err(shape_inference_error(
            layer,
            format!("expected integer keepdims attribute, got {other:?}"),
        )),
    }
}

pub(super) fn shape_elements(
    shape: &[usize],
    layer: &LayerSpec,
    label: &str,
) -> Result<usize, CostError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        checked_shape_mul(acc, *dim, layer)
            .map_err(|_| shape_inference_error(layer, format!("{label} element count overflow")))
    })
}

pub(super) fn checked_shape_mul(
    acc: usize,
    value: usize,
    layer: &LayerSpec,
) -> Result<usize, CostError> {
    acc.checked_mul(value).ok_or_else(|| {
        shape_inference_error(
            layer,
            format!("shape element count overflow while multiplying {acc} by {value}"),
        )
    })
}

pub(super) fn normalize_indices(
    layer: &LayerSpec,
    rank: usize,
    values: &[i64],
    attribute_name: &str,
) -> Result<Vec<usize>, CostError> {
    values
        .iter()
        .map(|value| normalize_index(layer, rank, *value, attribute_name))
        .collect()
}

pub(super) fn normalize_index(
    layer: &LayerSpec,
    rank: usize,
    value: i64,
    attribute_name: &str,
) -> Result<usize, CostError> {
    let normalized = if value < 0 {
        rank as i64 + value
    } else {
        value
    };
    if !(0..rank as i64).contains(&normalized) {
        return Err(shape_inference_error(
            layer,
            format!("{attribute_name} index {value} is out of range for rank-{rank} input"),
        ));
    }
    Ok(normalized as usize)
}
