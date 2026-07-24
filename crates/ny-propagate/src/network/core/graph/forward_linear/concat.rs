// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::{
    compose_forward_relaxation, resolve_input_shape, resolve_upstream_linear_bounds,
    sum_linear_bounds, wrap_forward_linear_error,
};
use crate::bounds::LinearBounds;
use crate::layers::{ConcatLayer, Layer};

pub(super) fn compose_concat_forward(
    node_name: &str,
    layer: &ConcatLayer,
    inputs: &[String],
    output_dim: usize,
    forward_bounds: &HashMap<String, LinearBounds>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    input_dim: usize,
) -> Result<LinearBounds> {
    if inputs.is_empty() {
        return Err(NyError::InvalidSpec(format!(
            "forward-linear bounds: Concat node '{node_name}' has no inputs"
        )));
    }

    let input_shapes = inputs
        .iter()
        .enumerate()
        .map(|(index, input_name)| {
            resolve_input_shape(
                input_name,
                layer.constant_input(index),
                layer.input_shape(index),
                ibp_node_bounds,
                input,
                node_name,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let identity = LinearBounds::identity(output_dim);
    let local_parts = layer
        .propagate_linear_nary(&identity, &input_shapes)
        .map_err(|error| {
            wrap_forward_linear_error(node_name, &Layer::Concat(layer.clone()), error)
        })?;

    let composed_parts = inputs
        .iter()
        .enumerate()
        .map(|(index, input_name)| {
            let upstream = resolve_upstream_linear_bounds(
                input_name,
                layer.constant_input(index),
                forward_bounds,
                input_dim,
                node_name,
            )?;
            compose_forward_relaxation(&local_parts[index], &upstream)
        })
        .collect::<Result<Vec<_>>>()?;

    sum_linear_bounds(&composed_parts)
}
